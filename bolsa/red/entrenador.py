"""Entrenamiento incremental, evaluación en vivo y madurez de la red neuronal.

Ciclo diario (lo ejecuta GitHub Actions):
  1. Con los precios nuevos, el maestro etiqueta los días cuyo futuro ya se conoce.
  2. La red entrena una sesión corta continuando desde los pesos guardados.
  3. La red puntúa el último día de cada acción y se registran sus 10 favoritas.
  4. Pasados `horizonte` días, esas predicciones se comparan con lo que pasó
     realmente (evaluación en vivo, sin trampas).
  5. Cuando las métricas superan los mínimos de `madurez`, la red se considera
     lista y sus recomendaciones entran en el puntaje final.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .caracteristicas import NOMBRES, caracteristicas_mercado, construir_caracteristicas
from .maestro import etiquetas_maestro
from .neuronal import RedNeuronal, entropia_cruzada

log = logging.getLogger(__name__)

MAX_HISTORIAL = 400
MAX_PREDICCIONES = 5000


@dataclass
class ConjuntoDatos:
    X: np.ndarray
    y: np.ndarray
    ganancia: np.ndarray
    fechas: np.ndarray
    tickers: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def auc_roc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rangos = pd.Series(p).rank(method="average").to_numpy()
    return float((rangos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metricas(y: np.ndarray, p: np.ndarray, ganancia: np.ndarray, fechas: np.ndarray,
             fraccion_top: float = 0.05, k: int = 10) -> dict:
    tasa = float(y.mean()) if len(y) else float("nan")
    top = np.argsort(-p)[: max(1, int(len(y) * fraccion_top))]
    precision_top = float(y[top].mean())
    tabla = pd.DataFrame({"f": fechas, "p": p, "y": y})
    top_diario = tabla.sort_values(["f", "p"], ascending=[True, False]).groupby("f", sort=False).head(k)
    p_seguro = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": auc_roc(y, p),
        "tasa_base": tasa,
        "precision_top": precision_top,
        "lift": precision_top / tasa if tasa > 0 else float("nan"),
        "precision_top10_diaria": float(top_diario["y"].mean()),
        "ganancia_top": float(np.mean(ganancia[top])),
        "ganancia_media": float(np.mean(ganancia)),
        "logloss": float(-np.mean(y * np.log(p_seguro) + (1 - y) * np.log(1 - p_seguro))),
    }


def _a_json(valor):
    if isinstance(valor, (np.floating, float)):
        return None if math.isnan(float(valor)) else round(float(valor), 5)
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, dict):
        return {k: _a_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_a_json(v) for v in valor]
    return valor


class EntrenadorRed:
    def __init__(self, dir_estado: str | Path, cfg: dict):
        self.cfg = cfg
        self.rcfg = cfg["red_neuronal"]
        dir_estado = Path(dir_estado)
        self.ruta_modelo = dir_estado / "red_neuronal.npz"
        self.ruta_meta = dir_estado / "red_neuronal.json"
        self.ruta_predicciones = dir_estado / "predicciones_red.json"
        self.meta = self._cargar_json(self.ruta_meta, None) or self._meta_nueva()
        self.predicciones: list[dict] = self._cargar_json(self.ruta_predicciones, [])
        self.red: RedNeuronal | None = None
        maestro_actual = self._meta_nueva()["maestro"]
        if self.meta.get("maestro") != maestro_actual:
            # Cambió la definición de "buen punto de compra": lo aprendido ya no sirve
            log.warning("Cambió la configuración del maestro: la red empieza a aprender de nuevo")
            self.meta = self._meta_nueva()
            self.predicciones = []
        elif self.ruta_modelo.exists():
            try:
                red = RedNeuronal.cargar(self.ruta_modelo)
                if red.n_entradas == len(NOMBRES) and red.capas == list(self.rcfg["capas"]):
                    self.red = red
                else:
                    log.warning("La arquitectura cambió: la red empieza a aprender de nuevo")
                    self.meta = self._meta_nueva()
            except Exception as exc:
                log.warning("No se pudo cargar la red (%s); se crea una nueva", exc)
                self.meta = self._meta_nueva()
        self._etiquetas: dict[str, pd.DataFrame] = {}
        self._ultimas: dict[str, tuple[pd.Timestamp, float, np.ndarray]] = {}

    @staticmethod
    def _cargar_json(ruta: Path, defecto):
        if ruta.exists():
            try:
                return json.loads(ruta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Archivo dañado: %s", ruta)
        return defecto

    def _meta_nueva(self) -> dict:
        return {
            "creada": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "caracteristicas": NOMBRES,
            "capas": list(self.rcfg["capas"]),
            "maestro": {k: self.rcfg[k] for k in ("horizonte", "ganancia_minima", "caida_maxima")},
            "sesiones": 0,
            "ejemplos_vistos": 0,
            "epocas_totales": 0,
            "ultimo_entrenamiento": None,
            "tasa_base": None,
            "historial": [],
            "vivo": {},
        }

    # ------------------------------------------------------------------
    def construir_dataset(self, enriquecidos: dict[str, pd.DataFrame],
                          mercado: pd.DataFrame | None) -> ConjuntoDatos:
        mercado_f = caracteristicas_mercado(mercado)
        partes_X, partes_y, partes_g, partes_f, partes_t = [], [], [], [], []
        self._etiquetas.clear()
        self._ultimas.clear()
        for ticker, df in enriquecidos.items():
            feats = construir_caracteristicas(df, mercado_f)
            et = etiquetas_maestro(df, int(self.rcfg["horizonte"]), float(self.rcfg["ganancia_minima"]),
                                   float(self.rcfg["caida_maxima"]))
            self._etiquetas[ticker] = et
            ultima = feats.iloc[-1].to_numpy()
            if not np.isnan(ultima).any():
                self._ultimas[ticker] = (df.index[-1], float(df["Close"].iloc[-1]), ultima)
            validas = feats.notna().all(axis=1).to_numpy() & et["etiqueta"].notna().to_numpy()
            if validas.sum() == 0:
                continue
            partes_X.append(feats.to_numpy()[validas])
            partes_y.append(et["etiqueta"].to_numpy()[validas])
            partes_g.append(et["ganancia"].to_numpy()[validas])
            partes_f.append(df.index.to_numpy()[validas].astype("datetime64[D]"))
            partes_t.append(np.full(validas.sum(), ticker, dtype=object))
        if not partes_X:
            vacio = np.zeros((0, len(NOMBRES)), dtype=np.float32)
            return ConjuntoDatos(vacio, np.zeros(0, np.float32), np.zeros(0), np.zeros(0, "datetime64[D]"),
                                 np.zeros(0, object))
        return ConjuntoDatos(
            np.vstack(partes_X).astype(np.float32),
            np.concatenate(partes_y).astype(np.float32),
            np.concatenate(partes_g),
            np.concatenate(partes_f),
            np.concatenate(partes_t),
        )

    # ------------------------------------------------------------------
    def entrenar(self, ds: ConjuntoDatos, minutos: float, max_epocas: int | None = None,
                 paciencia: int | None = None) -> dict | None:
        r = self.rcfg
        if len(ds) < 2000 or ds.y.sum() < 30:
            log.warning("Muy pocos ejemplos para entrenar (%d, %d positivos)", len(ds), int(ds.y.sum()))
            return None
        fechas = np.unique(ds.fechas)
        horizonte = int(r["horizonte"])
        dias_val = int(r["dias_validacion"])
        if len(fechas) < dias_val + horizonte + 100:
            dias_val = max(20, len(fechas) // 5)
        if len(fechas) < dias_val + horizonte + 40:
            log.warning("Historial demasiado corto para entrenar (%d fechas); usa un periodo más largo", len(fechas))
            return None
        inicio_val = fechas[-dias_val]
        fin_entreno = fechas[-(dias_val + horizonte)]  # hueco para que las etiquetas no se solapen
        en_entreno = ds.fechas < fin_entreno
        en_val = ds.fechas >= inicio_val

        if self.red is None:
            self.red = RedNeuronal(len(NOMBRES), r["capas"], semilla=42)
            self.red.normalizador.ajustar(ds.X[en_entreno])
            tasa_pos = float(np.clip(ds.y[en_entreno].mean(), 1e-3, 1 - 1e-3))
            self.red.b[-1][:] = math.log(tasa_pos / (1 - tasa_pos))
        norm = self.red.normalizador
        X_tr, y_tr = norm.transformar(ds.X[en_entreno]), ds.y[en_entreno]
        X_va, y_va = norm.transformar(ds.X[en_val]), ds.y[en_val]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            log.warning("Faltan ejemplos positivos o negativos para entrenar y validar")
            return None

        sesion = int(self.meta["sesiones"]) + 1
        tasa = float(r["tasa_aprendizaje"]) / math.sqrt(1 + sesion / 50)
        rng = np.random.default_rng(1000 + sesion)
        perdida_inicial = mejor_perdida = entropia_cruzada(self.red.logits(X_va), y_va)
        mejores_pesos = self.red.copiar_pesos()
        sin_mejora = epocas = ejemplos = 0
        inicio = time.monotonic()
        limite = minutos * 60
        max_epocas = int(max_epocas or r["max_epocas_por_sesion"])
        lote = int(r["batch"])
        muestras = min(len(X_tr), int(r["muestras_por_epoca"]))
        while epocas < max_epocas and time.monotonic() - inicio < limite:
            indices = rng.permutation(len(X_tr))[:muestras]
            for i in range(0, len(indices), lote):
                b = indices[i:i + lote]
                self.red.entrenar_lote(X_tr[b], y_tr[b], tasa=tasa, l2=float(r["l2"]))
                ejemplos += len(b)
                if time.monotonic() - inicio > limite:
                    break
            epocas += 1
            perdida = entropia_cruzada(self.red.logits(X_va), y_va)
            log.info("Sesión %d, época %d: pérdida validación %.5f", sesion, epocas, perdida)
            if perdida < mejor_perdida - 1e-5:
                mejor_perdida, mejores_pesos, sin_mejora = perdida, self.red.copiar_pesos(), 0
            else:
                sin_mejora += 1
                if sin_mejora >= int(paciencia or r["paciencia"]):
                    break
        self.red.restaurar_pesos(mejores_pesos)

        m = metricas(y_va, self.red.predecir(X_va), ds.ganancia[en_val], ds.fechas[en_val])
        # Referencia simple: ordenar por volatilidad (en cualquiera de los dos sentidos)
        auc_vol = auc_roc(y_va, ds.X[en_val][:, NOMBRES.index("volatilidad_21")])
        m["auc_referencia"] = max(auc_vol, 1 - auc_vol) if not math.isnan(auc_vol) else float("nan")
        registro = {
            "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "sesion": sesion,
            "epocas": epocas,
            "ejemplos": ejemplos,
            "minutos": round((time.monotonic() - inicio) / 60, 2),
            "n_entrenamiento": int(en_entreno.sum()),
            "n_validacion": int(en_val.sum()),
            "perdida_inicial": perdida_inicial,
            "perdida_final": mejor_perdida,
            **m,
        }
        self.meta["sesiones"] = sesion
        self.meta["ejemplos_vistos"] = int(self.meta["ejemplos_vistos"]) + ejemplos
        self.meta["epocas_totales"] = int(self.meta["epocas_totales"]) + epocas
        self.meta["ultimo_entrenamiento"] = registro["fecha"]
        self.meta["tasa_base"] = m["tasa_base"]
        self.meta["historial"] = (self.meta["historial"] + [_a_json(registro)])[-MAX_HISTORIAL:]
        log.info("Sesión %d: AUC %.3f, lift %.2f, %d ejemplos", sesion, m["auc"], m["lift"], ejemplos)
        return registro

    # ------------------------------------------------------------------
    def predecir_ultimas(self) -> pd.DataFrame:
        columnas = ["ticker", "probabilidad", "fecha", "precio"]
        if self.red is None or not self.red.normalizador.ajustado or not self._ultimas:
            return pd.DataFrame(columns=columnas)
        tickers = list(self._ultimas)
        X = np.vstack([self._ultimas[t][2] for t in tickers]).astype(np.float32)
        prob = self.red.predecir(self.red.normalizador.transformar(X))
        return pd.DataFrame({
            "ticker": tickers,
            "probabilidad": prob,
            "fecha": [self._ultimas[t][0] for t in tickers],
            "precio": [self._ultimas[t][1] for t in tickers],
        }).sort_values("probabilidad", ascending=False).reset_index(drop=True)

    def registrar_predicciones(self, predicciones: pd.DataFrame) -> list[dict]:
        if predicciones.empty:
            return []
        fecha = predicciones["fecha"].max()
        actuales = predicciones[predicciones["fecha"] == fecha].head(int(self.rcfg["top_predicciones"]))
        fecha_txt = pd.Timestamp(fecha).strftime("%Y-%m-%d")
        nuevas = [
            {"fecha": fecha_txt, "ticker": f.ticker, "probabilidad": round(float(f.probabilidad), 5),
             "precio": round(float(f.precio), 4), "sesion": int(self.meta["sesiones"])}
            for f in actuales.itertuples()
        ]
        self.predicciones = [p for p in self.predicciones if p["fecha"] != fecha_txt] + nuevas
        self.predicciones = self.predicciones[-MAX_PREDICCIONES:]
        return nuevas

    def _tasa_base_en(self, fecha: pd.Timestamp) -> float | None:
        valores = []
        for et in self._etiquetas.values():
            if fecha in et.index:
                v = et.at[fecha, "etiqueta"]
                if not pd.isna(v):
                    valores.append(float(v))
        return float(np.mean(valores)) if valores else None

    def evaluar_en_vivo(self) -> dict:
        """Compara las predicciones pasadas con lo que ocurrió realmente."""
        tasas: dict[pd.Timestamp, float | None] = {}
        for p in self.predicciones:
            if p.get("resultado") is not None:
                continue
            et = self._etiquetas.get(p["ticker"])
            fecha = pd.Timestamp(p["fecha"])
            if et is None or fecha not in et.index or pd.isna(et.at[fecha, "etiqueta"]):
                continue
            p["resultado"] = int(et.at[fecha, "etiqueta"])
            p["ganancia_max"] = round(float(et.at[fecha, "ganancia"]), 4)
            p["caida"] = round(float(et.at[fecha, "caida"]), 4)
            if fecha not in tasas:
                tasas[fecha] = self._tasa_base_en(fecha)
            p["tasa_base"] = None if tasas[fecha] is None else round(tasas[fecha], 4)
        evaluadas = [p for p in self.predicciones if p.get("resultado") is not None]
        if not evaluadas:
            self.meta["vivo"] = {"evaluadas": 0}
            return self.meta["vivo"]
        precision = float(np.mean([p["resultado"] for p in evaluadas]))
        bases = [p["tasa_base"] for p in evaluadas if p.get("tasa_base") is not None]
        base = float(np.mean(bases)) if bases else None
        self.meta["vivo"] = _a_json({
            "evaluadas": len(evaluadas),
            "aciertos": int(sum(p["resultado"] for p in evaluadas)),
            "precision": precision,
            "tasa_base": base,
            "ventaja": precision / base if base else None,
            "ganancia_max_media": float(np.mean([p["ganancia_max"] for p in evaluadas])),
            "pendientes": sum(1 for p in self.predicciones if p.get("resultado") is None),
        })
        return self.meta["vivo"]

    # ------------------------------------------------------------------
    def madurez(self) -> dict:
        req = self.rcfg["madurez"]
        recientes = self.meta["historial"][-5:]

        def media(clave):
            valores = [h[clave] for h in recientes if h.get(clave) is not None]
            return float(np.mean(valores)) if valores else None

        auc, lift = media("auc"), media("lift")
        ventajas = [h["auc"] - h["auc_referencia"] for h in recientes
                    if h.get("auc") is not None and h.get("auc_referencia") is not None]
        ventaja_ref = float(np.mean(ventajas)) if ventajas else None
        vivo = self.meta.get("vivo") or {}
        evaluadas = int(vivo.get("evaluadas") or 0)
        ventaja = vivo.get("ventaja")

        def progreso(actual, requerido, base=0.0):
            if actual is None:
                return 0.0
            return float(np.clip((actual - base) / (requerido - base), 0, 1)) if requerido > base else 1.0

        criterios = [
            ("Sesiones de entrenamiento", self.meta["sesiones"], req["sesiones_minimas"],
             progreso(self.meta["sesiones"], req["sesiones_minimas"])),
            ("AUC en validación (media últimas 5 sesiones)", auc, req["auc_minimo"],
             progreso(auc, req["auc_minimo"], 0.5)),
            ("Lift del 5% mejor valorado (validación)", lift, req["lift_minimo"],
             progreso(lift, req["lift_minimo"], 1.0)),
            ("Ventaja de AUC sobre la referencia simple (volatilidad)", ventaja_ref,
             req["ventaja_sobre_referencia"], progreso(ventaja_ref, req["ventaja_sobre_referencia"])),
            ("Predicciones evaluadas en vivo", evaluadas, req["predicciones_evaluadas_minimas"],
             progreso(evaluadas, req["predicciones_evaluadas_minimas"])),
            ("Ventaja en vivo (precisión / tasa base)", ventaja, req["ventaja_vivo_minima"],
             progreso(ventaja, req["ventaja_vivo_minima"], 1.0)),
        ]
        filas = [
            {"criterio": nombre, "actual": actual, "requerido": requerido,
             "cumple": actual is not None and actual >= requerido, "progreso": prog}
            for nombre, actual, requerido, prog in criterios
        ]
        total = float(np.mean([f["progreso"] for f in filas]))
        lista = all(f["cumple"] for f in filas)
        if lista:
            nivel = "✅ Lista"
        elif total >= 0.6:
            nivel = "🧪 Casi lista"
        elif total >= 0.25:
            nivel = "📚 En entrenamiento"
        else:
            nivel = "🐣 Aprendiz"
        return {"lista": lista, "nivel": nivel, "progreso": total, "criterios": filas}

    def guardar(self) -> None:
        self.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
        if self.red is not None:
            self.red.guardar(self.ruta_modelo)
        self.ruta_meta.write_text(json.dumps(_a_json(self.meta), ensure_ascii=False, indent=1), encoding="utf-8")
        self.ruta_predicciones.write_text(json.dumps(_a_json(self.predicciones), ensure_ascii=False),
                                          encoding="utf-8")

    @property
    def etiquetas(self) -> dict[str, pd.DataFrame]:
        return self._etiquetas
