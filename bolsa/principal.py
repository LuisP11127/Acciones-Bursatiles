"""Orquestación: escaneo diario completo y entrenamiento profundo de la red."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import graficos, reporte
from .cartera import (Cartera, alertas, cargar_posiciones_manuales, estado_tesis, importancia_accion,
                      razones_compra)
from .config import Rutas
from .datos import ProveedorSintetico, ProveedorYahoo
from .indicadores import agregar_indicadores
from .noticias import analizar_noticias
from .puntuacion import (analisis_tecnico, combinar, puntaje_analistas, puntaje_red,
                         recomendacion_texto)
from .red.caracteristicas import caracteristicas_mercado, construir_caracteristicas
from .red.entrenador import EntrenadorRed
from .red.maestro import operaciones_maestro
from .universo import obtener_universo

log = logging.getLogger(__name__)

DIAS_DATOS_VIEJOS = 7  # una acción sin cotizar en una semana no se considera
NOMBRES_INDICES = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "DIA": "Dow Jones", "IWM": "Russell 2000"}


class Mercado:
    """Datos descargados y enriquecidos con indicadores, compartidos por todo el proceso."""

    def __init__(self, cfg: dict, offline: bool, extra: list[str], proveedor=None):
        self.cfg = cfg
        self.offline = offline
        self.indices = [t.upper() for t in cfg["datos"]["indices"]]
        self.universo = obtener_universo(cfg, offline)
        for ticker in extra:
            self.universo.setdefault(ticker, {"nombre": ticker, "sector": "", "industria": "", "indices": []})
        if proveedor is not None:
            self.proveedor = proveedor
        else:
            self.proveedor = ProveedorSintetico(self.universo, self.indices) if offline else ProveedorYahoo()
        tickers = list(self.universo) + self.indices
        log.info("Descargando %d tickers (%s)...", len(tickers), cfg["datos"]["periodo"])
        precios = self.proveedor.precios(tickers, cfg["datos"]["periodo"])
        self.enriquecidos = {t: agregar_indicadores(df) for t, df in precios.items()}
        referencia = next((t for t in ["SPY", *self.indices] if t in self.enriquecidos), None)
        self.mercado = self.enriquecidos.get(referencia) if referencia else None
        if not self.enriquecidos:
            raise RuntimeError("No se pudo descargar ningún precio. Revisa la conexión con Yahoo Finance.")
        ultima = (self.mercado.index[-1] if self.mercado is not None
                  else max(df.index[-1] for df in self.enriquecidos.values()))
        self.fecha = ultima.strftime("%Y-%m-%d")
        limite = ultima - pd.Timedelta(days=DIAS_DATOS_VIEJOS)
        self.acciones = {t: df for t, df in self.enriquecidos.items()
                         if t not in self.indices and df.index[-1] >= limite}
        log.info("Datos al %s: %d acciones vigentes", self.fecha, len(self.acciones))

    def nombre(self, ticker: str, info: dict | None = None) -> str:
        return ((info or {}).get("nombre") or self.universo.get(ticker, {}).get("nombre")
                or NOMBRES_INDICES.get(ticker) or ticker)


# --------------------------------------------------------------------------
# Red neuronal
# --------------------------------------------------------------------------

def _paso_red(entrenador: EntrenadorRed, mercado: Mercado, minutos: float | None,
              max_epocas: int | None = None, paciencia: int | None = None) -> tuple[pd.DataFrame, dict]:
    ds = entrenador.construir_dataset(mercado.acciones, mercado.mercado)
    log.info("Red neuronal: %d ejemplos etiquetados (%.1f%% positivos)", len(ds),
             100 * float(ds.y.mean()) if len(ds) else 0)
    entrenador.evaluar_en_vivo()
    if minutos and minutos > 0:
        entrenador.entrenar(ds, minutos, max_epocas, paciencia)
    predicciones = entrenador.predecir_ultimas()
    if not predicciones.empty:
        predicciones = predicciones[predicciones["fecha"] == pd.Timestamp(mercado.fecha)].reset_index(drop=True)
    return predicciones, entrenador.madurez()


def _probabilidades_historicas(entrenador: EntrenadorRed, df: pd.DataFrame,
                               mercado: pd.DataFrame | None) -> pd.Series | None:
    red = entrenador.red
    if red is None or not red.normalizador.ajustado:
        return None
    feats = construir_caracteristicas(df, caracteristicas_mercado(mercado))
    validas = feats.notna().all(axis=1)
    if not validas.any():
        return None
    prob = red.predecir(red.normalizador.transformar(feats[validas].to_numpy()))
    return pd.Series(prob, index=feats.index[validas])


def _contexto_red(entrenador: EntrenadorRed, mercado: Mercado, predicciones: pd.DataFrame, madurez: dict,
                  tecnicos: dict, rutas: Rutas, analisis: dict | None = None) -> dict:
    tasa_base = entrenador.meta.get("tasa_base") or np.nan
    filas = []
    for p in predicciones.head(15).itertuples():
        info = (analisis or {}).get(p.ticker, {}).get("info")
        filas.append({
            "ticker": p.ticker,
            "nombre": mercado.nombre(p.ticker, info),
            "probabilidad": float(p.probabilidad),
            "lift": float(p.probabilidad) / tasa_base if tasa_base and tasa_base > 0 else np.nan,
            "precio": float(p.precio),
            "tendencia": tecnicos.get(p.ticker, {}).get("tendencia", "—"),
        })
    contexto = {
        "meta": entrenador.meta,
        "madurez": madurez,
        "predicciones": filas,
        "parametros": entrenador.red.n_parametros if entrenador.red else None,
        "ultimas_evaluadas": sorted((p for p in entrenador.predicciones if p.get("resultado") is not None),
                                    key=lambda p: (p["fecha"], p["probabilidad"]), reverse=True)[:15],
    }
    contexto["grafico_aprendizaje"] = None
    if graficos.grafico_aprendizaje(entrenador.meta.get("historial") or [], entrenador.rcfg["madurez"],
                                    rutas.graficos / "red_aprendizaje.png"):
        contexto["grafico_aprendizaje"] = "graficos/red_aprendizaje.png"
    ejemplo = filas[0]["ticker"] if filas else next(iter(mercado.acciones), None)
    contexto["grafico_maestro"] = None
    if ejemplo and ejemplo in entrenador.etiquetas:
        df = mercado.acciones[ejemplo]
        operaciones = operaciones_maestro(df, entrenador.etiquetas[ejemplo])
        graficos.grafico_maestro(df, ejemplo, operaciones, _probabilidades_historicas(entrenador, df, mercado.mercado),
                                 rutas.graficos / "red_maestro.png")
        contexto["grafico_maestro"] = "graficos/red_maestro.png"
        contexto["ticker_maestro"] = ejemplo
    return contexto


# --------------------------------------------------------------------------
# Escaneo diario
# --------------------------------------------------------------------------

def _analisis_completo(ticker: str, tecnico: dict, mercado: Mercado, cfg: dict, prob: float | None,
                       tasa_base: float | None, red_lista: bool) -> dict:
    info = mercado.proveedor.info(ticker)
    nombre = mercado.nombre(ticker, info)
    noticias = analizar_noticias(mercado.proveedor.noticias(ticker, nombre, cfg), cfg)
    analistas = puntaje_analistas(info, tecnico["precio"])
    componentes = {
        "tecnico": tecnico["puntaje_tecnico"],
        "momentum": tecnico["puntaje_momentum"],
        "noticias": noticias["puntaje"],
        "analistas": analistas["puntaje"] if analistas else None,
        "red_neuronal": puntaje_red(prob, tasa_base) if red_lista else None,
    }
    puntaje = combinar(componentes, cfg["pesos"])
    if analistas:
        potencial, base = analistas["potencial"], "precio objetivo de analistas"
    else:
        potencial, base = tecnico["potencial_tecnico"], tecnico["base_potencial"]
    meta = mercado.universo.get(ticker, {})
    return {
        "ticker": ticker,
        "nombre": nombre,
        "sector": info.get("sector") or meta.get("sector"),
        "meta": meta,
        "info": info,
        "tecnico": tecnico,
        "noticias": noticias,
        "analistas": analistas,
        "prob_red": prob,
        "lift_red": prob / tasa_base if prob is not None and tasa_base else None,
        "componentes": componentes,
        "puntaje": puntaje,
        "recomendacion": recomendacion_texto(puntaje),
        "potencial": potencial,
        "base_potencial": base,
    }


def _item_seguimiento(pos: dict, registro: dict | None, mercado: Mercado, rutas: Rutas, fecha: str,
                      prefijo: str, dias_grafico: int) -> dict:
    ticker = pos["ticker"]
    df = mercado.enriquecidos.get(ticker)
    precio = float(df["Close"].iloc[-1]) if df is not None else pos.get("precio_actual")
    pnl = precio / pos["precio_compra"] - 1 if precio else None
    item = {**pos, "precio_actual": precio, "pnl": pnl, "registro": registro,
            "nombre": pos.get("nombre") or mercado.nombre(ticker, (registro or {}).get("info"))}
    if pos.get("acciones") and precio:
        item["resultado_usd"] = (precio - pos["precio_compra"]) * float(pos["acciones"])
    item["dias"] = (pd.Timestamp(fecha) - pd.Timestamp(pos["fecha_compra"])).days if pos.get("fecha_compra") else "—"
    if registro:
        if not item.get("razones"):
            item["razones"], item["razones_auto"] = razones_compra(registro, False), True
        if not item.get("importancia"):
            item["importancia"], item["importancia_auto"] = importancia_accion(registro), True
    item["tesis"] = estado_tesis(registro)
    item["alertas"] = alertas(registro["tecnico"] if registro else None, df, pnl,
                              registro.get("noticias") if registro else None, pos.get("stop_loss"))
    if df is None:
        item["alertas"].insert(0, "⚠️ No se encontraron datos de precio para este ticker")
    else:
        marcas = [{"fecha": pos["fecha_compra"], "precio": pos["precio_compra"], "tipo": "compra"}] \
            if pos.get("fecha_compra") else []
        archivo = f"{prefijo}_{ticker}.png"
        graficos.grafico_accion(df, ticker, rutas.graficos / archivo, item["nombre"], dias=dias_grafico,
                                marcas=marcas, nota=f"Resultado {reporte.pct(pnl)}" if pnl is not None else "")
        item["grafico"] = f"graficos/{archivo}"
    return item


def _rendimiento_pasado(historial: list[dict], mercado: Mercado, fecha: str) -> list[dict]:
    filas = []
    spy = mercado.mercado["Close"] if mercado.mercado is not None else None
    for entrada in reversed(historial):
        if entrada["fecha"] == fecha:
            continue
        rendimientos = []
        for item in entrada["top"][:5]:
            df = mercado.enriquecidos.get(item["ticker"])
            if df is not None and item.get("precio"):
                rendimientos.append(float(df["Close"].iloc[-1]) / item["precio"] - 1)
        if not rendimientos:
            continue
        rend_mercado = None
        if spy is not None:
            previo = spy[spy.index <= pd.Timestamp(entrada["fecha"])]
            if len(previo):
                rend_mercado = float(spy.iloc[-1] / previo.iloc[-1] - 1)
        filas.append({"fecha": entrada["fecha"], "tickers": [i["ticker"] for i in entrada["top"][:5]],
                      "rendimiento": float(np.mean(rendimientos)), "mercado": rend_mercado})
        if len(filas) >= 10:
            break
    return filas


def _cargar_json(ruta: Path, defecto):
    if ruta.exists():
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Archivo dañado: %s", ruta)
    return defecto


def _limpiar_graficos(rutas: Rutas) -> None:
    for png in rutas.graficos.glob("*.png"):
        png.unlink()


def _generado() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def escanear(cfg: dict, salida: str | Path, offline: bool = False, minutos: float | None = None,
             proveedor=None) -> dict:
    rutas = Rutas(salida)
    ecfg = cfg["escaneo"]
    manuales = cargar_posiciones_manuales()
    cartera = Cartera.cargar(rutas.cartera, cfg)
    seguidos = list(dict.fromkeys(cartera.tickers() + [m["ticker"] for m in manuales]))
    mercado = Mercado(cfg, offline, seguidos, proveedor)
    fecha = mercado.fecha

    # 1) Red neuronal: evaluar, entrenar una sesión y predecir
    entrenador = EntrenadorRed(rutas.estado, cfg)
    minutos = cfg["red_neuronal"]["minutos_por_sesion"] if minutos is None else minutos
    predicciones, madurez = _paso_red(entrenador, mercado, minutos)
    red_lista = madurez["lista"]
    tasa_base = entrenador.meta.get("tasa_base")
    probabilidades = dict(zip(predicciones["ticker"], predicciones["probabilidad"]))
    entrenador.registrar_predicciones(predicciones)

    # 2) Fase 1: análisis técnico de todo el universo
    tecnicos: dict[str, dict] = {}
    for ticker, df in mercado.acciones.items():
        u = df.iloc[-1]
        seguido = ticker in seguidos
        if not seguido and (u["Close"] < ecfg["precio_minimo"]
                            or pd.isna(u["VolMA20"]) or u["VolMA20"] < ecfg["volumen_medio_minimo"]):
            continue
        tec = analisis_tecnico(df, mercado.mercado)
        if tec:
            tecnicos[ticker] = tec
    for ticker in seguidos:  # posiciones con datos algo viejos también se analizan
        if ticker not in tecnicos and ticker in mercado.enriquecidos and ticker not in mercado.indices:
            tec = analisis_tecnico(mercado.enriquecidos[ticker], mercado.mercado)
            if tec:
                tecnicos[ticker] = tec

    def preliminar(t: str) -> float:
        componentes = {"tecnico": tecnicos[t]["puntaje_tecnico"], "momentum": tecnicos[t]["puntaje_momentum"]}
        if red_lista:
            componentes["red_neuronal"] = puntaje_red(probabilidades.get(t), tasa_base)
        return combinar(componentes, cfg["pesos"])

    candidatos = sorted(tecnicos, key=preliminar, reverse=True)[: int(ecfg["candidatos_fase2"])]
    log.info("Fase 1: %d acciones analizadas, %d candidatos", len(tecnicos), len(candidatos))

    # 3) Fase 2: noticias, analistas y puntaje final
    analisis: dict[str, dict] = {}
    for ticker in dict.fromkeys(candidatos + [t for t in seguidos if t in tecnicos]):
        analisis[ticker] = _analisis_completo(ticker, tecnicos[ticker], mercado, cfg, probabilidades.get(ticker),
                                              tasa_base, red_lista)
    ranking = sorted((analisis[t] for t in candidatos), key=lambda r: r["puntaje"], reverse=True)
    top = ranking[: int(ecfg["top_oportunidades"])]

    # 4) Cartera simulada: ventas, compras y seguimiento
    eventos = cartera.revisar_ventas(fecha, mercado.enriquecidos)
    eventos += cartera.comprar(fecha, ranking, red_lista)
    cartera.actualizar_seguimiento(fecha, analisis)
    cartera.registrar_valor(fecha)

    # 5) Gráficos
    _limpiar_graficos(rutas)
    dias = int(ecfg["dias_grafico"])
    for r in top:
        archivo = f"{r['ticker']}.png"
        graficos.grafico_accion(mercado.enriquecidos[r["ticker"]], r["ticker"], rutas.graficos / archivo,
                                r["nombre"], dias=dias, nota=f"Puntaje {r['puntaje']:.0f} · {r['recomendacion']}")
        r["grafico"] = f"graficos/{archivo}"
    indices = []
    for ticker in mercado.indices:
        df = mercado.enriquecidos.get(ticker)
        tec = analisis_tecnico(df, mercado.mercado) if df is not None else None
        if not tec:
            continue
        archivo = f"indice_{ticker}.png"
        graficos.grafico_accion(df, ticker, rutas.graficos / archivo, mercado.nombre(ticker), dias=dias)
        indices.append({"ticker": ticker, "nombre": mercado.nombre(ticker), **tec, "grafico": f"graficos/{archivo}"})
    seguimiento = [_item_seguimiento(p, analisis.get(p["ticker"]), mercado, rutas, fecha, "seguimiento", dias)
                   for p in cartera.posiciones]
    seguimiento_manual = [_item_seguimiento(p, analisis.get(p["ticker"]), mercado, rutas, fecha, "mia", dias)
                          for p in manuales]
    grafico_cartera = graficos.grafico_cartera(
        cartera.historial, mercado.mercado["Close"] if mercado.mercado is not None else None,
        rutas.graficos / "cartera.png")

    # 6) Historial de recomendaciones
    historial = _cargar_json(rutas.historial_escaneos, [])
    historial = [h for h in historial if h["fecha"] != fecha] + [{
        "fecha": fecha,
        "top": [{"ticker": r["ticker"], "precio": r["tecnico"]["precio"], "puntaje": r["puntaje"]} for r in top],
    }]
    historial = historial[-400:]

    # 7) Reportes
    ctx = {
        "cfg": cfg,
        "fecha_datos": fecha,
        "generado": _generado(),
        "sintetico": mercado.proveedor.sintetico,
        "n_analizadas": len(tecnicos),
        "n_fase2": len(analisis),
        "indices": indices,
        "top": top,
        "red": _contexto_red(entrenador, mercado, predicciones, madurez, tecnicos, rutas, analisis),
        "cartera": {
            "resumen": cartera.resumen(),
            "capital_inicial": cartera.datos["capital_inicial"],
            "posiciones": seguimiento,
            "cerradas": cartera.cerradas,
            "eventos": eventos,
            "grafico": "graficos/cartera.png" if grafico_cartera else None,
        },
        "manuales": seguimiento_manual,
        "rendimiento_pasado": _rendimiento_pasado(historial, mercado, fecha),
    }
    reporte.escribir_panel(rutas.base / "README.md", ctx)
    reporte.escribir_oportunidades(rutas.base / "oportunidades.md", ctx)
    reporte.escribir_seguimiento(rutas.base / "seguimiento.md", ctx)
    reporte.escribir_red(rutas.base / "red_neuronal.md", ctx)
    _exportar_datos(rutas, tecnicos, analisis, top, mercado, probabilidades)

    # 8) Guardar estado para la próxima ejecución
    cartera.guardar(rutas.cartera)
    entrenador.guardar()
    rutas.historial_escaneos.write_text(json.dumps(historial, ensure_ascii=False), encoding="utf-8")
    log.info("Listo: %s", rutas.base / "README.md")
    return ctx


def _exportar_datos(rutas: Rutas, tecnicos: dict, analisis: dict, top: list[dict], mercado: Mercado,
                    probabilidades: dict) -> None:
    filas = []
    for ticker, t in tecnicos.items():
        a = analisis.get(ticker, {})
        filas.append({
            "ticker": ticker,
            "nombre": a.get("nombre") or mercado.nombre(ticker),
            "sector": a.get("sector") or mercado.universo.get(ticker, {}).get("sector"),
            "precio": t["precio"],
            "cambio_1d": round(t["cambio_1d"], 4),
            "ret_1m": round(t["ret_21"], 4),
            "ret_3m": round(t["ret_63"], 4),
            "rsi": round(t["rsi"], 1),
            "ma7": round(t["ma7"], 4),
            "ma25": round(t["ma25"], 4),
            "ma99": round(t["ma99"], 4),
            "tendencia": t["tendencia"],
            "puntaje_tecnico": t["puntaje_tecnico"],
            "puntaje_momentum": t["puntaje_momentum"],
            "puntaje_noticias": (a.get("componentes") or {}).get("noticias"),
            "puntaje_analistas": (a.get("componentes") or {}).get("analistas"),
            "prob_red": round(float(probabilidades[ticker]), 5) if ticker in probabilidades else None,
            "puntaje_total": a.get("puntaje"),
        })
    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla = tabla.sort_values(["puntaje_total", "puntaje_tecnico"], ascending=False, na_position="last")
    tabla.to_csv(rutas.datos / "escaneo.csv", index=False)
    resumen = [{
        "ticker": r["ticker"], "nombre": r["nombre"], "sector": r["sector"], "puntaje": r["puntaje"],
        "recomendacion": r["recomendacion"], "precio": r["tecnico"]["precio"], "potencial": r["potencial"],
        "componentes": r["componentes"], "senales_positivas": r["tecnico"]["senales_positivas"],
        "senales_negativas": r["tecnico"]["senales_negativas"],
        "noticias": [n.a_dict() for n in (r.get("noticias") or {}).get("noticias", [])],
    } for r in top]
    (rutas.datos / "oportunidades.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=str),
                                                    encoding="utf-8")


# --------------------------------------------------------------------------
# Entrenamiento profundo (semanal)
# --------------------------------------------------------------------------

def entrenar(cfg: dict, salida: str | Path, offline: bool = False, minutos: float | None = None,
             proveedor=None) -> dict:
    rutas = Rutas(salida)
    mercado = Mercado(cfg, offline, [], proveedor)
    entrenador = EntrenadorRed(rutas.estado, cfg)
    minutos = cfg["red_neuronal"]["minutos_sesion_profunda"] if minutos is None else minutos
    # Sesión larga: más épocas y más paciencia antes de la parada temprana
    predicciones, madurez = _paso_red(entrenador, mercado, minutos, max_epocas=200,
                                      paciencia=2 * int(cfg["red_neuronal"]["paciencia"]))
    tecnicos = {t: tec for t in predicciones["ticker"].head(15)
                if (tec := analisis_tecnico(mercado.acciones[t], mercado.mercado))}
    for png in ("red_aprendizaje.png", "red_maestro.png"):
        (rutas.graficos / png).unlink(missing_ok=True)
    ctx = {
        "cfg": cfg,
        "fecha_datos": mercado.fecha,
        "generado": _generado(),
        "sintetico": mercado.proveedor.sintetico,
        "red": _contexto_red(entrenador, mercado, predicciones, madurez, tecnicos, rutas),
    }
    reporte.escribir_red(rutas.base / "red_neuronal.md", ctx)
    entrenador.guardar()
    return ctx
