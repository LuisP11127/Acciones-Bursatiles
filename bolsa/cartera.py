"""Seguimiento de acciones compradas.

Hay dos tipos de posiciones:
- automáticas: la cartera simulada compra las mejores oportunidades y las
  vende con reglas fijas (stop-loss, objetivo, ruptura de tendencia). Cada
  compra guarda sus razones y por qué la empresa es importante.
- manuales: tus compras reales, declaradas en config/mis_acciones.yaml.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from .config import ruta_proyecto
from .indicadores import cruce_reciente

log = logging.getLogger(__name__)

SECTORES_ES = {
    "information technology": "Tecnología", "technology": "Tecnología",
    "health care": "Salud", "healthcare": "Salud",
    "financials": "Finanzas", "financial services": "Finanzas",
    "consumer discretionary": "Consumo discrecional", "consumer cyclical": "Consumo discrecional",
    "consumer staples": "Consumo básico", "consumer defensive": "Consumo básico",
    "communication services": "Comunicaciones", "industrials": "Industria",
    "energy": "Energía", "utilities": "Servicios públicos", "real estate": "Inmobiliario",
    "materials": "Materiales", "basic materials": "Materiales",
}
MAX_BITACORA = 60


def sector_es(sector: str | None) -> str:
    if not sector:
        return "sector no especificado"
    return SECTORES_ES.get(str(sector).strip().lower(), str(sector))


def formato_dinero(valor: float | None) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    return f"${valor:,.2f}"


def formato_capitalizacion(valor: float | None) -> str:
    if not valor:
        return "desconocida"
    if valor >= 1e12:
        return f"${valor / 1e12:.2f} billones"
    if valor >= 1e9:
        return f"${valor / 1e9:,.0f} mil millones"
    return f"${valor / 1e6:,.0f} millones"


def categoria_capitalizacion(valor: float | None) -> str:
    if not valor:
        return ""
    if valor >= 200e9:
        return "megacapitalización"
    if valor >= 10e9:
        return "gran capitalización"
    if valor >= 2e9:
        return "mediana capitalización"
    return "pequeña capitalización"


def primera_frase(texto: str | None, maximo: int = 220) -> str:
    if not texto:
        return ""
    frase = re.split(r"(?<=[.!?])\s", texto.strip(), maxsplit=1)[0]
    return frase if len(frase) <= maximo else frase[: maximo - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# Razones e importancia
# --------------------------------------------------------------------------

def razones_compra(registro: dict, red_lista: bool) -> list[str]:
    """Explica por qué se compra, a partir del análisis completo de la acción."""
    tec = registro["tecnico"]
    comp = registro["componentes"]
    partes = [f"{nombre} {valor:.0f}" for nombre, valor in (
        ("técnico", comp.get("tecnico")), ("momentum", comp.get("momentum")),
        ("noticias", comp.get("noticias")), ("analistas", comp.get("analistas")),
        ("red neuronal", comp.get("red_neuronal"))) if valor is not None]
    razones = [f"Puntaje total {registro['puntaje']:.0f}/100 ({registro['recomendacion']}): {', '.join(partes)}"]
    razones += tec["senales_positivas"][:5]
    noticias = registro.get("noticias") or {}
    if noticias.get("sentimiento") is not None:
        razones.append(
            f"Sentimiento de noticias {noticias['etiqueta'].lower()} ({noticias['sentimiento']:+.2f}) "
            f"en {len(noticias['noticias'])} titulares recientes "
            f"({noticias['positivas']} positivos, {noticias['negativas']} negativos)"
        )
        mejor = max(noticias["noticias"], key=lambda n: n.sentimiento, default=None)
        if mejor is not None and mejor.sentimiento > 0.2:
            razones.append(f"Titular destacado: “{mejor.titulo}” ({mejor.fuente})")
    analistas = registro.get("analistas")
    if analistas:
        texto = (f"Analistas: precio objetivo medio {formato_dinero(analistas['objetivo'])} "
                 f"({analistas['potencial'] * 100:+.1f}%) según {analistas['num_analistas']} opiniones")
        if analistas.get("recomendacion"):
            texto += f"; consenso {analistas['recomendacion']:.1f} (1 = compra fuerte, 5 = venta)"
        razones.append(texto)
    if red_lista and registro.get("prob_red") is not None:
        razones.append(f"La red neuronal estima {registro['prob_red'] * 100:.1f}% de probabilidad de que sea "
                       f"un buen punto de compra ({registro.get('lift_red', 0):.1f}× la media del mercado)")
    return razones


def importancia_accion(registro: dict, sectores_cartera: list[str] | None = None) -> list[str]:
    """Explica por qué la empresa es importante (para el mercado y para la cartera)."""
    info = registro.get("info") or {}
    meta = registro.get("meta") or {}
    sector = info.get("sector") or meta.get("sector") or registro.get("sector")
    industria = info.get("industria") or meta.get("industria")
    capitalizacion = info.get("capitalizacion")
    puntos = []
    descripcion = f"Empresa de {sector_es(sector)}"
    if industria:
        descripcion += f" ({industria})"
    if capitalizacion:
        descripcion += (f" con capitalización de {formato_capitalizacion(capitalizacion)}"
                        f" ({categoria_capitalizacion(capitalizacion)})")
    puntos.append(descripcion)
    indices = meta.get("indices") or []
    if indices:
        puntos.append(f"Forma parte del {' y del '.join(dict.fromkeys(indices))}: una de las empresas de referencia "
                      "de la bolsa de EE.UU.")
    if sectores_cartera is not None:
        iguales = sum(1 for s in sectores_cartera if sector_es(s) == sector_es(sector))
        if iguales == 0:
            puntos.append(f"Aporta diversificación: es la única posición de {sector_es(sector)} en la cartera")
        else:
            puntos.append(f"Refuerza la exposición a {sector_es(sector)} ({iguales + 1} posiciones en el sector)")
    puntos.append(f"Potencial estimado de subida: {registro['potencial'] * 100:+.1f}% ({registro['base_potencial']})")
    noticias = (registro.get("noticias") or {}).get("noticias") or []
    positivas = [n for n in noticias if n.sentimiento >= 0.3]
    if positivas:
        puntos.append(f"Catalizador reciente: “{positivas[0].titulo}”")
    negocio = primera_frase(info.get("resumen"))
    if negocio:
        puntos.append(f"Negocio: {negocio}")
    return puntos


def alertas(tecnico: dict | None, df: pd.DataFrame | None, pnl: float | None,
            noticias: dict | None, stop: float | None = None) -> list[str]:
    avisos = []
    if tecnico:
        if tecnico["precio"] < tecnico["ma99"]:
            avisos.append("⚠️ El precio está por debajo de la MA(99): la tendencia de fondo se debilitó")
        if tecnico["rsi"] > 75:
            avisos.append(f"⚠️ Sobrecompra: RSI {tecnico['rsi']:.0f}")
        if stop and tecnico["precio"] <= stop * 1.03:
            avisos.append(f"⚠️ A menos de 3% del stop-loss ({formato_dinero(stop)})")
    if df is not None and cruce_reciente(df["MA7"], df["MA25"], 3) < 0:
        avisos.append("⚠️ Cruce bajista reciente de la MA(7) bajo la MA(25)")
    if noticias and noticias.get("sentimiento") is not None and noticias["sentimiento"] <= -0.15:
        avisos.append(f"⚠️ Noticias negativas recientes (sentimiento {noticias['sentimiento']:+.2f})")
    if pnl is not None and pnl >= 0.20:
        avisos.append(f"💰 Ganancia de {pnl * 100:+.1f}%: considera asegurar parte del beneficio")
    if pnl is not None and pnl <= -0.10:
        avisos.append(f"🔻 Pérdida de {pnl * 100:.1f}%: revisa si la tesis de compra sigue en pie")
    return avisos


def estado_tesis(registro: dict | None) -> str:
    if not registro:
        return "Sin datos"
    tendencia = registro["tecnico"]["tendencia"]
    if tendencia == "Bajista":
        return "🔴 Rota: la tendencia pasó a bajista"
    if registro["puntaje"] >= 50 and tendencia in ("Alcista", "Alcista fuerte"):
        return "🟢 Vigente: la tendencia y el puntaje siguen respaldando la compra"
    return "🟡 Debilitada: vigilar de cerca"


# --------------------------------------------------------------------------
# Cartera simulada
# --------------------------------------------------------------------------

class Cartera:
    def __init__(self, datos: dict, cfg: dict):
        self.cfg = cfg["cartera"]
        self.datos = datos

    @classmethod
    def cargar(cls, ruta: Path, cfg: dict) -> "Cartera":
        if ruta.exists():
            try:
                return cls(json.loads(ruta.read_text(encoding="utf-8")), cfg)
            except json.JSONDecodeError:
                log.warning("cartera.json dañado; se crea una cartera nueva")
        capital = float(cfg["cartera"]["capital_inicial"])
        return cls({
            "creada": date.today().isoformat(),
            "capital_inicial": capital,
            "efectivo": capital,
            "posiciones": [],
            "cerradas": [],
            "historial": [],
        }, cfg)

    def guardar(self, ruta: Path) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(self.datos, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    @property
    def posiciones(self) -> list[dict]:
        return self.datos["posiciones"]

    @property
    def cerradas(self) -> list[dict]:
        return self.datos["cerradas"]

    @property
    def historial(self) -> list[dict]:
        return self.datos["historial"]

    def tickers(self) -> list[str]:
        return [p["ticker"] for p in self.posiciones]

    def valor_total(self) -> float:
        return self.datos["efectivo"] + sum(p["acciones"] * p.get("precio_actual", p["precio_compra"])
                                            for p in self.posiciones)

    # ------------------------------------------------------------------
    def revisar_ventas(self, fecha: str, enriquecidos: dict[str, pd.DataFrame]) -> list[dict]:
        eventos, siguen = [], []
        for pos in self.posiciones:
            df = enriquecidos.get(pos["ticker"])
            if df is None or df.empty:
                siguen.append(pos)
                continue
            u = df.iloc[-1]
            precio = float(u["Close"])
            pos["precio_actual"] = precio
            pos["fecha_actual"] = df.index[-1].strftime("%Y-%m-%d")
            pos["max_precio"] = max(float(pos.get("max_precio", precio)), precio)
            pnl = precio / pos["precio_compra"] - 1
            if pnl >= self.cfg["activar_stop_dinamico"]:
                nuevo_stop = pos["max_precio"] * (1 - self.cfg["stop_dinamico_pct"])
                pos["stop_loss"] = round(max(pos["stop_loss"], nuevo_stop), 4)
            dias = (pd.Timestamp(fecha) - pd.Timestamp(pos["fecha_compra"])).days
            motivo = None
            if precio <= pos["stop_loss"]:
                motivo = ("Stop dinámico: protección de ganancias" if pos["stop_loss"] > pos["precio_compra"]
                          else "Stop-loss alcanzado")
            elif precio >= pos["objetivo"]:
                motivo = "Objetivo de ganancia alcanzado"
            elif u["MA7"] < u["MA25"] and precio < u["MA99"]:
                motivo = "Ruptura de tendencia: MA(7) bajo MA(25) y precio bajo MA(99)"
            elif dias >= self.cfg["max_dias_sin_avance"] and pnl < 0.02:
                motivo = f"Sin avance tras {dias} días"
            if motivo:
                eventos.append(self._cerrar(pos, fecha, precio, motivo, dias))
            else:
                siguen.append(pos)
        self.datos["posiciones"] = siguen
        return eventos

    def _cerrar(self, pos: dict, fecha: str, precio: float, motivo: str, dias: int) -> dict:
        ingreso = pos["acciones"] * precio
        self.datos["efectivo"] = round(self.datos["efectivo"] + ingreso, 2)
        cerrada = {
            **{k: pos[k] for k in ("ticker", "nombre", "sector", "fecha_compra", "precio_compra", "acciones",
                                   "puntaje_compra", "razones", "importancia")},
            "fecha_venta": fecha,
            "precio_venta": round(precio, 4),
            "resultado_pct": round(precio / pos["precio_compra"] - 1, 4),
            "resultado_usd": round(ingreso - pos["coste"], 2),
            "motivo_venta": motivo,
            "dias": dias,
        }
        self.cerradas.append(cerrada)
        log.info("Venta %s: %s (%+.1f%%)", pos["ticker"], motivo, cerrada["resultado_pct"] * 100)
        return {"tipo": "venta", **cerrada}

    def comprar(self, fecha: str, ranking: list[dict], red_lista: bool) -> list[dict]:
        cfg = self.cfg
        compradas_hoy = sum(1 for p in self.posiciones if p["fecha_compra"] == fecha)
        vendidas_hoy = {c["ticker"] for c in self.cerradas if c["fecha_venta"] == fecha}
        eventos = []
        for registro in ranking:
            if len(self.posiciones) >= cfg["max_posiciones"] or compradas_hoy >= cfg["max_compras_por_dia"]:
                break
            ticker = registro["ticker"]
            if (registro["puntaje"] < cfg["puntaje_minimo_compra"] or ticker in self.tickers()
                    or ticker in vendidas_hoy):
                continue
            precio = float(registro["tecnico"]["precio"])
            tamano = min(self.valor_total() / cfg["max_posiciones"], self.datos["efectivo"])
            acciones = math.floor(tamano / precio)
            if acciones < 1:
                continue
            coste = round(acciones * precio, 2)
            atr = registro["tecnico"]["atr_pct"] * precio
            stop = min(precio * (1 - cfg["stop_loss_pct"]), precio - cfg["stop_atr"] * atr)
            sectores = [p.get("sector") for p in self.posiciones]
            posicion = {
                "ticker": ticker,
                "nombre": registro["nombre"],
                "sector": registro.get("sector"),
                "origen": "automática",
                "fecha_compra": fecha,
                "precio_compra": round(precio, 4),
                "acciones": acciones,
                "coste": coste,
                "puntaje_compra": registro["puntaje"],
                "recomendacion_compra": registro["recomendacion"],
                "razones": razones_compra(registro, red_lista),
                "importancia": importancia_accion(registro, sectores),
                "noticias_compra": [
                    {"titulo": n.titulo, "fuente": n.fuente, "enlace": n.enlace, "sentimiento": round(n.sentimiento, 2)}
                    for n in ((registro.get("noticias") or {}).get("noticias") or [])[:3]
                ],
                "stop_loss": round(max(stop, 0.01), 4),
                "objetivo": round(precio * (1 + cfg["objetivo_pct"]), 4),
                "max_precio": round(precio, 4),
                "precio_actual": round(precio, 4),
                "fecha_actual": fecha,
                "bitacora": [],
            }
            self.datos["efectivo"] = round(self.datos["efectivo"] - coste, 2)
            self.posiciones.append(posicion)
            compradas_hoy += 1
            eventos.append({"tipo": "compra", **posicion})
            log.info("Compra %s: %d acciones a %.2f (puntaje %.1f)", ticker, acciones, precio, registro["puntaje"])
        return eventos

    def actualizar_seguimiento(self, fecha: str, analisis: dict[str, dict]) -> None:
        for pos in self.posiciones:
            registro = analisis.get(pos["ticker"])
            if not registro:
                continue
            tec = registro["tecnico"]
            pnl = pos["precio_actual"] / pos["precio_compra"] - 1
            nota = (f"Tendencia {tec['tendencia'].lower()}; precio {'sobre' if tec['precio'] > tec['ma25'] else 'bajo'} "
                    f"la MA(25) y {'sobre' if tec['precio'] > tec['ma99'] else 'bajo'} la MA(99); RSI {tec['rsi']:.0f}; "
                    f"puntaje {registro['puntaje']:.0f}; noticias {(registro.get('noticias') or {}).get('etiqueta', 's/d').lower()}")
            entrada = {"fecha": fecha, "precio": round(pos["precio_actual"], 4), "pnl": round(pnl, 6), "nota": nota}
            bitacora = [b for b in pos.get("bitacora", []) if b["fecha"] != fecha] + [entrada]
            pos["bitacora"] = bitacora[-MAX_BITACORA:]

    def registrar_valor(self, fecha: str) -> None:
        valor = round(self.valor_total(), 2)
        self.datos["historial"] = [h for h in self.historial if h["fecha"] != fecha] + [{"fecha": fecha, "valor": valor}]

    def resumen(self) -> dict:
        valor = self.valor_total()
        capital = self.datos["capital_inicial"]
        ganadoras = [c for c in self.cerradas if c["resultado_pct"] > 0]
        return {
            "valor": valor,
            "efectivo": self.datos["efectivo"],
            "invertido": valor - self.datos["efectivo"],
            "rendimiento": valor / capital - 1,
            "posiciones": len(self.posiciones),
            "operaciones_cerradas": len(self.cerradas),
            "tasa_acierto": len(ganadoras) / len(self.cerradas) if self.cerradas else None,
        }


# --------------------------------------------------------------------------
# Posiciones manuales
# --------------------------------------------------------------------------

def cargar_posiciones_manuales(ruta: str = "config/mis_acciones.yaml") -> list[dict]:
    archivo = ruta_proyecto(ruta)
    if not archivo.exists():
        return []
    with open(archivo, encoding="utf-8") as fh:
        datos = yaml.safe_load(fh) or {}
    resultado = []
    for item in datos.get("posiciones") or []:
        try:
            fecha = item.get("fecha_compra")
            if isinstance(fecha, (date, datetime)):
                fecha = fecha.strftime("%Y-%m-%d")
            razones = item.get("razones") or []
            if isinstance(razones, str):
                razones = [razones]
            importancia = item.get("importancia") or []
            if isinstance(importancia, str):
                importancia = [importancia]
            resultado.append({
                "ticker": str(item["ticker"]).strip().upper().replace(".", "-"),
                "fecha_compra": str(fecha) if fecha else None,
                "precio_compra": float(item["precio_compra"]),
                "acciones": float(item.get("acciones") or 0),
                "razones": [str(r) for r in razones],
                "importancia": [str(i) for i in importancia],
                "origen": "manual",
            })
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Posición manual inválida %s: %s", item, exc)
    return resultado
