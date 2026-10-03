"""Noticias de cada acción y análisis de sentimiento.

Fuentes: Yahoo Finance (yfinance), Google News RSS y, opcionalmente, Finnhub
(si existe la variable de entorno FINNHUB_API_KEY).
"""

from __future__ import annotations

import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

log = logging.getLogger(__name__)

AGENTE = "Mozilla/5.0 (compatible; AccionesBursatiles/1.0)"

# Ajustes del léxico VADER para titulares financieros (escala -4 .. +4)
LEXICO_FINANCIERO = {
    "beat": 1.6, "beats": 1.8, "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0,
    "outperform": 1.8, "overweight": 1.4, "surge": 2.2, "surges": 2.2, "soar": 2.5,
    "soars": 2.5, "jump": 1.6, "jumps": 1.8, "rally": 1.8, "rallies": 1.8,
    "record": 1.4, "bullish": 2.4, "buyback": 1.6, "raises": 1.2, "raised": 1.0,
    "growth": 1.4, "profit": 1.4, "profitable": 1.8, "expands": 1.2, "expansion": 1.2,
    "partnership": 1.2, "approval": 1.8, "approved": 1.8, "breakthrough": 2.0,
    "gains": 1.4, "tops": 1.4, "strong": 1.4, "dividend": 0.8, "wins": 1.6,
    "miss": -1.8, "misses": -1.8, "missed": -1.8, "downgrade": -2.0, "downgrades": -2.0,
    "downgraded": -2.0, "underperform": -1.8, "underweight": -1.4, "plunge": -2.6,
    "plunges": -2.6, "tumble": -2.2, "tumbles": -2.2, "slump": -2.0, "slumps": -2.0,
    "sinks": -2.0, "falls": -1.2, "drops": -1.2, "bearish": -2.4, "lawsuit": -2.0,
    "sued": -2.0, "probe": -1.6, "investigation": -1.6, "recall": -1.8, "layoffs": -1.6,
    "cuts": -1.2, "bankruptcy": -3.4, "fraud": -3.0, "warning": -1.4, "weak": -1.6,
    "loss": -1.5, "losses": -1.5, "decline": -1.3, "declines": -1.3, "selloff": -2.0,
    "sell-off": -2.0, "halt": -1.6, "fined": -1.8, "delay": -1.0, "delays": -1.0,
    "antitrust": -1.4, "subpoena": -1.8, "short": -0.6,
}


@dataclass
class Noticia:
    titulo: str
    fuente: str = ""
    enlace: str = ""
    fecha: datetime | None = None
    resumen: str = ""
    sentimiento: float = 0.0
    peso: float = field(default=1.0, repr=False)

    def a_dict(self) -> dict:
        datos = asdict(self)
        datos["fecha"] = self.fecha.isoformat() if self.fecha else None
        datos.pop("peso", None)
        return datos


# --------------------------------------------------------------------------
# Sentimiento
# --------------------------------------------------------------------------

class AnalizadorSentimiento:
    """VADER con vocabulario financiero; si no está instalado usa solo el léxico."""

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
            self._vader.lexicon.update(LEXICO_FINANCIERO)
        except ImportError:  # pragma: no cover - dependencia opcional
            self._vader = None

    def puntuar(self, texto: str) -> float:
        """Devuelve un valor entre -1 (muy negativo) y +1 (muy positivo)."""
        if not texto:
            return 0.0
        if self._vader is not None:
            return float(self._vader.polarity_scores(texto)["compound"])
        palabras = re.findall(r"[a-z\-]+", texto.lower())
        total = sum(LEXICO_FINANCIERO.get(p, 0.0) for p in palabras)
        return total / math.sqrt(total * total + 15) if total else 0.0


_ANALIZADOR: AnalizadorSentimiento | None = None


def analizador() -> AnalizadorSentimiento:
    global _ANALIZADOR
    if _ANALIZADOR is None:
        _ANALIZADOR = AnalizadorSentimiento()
    return _ANALIZADOR


# --------------------------------------------------------------------------
# Fuentes
# --------------------------------------------------------------------------

def _fecha_iso(valor) -> datetime | None:
    if valor in (None, ""):
        return None
    try:
        if isinstance(valor, (int, float)):
            return datetime.fromtimestamp(valor, tz=timezone.utc)
        return datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
    except (ValueError, OSError):
        return None


def noticias_yahoo(ticker: str, maximo: int) -> list[Noticia]:
    import yfinance as yf

    try:
        crudas = yf.Ticker(ticker).news or []
    except Exception as exc:
        log.warning("Yahoo news falló para %s: %s", ticker, exc)
        return []
    resultado = []
    for item in crudas[:maximo]:
        contenido = item.get("content") if isinstance(item, dict) else None
        if contenido:  # formato nuevo de Yahoo (2025+)
            proveedor = contenido.get("provider") or {}
            url = (contenido.get("canonicalUrl") or {}).get("url") or (contenido.get("clickThroughUrl") or {}).get("url", "")
            resultado.append(Noticia(
                titulo=contenido.get("title", ""),
                fuente=proveedor.get("displayName", "Yahoo Finance"),
                enlace=url or "",
                fecha=_fecha_iso(contenido.get("pubDate") or contenido.get("displayTime")),
                resumen=contenido.get("summary", "") or "",
            ))
        elif isinstance(item, dict) and item.get("title"):  # formato antiguo
            resultado.append(Noticia(
                titulo=item["title"],
                fuente=item.get("publisher", "Yahoo Finance"),
                enlace=item.get("link", ""),
                fecha=_fecha_iso(item.get("providerPublishTime")),
            ))
    return [n for n in resultado if n.titulo]


def noticias_google(consulta: str, maximo: int) -> list[Noticia]:
    url = (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(consulta)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        respuesta = requests.get(url, headers={"User-Agent": AGENTE}, timeout=20)
        respuesta.raise_for_status()
        raiz = ET.fromstring(respuesta.content)
    except Exception as exc:
        log.warning("Google News falló para %r: %s", consulta, exc)
        return []
    resultado = []
    for item in raiz.iter("item"):
        titulo = (item.findtext("title") or "").strip()
        fuente = (item.findtext("source") or "").strip()
        # Google añade " - Fuente" al final del titular
        if fuente and titulo.endswith(f" - {fuente}"):
            titulo = titulo[: -len(fuente) - 3]
        fecha = None
        if item.findtext("pubDate"):
            try:
                fecha = parsedate_to_datetime(item.findtext("pubDate"))
            except (TypeError, ValueError):
                fecha = None
        resultado.append(Noticia(titulo=titulo, fuente=fuente or "Google News",
                                 enlace=item.findtext("link") or "", fecha=fecha))
        if len(resultado) >= maximo:
            break
    return resultado


def noticias_finnhub(ticker: str, dias: int, maximo: int) -> list[Noticia]:
    clave = os.environ.get("FINNHUB_API_KEY")
    if not clave:
        return []
    hoy = datetime.now(timezone.utc).date()
    try:
        respuesta = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": str(hoy - timedelta(days=dias)), "to": str(hoy), "token": clave},
            timeout=20,
        )
        respuesta.raise_for_status()
        crudas = respuesta.json() or []
    except Exception as exc:
        log.warning("Finnhub falló para %s: %s", ticker, exc)
        return []
    return [
        Noticia(titulo=n.get("headline", ""), fuente=n.get("source", "Finnhub"), enlace=n.get("url", ""),
                fecha=_fecha_iso(n.get("datetime")), resumen=n.get("summary", ""))
        for n in crudas[:maximo] if n.get("headline")
    ]


def _nombre_corto(nombre: str) -> str:
    """'Apple Inc.' -> 'Apple' (mejora la búsqueda en Google News)."""
    limpio = re.sub(r"[,\.()]|\b(Inc|Corp|Corporation|Co|Company|Ltd|plc|Holdings|Class [A-C]|N\.?V)\b", " ",
                    nombre or "")
    return " ".join(limpio.split()[:3])


def buscar_noticias(ticker: str, nombre: str, cfg: dict) -> list[Noticia]:
    ncfg = cfg["noticias"]
    maximo = int(ncfg["max_por_accion"])
    todas = noticias_yahoo(ticker, maximo) + noticias_finnhub(ticker, int(ncfg["dias_recientes"]), maximo)
    if ncfg.get("google_news", True) and len(todas) < maximo:
        consulta = f"{ticker} stock {_nombre_corto(nombre)}".strip()
        todas += noticias_google(consulta, maximo)
    return todas


# --------------------------------------------------------------------------
# Análisis
# --------------------------------------------------------------------------

def _clave_titulo(titulo: str) -> str:
    return re.sub(r"[^a-z0-9]", "", titulo.lower())[:80]


def analizar_noticias(lista: list[Noticia], cfg: dict, ahora: datetime | None = None) -> dict:
    """Puntúa cada titular y calcula un sentimiento ponderado por antigüedad."""
    ncfg = cfg["noticias"]
    ahora = ahora or datetime.now(timezone.utc)
    limite = ahora - timedelta(days=int(ncfg["dias_recientes"]))
    vida_media = float(ncfg.get("vida_media_dias", 3))
    vistas, filtradas = set(), []
    for n in lista:
        clave = _clave_titulo(n.titulo)
        if not clave or clave in vistas:
            continue
        if n.fecha is not None and n.fecha.tzinfo is None:
            n.fecha = n.fecha.replace(tzinfo=timezone.utc)
        if n.fecha is not None and n.fecha < limite:
            continue
        vistas.add(clave)
        n.sentimiento = analizador().puntuar(f"{n.titulo}. {n.resumen[:300]}" if n.resumen else n.titulo)
        edad = max(0.0, (ahora - n.fecha).total_seconds() / 86400) if n.fecha else vida_media * 2
        n.peso = 0.5 ** (edad / vida_media)
        filtradas.append(n)

    filtradas.sort(key=lambda n: n.fecha or limite, reverse=True)
    filtradas = filtradas[: int(ncfg["max_por_accion"])]
    if not filtradas:
        return {"noticias": [], "sentimiento": None, "puntaje": None, "positivas": 0,
                "negativas": 0, "etiqueta": "Sin noticias recientes"}

    peso_total = sum(n.peso for n in filtradas)
    medio = sum(n.sentimiento * n.peso for n in filtradas) / peso_total
    # Con pocas noticias el sentimiento se acerca a neutral
    ajustado = medio * len(filtradas) / (len(filtradas) + 2)
    positivas = sum(n.sentimiento >= 0.25 for n in filtradas)
    negativas = sum(n.sentimiento <= -0.25 for n in filtradas)
    if ajustado >= 0.25:
        etiqueta = "Muy positivo"
    elif ajustado >= 0.08:
        etiqueta = "Positivo"
    elif ajustado <= -0.25:
        etiqueta = "Muy negativo"
    elif ajustado <= -0.08:
        etiqueta = "Negativo"
    else:
        etiqueta = "Neutral"
    return {
        "noticias": filtradas,
        "sentimiento": round(ajustado, 3),
        "puntaje": round(max(0.0, min(100.0, 50 + 50 * ajustado * 1.6)), 1),
        "positivas": int(positivas),
        "negativas": int(negativas),
        "etiqueta": etiqueta,
    }
