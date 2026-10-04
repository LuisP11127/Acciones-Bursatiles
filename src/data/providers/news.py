"""Proveedores de noticias.

- YahooNewsProvider: noticias de Yahoo Finance (yfinance), sin clave.
- GoogleNewsRSSProvider: RSS público de Google News, sin clave.
- FinnhubNewsProvider: solo si existe la variable de entorno NEWS_API_KEY (Finnhub).
- SyntheticNewsProvider: SOLO desarrollo/pruebas; titulares marcados como sintéticos.

Cada proveedor devuelve [] si falla: una fuente caída no detiene el pipeline.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import numpy as np
import requests

from ...logging_utils import get_logger
from .base import NewsItem, NewsProvider

log = get_logger("NEWS")
AGENTE = "Mozilla/5.0 (compatible; QuantResearchBot/2.0; paper-trading research)"


def parse_datetime(value) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def short_company_name(name: str) -> str:
    """'Alphabet Inc. (Class A)' -> 'Alphabet' (mejora búsquedas y relevancia)."""
    limpio = re.sub(r"[,\.()]|\b(Inc|Corp|Corporation|Co|Company|Ltd|plc|Holdings|Group|Class [A-C]|N\.?V)\b",
                    " ", name or "")
    return " ".join(limpio.split()[:3])


class YahooNewsProvider(NewsProvider):
    name = "yahoo"

    def fetch(self, symbol: str, company_name: str, lookback_days: int, max_items: int) -> list[NewsItem]:
        try:
            import yfinance as yf

            crudas = yf.Ticker(symbol).news or []
        except Exception as exc:
            log.warning("Yahoo news falló para %s: %s", symbol, exc)
            return []
        items = []
        for item in crudas[: max_items * 2]:
            contenido = item.get("content") if isinstance(item, dict) else None
            if contenido:  # formato de Yahoo desde 2025
                proveedor = contenido.get("provider") or {}
                url = ((contenido.get("canonicalUrl") or {}).get("url")
                       or (contenido.get("clickThroughUrl") or {}).get("url") or "")
                items.append(NewsItem(symbol=symbol, source=proveedor.get("displayName") or "Yahoo Finance",
                                      title=(contenido.get("title") or "").strip(), url=url,
                                      published_at=parse_datetime(contenido.get("pubDate")
                                                                  or contenido.get("displayTime")),
                                      summary=(contenido.get("summary") or "")[:400], provider=self.name))
            elif isinstance(item, dict) and item.get("title"):  # formato antiguo
                items.append(NewsItem(symbol=symbol, source=item.get("publisher") or "Yahoo Finance",
                                      title=item["title"].strip(), url=item.get("link", ""),
                                      published_at=parse_datetime(item.get("providerPublishTime")),
                                      provider=self.name))
        return [n for n in items if n.title][:max_items]


class GoogleNewsRSSProvider(NewsProvider):
    name = "google_rss"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def fetch(self, symbol: str, company_name: str, lookback_days: int, max_items: int) -> list[NewsItem]:
        consulta = f"{symbol} stock {short_company_name(company_name)}".strip()
        url = (f"https://news.google.com/rss/search?q={quote_plus(consulta)}+when:{int(lookback_days)}d"
               "&hl=en-US&gl=US&ceid=US:en")
        try:
            respuesta = requests.get(url, headers={"User-Agent": AGENTE}, timeout=self.timeout)
            respuesta.raise_for_status()
            raiz = ET.fromstring(respuesta.content)
        except Exception as exc:
            log.warning("Google News falló para %s: %s", symbol, exc)
            return []
        items = []
        for item in raiz.iter("item"):
            titulo = (item.findtext("title") or "").strip()
            fuente = (item.findtext("source") or "").strip()
            if fuente and titulo.endswith(f" - {fuente}"):
                titulo = titulo[: -len(fuente) - 3]
            fecha = None
            if item.findtext("pubDate"):
                try:
                    fecha = parsedate_to_datetime(item.findtext("pubDate"))
                except (TypeError, ValueError):
                    fecha = None
            items.append(NewsItem(symbol=symbol, source=fuente or "Google News", title=titulo,
                                  url=item.findtext("link") or "", published_at=fecha, provider=self.name))
            if len(items) >= max_items:
                break
        return [n for n in items if n.title]


class FinnhubNewsProvider(NewsProvider):
    """Requiere NEWS_API_KEY (clave gratuita de finnhub.io). Sin clave no se usa."""

    name = "finnhub"

    def __init__(self, api_key: str | None, timeout: float = 15.0):
        self.api_key = api_key
        self.timeout = timeout

    def fetch(self, symbol: str, company_name: str, lookback_days: int, max_items: int) -> list[NewsItem]:
        if not self.api_key:
            return []
        hoy = datetime.now(timezone.utc).date()
        try:
            respuesta = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": symbol, "from": str(hoy - timedelta(days=lookback_days)), "to": str(hoy),
                        "token": self.api_key},
                timeout=self.timeout,
            )
            respuesta.raise_for_status()
            crudas = respuesta.json() or []
        except Exception as exc:  # no se registra la URL: contiene la clave
            log.warning("Finnhub falló para %s: %s", symbol, type(exc).__name__)
            return []
        return [
            NewsItem(symbol=symbol, source=n.get("source") or "Finnhub", title=(n.get("headline") or "").strip(),
                     url=n.get("url") or "", published_at=parse_datetime(n.get("datetime")),
                     summary=(n.get("summary") or "")[:400], provider=self.name)
            for n in crudas[:max_items] if n.get("headline")
        ]


_TITULARES_SINTETICOS = [
    "{n} beats earnings estimates and raises guidance",
    "Analysts upgrade {n} citing strong demand",
    "{n} misses revenue expectations, shares fall",
    "{n} downgraded on weaker outlook",
    "Regulators open probe into {n}",
    "{n} announces share buyback program",
    "{n} to present at investor conference",
]


class SyntheticNewsProvider(NewsProvider):
    """Titulares inventados para pruebas offline. La web los marca como SINTÉTICOS."""

    name = "synthetic"
    is_synthetic = True

    def __init__(self, now: datetime | None = None):
        self.now = now

    def fetch(self, symbol: str, company_name: str, lookback_days: int, max_items: int) -> list[NewsItem]:
        import zlib

        rng = np.random.default_rng(zlib.crc32(f"news{symbol}".encode()))
        ahora = self.now or datetime.now(timezone.utc)
        nombre = short_company_name(company_name) or symbol
        return [
            NewsItem(symbol=symbol, source="SINTÉTICO", title=f"[SINTÉTICO] {str(rng.choice(_TITULARES_SINTETICOS)).format(n=nombre)}",
                     url="", published_at=ahora - timedelta(hours=float(rng.uniform(1, 24 * lookback_days))),
                     provider=self.name)
            for _ in range(int(rng.integers(1, 6)))
        ]


def news_api_key() -> str | None:
    return os.environ.get("NEWS_API_KEY") or None
