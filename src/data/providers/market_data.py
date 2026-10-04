"""Proveedores de precios: yfinance (real) y sintético (solo desarrollo y pruebas)."""

from __future__ import annotations

import time
import zlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from .base import MARKET_COLUMNS, MarketDataProvider

log = get_logger("DATA")

_RENOMBRAR = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
    "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits",
}


def normalize_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Columnas estándar, índice de fechas sin zona horaria y ordenado."""
    if df is None or df.empty:
        return None
    df = df.rename(columns=_RENOMBRAR)
    if "close" not in df.columns:
        return None
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    for col in ("dividends", "splits"):
        if col not in df.columns:
            df[col] = 0.0
    df = df[[c for c in MARKET_COLUMNS if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "timestamp"
    df = df.dropna(how="all", subset=["open", "high", "low", "close"])
    return df.sort_index() if not df.empty else None


def split_download(raw: pd.DataFrame | None, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Separa la descarga en lote de yfinance (columnas MultiIndex) por símbolo."""
    resultado: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return resultado
    if isinstance(raw.columns, pd.MultiIndex):
        nivel0 = set(raw.columns.get_level_values(0))
        nivel1 = set(raw.columns.get_level_values(1))
        for symbol in symbols:
            if symbol in nivel0:
                sub = raw[symbol]
            elif symbol in nivel1:
                sub = raw.xs(symbol, axis=1, level=1)
            else:
                continue
            limpio = normalize_frame(sub)
            if limpio is not None:
                resultado[symbol] = limpio
    elif len(symbols) == 1:
        limpio = normalize_frame(raw)
        if limpio is not None:
            resultado[symbols[0]] = limpio
    return resultado


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance vía yfinance. No necesita clave de API."""

    name = "yfinance"

    def __init__(self, batch_size: int = 80, retries: int = 3, backoff: float = 5.0, timeout: float = 30.0):
        import yfinance  # noqa: F401  (falla pronto si no está instalado)

        self.batch_size = batch_size
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout

    def _download(self, symbols: list[str], start: date | None, end: date | None) -> pd.DataFrame | None:
        import yfinance as yf

        for intento in range(1, self.retries + 1):
            try:
                return yf.download(
                    symbols,
                    start=start.isoformat() if start else None,
                    end=(end + timedelta(days=1)).isoformat() if end else None,
                    period=None if start else "max",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    actions=True,
                    threads=True,
                    progress=False,
                    timeout=self.timeout,
                )
            except Exception as exc:  # errores de red, límites de peticiones...
                espera = self.backoff * intento
                log.warning("Descarga fallida (%d símbolos, intento %d/%d): %s; reintento en %.0fs",
                            len(symbols), intento, self.retries, exc, espera)
                time.sleep(espera)
        return None

    def fetch_history(self, symbols: list[str], start: date | None,
                      end: date | None = None) -> dict[str, pd.DataFrame]:
        symbols = list(dict.fromkeys(symbols))
        resultado: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), self.batch_size):
            lote = symbols[i:i + self.batch_size]
            resultado.update(split_download(self._download(lote, start, end), lote))
            log.info("Precios descargados: %d/%d", len(resultado), len(symbols))
            time.sleep(1.0)  # pausa entre lotes para no saturar al proveedor
        faltantes = [s for s in symbols if s not in resultado]
        if faltantes and len(faltantes) <= 40:
            for symbol in faltantes:  # segundo intento individual
                resultado.update(split_download(self._download([symbol], start, end), [symbol]))
        return resultado


# --------------------------------------------------------------------------
# Datos sintéticos: SOLO para desarrollo y pruebas sin conexión.
# Nunca se publican como datos reales (la web muestra un aviso).
# --------------------------------------------------------------------------

SYNTHETIC_ANCHOR_END = date(2027, 12, 31)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def synthetic_series(symbol: str, end: date, start: date = date(2010, 1, 4),
                     is_index: bool = False) -> pd.DataFrame:
    """Serie diaria reproducible con tendencias por tramos y dividendos.

    Se genera siempre hasta SYNTHETIC_ANCHOR_END y se recorta en `end`, así una
    fecha posterior no cambia el pasado. El cierre ajustado se calcula después
    del recorte (solo con dividendos conocidos hasta `end`, como Yahoo)."""
    rng = np.random.default_rng(_seed(symbol))
    fechas = pd.bdate_range(start=start, end=max(SYNTHETIC_ANCHOR_END, end))
    n = len(fechas)
    vol = (rng.uniform(0.12, 0.2) if is_index else rng.uniform(0.18, 0.5)) / np.sqrt(252)
    deriva = np.zeros(n)
    i = 0
    while i < n:
        largo = int(rng.integers(20, 120))
        deriva[i:i + largo] = rng.normal(0.0004, 0.0022)
        i += largo
    retornos = deriva + rng.standard_t(5, n) * vol * 0.8
    cierre = rng.uniform(20, 300) * np.exp(np.cumsum(retornos))
    apertura = cierre * np.exp(rng.normal(0, vol * 0.4, n))
    apertura[1:] = np.where(rng.random(n - 1) < 0.7, cierre[:-1] * np.exp(rng.normal(0, vol * 0.3, n - 1)),
                            apertura[1:])
    rango = np.abs(rng.normal(0, vol * 0.8, n)) * cierre
    maximo = np.maximum(apertura, cierre) + rango * rng.random(n)
    minimo = np.minimum(apertura, cierre) - rango * rng.random(n)
    volumen = rng.uniform(2e6, 3e7) * np.exp(rng.normal(0, 0.3, n)) * (1 + 15 * np.abs(retornos))
    dividendos = np.zeros(n)
    if not is_index and rng.random() < 0.6:
        trimestral = cierre * rng.uniform(0.002, 0.008)
        dividendos[60::63] = trimestral[60::63]
    df = pd.DataFrame({"open": apertura, "high": maximo, "low": minimo, "close": cierre,
                       "volume": volumen.round(), "dividends": dividendos, "splits": 0.0}, index=fechas)
    df = df[df.index <= pd.Timestamp(end)]
    # Cierre ajustado por dividendos (como Yahoo): factor acumulado hacia atrás
    cierre, dividendos = df["close"].to_numpy(), df["dividends"].to_numpy()
    factor = np.ones(len(df))
    for t in np.nonzero(dividendos)[0][::-1]:
        if t > 0:
            factor[:t] *= 1 - dividendos[t] / cierre[t - 1]
    df["adj_close"] = cierre * factor
    df.index.name = "timestamp"
    return df[MARKET_COLUMNS]


class SyntheticMarketProvider(MarketDataProvider):
    """Precios simulados reproducibles. `end` recorta la serie para simular el paso del tiempo."""

    name = "synthetic"
    is_synthetic = True

    def __init__(self, end: date | None = None, index_symbols: list[str] | None = None,
                 missing: list[str] | None = None):
        self.end = end or date.today()
        self.index_symbols = set(index_symbols or [])
        self.missing = set(missing or [])

    def fetch_history(self, symbols: list[str], start: date | None,
                      end: date | None = None) -> dict[str, pd.DataFrame]:
        fin = min(end or self.end, self.end)
        resultado = {}
        for symbol in dict.fromkeys(symbols):
            if symbol in self.missing:
                continue
            df = synthetic_series(symbol, fin, is_index=symbol in self.index_symbols)
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if not df.empty:
                resultado[symbol] = df
        return resultado
