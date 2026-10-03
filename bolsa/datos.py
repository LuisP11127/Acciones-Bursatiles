"""Proveedores de datos de mercado.

- `ProveedorYahoo`: datos reales (precios, ficha de la empresa y noticias) vía yfinance.
- `ProveedorSintetico`: datos simulados y reproducibles para pruebas sin conexión.
"""

from __future__ import annotations

import logging
import time
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import noticias as mod_noticias

log = logging.getLogger(__name__)

COLUMNAS = ["Open", "High", "Low", "Close", "Volume"]
MIN_FILAS = 120

CAMPOS_INFO = {
    "nombre": ("longName", "shortName"),
    "sector": ("sector",),
    "industria": ("industry",),
    "capitalizacion": ("marketCap",),
    "objetivo_medio": ("targetMeanPrice",),
    "objetivo_alto": ("targetHighPrice",),
    "objetivo_bajo": ("targetLowPrice",),
    "recomendacion": ("recommendationMean",),
    "recomendacion_clave": ("recommendationKey",),
    "num_analistas": ("numberOfAnalystOpinions",),
    "per": ("trailingPE",),
    "per_futuro": ("forwardPE",),
    "beta": ("beta",),
    "resumen": ("longBusinessSummary",),
}


def limpiar_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty or not set(COLUMNAS).issubset(df.columns):
        return None
    df = df[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in ("Open", "High", "Low"):
        df[col] = df[col].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    return df if len(df) >= MIN_FILAS else None


def _separar_descarga(crudo: pd.DataFrame, lote: list[str]) -> dict[str, pd.DataFrame]:
    salida: dict[str, pd.DataFrame] = {}
    if crudo is None or crudo.empty:
        return salida
    if isinstance(crudo.columns, pd.MultiIndex):
        nivel0 = set(crudo.columns.get_level_values(0))
        nivel1 = set(crudo.columns.get_level_values(1))
        for ticker in lote:
            if ticker in nivel0:
                sub = crudo[ticker]
            elif ticker in nivel1:
                sub = crudo.xs(ticker, axis=1, level=1)
            else:
                continue
            limpio = limpiar_ohlcv(sub)
            if limpio is not None:
                salida[ticker] = limpio
    elif len(lote) == 1:
        limpio = limpiar_ohlcv(crudo)
        if limpio is not None:
            salida[lote[0]] = limpio
    return salida


class ProveedorYahoo:
    sintetico = False

    def __init__(self, lote: int = 80, reintentos: int = 3):
        import yfinance  # noqa: F401  (falla pronto si no está instalado)

        self.lote = lote
        self.reintentos = reintentos

    def _descargar(self, tickers: list[str], periodo: str) -> pd.DataFrame | None:
        import yfinance as yf

        for intento in range(self.reintentos):
            try:
                return yf.download(
                    tickers,
                    period=periodo,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
            except Exception as exc:
                espera = 5 * (intento + 1)
                log.warning("Error descargando %d tickers (%s); reintento en %ss", len(tickers), exc, espera)
                time.sleep(espera)
        return None

    def precios(self, tickers: list[str], periodo: str = "5y") -> dict[str, pd.DataFrame]:
        tickers = list(dict.fromkeys(tickers))
        resultado: dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), self.lote):
            lote = tickers[i:i + self.lote]
            resultado.update(_separar_descarga(self._descargar(lote, periodo), lote))
            log.info("Precios descargados: %d/%d", len(resultado), len(tickers))
            time.sleep(1)  # pausa breve entre lotes para no saturar a Yahoo
        faltantes = [t for t in tickers if t not in resultado]
        if faltantes and len(faltantes) <= 60:
            # Segundo intento individual (a veces Yahoo omite tickers en lotes grandes)
            for ticker in faltantes:
                datos = _separar_descarga(self._descargar([ticker], periodo), [ticker])
                resultado.update(datos)
        if faltantes:
            perdidos = [t for t in tickers if t not in resultado]
            if perdidos:
                log.warning("Sin datos para %d tickers: %s", len(perdidos), ", ".join(perdidos[:20]))
        return resultado

    def info(self, ticker: str) -> dict:
        import yfinance as yf

        try:
            crudo = yf.Ticker(ticker).info or {}
        except Exception as exc:
            log.warning("No se pudo obtener la ficha de %s: %s", ticker, exc)
            return {}
        info = {}
        for campo, claves in CAMPOS_INFO.items():
            for clave in claves:
                if crudo.get(clave) not in (None, "", "None"):
                    info[campo] = crudo[clave]
                    break
        return info

    def noticias(self, ticker: str, nombre: str, cfg: dict) -> list[mod_noticias.Noticia]:
        return mod_noticias.buscar_noticias(ticker, nombre, cfg)


# --------------------------------------------------------------------------
# Datos sintéticos (modo --offline y pruebas)
# --------------------------------------------------------------------------

TITULARES_POSITIVOS = [
    "{n} beats earnings estimates and raises full-year guidance",
    "Analysts upgrade {n} to buy citing strong demand",
    "{n} shares surge after record quarterly revenue",
    "{n} announces $5 billion share buyback program",
    "{n} wins major contract, expands partnership",
]
TITULARES_NEGATIVOS = [
    "{n} misses revenue expectations, shares tumble",
    "{n} downgraded to sell on weak outlook",
    "Regulators open investigation into {n}",
    "{n} announces layoffs amid slowing growth",
    "{n} cuts guidance as costs rise",
]
TITULARES_NEUTROS = [
    "{n} to present at upcoming investor conference",
    "What to watch in {n} stock this week",
    "{n} schedules quarterly earnings release date",
]


def _semilla(texto: str) -> int:
    return zlib.crc32(texto.encode())


def serie_sintetica(ticker: str, dias: int = 1260, fin: pd.Timestamp | None = None,
                    es_indice: bool = False) -> pd.DataFrame:
    """Caminata aleatoria con tendencias por tramos, para probar todo el sistema."""
    rng = np.random.default_rng(_semilla(ticker))
    fin = (fin or pd.Timestamp.today()).normalize()
    fechas = pd.bdate_range(end=fin, periods=dias)
    vol_anual = rng.uniform(0.12, 0.2) if es_indice else rng.uniform(0.18, 0.5)
    vol = vol_anual / np.sqrt(252)
    deriva = np.zeros(dias)
    i = 0
    while i < dias:  # regímenes alcistas/bajistas de 20 a 120 días
        largo = int(rng.integers(20, 120))
        deriva[i:i + largo] = rng.normal(0.0004, 0.0025)
        i += largo
    retornos = deriva + rng.standard_t(5, dias) * vol * 0.8
    cierre = rng.uniform(20, 400) * np.exp(np.cumsum(retornos))
    apertura = cierre * np.exp(rng.normal(0, vol * 0.4, dias))
    apertura[1:] = np.where(rng.random(dias - 1) < 0.7, cierre[:-1] * np.exp(rng.normal(0, vol * 0.3, dias - 1)), apertura[1:])
    rango = np.abs(rng.normal(0, vol * 0.8, dias)) * cierre
    maximo = np.maximum(apertura, cierre) + rango * rng.random(dias)
    minimo = np.minimum(apertura, cierre) - rango * rng.random(dias)
    volumen_base = rng.uniform(2e6, 4e7)
    volumen = volumen_base * np.exp(rng.normal(0, 0.3, dias)) * (1 + 15 * np.abs(retornos))
    return pd.DataFrame(
        {"Open": apertura, "High": maximo, "Low": minimo, "Close": cierre, "Volume": volumen.round()},
        index=fechas,
    )


class ProveedorSintetico:
    sintetico = True

    def __init__(self, universo: dict[str, dict] | None = None, indices: list[str] | None = None,
                 dias: int = 1260, fin: str | pd.Timestamp | None = None):
        """`fin` recorta las series en esa fecha (las series son siempre las mismas,
        así se puede simular el paso de los días)."""
        self.universo = universo or {}
        self.indices = set(indices or [])
        self.dias = dias
        self.fin = pd.Timestamp(fin) if fin is not None else None

    def _serie(self, ticker: str) -> pd.DataFrame:
        df = serie_sintetica(ticker, self.dias, es_indice=ticker in self.indices)
        return df[df.index <= self.fin] if self.fin is not None else df

    def precios(self, tickers: list[str], periodo: str = "5y") -> dict[str, pd.DataFrame]:
        return {t: self._serie(t) for t in dict.fromkeys(tickers)}

    def info(self, ticker: str) -> dict:
        rng = np.random.default_rng(_semilla("info" + ticker))
        meta = self.universo.get(ticker, {})
        precio = float(self._serie(ticker)["Close"].iloc[-1])
        return {
            "nombre": meta.get("nombre", ticker),
            "sector": meta.get("sector", ""),
            "industria": meta.get("industria", ""),
            "capitalizacion": float(rng.uniform(5e9, 2e12)),
            "objetivo_medio": precio * float(rng.uniform(0.85, 1.35)),
            "recomendacion": float(rng.uniform(1.5, 3.5)),
            "num_analistas": int(rng.integers(3, 45)),
            "resumen": f"{meta.get('nombre', ticker)} is a synthetic company used for offline testing.",
        }

    def noticias(self, ticker: str, nombre: str, cfg: dict) -> list[mod_noticias.Noticia]:
        rng = np.random.default_rng(_semilla("news" + ticker))
        ahora = datetime.now(timezone.utc)
        sesgo = rng.uniform(-1, 1)
        resultado = []
        for _ in range(int(rng.integers(2, 8))):
            p = rng.random()
            if p < 0.35 + 0.25 * sesgo:
                plantilla = rng.choice(TITULARES_POSITIVOS)
            elif p < 0.7:
                plantilla = rng.choice(TITULARES_NEGATIVOS)
            else:
                plantilla = rng.choice(TITULARES_NEUTROS)
            resultado.append(mod_noticias.Noticia(
                titulo=str(plantilla).format(n=nombre or ticker),
                fuente="Datos sintéticos",
                enlace="",
                fecha=ahora - timedelta(hours=float(rng.uniform(1, 24 * 10))),
            ))
        return resultado
