"""Lista de acciones de EE.UU. a escanear (S&P 500, Nasdaq-100 y extras)."""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from .config import ruta_proyecto

log = logging.getLogger(__name__)

FUENTES_WIKIPEDIA = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500"),
    "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Nasdaq-100"),
}
AGENTE = "Mozilla/5.0 (compatible; AccionesBursatiles/1.0; +https://github.com)"

COLUMNAS_TICKER = ("Symbol", "Ticker", "Ticker symbol")
COLUMNAS_NOMBRE = ("Security", "Company", "Name")
COLUMNAS_SECTOR = ("GICS Sector", "Sector")
COLUMNAS_INDUSTRIA = ("GICS Sub-Industry", "GICS Sub-industry", "Industry")


def normalizar_ticker(ticker: str) -> str:
    """Yahoo usa guion en vez de punto: BRK.B -> BRK-B."""
    return str(ticker).strip().upper().replace(".", "-")


def _columna(df: pd.DataFrame, candidatas: tuple[str, ...]) -> str | None:
    for col in df.columns:
        if str(col).strip() in candidatas:
            return col
    return None


def _tabla_wikipedia(url: str, indice: str) -> dict[str, dict]:
    respuesta = requests.get(url, headers={"User-Agent": AGENTE}, timeout=30)
    respuesta.raise_for_status()
    for tabla in pd.read_html(io.StringIO(respuesta.text)):
        col_ticker = _columna(tabla, COLUMNAS_TICKER)
        if col_ticker is None or len(tabla) < 50:
            continue
        col_nombre = _columna(tabla, COLUMNAS_NOMBRE)
        col_sector = _columna(tabla, COLUMNAS_SECTOR)
        col_industria = _columna(tabla, COLUMNAS_INDUSTRIA)
        resultado = {}
        for _, fila in tabla.iterrows():
            ticker = normalizar_ticker(fila[col_ticker])
            if not ticker or ticker == "NAN":
                continue
            resultado[ticker] = {
                "nombre": str(fila[col_nombre]) if col_nombre else ticker,
                "sector": str(fila[col_sector]) if col_sector else "",
                "industria": str(fila[col_industria]) if col_industria else "",
                "indices": [indice],
            }
        return resultado
    raise ValueError(f"No se encontró la tabla de componentes en {url}")


def cargar_respaldo(ruta: str) -> dict[str, dict]:
    df = pd.read_csv(ruta_proyecto(ruta))
    return {
        normalizar_ticker(f.ticker): {
            "nombre": f.nombre,
            "sector": f.sector,
            "industria": "",
            "indices": [],
        }
        for f in df.itertuples()
    }


def obtener_universo(cfg: dict, offline: bool = False) -> dict[str, dict]:
    """Devuelve {ticker: {nombre, sector, industria, indices}}."""
    ucfg = cfg["universo"]
    universo: dict[str, dict] = {}
    if not offline:
        for fuente in ucfg.get("fuentes", []):
            if fuente not in FUENTES_WIKIPEDIA:
                log.warning("Fuente de universo desconocida: %s", fuente)
                continue
            url, indice = FUENTES_WIKIPEDIA[fuente]
            try:
                tabla = _tabla_wikipedia(url, indice)
            except Exception as exc:  # red caída, cambio de formato, etc.
                log.warning("No se pudo leer %s: %s", indice, exc)
                continue
            for ticker, meta in tabla.items():
                if ticker in universo:
                    universo[ticker]["indices"].extend(meta["indices"])
                else:
                    universo[ticker] = meta
            log.info("%s: %d componentes", indice, len(tabla))

    if not universo:
        log.warning("Usando la lista de respaldo %s", ucfg["respaldo"])
        universo = cargar_respaldo(ucfg["respaldo"])

    for ticker in ucfg.get("extra", []) or []:
        universo.setdefault(
            normalizar_ticker(ticker),
            {"nombre": ticker, "sector": "", "industria": "", "indices": []},
        )

    limite = int(ucfg.get("max_tickers") or 0)
    if limite > 0:
        universo = dict(list(universo.items())[:limite])
    return universo
