"""Universo de acciones: S&P 500 + Nasdaq-100 (Wikipedia) + símbolos adicionales.

El resultado se guarda en una copia versionada (data/universe/universe.csv) que
se usa como respaldo si la fuente no responde.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from ..config import Settings, resolve
from ..logging_utils import get_logger, summary

log = get_logger("UNIVERSE")

WIKIPEDIA = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500"),
    "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Nasdaq-100"),
}
AGENTE = "Mozilla/5.0 (compatible; QuantResearchBot/2.0)"
COLUMNAS = {
    "symbol": ("Symbol", "Ticker", "Ticker symbol"),
    "name": ("Security", "Company", "Name"),
    "sector": ("GICS Sector", "Sector"),
    "industry": ("GICS Sub-Industry", "GICS Sub-industry", "Industry"),
}
UNIVERSE_COLUMNS = ["symbol", "name", "sector", "industry", "indices"]


def normalize_symbol(symbol: str) -> str:
    """Formato de Yahoo: BRK.B -> BRK-B."""
    return str(symbol).strip().upper().replace(".", "-")


def _columna(tabla: pd.DataFrame, candidatas: tuple[str, ...]):
    for col in tabla.columns:
        if str(col).strip() in candidatas:
            return col
    return None


def fetch_wikipedia_index(url: str, index_name: str, timeout: float = 30.0) -> pd.DataFrame:
    respuesta = requests.get(url, headers={"User-Agent": AGENTE}, timeout=timeout)
    respuesta.raise_for_status()
    for tabla in pd.read_html(io.StringIO(respuesta.text)):
        col_symbol = _columna(tabla, COLUMNAS["symbol"])
        if col_symbol is None or len(tabla) < 50:
            continue
        filas = []
        for _, fila in tabla.iterrows():
            symbol = normalize_symbol(fila[col_symbol])
            if not symbol or symbol == "NAN":
                continue
            filas.append({
                "symbol": symbol,
                "name": str(fila[_columna(tabla, COLUMNAS["name"])]) if _columna(tabla, COLUMNAS["name"]) else symbol,
                "sector": str(fila[_columna(tabla, COLUMNAS["sector"])]) if _columna(tabla, COLUMNAS["sector"]) else "",
                "industry": (str(fila[_columna(tabla, COLUMNAS["industry"])])
                             if _columna(tabla, COLUMNAS["industry"]) else ""),
                "indices": index_name,
            })
        return pd.DataFrame(filas, columns=UNIVERSE_COLUMNS)
    raise ValueError(f"No se encontró la tabla de componentes en {url}")


def _combinar(tablas: list[pd.DataFrame]) -> pd.DataFrame:
    if not tablas:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    todas = pd.concat(tablas, ignore_index=True)
    indices = todas.groupby("symbol")["indices"].apply(lambda s: ";".join(dict.fromkeys(";".join(s).split(";"))))
    primera = todas.drop_duplicates("symbol").set_index("symbol")
    primera["indices"] = indices
    return primera.reset_index()[UNIVERSE_COLUMNS]


def _leer_csv(ruta) -> pd.DataFrame | None:
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta, dtype=str).fillna("")
    for col in UNIVERSE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df[UNIVERSE_COLUMNS]


def load_universe(settings: Settings, refresh: bool = True) -> pd.DataFrame:
    """Devuelve el universo (symbol, name, sector, industry, indices)."""
    ucfg = settings.universe["universe"]
    snapshot = resolve(ucfg["snapshot_file"], settings)
    universo = None
    if refresh and not settings.offline:
        tablas = []
        for clave, (url, nombre) in WIKIPEDIA.items():
            if not ucfg.get(clave, False):
                continue
            try:
                tabla = fetch_wikipedia_index(url, nombre)
                tablas.append(tabla)
                log.info("%s: %d componentes", nombre, len(tabla))
            except Exception as exc:
                summary.warning("UNIVERSE", f"No se pudo leer {nombre} de Wikipedia: {exc}")
        if tablas:
            universo = _combinar(tablas)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            universo.sort_values("symbol").to_csv(snapshot, index=False)
    if universo is None:
        universo = _leer_csv(snapshot)
        if universo is not None:
            log.info("Usando la copia versionada del universo (%d símbolos)", len(universo))
    if universo is None or universo.empty:
        universo = _leer_csv(resolve(ucfg["fallback_file"]))
        log.warning("Usando la lista de respaldo %s (%d símbolos)", ucfg["fallback_file"], len(universo))

    extra = [normalize_symbol(s) for s in ucfg.get("additional_symbols") or []]
    nuevos = [s for s in extra if s not in set(universo["symbol"])]
    if nuevos:
        universo = pd.concat([universo, pd.DataFrame({"symbol": nuevos, "name": nuevos, "sector": "",
                                                      "industry": "", "indices": "Adicional"})],
                             ignore_index=True)
    excluir = {normalize_symbol(s) for s in ucfg.get("exclude_symbols") or []}
    referencias = {normalize_symbol(s) for s in ucfg["reference_symbols"]}
    universo = universo[~universo["symbol"].isin(excluir | referencias)]
    universo = universo.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    limite = int(ucfg.get("max_symbols") or 0)
    if limite > 0:
        universo = universo.head(limite)
    summary.set("universe_size", len(universo))
    return universo
