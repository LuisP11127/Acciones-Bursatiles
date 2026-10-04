"""Lectura y escritura de datos: precios (csv.gz), JSON (atómico y sin NaN) y JSONL comprimido."""

from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]


def clean_for_json(obj, decimals: int | None = 6):
    """Convierte tipos de numpy/pandas y reemplaza NaN/inf por None."""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_for_json(v, decimals) for v in obj]
    if isinstance(obj, (np.floating, float)):
        valor = float(obj)
        if math.isnan(valor) or math.isinf(valor):
            return None
        return round(valor, decimals) if decimals is not None else valor
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist(), decimals)
    return obj


def write_json(path: str | Path, obj, compact: bool = False, decimals: int | None = 6) -> Path:
    """Escritura atómica; falla si quedara algún NaN (allow_nan=False)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    datos = clean_for_json(obj, decimals)
    texto = json.dumps(datos, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":") if compact else None, indent=None if compact else 1)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(texto)
    os.chmod(tmp, 0o644)  # legible por el servidor web (GitHub Pages)
    os.replace(tmp, path)
    return path


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def price_path(raw_dir: Path, symbol: str) -> Path:
    return Path(raw_dir) / f"{symbol}.csv.gz"


def save_prices(raw_dir: Path, symbol: str, df: pd.DataFrame) -> Path:
    path = price_path(raw_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    salida = df[[c for c in PRICE_COLUMNS if c in df.columns]].copy()
    salida.index.name = "timestamp"
    tmp = path.with_suffix(".tmp")
    salida.to_csv(tmp, compression="gzip", float_format="%.6f")
    os.replace(tmp, path)
    return path


def load_prices(raw_dir: Path, symbol: str) -> pd.DataFrame | None:
    path = price_path(raw_dir, symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"], compression="gzip")
    except (ValueError, OSError, EOFError):
        return None
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in ("dividends", "splits") else np.nan
    return df[PRICE_COLUMNS].sort_index()


def cached_symbols(raw_dir: Path) -> list[str]:
    return sorted(p.name[: -len(".csv.gz")] for p in Path(raw_dir).glob("*.csv.gz"))


def append_jsonl_gz(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(clean_for_json(record), ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl_gz(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    registros = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return registros
