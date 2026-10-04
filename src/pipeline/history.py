"""Snapshots diarios (qué pensaba el sistema cada día) y archivo de noticias.

data/history/AAAA-MM-DD/
  snapshot.json      metadatos de la ejecución + oportunidades + eventos de paper trading
  scores.csv.gz      puntuaciones de todas las acciones analizadas ese día
  details.json.gz    análisis completo (noticias, fundamentales, red, explicación) de las candidatas
data/news/AAAA-MM.jsonl.gz   noticias únicas con su sentimiento (para no depender de la fuente)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from ..data.storage import append_jsonl_gz, clean_for_json, read_json, read_jsonl_gz, write_json

SCORE_COLUMNS = ["symbol", "name", "sector", "price", "change_1d", "statistical_score", "trend_score",
                 "momentum_score", "technical_score", "volume_score", "risk_score", "history_score", "news_score",
                 "fundamental_score", "neural_score", "opportunity_probability", "confidence", "expected_return",
                 "expected_drawdown", "final_score", "eligible", "opportunity", "rank"]


def snapshot_dir(history_dir: Path, as_of) -> Path:
    return Path(history_dir) / pd.Timestamp(as_of).strftime("%Y-%m-%d")


def snapshot_exists(history_dir: Path, as_of) -> bool:
    return (snapshot_dir(history_dir, as_of) / "snapshot.json").exists()


def write_snapshot(history_dir: Path, as_of, snapshot: dict, scores: pd.DataFrame, details: dict) -> Path:
    destino = snapshot_dir(history_dir, as_of)
    destino.mkdir(parents=True, exist_ok=True)
    tabla = scores.reindex(columns=SCORE_COLUMNS)
    tabla.to_csv(destino / "scores.csv.gz", index=False, compression="gzip", float_format="%.5g")
    with gzip.open(destino / "details.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(clean_for_json(details), fh, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    write_json(destino / "snapshot.json", snapshot)  # se escribe al final: marca el snapshot como completo
    return destino


def list_snapshots(history_dir: Path) -> list[str]:
    if not Path(history_dir).exists():
        return []
    return sorted(p.name for p in Path(history_dir).iterdir() if (p / "snapshot.json").exists())


def load_snapshot(history_dir: Path, fecha: str) -> dict | None:
    return read_json(Path(history_dir) / fecha / "snapshot.json")


def load_scores(history_dir: Path, fecha: str) -> pd.DataFrame | None:
    ruta = Path(history_dir) / fecha / "scores.csv.gz"
    return pd.read_csv(ruta) if ruta.exists() else None


def load_details(history_dir: Path, fecha: str) -> dict:
    ruta = Path(history_dir) / fecha / "details.json.gz"
    if not ruta.exists():
        return {}
    with gzip.open(ruta, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def archive_news(news_dir: Path, items: list[dict]) -> int:
    """Añade al archivo mensual las noticias que aún no estaban (clave: símbolo + titular)."""
    por_mes: dict[str, list[dict]] = {}
    for item in items:
        fecha = item.get("published_at")
        if not fecha:
            continue
        por_mes.setdefault(str(fecha)[:7], []).append(item)
    nuevas = 0
    for mes, lista in por_mes.items():
        ruta = Path(news_dir) / f"{mes}.jsonl.gz"
        existentes = {(n.get("symbol"), n.get("key")) for n in read_jsonl_gz(ruta)}
        a_guardar = []
        for item in lista:
            clave = (item.get("symbol"), item.get("key"))
            if clave in existentes:
                continue
            existentes.add(clave)
            a_guardar.append(item)
        append_jsonl_gz(ruta, a_guardar)
        nuevas += len(a_guardar)
    return nuevas


def load_news_archive(news_dir: Path, months: int = 3) -> list[dict]:
    archivos = sorted(Path(news_dir).glob("*.jsonl.gz"))[-months:] if Path(news_dir).exists() else []
    noticias = []
    for ruta in archivos:
        noticias.extend(read_jsonl_gz(ruta))
    return noticias
