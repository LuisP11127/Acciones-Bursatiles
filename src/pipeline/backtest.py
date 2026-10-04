"""Backtest independiente (sin reentrenar).

Usa las predicciones fuera de muestra del último entrenamiento si existen y son
compatibles; si no, ejecuta solo las estrategias que no necesitan la red.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..backtesting.runner import run_backtest
from ..config import Settings
from ..logging_utils import get_logger, summary
from .train import OOS_FILE, load_analyses, save_backtest

log = get_logger("BACKTEST")


def run_backtest_pipeline(settings: Settings, update_data: bool = False, now: datetime | None = None) -> dict:
    summary.reset("backtest")
    paths = settings.paths
    paths.ensure()
    mercado, analisis = load_analyses(settings, update_data, now)
    oos, version = None, None
    ruta = paths.processed / OOS_FILE
    if ruta.exists():
        oos = pd.read_csv(ruta, parse_dates=["date"])
        version = str(oos["model_version"].iloc[0]) if "model_version" in oos and len(oos) else None
        log.info("Usando predicciones fuera de muestra de %s (%d filas)", version, len(oos))
    else:
        log.warning("No hay predicciones fuera de muestra: solo estrategias sin red neuronal")
    bt, trades = run_backtest(settings, analisis, mercado.calendar, mercado.frames[mercado.benchmark]["close"], oos,
                              model_version=version)
    save_backtest(paths, bt, trades, version)
    summary.set("status", "ok")
    summary.write(paths.results / "last_backtest_summary.json")
    return {"status": "ok", "period": bt["period"], "neural_available": bt["neural_available"]}
