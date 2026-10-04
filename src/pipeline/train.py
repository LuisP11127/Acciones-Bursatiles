"""Entrenamiento periódico de la red neuronal.

 1. Construye el dataset (features punto a punto + etiquetas futuras).
 2. Validación walk-forward (predicciones fuera de muestra año a año).
 3. Entrena el candidato a producción con los datos más recientes.
 4. Backtest de las estrategias con las predicciones fuera de muestra.
 5. Compara con el modelo en producción y solo lo sustituye si cumple los criterios.
 6. Guarda métricas, versión y, si se promueve, los pesos.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..backtesting.runner import run_backtest
from ..config import Settings
from ..data.market import load_market_data, update_market_data
from ..data.providers import create_market_provider
from ..data.storage import write_json
from ..data.universe import load_universe
from ..features.builder import FEATURE_COLUMNS
from ..features.labels import label_params
from ..logging_utils import get_logger, summary
from ..market_calendar import last_completed_session
from ..ml.dataset import build_panel_dataset
from ..ml.model import count_parameters
from ..ml.registry import ModelRegistry, evaluate_promotion
from ..ml.train import train_model
from ..ml.walkforward import neural_threshold, run_walk_forward
from .analysis import analyze_universe

log = get_logger("MODEL")
OOS_FILE = "oos_predictions.csv.gz"
BACKTEST_KEYS = ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "volatility", "win_rate",
                 "profit_factor", "trades", "avg_holding_days", "exposure")


def load_analyses(settings: Settings, update_data: bool, now: datetime | None = None, training: bool = True):
    universo = load_universe(settings, refresh=update_data)
    referencias = list(settings.universe["universe"]["reference_symbols"])
    simbolos = universo["symbol"].tolist()
    if update_data:
        cal = settings.data["market_calendar"]
        sesion = last_completed_session(now, cal["close_time"], int(cal["data_ready_delay_minutes"]), cal["timezone"])
        update_market_data(settings, simbolos + referencias, create_market_provider(settings, synthetic_end=sesion),
                           as_of=sesion)
    mercado = load_market_data(settings, simbolos)
    validos = mercado.usable_for_training() if training else mercado.usable_for_signals()
    validos = [s for s in validos if s not in referencias]
    return mercado, analyze_universe(mercado, validos, settings, with_labels=True)


def _huella(ds) -> str:
    h = hashlib.sha256()
    h.update(",".join(ds.symbols).encode())
    h.update(str(ds.dates.min()).encode() + str(ds.dates.max()).encode() + str(len(ds)).encode())
    h.update(np.nansum(ds.label).tobytes())
    h.update(np.round(ds.features[::997, :5].astype(np.float64), 5).tobytes())
    return h.hexdigest()[:16]


def _resumen_backtest(bt: dict) -> dict:
    salida = {}
    for grupo in ("strategies", "benchmarks"):
        for nombre, datos in bt.get(grupo, {}).items():
            if datos.get("metrics"):
                salida[nombre] = {k: datos["metrics"].get(k) for k in BACKTEST_KEYS if k in datos["metrics"]}
    return salida


def save_backtest(paths, bt: dict, trades: pd.DataFrame, version: str | None) -> None:
    write_json(paths.results / "backtest_latest.json", bt)
    if trades is not None and not trades.empty:
        trades.to_csv(paths.results / "backtest_trades.csv.gz", index=False, compression="gzip", float_format="%.6g")
    if version:
        write_json(paths.results / "backtests" / f"{version}.json",
                   {k: bt[k] for k in ("generated_at", "data_as_of", "period", "model_version", "neural_available")}
                   | {"summary": _resumen_backtest(bt)})


def run_training(settings: Settings, update_data: bool = True, now: datetime | None = None) -> dict:
    summary.reset("model_training")
    paths = settings.paths
    paths.ensure()
    mercado, analisis = load_analyses(settings, update_data, now)
    fcfg, nn_cfg, wf_cfg = settings.model["features"], settings.model["neural_network"], settings.model["walk_forward"]
    filtros = settings.universe["filters"]
    ds = build_panel_dataset(analisis, int(fcfg["lookback"]), float(filtros["min_price"]),
                             float(filtros["min_avg_dollar_volume"]))
    etiquetadas = ds.window_ok & ds.tradable & ~np.isnan(ds.label)
    summary.set("dataset_rows", int(len(ds)))
    summary.set("dataset_labeled_samples", int(etiquetadas.sum()))
    summary.set("dataset_symbols", len(ds.symbols))
    log.info("Dataset: %d filas, %d ejemplos etiquetados, %d acciones, tasa positiva %.3f", len(ds),
             etiquetadas.sum(), len(ds.symbols), float(np.nanmean(ds.label[etiquetadas])) if etiquetadas.any() else 0)

    registro = ModelRegistry(paths.models)
    version = registro.next_version()
    wf = run_walk_forward(ds, nn_cfg, wf_cfg, settings.rules)
    if wf.oos_predictions is None or wf.oos_predictions.empty:
        summary.warning("MODEL", "No hay historial suficiente para entrenar y validar la red")
        summary.set("status", "insufficient_history")
        summary.write(paths.results / "last_training_summary.json")
        return {"status": "insufficient_history"}
    oos = wf.oos_predictions.assign(model_version=version)
    oos.to_csv(paths.processed / OOS_FILE, index=False, compression="gzip", float_format="%.6g")

    # Candidato a producción: entrenado con todo el historial; el último año sirve para la parada temprana
    fechas = np.unique(ds.dates[etiquetadas])
    purga = int(wf_cfg["purge_days"])
    inicio_val = fechas[max(len(fechas) - 252, purga + 1)]
    fin_train = fechas[int(np.searchsorted(fechas, inicio_val)) - purga - 1]
    inicio_train = max(np.datetime64(str(wf_cfg["train_start"]), "D"), fechas[0])
    train_rows = ds.rows_between(inicio_train, fin_train, stride=int(nn_cfg["sample_stride"]))
    val_rows = ds.rows_between(inicio_val, fechas[-1])
    final = train_model(ds, train_rows, val_rows, nn_cfg, tag="[producción]")

    bt, trades = run_backtest(settings, analisis, mercado.calendar, mercado.frames[mercado.benchmark]["close"], oos,
                              model_version=version)
    umbral = neural_threshold(final.base_rate, settings.rules)
    candidato = {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "training_period": {"start": str(inicio_train), "end": str(fin_train)},
        "validation_period": {"start": str(inicio_val), "end": str(fechas[-1])},
        "data_as_of": mercado.as_of.strftime("%Y-%m-%d"),
        "features_version": fcfg["version"],
        "feature_names": list(FEATURE_COLUMNS),
        "lookback": int(fcfg["lookback"]),
        "label_definition": {**label_params(settings),
                             "description": "Operación simulada: entrada en la apertura de T+1; salida por stop-loss, "
                                            "take-profit o al cierre de T+H. Oportunidad si el retorno neto de "
                                            "costes >= min_return."},
        "hyperparameters": {k: nn_cfg[k] for k in ("architecture", "hidden_size", "num_layers", "dropout",
                                                   "learning_rate", "weight_decay", "batch_size", "max_epochs",
                                                   "patience", "sample_stride", "max_train_samples", "seed")},
        "n_parameters": count_parameters(final.model),
        "base_rate": final.base_rate,
        "threshold": umbral,
        "scaler": {"mean": final.mean.tolist(), "std": final.std.tolist()},
        "training": {"best_epoch": final.best_epoch, "epochs_run": final.epochs_run,
                     "train_samples": final.train_samples, "val_samples": final.val_samples,
                     "seconds": final.seconds, "history": final.history},
        "walk_forward": {"folds": wf.folds, "aggregate": wf.aggregate, "baselines": wf.baselines},
        "feature_importance": wf.feature_importance,
        "backtest": _resumen_backtest(bt),
        "dataset": {"rows": int(len(ds)), "symbols": len(ds.symbols), "labeled_samples": int(etiquetadas.sum()),
                    "positive_rate": float(np.nanmean(ds.label[etiquetadas])), "fingerprint": _huella(ds)},
        "config_hash": settings.config_hash(),
    }
    produccion = registro.production()
    promover, checks = evaluate_promotion(candidato, produccion, settings.model["promotion"], fcfg["version"])
    candidato["promotion_checks"] = checks
    candidato["compared_with"] = produccion["model_version"] if produccion else None
    registro.register(candidato, final.model, promote=promover)
    save_backtest(paths, bt, trades, version)
    write_json(paths.results / "training_latest.json", {
        "model_version": version, "promoted": promover, "checks": checks,
        "generated_at": candidato["created_at"], "data_as_of": candidato["data_as_of"],
        "walk_forward": candidato["walk_forward"], "feature_importance": wf.feature_importance,
        "production_version": registro.production_version,
    })
    summary.set("candidate_version", version)
    summary.set("promoted", promover)
    summary.set("production_version", registro.production_version)
    summary.set("oos_auc", round(float(wf.aggregate.get("roc_auc", float("nan"))), 4))
    summary.set("status", "ok")
    summary.write(paths.results / "last_training_summary.json")
    return {"status": "ok", "version": version, "promoted": promover, "checks": checks}
