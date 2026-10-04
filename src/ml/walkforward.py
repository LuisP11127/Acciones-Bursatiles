"""Validación walk-forward por años con purga.

Para cada año de test Y:
  entrenamiento: [train_start, inicio de la validación - purga]
  validación:    años Y-validation_years .. Y-1 (parada temprana) - purga antes del test
  test:          año Y (nunca visto durante el entrenamiento ni la validación)
La purga elimina las sesiones cuyas etiquetas (que miran H días adelante)
se solaparían con el periodo siguiente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .dataset import PanelDataset, fit_scaler
from .metrics import classification_metrics, roc_auc
from .numpy_mlp import NumpyMLP
from .train import TrainResult, predict, predict_windows, standardize, train_model

log = get_logger("MODEL")


@dataclass
class Fold:
    name: str
    train_start: np.datetime64
    train_end: np.datetime64
    val_start: np.datetime64
    val_end: np.datetime64
    test_start: np.datetime64
    test_end: np.datetime64

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, np.datetime64) else v) for k, v in self.__dict__.items()}


def neural_threshold(base_rate: float, strategy_cfg: dict) -> float:
    valor = strategy_cfg.get("min_neural_probability", "auto")
    if valor in (None, "auto"):
        return float(base_rate) * float(strategy_cfg.get("min_neural_lift", 1.15))
    return float(valor)


def make_folds(ds: PanelDataset, wf_cfg: dict) -> list[Fold]:
    etiquetadas = ds.window_ok & ~np.isnan(ds.label) & ds.tradable
    fechas = np.unique(ds.dates[etiquetadas])
    if len(fechas) < 600:
        return []
    purga = int(wf_cfg["purge_days"])
    anios_val = int(wf_cfg["validation_years"])
    inicio_train = max(np.datetime64(str(wf_cfg["train_start"]), "D"), fechas[0])
    anios = pd.DatetimeIndex(fechas.astype("datetime64[ns]")).year.to_numpy()
    ultimo = int(wf_cfg.get("last_test_year") or anios.max())
    folds = []
    for anio in range(int(wf_cfg["first_test_year"]), ultimo + 1):
        en_test = np.nonzero(anios == anio)[0]
        en_val = np.nonzero((anios >= anio - anios_val) & (anios < anio))[0]
        if len(en_test) < 40 or len(en_val) < 120:
            continue
        i_val, i_test = en_val[0], en_test[0]
        if i_val - purga - 1 <= 0 or i_test - purga - 1 <= i_val:
            continue
        fin_train = fechas[i_val - purga - 1]
        if (fin_train - inicio_train).astype(int) < 365 * 2:
            continue  # al menos dos años de entrenamiento
        folds.append(Fold(name=str(anio), train_start=inicio_train, train_end=fin_train, val_start=fechas[i_val],
                          val_end=fechas[i_test - purga - 1], test_start=fechas[i_test],
                          test_end=fechas[en_test[-1]]))
    maximo = int(wf_cfg.get("max_folds") or 0)
    return folds[-maximo:] if maximo > 0 else folds


def baseline_scores(ds: PanelDataset, train_rows: np.ndarray, test_rows: np.ndarray, seed: int) -> dict:
    """Benchmarks sencillos sobre el mismo test: score estadístico, momentum y
    regresión logística (features del último día, ajustada solo con entrenamiento)."""
    y = ds.label[test_rows]
    resultado = {
        "statistical_score": roc_auc(y, ds.stat_score[test_rows]),
        "momentum_3m": roc_auc(y, ds.momentum[test_rows]),
        "random": 0.5,
    }
    rng = np.random.default_rng(seed)
    filas = train_rows if len(train_rows) <= 200_000 else rng.choice(train_rows, 200_000, replace=False)
    media, desv = fit_scaler(ds.features, filas)
    X = np.clip((ds.features[filas] - media) / desv, -5, 5)
    logistica = NumpyMLP(X.shape[1], (), seed=seed).fit(X, np.nan_to_num(ds.label[filas]), epochs=3, seed=seed)
    X_test = np.clip((ds.features[test_rows] - media) / desv, -5, 5)
    resultado["logistic_regression"] = roc_auc(y, logistica.predict_proba(X_test))
    return resultado


def permutation_importance(result: TrainResult, ds: PanelDataset, rows: np.ndarray, seed: int,
                           max_rows: int = 8000) -> list[dict]:
    """Caída de AUC al desordenar cada feature (en toda la ventana) entre ejemplos."""
    rng = np.random.default_rng(seed)
    if len(rows) > max_rows:
        rows = np.sort(rng.choice(rows, max_rows, replace=False))
    y = ds.label[rows]
    if np.nansum(y) == 0 or np.nansum(y) == len(y):
        return []
    ventanas = standardize(ds.windows(rows), result.mean, result.std)
    base = roc_auc(y, predict_windows(result.model, ventanas)["probability"])
    importancias = []
    for j, nombre in enumerate(ds.feature_names):
        copia = ventanas.copy()
        copia[:, :, j] = copia[rng.permutation(len(copia)), :, j]
        auc = roc_auc(y, predict_windows(result.model, copia)["probability"])
        importancias.append({"feature": nombre, "auc_drop": float(base - auc)})
    importancias.sort(key=lambda d: d["auc_drop"], reverse=True)
    return importancias


@dataclass
class WalkForwardResult:
    folds: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    oos_predictions: pd.DataFrame | None = None
    feature_importance: list[dict] = field(default_factory=list)


def run_walk_forward(ds: PanelDataset, nn_cfg: dict, wf_cfg: dict, strategy_cfg: dict) -> WalkForwardResult:
    folds = make_folds(ds, wf_cfg)
    resultado = WalkForwardResult()
    if not folds:
        log.warning("Historial insuficiente para la validación walk-forward")
        return resultado
    seed = int(nn_cfg["seed"])
    stride = int(nn_cfg["sample_stride"])
    partes_oos = []
    ultimo: tuple[TrainResult, np.ndarray] | None = None
    for fold in folds:
        train_rows = ds.rows_between(fold.train_start, fold.train_end, stride=stride)
        val_rows = ds.rows_between(fold.val_start, fold.val_end)
        test_rows = ds.rows_between(fold.test_start, fold.test_end)
        if len(train_rows) < 500 or len(val_rows) < 100 or len(test_rows) < 100:
            log.warning("Fold %s omitido: pocos ejemplos", fold.name)
            continue
        log.info("Fold %s: entrenamiento %s→%s (%d), validación %s→%s (%d), test %s→%s (%d)", fold.name,
                 fold.train_start, fold.train_end, len(train_rows), fold.val_start, fold.val_end, len(val_rows),
                 fold.test_start, fold.test_end, len(test_rows))
        entrenado = train_model(ds, train_rows, val_rows, nn_cfg, seed=seed, tag=f"[fold {fold.name}]")
        pred = predict(entrenado.model, ds, test_rows, entrenado.mean, entrenado.std)
        umbral = neural_threshold(entrenado.base_rate, strategy_cfg)
        metricas = classification_metrics(ds.label[test_rows], pred["probability"], umbral,
                                          ds.trade_return[test_rows])
        base = baseline_scores(ds, train_rows, test_rows, seed)
        resultado.folds.append({**fold.to_dict(), "train_samples": entrenado.train_samples,
                                "val_samples": entrenado.val_samples, "test_samples": int(len(test_rows)),
                                "best_epoch": entrenado.best_epoch, "epochs_run": entrenado.epochs_run,
                                "train_base_rate": entrenado.base_rate, "seconds": entrenado.seconds,
                                "metrics": metricas, "baselines_auc": base})
        partes_oos.append(pd.DataFrame({
            "date": ds.dates[test_rows], "symbol": np.array(ds.symbols)[ds.symbol_idx[test_rows]],
            "probability": pred["probability"], "expected_return": pred["expected_return"],
            "expected_drawdown": pred["expected_drawdown"], "label": ds.label[test_rows],
            "trade_return": ds.trade_return[test_rows], "fold": fold.name,
            "train_base_rate": entrenado.base_rate, "threshold": umbral,
        }))
        ultimo = (entrenado, test_rows)
    if not partes_oos:
        return resultado
    oos = pd.concat(partes_oos, ignore_index=True)
    resultado.oos_predictions = oos
    umbral_medio = float(np.mean([f["metrics"]["threshold"] for f in resultado.folds]))
    resultado.aggregate = classification_metrics(oos["label"].to_numpy(), oos["probability"].to_numpy(),
                                                 umbral_medio, oos["trade_return"].to_numpy())
    resultado.aggregate["folds"] = len(resultado.folds)
    resultado.aggregate["mean_fold_auc"] = float(np.nanmean([f["metrics"]["roc_auc"] for f in resultado.folds]))
    resultado.aggregate["oos_start"] = str(oos["date"].min())
    resultado.aggregate["oos_end"] = str(oos["date"].max())
    resultado.baselines = {
        nombre: float(np.nanmean([f["baselines_auc"][nombre] for f in resultado.folds]))
        for nombre in resultado.folds[0]["baselines_auc"]
    }
    if ultimo is not None:
        resultado.feature_importance = permutation_importance(ultimo[0], ds, ultimo[1], seed)
    return resultado
