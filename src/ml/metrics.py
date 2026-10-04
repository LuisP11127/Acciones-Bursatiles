"""Métricas de clasificación (no solo accuracy)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    ok = ~np.isnan(score) & ~np.isnan(y)
    y, score = y[ok], score[ok]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rangos = pd.Series(score).rank(method="average").to_numpy()
    return float((rangos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def classification_metrics(y: np.ndarray, prob: np.ndarray, threshold: float,
                           trade_return: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    pred = prob >= threshold
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and not math.isnan(precision) \
        and not math.isnan(recall) and (precision + recall) else float("nan")
    base = float(y.mean()) if len(y) else float("nan")
    decil = np.argsort(-prob)[: max(1, len(prob) // 10)]
    p_clip = np.clip(prob, 1e-6, 1 - 1e-6)
    resultado = {
        "n": int(len(y)),
        "base_rate": base,
        "threshold": float(threshold),
        "predicted_positive_rate": float(pred.mean()) if len(pred) else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(y, prob),
        "brier": float(np.mean((prob - y) ** 2)) if len(y) else float("nan"),
        "log_loss": float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))) if len(y) else float("nan"),
        "top_decile_precision": float(y[decil].mean()) if len(y) else float("nan"),
    }
    resultado["top_decile_lift"] = (resultado["top_decile_precision"] / base) if base else float("nan")
    if trade_return is not None and len(trade_return):
        r = np.asarray(trade_return, dtype=float)
        resultado["avg_trade_return_all"] = float(np.nanmean(r))
        resultado["avg_trade_return_top_decile"] = float(np.nanmean(r[decil]))
        resultado["avg_trade_return_selected"] = float(np.nanmean(r[pred])) if pred.any() else float("nan")
    return resultado
