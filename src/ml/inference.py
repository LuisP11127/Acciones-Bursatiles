"""Inferencia del modelo en producción y explicación por sensibilidad.

La explicación indica qué variables movieron la probabilidad del MODELO
(oclusión: se sustituye cada variable por su valor medio de entrenamiento en toda
la ventana y se mide el cambio). Es una explicación del modelo, no una
afirmación sobre la empresa.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..features.builder import FEATURE_COLUMNS, FEATURE_LABELS
from ..logging_utils import get_logger
from .dataset import last_window

log = get_logger("MODEL")
UNCERTAINTY_SCALE = 0.10  # desviación entre pasadas con dropout que equivale a confianza 0


def neural_score(probability: float | None, base_rate: float | None) -> float | None:
    """0-100 según cuántas veces supera la probabilidad a la tasa base (lift)."""
    if probability is None or not base_rate or base_rate <= 0:
        return None
    lift = max(float(probability), 1e-6) / float(base_rate)
    return round(50 + 50 * math.tanh(math.log2(lift)), 1)


def compatible(meta: dict | None, features_version: str, lookback: int) -> tuple[bool, str]:
    if not meta:
        return False, "No hay modelo neuronal en producción todavía."
    if meta.get("features_version") != features_version:
        return False, (f"El modelo {meta['model_version']} usa features {meta.get('features_version')} y la "
                       f"versión actual es {features_version}: hace falta reentrenar.")
    if list(meta.get("feature_names", [])) != list(FEATURE_COLUMNS) or int(meta.get("lookback", -1)) != lookback:
        return False, f"El modelo {meta['model_version']} no coincide con las features actuales."
    return True, ""


def predict_latest(model, meta: dict, features: dict[str, pd.DataFrame], mc_passes: int = 10,
                   explain: list[str] | None = None, top_k: int = 5) -> pd.DataFrame:
    """Predicción para la última sesión de cada símbolo (solo datos hasta esa sesión)."""
    from .train import predict_windows, standardize

    lookback = int(meta["lookback"])
    simbolos, ventanas = [], []
    for symbol, f in features.items():
        v = last_window(f, lookback)
        if v is not None:
            simbolos.append(symbol)
            ventanas.append(v)
    columnas = ["symbol", "probability", "probability_std", "confidence", "expected_return",
                "expected_drawdown", "lift", "neural_score", "explanation"]
    if not simbolos:
        return pd.DataFrame(columns=columnas)
    estandar = standardize(np.stack(ventanas), meta["_mean"], meta["_std"])
    pred = predict_windows(model, estandar, mc_passes=mc_passes)
    base = float(meta["base_rate"])
    tabla = pd.DataFrame({
        "symbol": simbolos,
        "probability": pred["probability"],
        "probability_std": pred.get("probability_std", np.full(len(simbolos), np.nan)),
        "expected_return": pred["expected_return"],
        "expected_drawdown": pred["expected_drawdown"],
    })
    tabla["confidence"] = np.clip(1 - tabla["probability_std"] / UNCERTAINTY_SCALE, 0, 1)
    tabla["lift"] = tabla["probability"] / base
    tabla["neural_score"] = [neural_score(p, base) for p in tabla["probability"]]
    tabla["explanation"] = None
    if explain:
        objetivo = [i for i, s in enumerate(simbolos) if s in set(explain)]
        if objetivo:
            explicaciones = sensitivity(model, estandar[objetivo], np.stack(ventanas)[objetivo], top_k=top_k)
            for i, exp in zip(objetivo, explicaciones):
                tabla.at[i, "explanation"] = exp
    return tabla[columnas]


def sensitivity(model, windows_std: np.ndarray, windows_raw: np.ndarray, top_k: int = 5) -> list[dict]:
    from .train import predict_windows

    n, _, f = windows_std.shape
    base = predict_windows(model, windows_std)["probability"]
    ocluidas = np.repeat(windows_std, f, axis=0)
    for j in range(f):
        ocluidas[j::f, :, j] = 0.0  # valor medio de entrenamiento
    p = predict_windows(model, ocluidas)["probability"].reshape(n, f)
    salida = []
    for i in range(n):
        efectos = base[i] - p[i]  # >0: la variable elevó la probabilidad
        orden = np.argsort(-np.abs(efectos))[:top_k]
        salida.append({
            "method": "sensibilidad por oclusión",
            "disclaimer": "Explicación del modelo: indica qué variables movieron su probabilidad. "
                          "No es una afirmación objetiva sobre la empresa.",
            "base_probability": float(base[i]),
            "factors": [{
                "feature": FEATURE_COLUMNS[j], "label": FEATURE_LABELS.get(FEATURE_COLUMNS[j], FEATURE_COLUMNS[j]),
                "effect": float(efectos[j]), "direction": "sube" if efectos[j] > 0 else "baja",
                "value": float(windows_raw[i, -1, j]),
            } for j in orden],
        })
    return salida
