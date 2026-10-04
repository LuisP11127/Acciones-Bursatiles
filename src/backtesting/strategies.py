"""Definición de las estrategias comparadas en el backtesting.

Noticias y fundamentales no tienen historial punto a punto en fuentes gratuitas:
en el backtesting el final_score se calcula con los componentes que sí lo
tienen (statistical, neural fuera de muestra y risk), repartiendo los pesos de
config/model.yaml entre ellos. Se indica en los resultados.
"""

from __future__ import annotations

import numpy as np

from .engine import Panel

BACKTEST_COMPONENTS = ("statistical", "neural", "risk")


def neural_score_matrix(prob: np.ndarray, base: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.maximum(prob, 1e-6) / base
        return 50 + 50 * np.tanh(np.log2(lift))


def weighted(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    suma = None
    peso = None
    for nombre, valores in components.items():
        w = float(weights.get(nombre, 0))
        if w <= 0:
            continue
        disponible = np.isfinite(valores)
        aporte = np.where(disponible, valores * w, 0.0)
        pw = np.where(disponible, w, 0.0)
        suma = aporte if suma is None else suma + aporte
        peso = pw if peso is None else peso + pw
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(peso > 0, suma / peso, np.nan)


def strategy_signals(name: str, panel: Panel, settings, prob: np.ndarray | None = None,
                     threshold: np.ndarray | None = None, base: np.ndarray | None = None):
    """Devuelve (señal de entrada booleana, score para ordenar candidatas), ambas (T, N)."""
    reglas = settings.strategy["strategy"]
    pesos = settings.model["ranking"]["weights"]
    stat = panel.columns["statistical_score"]
    riesgo = panel.columns["risk_score"]
    min_final, min_stat = float(reglas["min_final_score"]), float(reglas["min_statistical_score"])
    if name == "combined":
        if prob is None:
            raise ValueError("La estrategia combinada necesita predicciones fuera de muestra")
        nn = neural_score_matrix(prob, base)
        final = weighted({"statistical": stat, "neural": nn, "risk": riesgo}, pesos)
        tiene_nn = np.isfinite(prob)
        entrada = (final >= min_final) & (stat >= min_stat) & tiene_nn & (prob >= threshold)
        return np.nan_to_num(entrada, nan=False).astype(bool), final
    if name == "statistical_only":
        final = weighted({"statistical": stat, "risk": riesgo}, pesos)
        entrada = (final >= min_final) & (stat >= min_stat)
        return np.nan_to_num(entrada, nan=False).astype(bool), final
    if name == "neural_only":
        if prob is None:
            raise ValueError("La estrategia neuronal necesita predicciones fuera de muestra")
        entrada = np.isfinite(prob) & (prob >= threshold)
        return entrada.astype(bool), prob
    if name == "ma_crossover":
        entrada = (panel.columns["cross_ma7_ma25"] > 0) & (panel.columns["above_ma99"] > 0)
        return np.nan_to_num(entrada, nan=False).astype(bool), panel.columns["ret63"]
    raise ValueError(f"Estrategia desconocida: {name}")


STRATEGY_DESCRIPTIONS = {
    "combined": "Score final (estadístico + red neuronal fuera de muestra + riesgo) con los umbrales de "
                "strategy.yaml y probabilidad de la red sobre su umbral.",
    "statistical_only": "Solo el score estadístico y el de riesgo, con los mismos umbrales.",
    "neural_only": "Solo la probabilidad de la red neuronal (fuera de muestra) sobre su umbral.",
    "ma_crossover": "Regla sencilla: la MA(7) cruza por encima de la MA(25) con el precio sobre la MA(99).",
    "buy_hold_benchmark": "Comprar y mantener el índice de referencia (SPY).",
    "equal_weight_universe": "Cartera equiponderada de todas las acciones operables del universo.",
}
