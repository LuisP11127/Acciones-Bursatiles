"""Análisis fundamental opcional.

Cada métrica disponible se traduce a 0-100 con rangos fijos y simples; el
fundamental_score es la media de las disponibles (si hay al menos `min_metrics`).
Es una heurística interna, no una valoración de la empresa. Estos datos solo
existen en el presente (sin historial punto a punto), por eso no se usan en el
backtesting ni en el entrenamiento.
"""

from __future__ import annotations

import numpy as np

METRICS = ["revenue_growth", "earnings_growth", "profit_margin", "pe", "peg", "debt_to_equity",
           "free_cash_flow", "roe", "roic", "market_cap"]
LABELS = {
    "revenue_growth": "Crecimiento de ingresos", "earnings_growth": "Crecimiento de beneficios",
    "profit_margin": "Margen de beneficio", "pe": "PER", "peg": "PEG", "debt_to_equity": "Deuda / patrimonio",
    "free_cash_flow": "Flujo de caja libre", "roe": "ROE", "roic": "ROIC", "market_cap": "Capitalización",
}


def _tramos(x: float, puntos: list[tuple[float, float]]) -> float:
    xs, ys = zip(*puntos)
    return float(np.interp(x, xs, ys))


def metric_scores(f: dict) -> dict[str, float]:
    s: dict[str, float] = {}

    def num(clave):
        try:
            v = float(f.get(clave))
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    if (v := num("revenue_growth")) is not None:
        s["revenue_growth"] = _tramos(v, [(-0.2, 10), (0, 40), (0.1, 65), (0.25, 85), (0.5, 95)])
    if (v := num("earnings_growth")) is not None:
        s["earnings_growth"] = _tramos(v, [(-0.5, 10), (0, 40), (0.15, 65), (0.4, 85), (1.0, 95)])
    if (v := num("profit_margin")) is not None:
        s["profit_margin"] = _tramos(v, [(-0.1, 10), (0, 30), (0.1, 55), (0.2, 70), (0.35, 90)])
    if (v := num("pe")) is not None:
        s["pe"] = 20.0 if v <= 0 else _tramos(v, [(5, 70), (15, 75), (25, 65), (40, 50), (80, 30), (150, 15)])
    if (v := num("peg")) is not None:
        s["peg"] = 30.0 if v <= 0 else _tramos(v, [(0.5, 85), (1, 75), (2, 55), (3, 40), (5, 25)])
    if (v := num("debt_to_equity")) is not None:
        s["debt_to_equity"] = _tramos(max(v, 0), [(0, 85), (0.5, 75), (1, 60), (2, 40), (4, 20)])
    if (v := num("free_cash_flow")) is not None:
        cap = num("market_cap")
        if cap and cap > 0:
            s["free_cash_flow"] = _tramos(v / cap, [(-0.05, 15), (0, 35), (0.02, 55), (0.05, 75), (0.1, 90)])
        else:
            s["free_cash_flow"] = 65.0 if v > 0 else 25.0
    for clave in ("roe", "roic"):
        if (v := num(clave)) is not None:
            s[clave] = _tramos(v, [(-0.1, 10), (0, 30), (0.1, 55), (0.2, 75), (0.35, 90)])
    return s


def fundamental_analysis(f: dict, min_metrics: int = 3) -> dict:
    puntos = metric_scores(f)
    disponibles = [m for m in METRICS if f.get(m) is not None]
    no_disponibles = [m for m in METRICS if f.get(m) is None]
    score = round(float(np.mean(list(puntos.values()))), 1) if len(puntos) >= min_metrics else None
    return {
        "fundamental_score": score,
        "metric_scores": {k: round(v, 1) for k, v in puntos.items()},
        "values": {m: f.get(m) for m in disponibles},
        "available": disponibles,
        "unavailable": no_disponibles,
        "note": ("Datos del proveedor en la fecha de actualización; sin historial punto a punto, por lo que "
                 "no se usan en backtesting." if score is not None else
                 "Datos fundamentales insuficientes en el proveedor para calcular un score."),
    }
