"""Ranking Engine: pesos de config/model.yaml y reparto cuando falta un componente."""

import numpy as np
import pandas as pd

from src.explanations.engine import explain_symbol
from src.ranking.engine import combine_scores


def test_combine_scores_reweights_missing_components(settings):
    pesos = settings.model["ranking"]["weights"]
    todos = combine_scores({"statistical": 80, "news": 60, "fundamental": 70, "neural": 90, "risk": 50}, pesos)
    total = sum(pesos.values())
    esperado = sum(v * pesos[k] / total for k, v in {"statistical": 80, "news": 60, "fundamental": 70,
                                                     "neural": 90, "risk": 50}.items())
    assert np.isclose(todos["final_score"], esperado, atol=0.06)
    parcial = combine_scores({"statistical": 80, "news": None, "fundamental": None, "neural": None, "risk": 50}, pesos)
    w = pesos["statistical"] + pesos["risk"]
    assert np.isclose(parcial["final_score"], (80 * pesos["statistical"] + 50 * pesos["risk"]) / w, atol=0.06)
    assert set(parcial["missing"]) == {"news", "fundamental", "neural"}
    assert np.isclose(sum(parcial["weights_used"].values()), 1.0)
    assert combine_scores({}, pesos)["final_score"] is None


def test_explanations_are_built_from_real_values(settings):
    ind = pd.Series({"close": 110.0, "ma7": 108.0, "ma25": 105.0, "ma99": 100.0, "bars_since_cross_7_25": 2,
                     "slope_ma99": 0.01, "ret63": 0.12, "rel_strength63": 0.05, "macd_hist": 0.4, "rsi14": 58,
                     "dist_ma25": 110 / 105 - 1, "rel_volume": 1.6, "ret1": 0.01, "vol_trend": 1.3, "ret5": 0.02,
                     "vol20": 0.22, "drawdown252": -0.03, "dist_high252": -0.02, "bars_since_golden_25_99": 100})
    scores = pd.Series({"history_score": 61.0})
    exp = explain_symbol(ind, scores, settings)
    textos = " ".join(r["text"] for r in exp["reasons"])
    assert "+10.0% sobre la MA(99) (110.00 vs 100.00)" in textos
    assert "cruzó por encima de la MA(25) hace 2 sesiones" in textos
    assert all(r["why_it_matters"] for r in exp["reasons"])
    assert exp["summary"].startswith("Seleccionada porque:")
    assert "no es una afirmación objetiva" in exp["disclaimers"]["model"].lower()
    baja = explain_symbol(ind.copy().replace({110.0: 90.0}), scores, settings)
    assert any(r["code"] == "price_below_ma99" for r in baja["risks"])
