"""Features (punto a punto) y etiquetas de entrenamiento (triple barrera)."""

from datetime import timedelta

import numpy as np
import pandas as pd

from src.features.builder import FEATURE_COLUMNS, build_feature_frame, decision_cutoff, news_features
from src.features.labels import LABEL_COLUMNS, opportunity_labels
from src.indicators.technical import add_market_context, compute_indicators

PARAMS = dict(horizon=5, stop_loss=0.05, take_profit=0.10, cost_per_side=0.0, min_return=0.02)


def bars(rows, start="2024-01-01"):
    """rows = [(open, high, low, close), ...]"""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.bdate_range(start, periods=len(rows)))
    df["volume"] = 1e6
    return df


def test_feature_frame_columns_and_types(frame, market_frame):
    feats = build_feature_frame(add_market_context(compute_indicators(frame), compute_indicators(market_frame)))
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.dtypes.eq(np.float32).all()
    assert feats.dropna().shape[0] > 0.9 * len(feats)


def test_features_are_scale_invariant(frame, market_frame):
    """Multiplicar todos los precios por una constante (p. ej., un ajuste por split) no cambia las features."""
    mercado = compute_indicators(market_frame)
    a = build_feature_frame(add_market_context(compute_indicators(frame), mercado))
    escalado = frame.copy()
    for col in ("open", "high", "low", "close", "raw_close"):
        escalado[col] = escalado[col] * 7.3
    b = build_feature_frame(add_market_context(compute_indicators(escalado), mercado))
    pd.testing.assert_frame_equal(a, b, atol=1e-4, rtol=1e-4)


def test_label_take_profit_stop_and_time_exit():
    filas = [(100, 100, 100, 100)] + [(100, 101, 99, 100)] * 2 + [(101, 111, 100, 110)] + [(110, 110, 109, 110)] * 6
    et = opportunity_labels(bars(filas), **PARAMS)
    assert et.loc[et.index[0], "exit_type"] == 1 and np.isclose(et.loc[et.index[0], "trade_return"], 0.10)
    filas = [(100, 100, 100, 100)] + [(100, 101, 99, 100), (99, 99, 94, 95)] + [(95, 96, 94, 95)] * 7
    et = opportunity_labels(bars(filas), **PARAMS)
    assert et.loc[et.index[0], "exit_type"] == -1 and np.isclose(et.loc[et.index[0], "trade_return"], -0.05)
    assert et.loc[et.index[0], "label"] == 0
    filas = [(100, 100, 100, 100)] + [(100, 101, 99, 100.5)] * 4 + [(100, 104, 100, 103)] + [(103, 103, 103, 103)] * 4
    et = opportunity_labels(bars(filas), **PARAMS)
    assert et.loc[et.index[0], "exit_type"] == 0 and np.isclose(et.loc[et.index[0], "trade_return"], 0.03)
    assert et.loc[et.index[0], "holding_days"] == 5 and et.loc[et.index[0], "label"] == 1


def test_label_conservative_same_bar_and_gaps():
    # Stop y objetivo en la misma vela: se asume el stop
    filas = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 115, 90, 100)] + [(100, 100, 100, 100)] * 6
    et = opportunity_labels(bars(filas), **PARAMS)
    assert et.loc[et.index[0], "exit_type"] == -1
    # Gap bajista a través del stop: se sale a la apertura (peor que el stop)
    filas = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (90, 91, 89, 90)] + [(90, 90, 90, 90)] * 6
    et = opportunity_labels(bars(filas), **PARAMS)
    assert np.isclose(et.loc[et.index[0], "trade_return"], -0.10)
    # Gap alcista a través del objetivo: se sale a la apertura
    filas = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (112, 113, 111, 112)] + [(112, 112, 112, 112)] * 6
    et = opportunity_labels(bars(filas), **PARAMS)
    assert np.isclose(et.loc[et.index[0], "trade_return"], 0.12) and et.loc[et.index[0], "exit_type"] == 1


def test_label_costs_and_future_unknown():
    filas = [(100, 100, 100, 100)] * 12
    et = opportunity_labels(bars(filas), horizon=5, stop_loss=0.05, take_profit=0.10, cost_per_side=0.001,
                            min_return=0.0)
    assert np.isclose(et["trade_return"].iloc[0], 0.999 / 1.001 - 1)
    assert et["label"].iloc[0] == 0          # con costes, quedarse plano no es oportunidad
    assert et["label"].iloc[-5:].isna().all()  # las últimas H sesiones no tienen futuro aún
    assert et["label"].iloc[:-5].notna().all()
    assert list(et.columns) == LABEL_COLUMNS


def test_news_features_respect_decision_cutoff():
    sesiones = pd.bdate_range("2026-09-28", periods=5)
    cierre = decision_cutoff(sesiones[2])
    noticias = [{"published_at": (cierre - timedelta(hours=2)).isoformat(), "sentiment": 0.8},
                {"published_at": (cierre + timedelta(minutes=30)).isoformat(), "sentiment": -0.9}]
    f = news_features(noticias, sesiones)
    assert np.isclose(f.loc[sesiones[2], "news_sentiment_7d"], 0.8)       # la posterior al cierre no cuenta
    assert np.isclose(f.loc[sesiones[3], "news_sentiment_7d"], (0.8 - 0.9) / 2)
    assert np.isnan(f.loc[sesiones[0], "news_sentiment_7d"])
