"""Technical Analysis Engine."""

import numpy as np
import pandas as pd

from src.indicators.technical import add_market_context, bars_since, compute_indicators, crossover, rsi, sma
from conftest import ohlc_from_closes


def test_moving_averages_7_25_99():
    ind = compute_indicators(ohlc_from_closes(np.arange(1, 251)))
    assert ind["ma7"].iloc[-1] == np.mean(np.arange(244, 251))
    assert ind["ma25"].iloc[-1] == np.mean(np.arange(226, 251))
    assert ind["ma99"].iloc[-1] == np.mean(np.arange(152, 251))
    assert ind["ma99"].iloc[:98].isna().all() and ind["ma99"].notna().iloc[98]
    assert np.isclose(ind["close_ma7"].iloc[-1], 250 / ind["ma7"].iloc[-1])
    assert ind["ma7_gt_ma25"].iloc[-1] == 1 and ind["ma25_gt_ma99"].iloc[-1] == 1 and ind["above_ma99"].iloc[-1] == 1


def test_indicator_columns_present(frame):
    ind = compute_indicators(frame)
    for col in ("sma50", "ema12", "rsi14", "macd", "macd_signal", "macd_hist", "atr14", "bb_upper", "bb_lower",
                "mom10", "roc20", "vol20", "rel_volume", "drawdown252", "dist_ma7", "dist_ma25", "dist_ma99",
                "cross_ma7_ma25", "cross_ma25_ma99", "slope_ma7", "slope_ma25", "slope_ma99"):
        assert col in ind.columns
    assert ind["rsi14"].dropna().between(0, 100).all()
    assert (ind["drawdown252"].dropna() <= 1e-12).all()


def test_rsi_extremes_and_crossovers():
    assert rsi(pd.Series(np.arange(1, 60, dtype=float))).iloc[-1] == 100
    assert rsi(pd.Series(np.arange(60, 1, -1, dtype=float))).iloc[-1] < 1
    rapida = pd.Series([1, 1, 3, 3, 1], dtype=float)
    lenta = pd.Series([2, 2, 2, 2, 2], dtype=float)
    assert crossover(rapida, lenta).tolist() == [0, 0, 1, 0, -1]
    assert bars_since(pd.Series([False, True, False, False])).tolist() == [250, 0, 1, 2]
    assert sma(pd.Series([1.0, 2.0]), 5).isna().all()


def test_market_context_alignment(frame, market_frame):
    ind = add_market_context(compute_indicators(frame), compute_indicators(market_frame))
    assert ind["mkt_ret21"].notna().sum() > 1000
    assert np.allclose(ind["rel_strength63"].dropna(), (ind["ret63"] - ind["mkt_ret63"]).dropna())
