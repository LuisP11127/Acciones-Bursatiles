"""Backtesting Engine: coherencia con las etiquetas, ejecución en T+1 y métricas."""

import numpy as np
import pandas as pd

from src.backtesting.engine import Panel, Rules, buy_and_hold, check_exit, equal_weight, run_strategy
from src.backtesting.metrics import equity_metrics, trade_metrics, yearly_returns
from src.features.labels import opportunity_labels

RULES = Rules(holding_period_days=5, stop_loss=0.05, take_profit=0.10, exit_on_trend_break=False,
              max_entry_gap=None, max_positions=100, max_new_positions_per_day=100, initial_capital=100_000,
              cost_per_side=0.001)


def panel_from(frames: dict[str, pd.DataFrame]) -> Panel:
    fechas = frames[next(iter(frames))].index
    simbolos = sorted(frames)
    mat = {c: np.column_stack([frames[s][c].to_numpy() for s in simbolos]) for c in ("open", "high", "low", "close")}
    T, N = len(fechas), len(simbolos)
    return Panel(dates=fechas, symbols=simbolos, open=mat["open"], high=mat["high"], low=mat["low"],
                 close=mat["close"], trend_break=np.zeros((T, N), bool), tradable=np.ones((T, N), bool))


def test_engine_matches_label_simulation(frame):
    """Sin límites de cartera, cada operación del motor reproduce la etiqueta del mismo día."""
    df = frame.iloc[-400:]
    panel = panel_from({"X": df})
    senal = np.zeros((len(df), 1), bool)
    dias = [10, 60, 120, 200, 260, 330]
    senal[dias, 0] = True
    res = run_strategy(panel, senal, np.ones_like(senal, float), RULES, 0, len(df) - 1, "test")
    et = opportunity_labels(df, horizon=5, stop_loss=0.05, take_profit=0.10, cost_per_side=0.001, min_return=0.0)
    assert len(res.trades) == len(dias)
    for d, (_, op) in zip(dias, res.trades.iterrows()):
        assert op["entry_date"] == df.index[d + 1]                 # se ejecuta en la apertura de T+1
        assert np.isclose(op["entry_price"], df["open"].iloc[d + 1])
        assert np.isclose(op["return"], et["trade_return"].iloc[d], atol=1e-9)
        assert op["holding_days"] == et["holding_days"].iloc[d]


def test_entry_gap_invalidation_and_capacity(frame):
    base = frame.iloc[-100:]
    df = base.copy()
    df.iloc[11, df.columns.get_loc("open")] = df["close"].iloc[10] * 1.2   # gap del 20% tras la señal (solo X)
    panel = panel_from({"X": df, "Y": base * 1.01, "Z": base * 0.99})
    reglas = Rules(**{**RULES.__dict__, "max_entry_gap": 0.05, "max_positions": 2, "max_new_positions_per_day": 2})
    senal = np.zeros((len(df), 3), bool)
    senal[10, :] = True
    orden = np.tile(np.array([3.0, 2.0, 1.0]), (len(df), 1))
    res = run_strategy(panel, senal, orden, reglas, 0, len(df) - 1, "test")
    # X invalidada por gap; Y entra; Z no entra por la capacidad diaria (2 candidatas: X e Y)
    assert res.invalidated == 1
    assert list(res.trades["symbol"]) == ["Y"]


def test_check_exit_order():
    r = RULES
    assert check_exit(90, 95, 89, 94, 95, 110, 2, False, False, r) == (90, "stop_loss")       # gap bajo el stop
    assert check_exit(100, 112, 94, 100, 95, 110, 2, False, False, r) == (95, "stop_loss")    # ambos: stop primero
    assert check_exit(100, 111, 99, 105, 95, 110, 2, False, False, r) == (110, "take_profit")
    assert check_exit(100, 101, 99, 100.5, 95, 110, 5, False, False, r) == (100.5, "time")
    assert check_exit(100, 101, 99, 100.5, 95, 110, 2, False, True, r) == (100, "trend_break")
    assert check_exit(100, 101, 99, 100.5, 95, 110, 2, False, False, r) is None


def test_equity_metrics_known_values():
    fechas = pd.bdate_range("2020-01-01", periods=253)
    curva = pd.Series(100 * 1.001 ** np.arange(253), index=fechas)
    m = equity_metrics(curva)
    assert np.isclose(m["total_return"], 1.001 ** 252 - 1)
    assert m["max_drawdown"] == 0 and m["volatility"] < 1e-9
    caida = pd.Series([100, 120, 90, 130], index=pd.bdate_range("2020-01-01", periods=4))
    assert np.isclose(equity_metrics(caida)["max_drawdown"], 90 / 120 - 1)
    ops = pd.DataFrame({"return": [0.1, -0.05, 0.02, -0.01], "pnl": [100, -50, 20, -10], "holding_days": [5, 3, 4, 2]})
    tm = trade_metrics(ops)
    assert tm["trades"] == 4 and np.isclose(tm["win_rate"], 0.5) and np.isclose(tm["profit_factor"], 120 / 60)
    assert np.isclose(tm["avg_gain"], 0.06) and np.isclose(tm["avg_loss"], -0.03)
    anual = yearly_returns(pd.Series([100, 110, 121], index=pd.to_datetime(["2020-12-30", "2020-12-31", "2021-12-31"])))
    assert np.isclose(anual[-1]["return"], 0.1)


def test_benchmarks_start_at_capital(frame):
    df = frame.iloc[-50:]
    panel = panel_from({"X": df, "Y": df * 1.2})
    bh = buy_and_hold(df["close"], df.index[10:], 1000)
    ew = equal_weight(panel, 10, 49, 1000)
    assert np.isclose(bh.iloc[0], 1000) and np.isclose(ew.iloc[0], 1000)
    assert np.isclose(ew.iloc[-1] / 1000, df["close"].iloc[49] / df["close"].iloc[10], rtol=1e-6)
