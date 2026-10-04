"""Paper Trading / Tracking Engine (solo simulación)."""

import numpy as np
import pandas as pd
import pytest

from src.trading.broker import PaperBroker, RealExecutionDisabled, get_broker
from src.trading.manual import load_manual_positions
from src.trading.paper import PaperTrader, entry_decision


def raw(rows, start="2026-09-01"):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.bdate_range(start, periods=len(rows)))
    df["volume"], df["splits"], df["dividends"] = 1e6, 0.0, 0.0
    return df


def candidate(symbol="AAA", price=100.0, final=80.0, stat=70.0, prob=None):
    return {"symbol": symbol, "name": symbol, "price": price, "final_score": final, "statistical_score": stat,
            "opportunity_probability": prob, "neural_threshold": 0.45 if prob is not None else None,
            "explanation": {"summary": "Seleccionada porque: test", "reasons": [
                {"text": "Precio sobre la MA(99)", "why_it_matters": "Tendencia de largo plazo"}], "risks": []}}


def run(trader, df, dia, symbol="AAA"):
    fecha = df.index[dia]
    return trader.process(fecha, {symbol: df.loc[:fecha]}, {symbol: pd.Series(False, index=df.index)},
                          df.index)


def test_lifecycle_pending_open_take_profit(settings):
    trader = PaperTrader(settings.paths.state, settings)
    df = raw([(100, 101, 99, 100)] + [(100.5, 102, 99.5, 101)] + [(101, 104, 100, 103)] * 3 + [(110, 116, 109, 115)])
    eventos = trader.create_signals(df.index[0], [candidate()], model_available=False, model_version=None)
    assert eventos[0]["type"] == "signal" and trader.trades[0]["status"] == "PENDING"
    run(trader, df, 1)
    t = trader.trades[0]
    assert t["status"] == "OPEN" and t["entry_date"] == df.index[1].strftime("%Y-%m-%d")
    assert t["entry_price"] == 100.5          # apertura de la sesión siguiente a la señal
    assert t["reasons"] == ["Precio sobre la MA(99)"] and t["why_it_matters"] == ["Tendencia de largo plazo"]
    run(trader, df, 5)
    assert t["status"] == "CLOSED" and t["exit_reason"] == "Take-profit"
    assert np.isclose(t["exit_price"], 100.5 * 1.15)
    assert t["actual_return"] > 0.14 and t["holding_days"] == 5
    trader.save()
    assert PaperTrader(settings.paths.state, settings).trades[0]["status"] == "CLOSED"


def test_stop_loss_and_time_exit(settings):
    trader = PaperTrader(settings.paths.state, settings)
    df = raw([(100, 101, 99, 100), (100, 101, 99, 100), (99, 99, 90, 91)] + [(91, 92, 90, 91)] * 3)
    trader.create_signals(df.index[0], [candidate()], False, None)
    run(trader, df, 5)
    t = trader.trades[0]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "Stop-loss" and np.isclose(t["exit_price"], 92.0)
    reglas = settings.rules
    trader2 = PaperTrader(settings.paths.state / "b", settings)
    plano = raw([(100, 101, 99, 100)] * (int(reglas["holding_period_days"]) + 3))
    trader2.create_signals(plano.index[0], [candidate()], False, None)
    run(trader2, plano, len(plano) - 1)
    assert trader2.trades[0]["exit_reason"] == "Fin del periodo de mantenimiento"
    assert trader2.trades[0]["holding_days"] == int(reglas["holding_period_days"])


def test_invalidation_by_gap_and_split_adjustment(settings):
    trader = PaperTrader(settings.paths.state, settings)
    df = raw([(100, 101, 99, 100), (120, 121, 119, 120)])
    trader.create_signals(df.index[0], [candidate()], False, None)
    run(trader, df, 1)
    assert trader.trades[0]["status"] == "INVALIDATED" and "Abrió" in trader.trades[0]["invalidation_reason"]

    trader = PaperTrader(settings.paths.state / "s", settings)
    df = raw([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
    trader.create_signals(df.index[0], [candidate()], False, None)
    run(trader, df, 1)
    acciones = trader.trades[0]["shares"]
    # split 4:1 posterior: el proveedor reescribe la historia en precios divididos por 4
    df2 = df / 4
    df2["volume"], df2["dividends"], df2["splits"] = 4e6, 0.0, 0.0
    df2 = pd.concat([df2, raw([(25, 25.2, 24.8, 25)], start=df.index[-1] + pd.offsets.BDay())])
    df2.iloc[-1, df2.columns.get_loc("splits")] = 4.0
    run(trader, df2, len(df2) - 1)
    t = trader.trades[0]
    assert t["status"] == "OPEN" and np.isclose(t["entry_price"], 25.0) and np.isclose(t["shares"], acciones * 4)


def test_entry_rules_and_modes(settings):
    reglas = settings.rules
    assert entry_decision(candidate(final=80, stat=70), reglas, model_available=False)[0]
    assert not entry_decision(candidate(final=50), reglas, False)[0]
    assert not entry_decision(candidate(stat=40), reglas, False)[0]
    assert not entry_decision(candidate(prob=0.30), reglas, model_available=True)[0]
    assert entry_decision(candidate(prob=0.60), reglas, model_available=True)[0]
    exigente = dict(reglas, require_neural_model=True)
    assert not entry_decision(candidate(), exigente, model_available=False)[0]


def test_capacity_and_no_duplicates(settings):
    trader = PaperTrader(settings.paths.state, settings)
    fecha = pd.Timestamp("2026-09-01")
    cands = [candidate(f"S{i}") for i in range(10)] + [candidate("S0")]
    eventos = trader.create_signals(fecha, cands, False, None)
    assert len(eventos) == int(settings.rules["max_new_positions_per_day"])
    assert trader.create_signals(fecha, cands, False, None) == []   # misma sesión: no duplica


def test_broker_never_executes_real_orders():
    assert isinstance(get_broker("PAPER_TRADING"), PaperBroker)
    assert get_broker("RESEARCH").submit_order("AAPL", "buy", 1)["simulated"] is True
    with pytest.raises(RealExecutionDisabled):
        get_broker("LIVE")


def test_manual_positions(tmp_path):
    archivo = tmp_path / "mis.yaml"
    archivo.write_text("posiciones:\n  - ticker: brk.b\n    fecha_compra: 2026-01-02\n    precio_compra: 400\n"
                       "    acciones: 2\n    razones: Valor a largo plazo\n  - ticker: MALA\n", encoding="utf-8")
    posiciones = load_manual_positions(str(archivo))
    assert len(posiciones) == 1 and posiciones[0]["symbol"] == "BRK-B" and posiciones[0]["reasons"] == ["Valor a largo plazo"]
