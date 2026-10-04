"""Data Engine: validación, descarga incremental, proveedores, universo, calendario y almacenamiento."""

import types
from datetime import date

import numpy as np
import pandas as pd

from src.data import universe as universe_mod
from src.data.market import merge_incremental
from src.data.providers.market_data import SyntheticMarketProvider, split_download
from src.data.storage import load_prices, save_prices, write_json
from src.data.validation import adjusted_ohlcv, validate_prices
from src.market_calendar import is_trading_day, last_completed_session, nyse_holidays

VCFG = {"max_daily_move": 0.6, "spike_reversal_move": 0.35, "max_suspicious_days": 3, "max_missing_ratio": 0.03,
        "max_gap_sessions": 10, "ohlc_tolerance": 0.01}


def raw_frame(n=400, start="2023-01-02"):
    fechas = pd.bdate_range(start, periods=n)
    c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "adj_close": c * 0.98,
                         "volume": 1e6, "dividends": 0.0, "splits": 0.0}, index=fechas)


def test_validation_cleans_duplicates_invalid_rows_and_order():
    df = raw_frame()
    df = pd.concat([df, df.iloc[[5]]])                     # fecha duplicada
    df.iloc[10, df.columns.get_loc("close")] = -3           # precio inválido
    df.iloc[20, df.columns.get_loc("volume")] = -1          # volumen inválido
    df.iloc[30, df.columns.get_loc("high")] = df.iloc[30]["close"] * 0.999   # incoherencia OHLC leve
    df = df.iloc[::-1]                                      # orden inverso
    limpio, rep = validate_prices("X", df, VCFG, min_history=100)
    assert limpio.index.is_monotonic_increasing and not limpio.index.duplicated().any()
    assert rep.duplicates_removed == 1
    assert rep.invalid_rows_removed == 2
    assert rep.ohlc_fixed == 1
    assert (limpio["high"] >= limpio[["open", "close"]].max(axis=1)).all()
    assert rep.usable_for_training and rep.usable_for_signals


def test_validation_removes_spike_and_rejects_extreme_moves():
    df = raw_frame()
    df.iloc[100, df.columns.get_loc("adj_close")] *= 3      # pico que se revierte al día siguiente
    limpio, rep = validate_prices("X", df, VCFG, min_history=100)
    assert rep.spikes_removed == 1 and len(limpio) == len(df) - 1
    malo = raw_frame()
    for i in (50, 100, 150, 200):                          # saltos permanentes sin split
        malo.iloc[i:, malo.columns.get_loc("adj_close")] *= 2.0
    _, rep = validate_prices("X", malo, VCFG, min_history=100)
    assert not rep.usable_for_training and rep.suspicious_moves == 4


def test_validation_gaps_history_and_staleness():
    df = raw_frame(300)
    calendario = pd.bdate_range(df.index[0], periods=330)
    hueco = df.drop(df.index[100:115])
    _, rep = validate_prices("X", hueco, VCFG, min_history=100, calendar=calendario[:300])
    assert not rep.usable_for_signals and rep.max_gap_sessions == 15
    _, rep = validate_prices("X", df, VCFG, min_history=500)
    assert not rep.usable_for_training  # historial insuficiente
    _, rep = validate_prices("X", df, VCFG, min_history=100, calendar=calendario, as_of=calendario[-1],
                             max_stale_sessions=5)
    assert rep.usable_for_training and not rep.usable_for_signals  # desactualizada: no genera señales
    assert validate_prices("X", None, VCFG)[0] is None


def test_adjusted_ohlcv_uses_adjustment_factor():
    df = raw_frame()
    adj = adjusted_ohlcv(df)
    assert np.allclose(adj["close"], df["adj_close"])
    assert np.allclose(adj["open"], df["open"] * 0.98)
    assert np.allclose(adj["raw_close"], df["close"])


def test_merge_incremental_detects_adjustment_changes():
    cache = raw_frame(300)
    nuevo = raw_frame(310).iloc[280:]
    unido = merge_incremental(cache, nuevo)
    assert unido is not None and len(unido) == 310
    revisado = nuevo.copy()
    revisado["adj_close"] *= 0.99   # dividendo nuevo: el ajuste histórico cambió
    assert merge_incremental(cache, revisado) is None
    con_split = nuevo.copy()
    con_split.iloc[-1, con_split.columns.get_loc("splits")] = 4.0
    assert merge_incremental(cache, con_split) is None


def test_split_download_parses_yfinance_layouts():
    def yahoo(n=150):
        fechas = pd.bdate_range("2025-01-01", periods=n, tz="America/New_York")
        c = np.linspace(100, 120, n)
        return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Adj Close": c * 0.99,
                             "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0}, index=fechas)
    crudo = pd.concat({"AAA": yahoo(), "BBB": yahoo()}, axis=1)
    datos = split_download(crudo, ["AAA", "BBB", "FALTA"])
    assert set(datos) == {"AAA", "BBB"}
    assert list(datos["AAA"].columns) == ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
    assert datos["AAA"].index.tz is None
    assert "AAA" in split_download(crudo.swaplevel(axis=1), ["AAA"])


def test_synthetic_provider_is_consistent_over_time():
    a = SyntheticMarketProvider(end=date(2025, 6, 30)).fetch_history(["AAA"], None)["AAA"]
    b = SyntheticMarketProvider(end=date(2026, 6, 30)).fetch_history(["AAA"], None)["AAA"]
    assert np.allclose(a["close"], b.loc[a.index, "close"])
    assert SyntheticMarketProvider(missing=["AAA"]).fetch_history(["AAA"], None) == {}


def test_universe_from_wikipedia_with_fallback(monkeypatch, settings):
    filas = "".join(f"<tr><td>T{i}.B</td><td>Empresa {i}</td><td>Industrials</td><td>Machinery</td></tr>"
                    for i in range(60))
    html = ("<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>"
            f"{filas}</table>")
    monkeypatch.setattr(universe_mod.requests, "get",
                        lambda *a, **k: types.SimpleNamespace(text=html, raise_for_status=lambda: None))
    settings.offline = False
    settings.universe["universe"]["max_symbols"] = 0
    settings.universe["universe"]["additional_symbols"] = ["zzz"]
    tabla = universe_mod.load_universe(settings)
    assert "T0-B" in set(tabla["symbol"]) and "ZZZ" in set(tabla["symbol"])
    assert tabla.set_index("symbol").loc["T0-B", "indices"] == "S&P 500;Nasdaq-100"
    assert (settings.paths.universe / "universe.csv").exists()  # copia versionada

    def falla(*a, **k):
        raise ConnectionError("sin red")
    monkeypatch.setattr(universe_mod.requests, "get", falla)
    assert "T0-B" in set(universe_mod.load_universe(settings)["symbol"])  # usa la copia versionada


def test_nyse_calendar():
    festivos = nyse_holidays(2026)
    assert date(2026, 4, 3) in festivos       # Viernes Santo
    assert date(2026, 7, 3) in festivos       # 4 de julio observado (cae en sábado)
    assert date(2026, 11, 26) in festivos     # Acción de Gracias
    assert not is_trading_day(date(2026, 10, 3))   # sábado
    from datetime import datetime, timezone
    # 21:00 UTC = 17:00 en Nueva York (horario de verano): la sesión del día ya cerró
    assert last_completed_session(datetime(2026, 10, 2, 21, 0, tzinfo=timezone.utc)) == date(2026, 10, 2)
    assert last_completed_session(datetime(2026, 10, 2, 19, 0, tzinfo=timezone.utc)) == date(2026, 10, 1)


def test_storage_roundtrip_and_nan_guard(tmp_path):
    df = raw_frame(50)
    save_prices(tmp_path, "AAA", df)
    leido = load_prices(tmp_path, "AAA")
    assert np.allclose(leido["close"], df["close"]) and len(leido) == 50
    ruta = write_json(tmp_path / "x.json", {"a": float("nan"), "b": np.float32(1.5)})
    assert ruta.read_text().replace(" ", "").replace("\n", "") == '{"a":null,"b":1.5}'
