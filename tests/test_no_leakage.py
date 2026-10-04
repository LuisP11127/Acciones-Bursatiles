"""Detección de fugas de información (look-ahead bias).

Principio: todo lo que se usa para decidir en T (indicadores, features, scores,
contexto de mercado, noticias, liquidez) debe ser IDÉNTICO si se cambia o se
elimina cualquier dato posterior a T. Solo las etiquetas pueden mirar al futuro.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.providers.base import NewsItem
from src.features.builder import FEATURE_COLUMNS
from src.features.labels import LABEL_COLUMNS
from src.ml.dataset import build_panel_dataset, fit_scaler
from src.ml.walkforward import make_folds
from src.news.scoring import score_news
from src.pipeline.analysis import analyze_symbol

CORTE = -300  # T = 300 sesiones antes del final


def _futuro_alterado(frame: pd.DataFrame, semilla: int = 1) -> pd.DataFrame:
    """Sustituye todo lo posterior a T por otra trayectoria."""
    alterado = frame.copy()
    rng = np.random.default_rng(semilla)
    n = len(alterado) + CORTE + 1
    factor = np.exp(np.cumsum(rng.normal(0.01, 0.05, len(alterado) - n)))
    for col in ("open", "high", "low", "close", "raw_close"):
        alterado.iloc[n:, alterado.columns.get_loc(col)] = alterado[col].iloc[n:].to_numpy() * factor
    alterado.iloc[n:, alterado.columns.get_loc("volume")] *= 5
    return alterado


@pytest.fixture
def pair(frame, market_frame, settings):
    original = analyze_symbol("TEST", frame, market_frame, settings)
    alterado = analyze_symbol("TEST", _futuro_alterado(frame), _futuro_alterado(market_frame, 2), settings)
    truncado = analyze_symbol("TEST", frame.iloc[:CORTE + 1], market_frame.iloc[:CORTE + 1], settings)
    return original, alterado, truncado


def test_indicators_do_not_use_future(pair):
    original, alterado, truncado = pair
    t = original.ind.index[CORTE]
    pd.testing.assert_frame_equal(original.ind.loc[:t], alterado.ind.loc[:t])
    pd.testing.assert_frame_equal(original.ind.loc[:t], truncado.ind.loc[:t])


def test_features_do_not_use_future(pair):
    original, alterado, truncado = pair
    t = original.features.index[CORTE]
    pd.testing.assert_frame_equal(original.features.loc[:t], alterado.features.loc[:t])
    pd.testing.assert_frame_equal(original.features.loc[:t], truncado.features.loc[:t])


def test_statistical_scores_do_not_use_future(pair):
    """Incluye el componente histórico, que usa resultados de operaciones pasadas."""
    original, alterado, truncado = pair
    t = original.scores.index[CORTE]
    pd.testing.assert_frame_equal(original.scores.loc[:t], alterado.scores.loc[:t])
    pd.testing.assert_frame_equal(original.scores.loc[:t], truncado.scores.loc[:t])


def test_liquidity_filter_is_point_in_time(pair):
    original, alterado, _ = pair
    t = original.ind.index[CORTE]
    a = original.tradable(5, 1e6).loc[:t]
    b = alterado.tradable(5, 1e6).loc[:t]
    pd.testing.assert_series_equal(a, b)


def test_labels_are_the_only_forward_looking_data(pair, settings):
    original, alterado, _ = pair
    t = original.labels.index[CORTE - 30]
    horizonte = int(settings.rules["holding_period_days"])
    # La etiqueta SÍ cambia al alterar el futuro (mira H sesiones adelante)...
    cerca = original.labels.index[CORTE - horizonte // 2]
    assert not np.allclose(original.labels.loc[cerca:, "trade_return"].dropna().head(5),
                           alterado.labels.loc[cerca:, "trade_return"].dropna().head(5))
    # ...pero las etiquetas que terminan antes de T no cambian.
    pd.testing.assert_frame_equal(original.labels.loc[:t], alterado.labels.loc[:t])
    # Y ninguna columna de etiqueta (retorno futuro, máxima subida, drawdown) es una feature.
    assert not set(LABEL_COLUMNS) & set(FEATURE_COLUMNS)
    assert not any(k in c for c in FEATURE_COLUMNS for k in ("future", "label", "mfe", "mae", "trade_return"))


def test_windows_only_look_backwards(frame, market_frame, settings):
    a = analyze_symbol("TEST", frame, market_frame, settings)
    ds = build_panel_dataset({"TEST": a}, lookback=10, min_price=0, min_dollar_volume=0)
    fila = int(np.nonzero(ds.window_ok)[0][500])
    ventana = ds.windows(np.array([fila]))[0]
    assert np.allclose(ventana[-1], ds.features[fila])           # termina en T
    assert np.allclose(ventana[0], ds.features[fila - 9])        # empieza 9 sesiones antes
    primera = int(np.argmax(ds.window_ok))
    assert not np.isnan(a.features.iloc[primera - 9:primera + 1].to_numpy()).any()   # ventana completa y sin NaN
    assert np.isnan(a.features.iloc[:primera].to_numpy()).any(axis=1)[primera - 10]   # la fila anterior no lo estaba


def test_windows_never_cross_symbols(frame, market_frame, settings):
    a = analyze_symbol("AAA", frame, market_frame, settings)
    b = analyze_symbol("BBB", frame * 1.5, market_frame, settings)
    ds = build_panel_dataset({"AAA": a, "BBB": b}, lookback=20, min_price=0, min_dollar_volume=0)
    inicio_b = int(np.argmax(ds.symbol_idx == 1))
    assert not ds.window_ok[inicio_b:inicio_b + 19].any()   # las primeras filas de BBB no usan filas de AAA


def test_scaler_fitted_only_on_training_rows():
    features = np.vstack([np.zeros((100, 3)), np.full((100, 3), 50.0)]).astype(np.float32)
    media, desv = fit_scaler(features, np.arange(100))
    assert np.allclose(media, 0) and np.allclose(desv, 1)  # el periodo de test (valor 50) no influye


def test_walk_forward_folds_are_separated_and_purged(settings, frame, market_frame):
    analisis = {f"S{i}": analyze_symbol(f"S{i}", frame * (1 + i / 10), market_frame, settings) for i in range(2)}
    ds = build_panel_dataset(analisis, lookback=40, min_price=0, min_dollar_volume=0)
    wf = dict(settings.model["walk_forward"], first_test_year=2022)
    folds = make_folds(ds, wf)
    assert len(folds) >= 3
    sesiones = np.unique(ds.dates)
    purga = int(wf["purge_days"])
    horizonte = int(settings.rules["holding_period_days"])
    assert purga >= horizonte + 1
    for f in folds:
        assert f.train_end < f.val_start < f.val_end < f.test_start <= f.test_end
        # sesiones entre el final del entrenamiento y el inicio de la validación >= purga
        assert np.sum((sesiones > f.train_end) & (sesiones < f.val_start)) >= purga
        assert np.sum((sesiones > f.val_end) & (sesiones < f.test_start)) >= purga
        # ninguna etiqueta de entrenamiento mira dentro del periodo de validación
        filas = ds.rows_between(f.train_start, f.train_end)
        ultima = ds.dates[filas].max()
        assert np.sum((sesiones > ultima) & (sesiones < f.val_start)) >= horizonte


def test_news_published_after_decision_are_ignored():
    decision = datetime(2026, 10, 2, 20, 0, tzinfo=timezone.utc)
    items = [NewsItem("AAPL", "R", "Apple beats estimates", "", decision - timedelta(hours=3)),
             NewsItem("AAPL", "R", "Apple plunges after fraud probe", "", decision + timedelta(hours=1))]
    resultado = score_news(items, "AAPL", "Apple Inc.", {"lookback_days": 14, "half_life_days": 3,
                                                          "max_items_per_symbol": 15}, as_of=decision)
    assert resultado["count"] == 1 and resultado["sentiment"] > 0
    assert all("plunges" not in n["title"] for n in resultado["items"])
