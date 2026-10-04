"""Pipeline completo con datos sintéticos: actualización diaria, entrenamiento, backtest y exportación."""

import json
from datetime import datetime, timezone

import pytest

from src.data.storage import read_json
from src.pipeline.backtest import run_backtest_pipeline
from src.pipeline.export import run_export
from src.pipeline.history import list_snapshots, load_scores
from src.pipeline.train import run_training
from src.pipeline.update import run_daily_update

QUICK = {"model": {"neural_network": {"max_epochs": 1, "max_train_samples": 8000, "hidden_size": 8,
                                      "mc_dropout_passes": 2},
                   "walk_forward": {"max_folds": 2}, "promotion": {"min_auc_advantage_vs_baselines": -1.0,
                                                                   "min_auc": 0.0, "min_top_decile_lift": 0.0}}}


@pytest.fixture
def quick_settings(tmp_path):
    from src.config import load_settings
    return load_settings(data_dir=tmp_path / "data", frontend_data_dir=tmp_path / "web",
                         overrides={"universe": {"universe": {"max_symbols": 10}}, **QUICK}, offline=True)


def _no_nan(obj):
    texto = json.dumps(obj)
    assert "NaN" not in texto and "Infinity" not in texto


def test_full_pipeline_offline(quick_settings):
    s = quick_settings
    d1 = datetime(2026, 9, 29, 22, 30, tzinfo=timezone.utc)
    r1 = run_daily_update(s, now=d1)
    assert r1["status"] == "ok" and r1["as_of"] == "2026-09-29"
    assert run_daily_update(s, now=d1)["status"] == "skipped"          # idempotente
    entrenamiento = run_training(s, update_data=False)
    assert entrenamiento["status"] == "ok" and entrenamiento["promoted"]
    registro = read_json(s.paths.models / "registry.json")
    assert registro["production"] == "model_v001"
    v = registro["versions"][0]
    for clave in ("model_version", "training_date", "training_period", "features_version", "walk_forward",
                  "backtest", "label_definition"):
        assert clave in v
    r2 = run_daily_update(s, now=datetime(2026, 9, 30, 22, 30, tzinfo=timezone.utc))
    assert r2["status"] == "ok"
    assert list_snapshots(s.paths.history) == ["2026-09-29", "2026-09-30"]
    scores = load_scores(s.paths.history, "2026-09-30")
    assert scores["opportunity_probability"].notna().any()             # la red en producción ya puntúa
    bt = run_backtest_pipeline(s)
    assert bt["neural_available"]
    resultado = run_export(s)
    web = s.paths.frontend_data
    for nombre in ("market", "opportunities", "tracking", "history", "backtest", "model", "news"):
        datos = json.loads((web / f"{nombre}.json").read_text())
        _no_nan(datos)
        assert "generated_at" in datos or "exported_at" in datos
    mercado = json.loads((web / "market.json").read_text())
    assert mercado["status"]["state"] == "synthetic" and "No ejecuta operaciones reales" in mercado["disclaimer"]
    assert mercado["model"]["production_version"] == "model_v001"
    indice = json.loads((web / "stocks" / "index.json").read_text())
    assert len(indice["stocks"]) == resultado["stocks"] > 0
    pagina = json.loads((web / "stocks" / f"{indice['stocks'][0]['symbol']}.json").read_text())
    for clave in ("series", "oscillators", "scores", "explanation", "neural", "trades", "news_archive"):
        assert clave in pagina
    assert {"ma7", "ma25", "ma99", "volume"} <= set(pagina["series"])
    assert len(pagina["series"]["dt"]) == len(pagina["series"]["close"])
    backtest = json.loads((web / "backtest.json").read_text())
    assert backtest["available"] and "buy_hold_benchmark" in backtest["benchmarks"]


def test_export_without_any_data(quick_settings):
    resultado = run_export(quick_settings)
    assert resultado["status"] == "no_data"
    mercado = json.loads((quick_settings.paths.frontend_data / "market.json").read_text())
    assert mercado["status"]["state"] == "no_data"
    assert json.loads((quick_settings.paths.frontend_data / "backtest.json").read_text())["available"] is False


def test_research_mode_creates_no_trades(quick_settings):
    quick_settings.strategy["mode"] = "RESEARCH"
    run_daily_update(quick_settings, now=datetime(2026, 9, 29, 22, 30, tzinfo=timezone.utc))
    assert read_json(quick_settings.paths.state / "trades.json")["trades"] == []
