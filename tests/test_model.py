"""Neural Network Engine: modelo, entrenamiento reproducible, registro y promoción, inferencia."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.ml.dataset import PanelDataset
from src.ml.inference import neural_score, predict_latest
from src.ml.metrics import classification_metrics, roc_auc
from src.ml.model import OpportunityNet
from src.ml.numpy_mlp import NumpyMLP
from src.ml.registry import ModelRegistry, evaluate_promotion
from src.ml.train import predict, train_model

NN = {"architecture": "gru", "hidden_size": 16, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.003,
      "weight_decay": 0.0001, "batch_size": 256, "max_epochs": 6, "patience": 3, "sample_stride": 1,
      "max_train_samples": 20000, "return_loss_weight": 0.3, "drawdown_loss_weight": 0.3, "mc_dropout_passes": 5,
      "seed": 7, "num_threads": 1}


def synthetic_dataset(n_symbols=4, n=1500, lookback=10, seed=0) -> PanelDataset:
    """La etiqueta depende de la tendencia de la feature 0 en las últimas sesiones (patrón temporal)."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for _ in range(n_symbols):
        x = rng.normal(0, 1, (n, 5)).astype(np.float32)
        x[:, 0] = np.convolve(rng.normal(0, 1, n), np.ones(5) / 5, mode="same")
        tendencia = pd.Series(x[:, 0]).rolling(5).mean().to_numpy()
        labels.append((tendencia > 0.1).astype(np.float32))
        feats.append(x)
    R = n_symbols * n
    return PanelDataset(
        features=np.concatenate(feats), label=np.concatenate(labels), trade_return=np.concatenate(labels) * 0.1 - 0.02,
        mae=np.full(R, -0.03, np.float32), dates=np.tile(np.arange(n).astype("datetime64[D]"), n_symbols),
        symbol_idx=np.repeat(np.arange(n_symbols), n).astype(np.int32), row_in_symbol=np.tile(np.arange(n), n_symbols),
        window_ok=np.tile(np.arange(n) >= lookback, n_symbols), tradable=np.ones(R, bool),
        stat_score=np.zeros(R, np.float32), momentum=np.zeros(R, np.float32), symbols=[f"S{i}" for i in range(n_symbols)],
        feature_names=[f"f{i}" for i in range(5)], lookback=lookback)


def test_model_shapes():
    model = OpportunityNet(5, 8, 1, 0.1, "gru")
    p, r, d = model(torch.zeros(3, 10, 5))
    assert p.shape == r.shape == d.shape == (3,)
    assert OpportunityNet(5, 8, 2, 0.1, "lstm")(torch.zeros(2, 4, 5))[0].shape == (2,)
    with pytest.raises(ValueError):
        OpportunityNet(5, architecture="transformer")


def test_gru_learns_temporal_pattern_and_is_reproducible():
    ds = synthetic_dataset()
    entreno = np.nonzero(ds.window_ok & (ds.dates < np.datetime64(1000, "D")))[0]
    valid = np.nonzero(ds.window_ok & (ds.dates >= np.datetime64(1000, "D")) & (ds.dates < np.datetime64(1200, "D")))[0]
    test = np.nonzero(ds.window_ok & (ds.dates >= np.datetime64(1200, "D")))[0]
    a = train_model(ds, entreno, valid, NN)
    pred = predict(a.model, ds, test, a.mean, a.std)
    assert roc_auc(ds.label[test], pred["probability"]) > 0.85
    b = train_model(ds, entreno, valid, NN)
    assert np.allclose(pred["probability"], predict(b.model, ds, test, b.mean, b.std)["probability"], atol=1e-5)
    mc = predict(a.model, ds, test[:50], a.mean, a.std, mc_passes=5)
    assert mc["probability_std"].shape == (50,) and (mc["probability_std"] >= 0).all()


def test_classification_metrics_and_neural_score():
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.3, 0.2, 0.1, 0.6, 0.2])
    m = classification_metrics(y, p, threshold=0.5)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3) and np.isclose(m["f1"], 2 / 3)
    assert np.isclose(m["base_rate"], 0.3) and m["roc_auc"] > 0.9
    assert neural_score(0.3, 0.3) == 50 and neural_score(0.6, 0.3) > 80 and neural_score(0.15, 0.3) < 20
    assert neural_score(0.5, None) is None


def test_numpy_logistic_baseline_learns():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (4000, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    modelo = NumpyMLP(3, ()).fit(X, y, epochs=5)
    assert roc_auc(y, modelo.predict_proba(X)) > 0.95


def _candidato(auc, lift=1.3, bases=None, sharpe=1.0):
    return {"model_version": "model_v002", "features_version": "f1",
            "walk_forward": {"aggregate": {"roc_auc": auc, "top_decile_lift": lift, "n": 10000},
                             "baselines": bases or {"statistical_score": 0.53, "random": 0.5}},
            "backtest": {"combined": {"sharpe": sharpe}}}


CFG = {"min_auc": 0.52, "min_top_decile_lift": 1.05, "min_auc_advantage_vs_baselines": 0.0,
       "max_auc_drop_vs_production": 0.01, "max_sharpe_drop_vs_production": 0.1, "min_test_samples": 2000}


def test_promotion_rules():
    ok, _ = evaluate_promotion(_candidato(0.56), None, CFG, "f1")
    assert ok
    ok, checks = evaluate_promotion(_candidato(0.51), None, CFG, "f1")
    assert not ok and any(c["name"] == "auc_minimo" and not c["passed"] for c in checks)
    ok, _ = evaluate_promotion(_candidato(0.55, bases={"statistical_score": 0.57}), None, CFG, "f1")
    assert not ok  # no supera al benchmark sencillo
    produccion = _candidato(0.60) | {"model_version": "model_v001"}
    ok, checks = evaluate_promotion(_candidato(0.56), produccion, CFG, "f1")
    assert not ok and any(c["name"] == "no_empeora_auc" and not c["passed"] for c in checks)
    ok, _ = evaluate_promotion(_candidato(0.595, sharpe=0.5), produccion, CFG, "f1")
    assert not ok  # empeora claramente el Sharpe del backtest
    ok, _ = evaluate_promotion(_candidato(0.56), produccion | {"features_version": "f0"}, CFG, "f1")
    assert ok  # un modelo con otras features no es comparable


def test_registry_versions_and_production_roundtrip(tmp_path):
    reg = ModelRegistry(tmp_path)
    assert reg.next_version() == "model_v001" and reg.load_production() == (None, None)
    model = OpportunityNet(5, 8, 1, 0.1, "gru")
    meta = {"model_version": "model_v001", "features_version": "f1", "feature_names": [f"f{i}" for i in range(5)],
            "lookback": 10, "base_rate": 0.4, "hyperparameters": {"hidden_size": 8, "num_layers": 1, "dropout": 0.1,
                                                                 "architecture": "gru"},
            "scaler": {"mean": [0.0] * 5, "std": [1.0] * 5}}
    reg.register(meta, model, promote=True)
    reg.register(dict(meta, model_version="model_v002"), None, promote=False)
    reg2 = ModelRegistry(tmp_path)
    assert reg2.production_version == "model_v001" and reg2.next_version() == "model_v003"
    assert reg2.get("model_v002")["status"] == "rejected"
    cargado, m = reg2.load_production()
    x = torch.randn(2, 10, 5)
    assert torch.allclose(cargado(x)[0], model.eval()(x)[0])
    assert not (tmp_path / "model_v002").exists()  # los rechazados solo guardan métricas


def test_predict_latest_with_sensitivity_explanation():
    model = OpportunityNet(39, 8, 1, 0.1, "gru")
    from src.features.builder import FEATURE_COLUMNS
    meta = {"lookback": 10, "base_rate": 0.4, "_mean": np.zeros(39, np.float32), "_std": np.ones(39, np.float32)}
    rng = np.random.default_rng(0)
    feats = {s: pd.DataFrame(rng.normal(0, 1, (30, 39)).astype(np.float32), columns=FEATURE_COLUMNS) for s in ("A", "B")}
    feats["C"] = feats["A"].copy()
    feats["C"].iloc[-1, 0] = np.nan  # ventana incompleta: no se predice
    tabla = predict_latest(model, meta, feats, mc_passes=3, explain=["A"])
    assert list(tabla["symbol"]) == ["A", "B"]
    exp = tabla.set_index("symbol").loc["A", "explanation"]
    assert len(exp["factors"]) == 5 and "no es una afirmación" in exp["disclaimer"].lower()
    assert tabla["confidence"].between(0, 1).all()
