"""Entrenamiento reproducible de la red: AdamW (regularización L2), dropout,
recorte de gradiente y parada temprana con validación temporal."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..logging_utils import get_logger
from .dataset import PanelDataset, fit_scaler
from .metrics import roc_auc
from .model import TARGET_SCALE, OpportunityNet

log = get_logger("MODEL")


def set_reproducible(seed: int, num_threads: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_threads and num_threads > 0:
        torch.set_num_threads(int(num_threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class TrainResult:
    model: OpportunityNet
    mean: np.ndarray
    std: np.ndarray
    base_rate: float
    best_epoch: int
    epochs_run: int
    history: list[dict] = field(default_factory=list)
    train_samples: int = 0
    val_samples: int = 0
    seconds: float = 0.0


def _tensores(ds: PanelDataset, filas: np.ndarray, mean, std):
    x = torch.from_numpy(ds.windows(filas, mean, std))
    y = torch.from_numpy(np.nan_to_num(ds.label[filas]).astype(np.float32))
    r = torch.from_numpy(np.clip(np.nan_to_num(ds.trade_return[filas]) / TARGET_SCALE, -5, 5).astype(np.float32))
    d = torch.from_numpy(np.clip(np.nan_to_num(ds.mae[filas]) / TARGET_SCALE, -5, 5).astype(np.float32))
    return x, y, r, d


@torch.no_grad()
def _predict_batches(model: OpportunityNet, lotes, mc_passes: int = 0) -> dict[str, np.ndarray]:
    probs, rets, dds, desv = [], [], [], []
    model.eval()
    for ventanas in lotes:
        x = torch.from_numpy(ventanas)
        logit, r, d = model(x)
        probs.append(torch.sigmoid(logit).numpy())
        rets.append((r * TARGET_SCALE).numpy())
        dds.append((d * TARGET_SCALE).numpy())
        if mc_passes > 0:  # dropout activo: dispersión entre pasadas = incertidumbre
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            pasadas = torch.stack([torch.sigmoid(model(x)[0]) for _ in range(mc_passes)])
            model.eval()
            desv.append(pasadas.std(dim=0).numpy())
    vacio = np.zeros(0, np.float32)
    salida = {
        "probability": np.concatenate(probs) if probs else vacio,
        "expected_return": np.concatenate(rets) if rets else vacio,
        "expected_drawdown": np.concatenate(dds) if dds else vacio,
    }
    if mc_passes > 0:
        salida["probability_std"] = np.concatenate(desv) if desv else vacio
    return salida


def predict(model: OpportunityNet, ds: PanelDataset, rows: np.ndarray, mean: np.ndarray, std: np.ndarray,
            mc_passes: int = 0, batch_size: int = 2048) -> dict[str, np.ndarray]:
    """Predicciones para filas del dataset (ventanas construidas por lotes)."""
    lotes = (ds.windows(rows[i:i + batch_size], mean, std) for i in range(0, len(rows), batch_size))
    return _predict_batches(model, lotes, mc_passes)


def predict_windows(model: OpportunityNet, windows_std: np.ndarray, mc_passes: int = 0,
                    batch_size: int = 2048) -> dict[str, np.ndarray]:
    """Predicciones para ventanas ya normalizadas (B, lookback, F)."""
    lotes = (windows_std[i:i + batch_size].astype(np.float32, copy=False)
             for i in range(0, len(windows_std), batch_size))
    return _predict_batches(model, lotes, mc_passes)


def standardize(windows: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float = 5.0) -> np.ndarray:
    return np.clip((windows - mean) / std, -clip, clip).astype(np.float32)


def train_model(ds: PanelDataset, train_rows: np.ndarray, val_rows: np.ndarray, cfg: dict,
                seed: int | None = None, tag: str = "") -> TrainResult:
    seed = int(cfg["seed"] if seed is None else seed)
    set_reproducible(seed, int(cfg.get("num_threads", 0)))
    inicio = time.monotonic()
    rng = np.random.default_rng(seed)
    if len(train_rows) > int(cfg["max_train_samples"]):
        train_rows = np.sort(rng.choice(train_rows, int(cfg["max_train_samples"]), replace=False))
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError("Sin ejemplos de entrenamiento o validación")
    mean, std = fit_scaler(ds.features, train_rows)
    base = float(np.clip(np.nanmean(ds.label[train_rows]), 1e-3, 1 - 1e-3))
    model = OpportunityNet(len(ds.feature_names), int(cfg["hidden_size"]), int(cfg["num_layers"]),
                           float(cfg["dropout"]), str(cfg["architecture"]))
    with torch.no_grad():
        model.head_prob.bias.fill_(math.log(base / (1 - base)))
    optimizador = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                    weight_decay=float(cfg["weight_decay"]))
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss(beta=1.0)
    w_r, w_d = float(cfg["return_loss_weight"]), float(cfg["drawdown_loss_weight"])
    lote = int(cfg["batch_size"])
    val_eval = val_rows if len(val_rows) <= 120_000 else np.sort(rng.choice(val_rows, 120_000, replace=False))
    y_val = np.nan_to_num(ds.label[val_eval])

    def perdida_validacion() -> tuple[float, float]:
        pred = predict(model, ds, val_eval, mean, std)
        p = np.clip(pred["probability"], 1e-6, 1 - 1e-6)
        return float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))), roc_auc(y_val, p)

    mejor_perdida, mejor_auc = perdida_validacion()
    mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
    mejor_epoca, sin_mejora, historial = 0, 0, [{"epoch": 0, "val_loss": mejor_perdida, "val_auc": mejor_auc}]
    epocas = 0
    for epoca in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        orden = rng.permutation(train_rows)
        perdidas = []
        for i in range(0, len(orden), lote):
            x, y, r, d = _tensores(ds, orden[i:i + lote], mean, std)
            logit, pr, pd_ = model(x)
            perdida = bce(logit, y) + w_r * huber(pr, r) + w_d * huber(pd_, d)
            optimizador.zero_grad()
            perdida.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizador.step()
            perdidas.append(perdida.item())
        epocas = epoca
        val_loss, val_auc = perdida_validacion()
        historial.append({"epoch": epoca, "train_loss": float(np.mean(perdidas)), "val_loss": val_loss,
                          "val_auc": val_auc})
        log.info("%s época %d: pérdida entrenamiento %.4f · validación %.4f · AUC val %.3f",
                 tag, epoca, np.mean(perdidas), val_loss, val_auc)
        if val_loss < mejor_perdida - 1e-4:
            mejor_perdida, mejor_epoca, sin_mejora = val_loss, epoca, 0
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= int(cfg["patience"]):
                log.info("%s parada temprana en la época %d (mejor: %d)", tag, epoca, mejor_epoca)
                break
    model.load_state_dict(mejor_estado)
    model.eval()
    return TrainResult(model=model, mean=mean, std=std, base_rate=base, best_epoch=mejor_epoca,
                       epochs_run=epocas, history=historial, train_samples=int(len(train_rows)),
                       val_samples=int(len(val_rows)), seconds=round(time.monotonic() - inicio, 1))
