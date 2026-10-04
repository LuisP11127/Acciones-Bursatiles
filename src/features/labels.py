"""Etiquetas de entrenamiento: ¿fue una buena oportunidad entrar después de T?

ESTE MÓDULO USA DATOS FUTUROS y solo se usa para construir etiquetas durante el
entrenamiento/evaluación histórica. Nunca alimenta las features.

Simula la operación de la estrategia (triple barrera):
  - decisión al cierre de T, entrada en la APERTURA de T+1;
  - salida por stop-loss, take-profit o al cierre de T+H (H = holding_period_days);
  - si stop y take-profit se tocan el mismo día se asume el stop (conservador),
    salvo que la apertura ya supere el take-profit;
  - gaps: si la apertura salta el nivel, se sale a la apertura;
  - costes de comisión y deslizamiento en la entrada y la salida.
Etiqueta = 1 si el retorno neto >= min_trade_return.
Las últimas H sesiones no tienen etiqueta (su futuro aún no existe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

LABEL_COLUMNS = ["label", "trade_return", "mae", "mfe", "holding_days", "exit_type"]
EXIT_TYPES = {1: "take_profit", -1: "stop_loss", 0: "time"}


def opportunity_labels(ohlc: pd.DataFrame, horizon: int, stop_loss: float, take_profit: float,
                       cost_per_side: float, min_return: float) -> pd.DataFrame:
    n = len(ohlc)
    salida = pd.DataFrame(np.nan, index=ohlc.index, columns=LABEL_COLUMNS)
    H = int(horizon)
    if n < H + 2:
        return salida
    o = ohlc["open"].to_numpy(dtype=float)
    h = ohlc["high"].to_numpy(dtype=float)
    l = ohlc["low"].to_numpy(dtype=float)
    c = ohlc["close"].to_numpy(dtype=float)
    m = n - H                                   # decisiones T = 0..m-1 (necesitan T+1..T+H)
    filas = np.arange(m)
    v_o = sliding_window_view(o[1:], H)[:m]     # fila T: barras T+1 .. T+H
    v_h = sliding_window_view(h[1:], H)[:m]
    v_l = sliding_window_view(l[1:], H)[:m]
    entrada = v_o[:, 0]
    nivel_sl = entrada * (1 - stop_loss)
    nivel_tp = entrada * (1 + take_profit)
    toca_sl = v_l <= nivel_sl[:, None]
    toca_tp = v_h >= nivel_tp[:, None]
    primero_sl = np.where(toca_sl.any(axis=1), toca_sl.argmax(axis=1), H)
    primero_tp = np.where(toca_tp.any(axis=1), toca_tp.argmax(axis=1), H)
    k_sl = np.minimum(primero_sl, H - 1)
    k_tp = np.minimum(primero_tp, H - 1)
    apertura_sl = v_o[filas, k_sl]
    apertura_tp = v_o[filas, k_tp]
    precio_sl = np.where(apertura_sl < nivel_sl, apertura_sl, nivel_sl)   # gap bajista: sale a la apertura
    precio_tp = np.where(apertura_tp > nivel_tp, apertura_tp, nivel_tp)   # gap alcista: sale a la apertura
    mismo_dia = (primero_sl == primero_tp) & (primero_sl < H)
    gap_tp = mismo_dia & (apertura_tp >= nivel_tp)
    es_stop = (primero_sl <= primero_tp) & (primero_sl < H) & ~gap_tp
    es_tp = ((primero_tp < primero_sl) & (primero_tp < H)) | gap_tp
    cierre_final = c[H:H + m]                   # cierre de T+H
    precio_salida = np.where(es_stop, precio_sl, np.where(es_tp, precio_tp, cierre_final))
    k_salida = np.where(es_stop, k_sl, np.where(es_tp, k_tp, H - 1))
    neto = (precio_salida * (1 - cost_per_side)) / (entrada * (1 + cost_per_side)) - 1
    hasta_salida = np.arange(H)[None, :] <= k_salida[:, None]
    mae = np.where(hasta_salida, v_l, np.inf).min(axis=1) / entrada - 1
    mfe = np.where(hasta_salida, v_h, -np.inf).max(axis=1) / entrada - 1
    tipo = np.where(es_stop, -1, np.where(es_tp, 1, 0))
    salida.iloc[:m] = np.column_stack([(neto >= min_return).astype(float), neto, mae, mfe, k_salida + 1, tipo])
    return salida


def label_params(settings) -> dict:
    """Parámetros de la etiqueta tomados de strategy.yaml y model.yaml."""
    reglas = settings.strategy["strategy"]
    return {
        "horizon": int(reglas["holding_period_days"]),
        "stop_loss": float(reglas["stop_loss"]),
        "take_profit": float(reglas["take_profit"]),
        "cost_per_side": (float(reglas["commission_bps"]) + float(reglas["slippage_bps"])) / 10_000,
        "min_return": float(settings.model["labels"]["min_trade_return"]),
    }
