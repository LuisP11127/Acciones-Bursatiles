"""Dataset de secuencias para la red neuronal.

Todas las filas (sesiones) de todas las acciones se concatenan en una matriz;
cada ejemplo es la ventana de las `lookback` sesiones que terminan en T (solo
pasado). Las ventanas se construyen al vuelo por lotes para no duplicar memoria.
La normalización se ajusta SOLO con filas de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.builder import FEATURE_COLUMNS
from ..pipeline.analysis import SymbolAnalysis


@dataclass
class PanelDataset:
    features: np.ndarray      # (R, F) float32
    label: np.ndarray         # (R,) 1/0, NaN si el futuro aún no se conoce
    trade_return: np.ndarray  # (R,) retorno neto de la operación simulada
    mae: np.ndarray           # (R,) peor caída durante la operación
    dates: np.ndarray         # (R,) datetime64[D]
    symbol_idx: np.ndarray    # (R,) int32
    row_in_symbol: np.ndarray  # (R,) posición dentro de la serie del símbolo
    window_ok: np.ndarray     # (R,) ventana completa y sin NaN, dentro del mismo símbolo
    tradable: np.ndarray      # (R,) filtro de liquidez punto a punto
    stat_score: np.ndarray    # (R,) para comparar con benchmarks sencillos
    momentum: np.ndarray      # (R,) retorno de 3 meses (benchmark)
    symbols: list[str]
    feature_names: list[str]
    lookback: int

    def __len__(self) -> int:
        return len(self.label)

    def windows(self, rows: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None,
                clip: float = 5.0) -> np.ndarray:
        """(B, lookback, F) con las sesiones T-lookback+1 .. T de cada fila T."""
        offsets = np.arange(-self.lookback + 1, 1)
        ventanas = self.features[np.asarray(rows)[:, None] + offsets[None, :]]
        if mean is not None:
            ventanas = np.clip((ventanas - mean) / std, -clip, clip)
        return ventanas.astype(np.float32, copy=False)

    def rows_between(self, start: np.datetime64 | None, end: np.datetime64 | None,
                     require_label: bool = True, stride: int = 1) -> np.ndarray:
        mascara = self.window_ok & self.tradable
        if require_label:
            mascara &= ~np.isnan(self.label)
        if start is not None:
            mascara &= self.dates >= start
        if end is not None:
            mascara &= self.dates <= end
        if stride > 1:
            mascara &= (self.row_in_symbol % stride) == 0
        return np.nonzero(mascara)[0]


def build_panel_dataset(analyses: dict[str, SymbolAnalysis], lookback: int, min_price: float,
                        min_dollar_volume: float, symbols: list[str] | None = None) -> PanelDataset:
    feats, labels, rets, maes, fechas, sym_idx, filas, tradables, stats, moms = ([] for _ in range(10))
    ventana_ok = []
    nombres = []
    for i, symbol in enumerate(symbols or sorted(analyses)):
        a = analyses.get(symbol)
        if a is None:
            continue
        x = a.features[FEATURE_COLUMNS].to_numpy(np.float32)
        n = len(x)
        valido = ~np.isnan(x).any(axis=1)
        # longitud de la racha de filas válidas que termina en cada fila
        posiciones = np.arange(n)
        ultima_invalida = np.maximum.accumulate(np.where(~valido, posiciones, -1))
        racha = posiciones - ultima_invalida
        etiquetas = a.labels.reindex(a.features.index)
        feats.append(np.nan_to_num(x, nan=0.0))
        labels.append(etiquetas["label"].to_numpy(np.float32) if "label" in etiquetas else np.full(n, np.nan, np.float32))
        rets.append(etiquetas["trade_return"].to_numpy(np.float32) if "trade_return" in etiquetas
                    else np.full(n, np.nan, np.float32))
        maes.append(etiquetas["mae"].to_numpy(np.float32) if "mae" in etiquetas else np.full(n, np.nan, np.float32))
        fechas.append(a.features.index.to_numpy().astype("datetime64[D]"))
        sym_idx.append(np.full(n, len(nombres), dtype=np.int32))
        filas.append(np.arange(n, dtype=np.int32))
        ventana_ok.append(racha >= lookback)
        tradables.append(a.tradable(min_price, min_dollar_volume).fillna(False).to_numpy(bool))
        stats.append(a.scores["statistical_score"].to_numpy(np.float32))
        moms.append(a.ind["ret63"].to_numpy(np.float32))
        nombres.append(symbol)
    if not feats:
        raise ValueError("No hay datos para construir el dataset")
    return PanelDataset(
        features=np.concatenate(feats), label=np.concatenate(labels), trade_return=np.concatenate(rets),
        mae=np.concatenate(maes), dates=np.concatenate(fechas), symbol_idx=np.concatenate(sym_idx),
        row_in_symbol=np.concatenate(filas), window_ok=np.concatenate(ventana_ok),
        tradable=np.concatenate(tradables), stat_score=np.concatenate(stats), momentum=np.concatenate(moms),
        symbols=nombres, feature_names=list(FEATURE_COLUMNS), lookback=int(lookback),
    )


def fit_scaler(features: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviación SOLO de las filas de entrenamiento."""
    datos = features[rows]
    media = datos.mean(axis=0).astype(np.float32)
    desv = datos.std(axis=0).astype(np.float32)
    return media, np.where(desv < 1e-8, 1.0, desv).astype(np.float32)


def last_window(features: pd.DataFrame, lookback: int) -> np.ndarray | None:
    """Ventana de las últimas `lookback` sesiones (para predecir hoy) o None si faltan datos."""
    x = features[FEATURE_COLUMNS].to_numpy(np.float32)[-lookback:]
    if len(x) < lookback or np.isnan(x).any():
        return None
    return x
