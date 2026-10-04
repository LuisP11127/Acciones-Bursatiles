"""Análisis por acción: indicadores + contexto de mercado + features + etiquetas +
scores estadísticos. Lo comparten la actualización diaria, el entrenamiento,
el backtesting y la exportación para la web (un único cálculo, mismas reglas)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.market import MarketData
from ..features.builder import build_feature_frame
from ..features.labels import label_params, opportunity_labels
from ..indicators.technical import add_market_context, compute_indicators
from ..logging_utils import get_logger, summary
from ..statistical.scoring import compute_statistical_scores

log = get_logger("FEATURES")

# Columnas que se conservan (float32) para explicaciones, backtesting y gráficos
KEEP_COLUMNS = [
    "open", "high", "low", "close", "volume", "raw_close", "ma7", "ma25", "ma99", "sma50", "sma200",
    "rsi14", "macd", "macd_signal", "macd_hist", "atr_pct", "vol20", "rel_volume", "vol_trend",
    "up_volume_ratio", "dollar_volume20", "dist_ma7", "dist_ma25", "dist_ma99", "ma7_gt_ma25",
    "ma25_gt_ma99", "above_ma99", "cross_ma7_ma25", "cross_ma25_ma99", "bars_since_cross_7_25",
    "bars_since_golden_25_99", "slope_ma7", "slope_ma25", "slope_ma99", "ret1", "ret5", "ret21", "ret63",
    "drawdown252", "dist_high252", "high252", "low252", "bb_pctb", "bb_upper", "bb_lower", "rel_strength63",
    "mom10", "roc20",
]


@dataclass
class SymbolAnalysis:
    symbol: str
    ind: pd.DataFrame       # indicadores (subconjunto KEEP_COLUMNS)
    features: pd.DataFrame  # features del modelo (punto a punto)
    labels: pd.DataFrame    # etiquetas futuras (solo entrenamiento/evaluación)
    scores: pd.DataFrame    # componentes del statistical_score

    def tradable(self, min_price: float, min_dollar_volume: float) -> pd.Series:
        """Filtro de liquidez punto a punto (solo con datos hasta cada fecha)."""
        return (self.ind["raw_close"] >= min_price) & (self.ind["dollar_volume20"] >= min_dollar_volume)


def analyze_symbol(symbol: str, frame: pd.DataFrame, market: pd.DataFrame | None, settings,
                   with_labels: bool = True) -> SymbolAnalysis:
    ind = add_market_context(compute_indicators(frame), market)
    features = build_feature_frame(ind)
    params = label_params(settings)
    labels = opportunity_labels(frame, **params) if with_labels else pd.DataFrame(index=frame.index)
    scores = compute_statistical_scores(ind, labels if with_labels else None, settings)
    columnas = [c for c in KEEP_COLUMNS if c in ind.columns]
    return SymbolAnalysis(symbol=symbol, ind=ind[columnas].astype(np.float32), features=features,
                          labels=labels.astype(np.float32), scores=scores.astype(np.float32))


def analyze_universe(market_data: MarketData, symbols: list[str], settings,
                     with_labels: bool = True) -> dict[str, SymbolAnalysis]:
    """Analiza cada símbolo; un error en uno no detiene el resto."""
    mercado = market_data.frames.get(market_data.benchmark)
    resultado: dict[str, SymbolAnalysis] = {}
    for symbol in symbols:
        frame = market_data.frames.get(symbol)
        if frame is None:
            continue
        try:
            resultado[symbol] = analyze_symbol(symbol, frame, mercado, settings, with_labels)
        except Exception as exc:  # una acción problemática no detiene el análisis
            summary.error("FEATURES", f"análisis fallido: {exc}", symbol)
    summary.count("symbols_analyzed", len(resultado))
    log.info("Indicadores, features y scores calculados para %d símbolos", len(resultado))
    return resultado
