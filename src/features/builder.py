"""Features para el machine learning (punto a punto en el tiempo).

Todas son relativas (retornos, ratios, osciladores normalizados) para que una
acción de $20 y otra de $900 se comparen en la misma escala y para que el
ajuste de precios por dividendos/splits no filtre información. El OHLCV entra
normalizado (retorno, gap, cuerpo y mechas de la vela, volumen relativo).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from ..market_calendar import NY

FEATURE_GROUPS: dict[str, list[str]] = {
    "ohlcv": ["ret1", "gap", "body", "upper_shadow", "lower_shadow", "rel_volume_log"],
    "momentum": ["ret5", "ret21", "ret63", "mom10", "roc20"],
    "moving_averages": ["dist_ma7", "dist_ma25", "dist_ma99", "ma7_ma25", "ma25_ma99", "slope_ma7",
                        "slope_ma25", "slope_ma99", "ma7_gt_ma25", "ma25_gt_ma99", "cross_recent_7_25",
                        "bars_since_cross_7_25_n"],
    "oscillators": ["rsi_n", "macd_n", "macd_hist_n", "bb_pctb", "bb_width"],
    "risk": ["atr_pct", "vol20", "vol60", "drawdown252", "dist_low252_log"],
    "volume": ["vol_trend_log", "up_volume_ratio_log"],
    "market": ["mkt_ret21", "mkt_dist_ma99", "mkt_vol20", "rel_strength63"],
}
FEATURE_COLUMNS: list[str] = [c for grupo in FEATURE_GROUPS.values() for c in grupo]
NEWS_FEATURE_COLUMNS = ["news_sentiment_7d", "news_count_7d_log"]

FEATURE_LABELS = {
    "ret1": "Retorno del día", "gap": "Gap de apertura", "body": "Cuerpo de la vela",
    "upper_shadow": "Mecha superior", "lower_shadow": "Mecha inferior", "rel_volume_log": "Volumen relativo",
    "ret5": "Retorno 5 sesiones", "ret21": "Retorno 1 mes", "ret63": "Retorno 3 meses",
    "mom10": "Momentum 10 sesiones", "roc20": "ROC 20 sesiones",
    "dist_ma7": "Distancia a la MA(7)", "dist_ma25": "Distancia a la MA(25)", "dist_ma99": "Distancia a la MA(99)",
    "ma7_ma25": "MA(7) frente a MA(25)", "ma25_ma99": "MA(25) frente a MA(99)",
    "slope_ma7": "Pendiente de la MA(7)", "slope_ma25": "Pendiente de la MA(25)", "slope_ma99": "Pendiente de la MA(99)",
    "ma7_gt_ma25": "MA(7) > MA(25)", "ma25_gt_ma99": "MA(25) > MA(99)",
    "cross_recent_7_25": "Cruce reciente MA(7)/MA(25)", "bars_since_cross_7_25_n": "Sesiones desde el último cruce",
    "rsi_n": "RSI(14)", "macd_n": "MACD", "macd_hist_n": "Histograma MACD", "bb_pctb": "Posición en Bollinger",
    "bb_width": "Ancho de Bollinger", "atr_pct": "ATR relativo", "vol20": "Volatilidad 20 sesiones",
    "vol60": "Volatilidad 60 sesiones", "drawdown252": "Drawdown 52 semanas",
    "dist_low252_log": "Distancia al mínimo de 52 semanas", "vol_trend_log": "Tendencia del volumen",
    "up_volume_ratio_log": "Volumen en subidas vs bajadas", "mkt_ret21": "Retorno del S&P 500 (1 mes)",
    "mkt_dist_ma99": "S&P 500 frente a su MA(99)", "mkt_vol20": "Volatilidad del S&P 500",
    "rel_strength63": "Fuerza relativa vs S&P 500 (3 meses)",
    "news_sentiment_7d": "Sentimiento de noticias (7 días)", "news_count_7d_log": "Número de noticias (7 días)",
}


def _log_ratio(serie: pd.Series) -> pd.Series:
    return np.log(serie.where(serie > 0)).clip(-3, 3)


def build_feature_frame(ind: pd.DataFrame) -> pd.DataFrame:
    """`ind` = salida de compute_indicators + add_market_context."""
    f = pd.DataFrame(index=ind.index)
    for col in ("ret1", "gap", "body", "upper_shadow", "lower_shadow", "ret5", "ret21", "ret63", "mom10",
                "roc20", "dist_ma7", "dist_ma25", "dist_ma99", "ma7_ma25", "ma25_ma99", "slope_ma7",
                "slope_ma25", "slope_ma99", "ma7_gt_ma25", "ma25_gt_ma99", "bb_width", "atr_pct", "vol20",
                "vol60", "drawdown252", "mkt_ret21", "mkt_dist_ma99", "mkt_vol20", "rel_strength63"):
        f[col] = ind[col]
    f["rel_volume_log"] = _log_ratio(ind["rel_volume"])
    f["cross_recent_7_25"] = ind["cross_ma7_ma25"].rolling(5, min_periods=1).sum().clip(-1, 1)
    f["bars_since_cross_7_25_n"] = ind["bars_since_cross_7_25"].clip(upper=60) / 60
    f["rsi_n"] = (ind["rsi14"] - 50) / 50
    f["macd_n"] = ind["macd"] / ind["close"]
    f["macd_hist_n"] = ind["macd_hist"] / ind["close"]
    f["bb_pctb"] = ind["bb_pctb"].clip(-1, 2)
    f["dist_low252_log"] = np.log1p(ind["dist_low252"].clip(lower=-0.99))
    f["vol_trend_log"] = _log_ratio(ind["vol_trend"])
    f["up_volume_ratio_log"] = _log_ratio(ind["up_volume_ratio"])
    return f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def decision_cutoff(session: pd.Timestamp) -> datetime:
    """Instante de decisión de la sesión T: su cierre (16:00 Nueva York).
    Las noticias publicadas después NO pueden usarse para decidir en T."""
    return datetime.combine(pd.Timestamp(session).date(), time(16, 0), tzinfo=NY)


def news_features(news: list[dict], sessions: pd.DatetimeIndex, window_days: int = 7) -> pd.DataFrame:
    """Features de noticias punto a punto: en la sesión T solo cuenta lo publicado
    hasta el cierre de T (y dentro de la ventana). `news`: dicts con published_at y sentiment."""
    salida = pd.DataFrame(np.nan, index=sessions, columns=NEWS_FEATURE_COLUMNS, dtype=float)
    registros = []
    for n in news:
        fecha = n.get("published_at")
        if isinstance(fecha, str):
            try:
                fecha = datetime.fromisoformat(fecha.replace("Z", "+00:00"))
            except ValueError:
                continue
        if fecha is None or n.get("sentiment") is None:
            continue
        registros.append((fecha if fecha.tzinfo else fecha.replace(tzinfo=NY), float(n["sentiment"])))
    if not registros:
        return salida
    registros.sort()
    fechas = [r[0] for r in registros]
    sentimientos = np.array([r[1] for r in registros])
    import bisect

    for sesion in sessions:
        corte = decision_cutoff(sesion)
        inicio = corte - timedelta(days=window_days)
        i0 = bisect.bisect_right(fechas, inicio)
        i1 = bisect.bisect_right(fechas, corte)
        if i1 > i0:
            salida.loc[sesion, "news_sentiment_7d"] = float(sentimientos[i0:i1].mean())
            salida.loc[sesion, "news_count_7d_log"] = float(np.log1p(i1 - i0))
        else:
            salida.loc[sesion, "news_count_7d_log"] = 0.0
    return salida
