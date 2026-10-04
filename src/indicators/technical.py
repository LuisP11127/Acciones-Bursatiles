"""Indicadores técnicos. Todos son causales: el valor en T usa solo datos hasta T
(medias móviles hacia atrás, EWM sin centrar). Las pruebas de fuga lo verifican."""

from __future__ import annotations

import numpy as np
import pandas as pd

MOVING_AVERAGES = (7, 25, 99)
TRADING_DAYS = 252


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder."""
    delta = close.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perdida = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    valor = 100 - 100 / (1 + rs)
    return valor.where(perdida != 0, 100.0).where(ganancia.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    linea = ema(close, fast) - ema(close, slow)
    senal = linea.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return linea, senal, linea - senal


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    previo = close.shift(1)
    rango = pd.concat([high - low, (high - previo).abs(), (low - previo).abs()], axis=1).max(axis=1)
    return rango.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 el día que `fast` cruza por encima de `slow`, -1 por debajo, 0 si no."""
    arriba = (fast > slow).astype(float).where(fast.notna() & slow.notna())
    cambio = arriba.diff()
    return cambio.fillna(0.0).clip(-1, 1)


def bars_since(event: pd.Series, cap: int = 250) -> pd.Series:
    """Sesiones desde el último evento (True). Antes del primero: `cap`."""
    grupos = event.astype(bool).cumsum()
    contador = event.groupby(grupos).cumcount().astype(float)
    contador[grupos == 0] = cap
    return contador.clip(upper=cap)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade los indicadores a un DataFrame OHLCV (ajustado). Devuelve una copia."""
    df = df.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    for n in MOVING_AVERAGES:
        df[f"ma{n}"] = sma(c, n)
    df["sma50"] = sma(c, 50)
    df["sma200"] = sma(c, 200)
    df["ema12"] = ema(c, 12)
    df["ema26"] = ema(c, 26)
    df["ema50"] = ema(c, 50)

    df["rsi14"] = rsi(c, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["atr14"] = atr(h, l, c, 14)
    df["atr_pct"] = df["atr14"] / c

    media20 = sma(c, 20)
    desv20 = c.rolling(20, min_periods=20).std()
    df["bb_mid"] = media20
    df["bb_upper"] = media20 + 2 * desv20
    df["bb_lower"] = media20 - 2 * desv20
    ancho = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pctb"] = (c - df["bb_lower"]) / ancho
    df["bb_width"] = ancho / media20

    df["mom10"] = c / c.shift(10) - 1
    df["roc20"] = c.pct_change(20)
    for n in (1, 5, 21, 63, 126):
        df[f"ret{n}"] = c.pct_change(n)
    logret = np.log(c / c.shift(1))
    df["vol20"] = logret.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    df["vol60"] = logret.rolling(60, min_periods=60).std() * np.sqrt(TRADING_DAYS)

    df["vol_avg5"] = sma(v, 5)
    df["vol_avg20"] = sma(v, 20)
    df["vol_avg50"] = sma(v, 50)
    df["rel_volume"] = v / df["vol_avg20"].replace(0, np.nan)
    df["vol_trend"] = df["vol_avg5"] / df["vol_avg50"].replace(0, np.nan)
    sube = (c > c.shift(1)).astype(float)
    vol_sube = (v * sube).rolling(20, min_periods=20).sum()
    vol_baja = (v * (1 - sube)).rolling(20, min_periods=20).sum()
    df["up_volume_ratio"] = vol_sube / vol_baja.replace(0, np.nan)
    df["dollar_volume20"] = sma(df["raw_close"] * v if "raw_close" in df else c * v, 20)

    df["high252"] = h.rolling(TRADING_DAYS, min_periods=126).max()
    df["low252"] = l.rolling(TRADING_DAYS, min_periods=126).min()
    df["drawdown252"] = c / c.rolling(TRADING_DAYS, min_periods=126).max() - 1
    df["drawdown"] = c / c.cummax() - 1  # desde el máximo histórico conocido hasta T
    df["dist_high252"] = c / df["high252"] - 1
    df["dist_low252"] = c / df["low252"] - 1

    for n in MOVING_AVERAGES:
        df[f"close_ma{n}"] = c / df[f"ma{n}"]
        df[f"dist_ma{n}"] = df[f"close_ma{n}"] - 1
    df["ma7_ma25"] = df["ma7"] / df["ma25"] - 1
    df["ma25_ma99"] = df["ma25"] / df["ma99"] - 1
    df["ma7_gt_ma25"] = (df["ma7"] > df["ma25"]).astype(float).where(df["ma25"].notna())
    df["ma25_gt_ma99"] = (df["ma25"] > df["ma99"]).astype(float).where(df["ma99"].notna())
    df["above_ma99"] = (c > df["ma99"]).astype(float).where(df["ma99"].notna())
    df["cross_ma7_ma25"] = crossover(df["ma7"], df["ma25"])
    df["cross_ma25_ma99"] = crossover(df["ma25"], df["ma99"])
    df["bars_since_cross_7_25"] = bars_since(df["cross_ma7_ma25"] != 0)
    df["bars_since_golden_25_99"] = bars_since(df["cross_ma25_ma99"] > 0)
    df["slope_ma7"] = df["ma7"].pct_change(3)
    df["slope_ma25"] = df["ma25"].pct_change(5)
    df["slope_ma99"] = df["ma99"].pct_change(10)

    df["gap"] = o / c.shift(1) - 1
    df["body"] = (c - o) / o
    df["upper_shadow"] = (h - np.maximum(o, c)) / c
    df["lower_shadow"] = (np.minimum(o, c) - l) / c
    return df.replace([np.inf, -np.inf], np.nan)


def add_market_context(df: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    """Contexto del índice de referencia (alineado por fecha, sin mirar al futuro)."""
    df = df.copy()
    if market is None or market.empty:
        for col in ("mkt_ret21", "mkt_ret63", "mkt_dist_ma99", "mkt_vol20"):
            df[col] = np.nan
        df["rel_strength63"] = np.nan
        return df
    m = pd.DataFrame(index=market.index)
    m["mkt_ret21"] = market["close"].pct_change(21)
    m["mkt_ret63"] = market["close"].pct_change(63)
    m["mkt_dist_ma99"] = market["close"] / sma(market["close"], 99) - 1
    m["mkt_vol20"] = np.log(market["close"] / market["close"].shift(1)).rolling(20, min_periods=20).std() \
        * np.sqrt(TRADING_DAYS)
    alineado = m.reindex(df.index).ffill(limit=5)
    for col in m.columns:
        df[col] = alineado[col]
    df["rel_strength63"] = df["ret63"] - df["mkt_ret63"]
    return df
