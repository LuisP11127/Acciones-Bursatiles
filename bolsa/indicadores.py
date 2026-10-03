"""Indicadores técnicos: medias móviles MA(7), MA(25), MA(99), RSI, MACD, ATR..."""

from __future__ import annotations

import numpy as np
import pandas as pd

MEDIAS = (7, 25, 99)


def sma(serie: pd.Series, n: int) -> pd.Series:
    return serie.rolling(n, min_periods=n).mean()


def ema(serie: pd.Series, n: int) -> pd.Series:
    return serie.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(cierre: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder."""
    delta = cierre.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perdida = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    resultado = 100 - 100 / (1 + rs)
    # Sin pérdidas en el periodo: RSI = 100
    return resultado.where(perdida != 0, 100.0).where(ganancia.notna())


def macd(cierre: pd.Series, rapida: int = 12, lenta: int = 26, senal: int = 9):
    linea = ema(cierre, rapida) - ema(cierre, lenta)
    linea_senal = linea.ewm(span=senal, adjust=False, min_periods=senal).mean()
    return linea, linea_senal, linea - linea_senal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    cierre_prev = df["Close"].shift(1)
    rango = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - cierre_prev).abs(),
            (df["Low"] - cierre_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rango.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger_pctb(cierre: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    media = sma(cierre, n)
    desv = cierre.rolling(n, min_periods=n).std()
    ancho = (2 * k * desv).replace(0, np.nan)
    return (cierre - (media - k * desv)) / ancho


def agregar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del DataFrame OHLCV con todos los indicadores."""
    df = df.copy()
    cierre = df["Close"]
    for n in MEDIAS:
        df[f"MA{n}"] = sma(cierre, n)
    df["RSI14"] = rsi(cierre)
    df["MACD"], df["MACD_senal"], df["MACD_hist"] = macd(cierre)
    df["ATR14"] = atr(df)
    df["VolMA5"] = sma(df["Volume"], 5)
    df["VolMA20"] = sma(df["Volume"], 20)
    df["VolMA50"] = sma(df["Volume"], 50)
    df["Max252"] = df["High"].rolling(252, min_periods=126).max()
    df["Min252"] = df["Low"].rolling(252, min_periods=126).min()
    df["BB_pctb"] = bollinger_pctb(cierre)
    return df


def cruce_reciente(rapida: pd.Series, lenta: pd.Series, dias: int) -> int:
    """+1 si `rapida` cruzó por encima de `lenta` en los últimos `dias`,
    -1 si cruzó por debajo, 0 si no hubo cruce (gana el más reciente)."""
    diferencia = (rapida - lenta).dropna()
    if len(diferencia) < 2:
        return 0
    signo = np.sign(diferencia.to_numpy())
    ventana = signo[-(dias + 1):]
    for i in range(len(ventana) - 1, 0, -1):
        if ventana[i] > 0 >= ventana[i - 1]:
            return 1
        if ventana[i] < 0 <= ventana[i - 1]:
            return -1
    return 0
