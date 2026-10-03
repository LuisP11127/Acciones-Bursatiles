"""Variables de entrada de la red neuronal.

Todas son relativas (porcentajes, ratios), así una acción de $20 y otra de
$900 se comparan en la misma escala. Solo usan información disponible al
cierre de cada día: nada del futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicadores import agregar_indicadores

NOMBRES = [
    "dist_ma7", "dist_ma25", "dist_ma99", "ma7_vs_ma25", "ma25_vs_ma99",
    "pend_ma7", "pend_ma25", "pend_ma99",
    "ret_1", "ret_5", "ret_10", "ret_21", "ret_63",
    "rsi", "macd_hist", "volatilidad_21", "atr",
    "volumen_20", "volumen_5_50",
    "posicion_52s", "dist_max_52s", "dist_min_52s", "bollinger",
    "cuerpo_vela", "mecha_superior", "mecha_inferior", "cruce_7_25",
    "mercado_dist_ma99", "mercado_ret_21", "mercado_volatilidad_21", "fuerza_relativa_63",
]


def caracteristicas_mercado(mercado: pd.DataFrame | None) -> pd.DataFrame | None:
    if mercado is None or len(mercado) < 100:
        return None
    cierre = mercado["Close"]
    return pd.DataFrame({
        "mercado_dist_ma99": cierre / cierre.rolling(99, min_periods=99).mean() - 1,
        "mercado_ret_21": cierre.pct_change(21),
        "mercado_volatilidad_21": cierre.pct_change().rolling(21, min_periods=21).std(),
        "mercado_ret_63": cierre.pct_change(63),
    })


def construir_caracteristicas(df: pd.DataFrame, mercado: pd.DataFrame | None = None) -> pd.DataFrame:
    """`mercado` es la salida de caracteristicas_mercado (S&P 500)."""
    if "MA99" not in df.columns:
        df = agregar_indicadores(df)
    c = df["Close"]
    ret_diario = c.pct_change()
    seguro = lambda s: s.replace(0, np.nan)  # noqa: E731
    f = pd.DataFrame(index=df.index)
    f["dist_ma7"] = c / df["MA7"] - 1
    f["dist_ma25"] = c / df["MA25"] - 1
    f["dist_ma99"] = c / df["MA99"] - 1
    f["ma7_vs_ma25"] = df["MA7"] / df["MA25"] - 1
    f["ma25_vs_ma99"] = df["MA25"] / df["MA99"] - 1
    f["pend_ma7"] = df["MA7"].pct_change(3)
    f["pend_ma25"] = df["MA25"].pct_change(5)
    f["pend_ma99"] = df["MA99"].pct_change(10)
    for n in (1, 5, 10, 21, 63):
        f[f"ret_{n}"] = c.pct_change(n)
    f["rsi"] = (df["RSI14"] - 50) / 50
    f["macd_hist"] = df["MACD_hist"] / c
    f["volatilidad_21"] = ret_diario.rolling(21, min_periods=21).std()
    f["atr"] = df["ATR14"] / c
    f["volumen_20"] = np.log((df["Volume"] + 1) / (df["VolMA20"] + 1))
    f["volumen_5_50"] = np.log((df["VolMA5"] + 1) / (df["VolMA50"] + 1))
    rango = seguro(df["Max252"] - df["Min252"])
    f["posicion_52s"] = (c - df["Min252"]) / rango
    f["dist_max_52s"] = c / df["Max252"] - 1
    f["dist_min_52s"] = np.log(c / df["Min252"])
    f["bollinger"] = df["BB_pctb"].clip(-1, 2)
    f["cuerpo_vela"] = (c - df["Open"]) / df["Open"]
    f["mecha_superior"] = (df["High"] - np.maximum(df["Open"], c)) / c
    f["mecha_inferior"] = (np.minimum(df["Open"], c) - df["Low"]) / c
    signo = np.sign(df["MA7"] - df["MA25"])
    cruce = signo.diff().fillna(0) / 2  # +1 cruce alcista, -1 bajista
    f["cruce_7_25"] = cruce.rolling(5, min_periods=1).sum().clip(-1, 1)
    if mercado is not None:
        m = mercado.reindex(df.index).ffill()
        f["mercado_dist_ma99"] = m["mercado_dist_ma99"]
        f["mercado_ret_21"] = m["mercado_ret_21"]
        f["mercado_volatilidad_21"] = m["mercado_volatilidad_21"]
        f["fuerza_relativa_63"] = f["ret_63"] - m["mercado_ret_63"]
    else:
        for col in ("mercado_dist_ma99", "mercado_ret_21", "mercado_volatilidad_21"):
            f[col] = 0.0
        f["fuerza_relativa_63"] = f["ret_63"]
    # Las filas sin historial suficiente quedan con NaN y se descartan al entrenar
    return f[NOMBRES].replace([np.inf, -np.inf], np.nan).astype(np.float32)
