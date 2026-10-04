"""Statistical Scoring Engine.

Combina varios factores en una métrica interna de 0 a 100. NO es una verdad
objetiva ni una probabilidad: es una forma ordenada de resumir la situación
técnica de la acción según reglas fijas y transparentes.

Todo está vectorizado sobre la serie completa y es causal: el score de la
sesión T usa solo datos hasta T. El componente "history" usa resultados de
operaciones pasadas ya cerradas en T (etiquetas desplazadas H sesiones).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ["trend", "momentum", "technical", "volume", "risk", "history"]


def _clip(x, lo: float = 0.0, hi: float = 100.0):
    return np.clip(x, lo, hi)


def trend_score(ind: pd.DataFrame) -> pd.Series:
    s = (30 * ind["above_ma99"] + 20 * ind["ma25_gt_ma99"] + 20 * ind["ma7_gt_ma25"]
         + 15 * (ind["slope_ma99"] > 0) + 15 * (ind["slope_ma25"] > 0))
    return s.where(ind["ma99"].notna() & ind["slope_ma99"].notna())


def momentum_score(ind: pd.DataFrame) -> pd.Series:
    r63 = _clip(50 + ind["ret63"] * 200)
    r21 = _clip(50 + ind["ret21"] * 300)
    hist = ind["macd_hist"]
    macd_c = 50 + 25 * np.sign(hist) + 25 * np.sign(hist - hist.shift(1))
    partes = [r63, r21, macd_c]
    if ind["rel_strength63"].notna().any():
        partes.append(_clip(50 + ind["rel_strength63"] * 200))
    return pd.concat(partes, axis=1).mean(axis=1, skipna=False).where(ind["ret63"].notna())


def technical_score(ind: pd.DataFrame) -> pd.Series:
    rsi_c = _clip(100 - (ind["rsi14"] - 55).abs() * 2.5)
    bb_c = _clip(100 - (ind["bb_pctb"] - 0.55).abs() * 120)
    dist = ind["dist_ma25"]
    pull_c = pd.Series(np.where(dist < 0, 40.0, _clip(90 - np.maximum(dist - 0.03, 0) * 700, 10, 90)),
                       index=ind.index).where(dist.notna())
    reciente = ind["cross_ma7_ma25"].rolling(5, min_periods=1).sum().clip(-1, 1)
    cross_c = 50 + 40 * reciente
    return pd.concat([rsi_c, bb_c, pull_c, cross_c], axis=1).mean(axis=1, skipna=False)


def volume_score(ind: pd.DataFrame) -> pd.Series:
    upr = _clip(50 + np.log(ind["up_volume_ratio"].where(ind["up_volume_ratio"] > 0)) * 60)
    tendencia = np.log(ind["vol_trend"].where(ind["vol_trend"] > 0))
    direccion = np.where(ind["ret5"] >= 0, 1.0, -1.0)
    vt = _clip(50 + tendencia * 80 * direccion)
    return pd.concat([upr, pd.Series(vt, index=ind.index)], axis=1).mean(axis=1, skipna=False)


def risk_score(ind: pd.DataFrame) -> pd.Series:
    """Más alto = menos riesgo (volatilidad, drawdown y ATR bajos)."""
    vol_c = _clip(105 - ind["vol20"] * 120)
    dd_c = _clip(85 + ind["drawdown252"] * 150)
    atr_c = _clip(105 - ind["atr_pct"] * 1333)
    return pd.concat([vol_c, dd_c, atr_c], axis=1).mean(axis=1, skipna=False)


def history_score(ind: pd.DataFrame, labels: pd.DataFrame | None, horizon: int, window: int,
                  min_obs: int) -> pd.Series:
    """% de señales pasadas en tendencia alcista (MA7>MA25>MA99) que resultaron
    oportunidades, usando solo operaciones cuyo resultado ya se conocía en T."""
    if labels is None:
        return pd.Series(np.nan, index=ind.index)
    condicion = ((ind["ma7"] > ind["ma25"]) & (ind["ma25"] > ind["ma99"])).astype(float)
    etiqueta = labels["label"]
    valido = etiqueta.notna() & ind["ma99"].notna()
    # La etiqueta de t se conoce al cierre de t+H: se desplaza H sesiones
    exitos = (etiqueta.fillna(0) * condicion * valido).shift(horizon)
    casos = (condicion * valido).shift(horizon)
    suma_exitos = exitos.rolling(window, min_periods=1).sum()
    suma_casos = casos.rolling(window, min_periods=1).sum()
    tasa = suma_exitos / suma_casos.where(suma_casos >= min_obs)
    return 100 * tasa


def compute_statistical_scores(ind: pd.DataFrame, labels: pd.DataFrame | None, settings) -> pd.DataFrame:
    """Devuelve trend_score, momentum_score, technical_score, volume_score,
    risk_score, history_score y statistical_score (media ponderada de los disponibles)."""
    scfg = settings.model["statistical"]
    horizonte = int(settings.strategy["strategy"]["holding_period_days"])
    partes = pd.DataFrame({
        "trend_score": trend_score(ind),
        "momentum_score": momentum_score(ind),
        "technical_score": technical_score(ind),
        "volume_score": volume_score(ind),
        "risk_score": risk_score(ind),
        "history_score": history_score(ind, labels, horizonte, int(scfg["history_window_days"]),
                                       int(scfg["history_min_observations"])),
    }, index=ind.index)
    pesos = scfg["weights"]
    suma = pd.Series(0.0, index=ind.index)
    peso_total = pd.Series(0.0, index=ind.index)
    for comp in COMPONENTS:
        w = float(pesos.get(comp, 0))
        valores = partes[f"{comp}_score"]
        disponible = valores.notna()
        suma += np.where(disponible, valores.fillna(0) * w, 0.0)
        peso_total += np.where(disponible, w, 0.0)
    core = partes[["trend_score", "momentum_score", "technical_score", "risk_score"]].notna().all(axis=1)
    partes["statistical_score"] = (suma / peso_total.replace(0, np.nan)).where(core)
    return partes.astype(float)
