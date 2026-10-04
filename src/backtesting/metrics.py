"""Métricas de rendimiento de una curva de equity y de sus operaciones."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _nan():
    return float("nan")


def equity_metrics(equity: pd.Series, risk_free: float = 0.0) -> dict:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return {"total_return": _nan(), "cagr": _nan(), "volatility": _nan(), "sharpe": _nan(),
                "sortino": _nan(), "max_drawdown": _nan(), "days": int(len(equity))}
    retornos = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    anios = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / anios) - 1) if equity.iloc[-1] > 0 else -1.0
    exceso = retornos - risk_free / TRADING_DAYS
    desv = float(retornos.std(ddof=1)) if len(retornos) > 1 else _nan()
    abajo = exceso[exceso < 0]
    desv_abajo = float(np.sqrt((abajo ** 2).sum() / max(len(exceso), 1)))
    sharpe = float(exceso.mean() / desv * math.sqrt(TRADING_DAYS)) if desv and desv > 0 else _nan()
    sortino = float(exceso.mean() / desv_abajo * math.sqrt(TRADING_DAYS)) if desv_abajo > 0 else _nan()
    caida = equity / equity.cummax() - 1
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": desv * math.sqrt(TRADING_DAYS) if desv == desv else _nan(),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(caida.min()),
        "days": int(len(equity)),
        "start": equity.index[0].strftime("%Y-%m-%d"),
        "end": equity.index[-1].strftime("%Y-%m-%d"),
    }


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0, "win_rate": _nan(), "avg_return": _nan(), "median_return": _nan(),
                "avg_gain": _nan(), "avg_loss": _nan(), "profit_factor": _nan(), "avg_holding_days": _nan(),
                "best_trade": _nan(), "worst_trade": _nan()}
    r = trades["return"].astype(float)
    pnl = trades["pnl"].astype(float)
    ganancias, perdidas = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades": int(len(trades)),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "avg_gain": float(r[r > 0].mean()) if (r > 0).any() else _nan(),
        "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else _nan(),
        "profit_factor": float(ganancias / perdidas) if perdidas > 0 else (float("inf") if ganancias > 0 else _nan()),
        "avg_holding_days": float(trades["holding_days"].mean()),
        "best_trade": float(r.max()),
        "worst_trade": float(r.min()),
    }


def yearly_returns(equity: pd.Series) -> list[dict]:
    equity = equity.dropna()
    if len(equity) < 2:
        return []
    salida = []
    for anio, serie in equity.groupby(equity.index.year):
        previo = equity[equity.index < serie.index[0]]
        inicio = previo.iloc[-1] if len(previo) else serie.iloc[0]
        salida.append({"year": int(anio), "return": float(serie.iloc[-1] / inicio - 1),
                       "partial": bool(len(serie) < 240)})
    return salida


def full_metrics(equity: pd.Series, trades: pd.DataFrame | None = None, exposure: pd.Series | None = None,
                 risk_free: float = 0.0) -> dict:
    resultado = equity_metrics(equity, risk_free)
    if trades is not None:
        resultado.update(trade_metrics(trades))
    if exposure is not None and len(exposure):
        resultado["exposure"] = float(exposure.mean())
    resultado["yearly"] = yearly_returns(equity)
    return resultado
