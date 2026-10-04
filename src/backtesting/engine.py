"""Backtesting de cartera, día a día y sin información futura.

Secuencia de cada sesión t:
  1. Se ejecutan a la APERTURA las entradas decididas al cierre de t-1
     (se invalidan si la apertura supera el cierre de la señal en más de max_entry_gap).
  2. Se revisan salidas: ruptura de tendencia marcada el día anterior (a la apertura),
     gaps a través del stop/objetivo, stop/objetivo intradía (si se tocan ambos se
     asume el stop) y salida por tiempo al cierre al cumplir holding_period_days.
  3. Valoración de la cartera al cierre.
  4. Con datos hasta el cierre de t se eligen las entradas para t+1.
Comisión + deslizamiento se aplican en cada entrada y salida.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..pipeline.analysis import SymbolAnalysis


@dataclass
class Rules:
    holding_period_days: int
    stop_loss: float
    take_profit: float
    exit_on_trend_break: bool
    max_entry_gap: float | None
    max_positions: int
    max_new_positions_per_day: int
    initial_capital: float
    cost_per_side: float

    @classmethod
    def from_settings(cls, settings) -> "Rules":
        r = settings.strategy["strategy"]
        return cls(
            holding_period_days=int(r["holding_period_days"]), stop_loss=float(r["stop_loss"]),
            take_profit=float(r["take_profit"]), exit_on_trend_break=bool(r["exit_on_trend_break"]),
            max_entry_gap=float(r["max_entry_gap"]) if r.get("max_entry_gap") is not None else None,
            max_positions=int(r["max_positions"]), max_new_positions_per_day=int(r["max_new_positions_per_day"]),
            initial_capital=float(r["initial_capital"]),
            cost_per_side=(float(r["commission_bps"]) + float(r["slippage_bps"])) / 10_000,
        )


def check_exit(bar_open: float, bar_high: float, bar_low: float, bar_close: float, stop: float, target: float,
               bars_held: int, is_entry_bar: bool, pending_trend_exit: bool, rules: Rules) -> tuple[float, str] | None:
    """Regla de salida compartida por el backtesting y el paper trading.
    `bars_held` incluye la sesión actual (la de entrada cuenta como 1)."""
    if pending_trend_exit:
        return bar_open, "trend_break"
    if not is_entry_bar:
        if bar_open <= stop:
            return bar_open, "stop_loss"
        if bar_open >= target:
            return bar_open, "take_profit"
    if bar_low <= stop:
        return stop, "stop_loss"
    if bar_high >= target:
        return target, "take_profit"
    if bars_held >= rules.holding_period_days:
        return bar_close, "time"
    return None


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    symbols: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    trend_break: np.ndarray
    tradable: np.ndarray
    columns: dict[str, np.ndarray] = field(default_factory=dict)


def build_panel(analyses: dict[str, SymbolAnalysis], calendar: pd.DatetimeIndex, min_price: float,
                min_dollar_volume: float, extra: dict[str, str] | None = None) -> Panel:
    """Matrices (fechas × símbolos). `extra` = {nombre: columna de scores/ind} a incluir."""
    simbolos = sorted(analyses)
    T, N = len(calendar), len(simbolos)

    def matriz(dtype=np.float64, fill=np.nan):
        return np.full((T, N), fill, dtype=dtype)

    o, h, l, c = matriz(), matriz(), matriz(), matriz()
    tendencia = matriz(bool, False)
    operable = matriz(bool, False)
    extras = {nombre: matriz(np.float32) for nombre in (extra or {})}
    for j, s in enumerate(simbolos):
        a = analyses[s]
        ind = a.ind.reindex(calendar)
        o[:, j], h[:, j] = ind["open"].to_numpy(), ind["high"].to_numpy()
        l[:, j], c[:, j] = ind["low"].to_numpy(), ind["close"].to_numpy()
        tendencia[:, j] = ((ind["ma7"] < ind["ma25"]) & (ind["close"] < ind["ma99"])).fillna(False).to_numpy()
        operable[:, j] = a.tradable(min_price, min_dollar_volume).reindex(calendar).fillna(False).to_numpy(bool)
        for nombre, col in (extra or {}).items():
            fuente = a.scores if col in a.scores.columns else a.ind
            extras[nombre][:, j] = fuente[col].reindex(calendar).to_numpy(np.float32)
    return Panel(dates=calendar, symbols=simbolos, open=o, high=h, low=l, close=c, trend_break=tendencia,
                 tradable=operable, columns=extras)


@dataclass
class StrategyResult:
    name: str
    equity: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    invalidated: int = 0


def run_strategy(panel: Panel, entry_signal: np.ndarray, rank: np.ndarray, rules: Rules, start: int, end: int,
                 name: str) -> StrategyResult:
    """Simula la estrategia entre los índices de fecha [start, end]."""
    T, N = panel.close.shape
    efectivo = rules.initial_capital
    posiciones: dict[int, dict] = {}
    pendientes: list[tuple[int, int, float, float]] = []
    ultimo_cierre = np.full(N, np.nan)
    if start > 0:
        previos = panel.close[:start]
        validos = ~np.isnan(previos)
        for j in range(N):
            idx = np.nonzero(validos[:, j])[0]
            if len(idx):
                ultimo_cierre[j] = previos[idx[-1], j]
    curva, exposicion, operaciones = [], [], []
    invalidadas = 0
    equity_previa = rules.initial_capital
    c_costo = rules.cost_per_side
    for t in range(start, end + 1):
        o, h, l, c = panel.open[t], panel.high[t], panel.low[t], panel.close[t]
        # 1) entradas a la apertura
        for j, t_senal, cierre_senal, score in pendientes:
            if j in posiciones or len(posiciones) >= rules.max_positions:
                continue
            if not np.isfinite(o[j]):
                invalidadas += 1
                continue
            if rules.max_entry_gap is not None and o[j] > cierre_senal * (1 + rules.max_entry_gap):
                invalidadas += 1
                continue
            asignacion = min(equity_previa / rules.max_positions, efectivo)
            if asignacion <= 1.0:
                continue
            acciones = asignacion / (o[j] * (1 + c_costo))
            efectivo -= acciones * o[j] * (1 + c_costo)
            posiciones[j] = {"entry_idx": t, "signal_idx": t_senal, "entry": o[j], "shares": acciones,
                             "stop": o[j] * (1 - rules.stop_loss), "target": o[j] * (1 + rules.take_profit),
                             "bars": 0, "min_low": o[j], "pending_trend": False, "score": score}
        pendientes = []
        # 2) salidas
        for j in list(posiciones):
            pos = posiciones[j]
            if not np.isfinite(o[j]):
                continue  # sin sesión para esta acción: no se puede operar hoy
            pos["bars"] += 1
            pos["min_low"] = min(pos["min_low"], l[j])
            salida = check_exit(o[j], h[j], l[j], c[j], pos["stop"], pos["target"], pos["bars"],
                                pos["entry_idx"] == t, pos["pending_trend"], rules)
            if salida is None:
                if rules.exit_on_trend_break and panel.trend_break[t, j]:
                    pos["pending_trend"] = True
                continue
            precio, motivo = salida
            efectivo += pos["shares"] * precio * (1 - c_costo)
            pnl = pos["shares"] * (precio * (1 - c_costo) - pos["entry"] * (1 + c_costo))
            operaciones.append({
                "symbol": panel.symbols[j], "signal_date": panel.dates[pos["signal_idx"]],
                "entry_date": panel.dates[pos["entry_idx"]], "entry_price": pos["entry"],
                "exit_date": panel.dates[t], "exit_price": precio, "exit_reason": motivo,
                "return": (precio * (1 - c_costo)) / (pos["entry"] * (1 + c_costo)) - 1, "pnl": pnl,
                "holding_days": pos["bars"], "mae": pos["min_low"] / pos["entry"] - 1, "score": pos["score"],
            })
            del posiciones[j]
        # 3) valoración al cierre
        ultimo_cierre = np.where(np.isfinite(c), c, ultimo_cierre)
        invertido = sum(p["shares"] * ultimo_cierre[j] for j, p in posiciones.items()
                        if np.isfinite(ultimo_cierre[j]))
        equity = efectivo + invertido
        curva.append(equity)
        exposicion.append(invertido / equity if equity > 0 else 0.0)
        equity_previa = equity
        # 4) nuevas señales para la apertura siguiente
        if t < end:
            huecos = min(rules.max_positions - len(posiciones), rules.max_new_positions_per_day)
            if huecos > 0:
                candidatas = entry_signal[t] & panel.tradable[t] & np.isfinite(c)
                if posiciones:
                    candidatas[list(posiciones)] = False
                idx = np.nonzero(candidatas)[0]
                if len(idx):
                    orden = idx[np.argsort(-np.nan_to_num(rank[t, idx], nan=-np.inf), kind="stable")][:huecos]
                    pendientes = [(int(j), t, float(c[j]), float(rank[t, j])) for j in orden]
    # cierre de las posiciones abiertas al final del periodo
    for j, pos in posiciones.items():
        precio = ultimo_cierre[j]
        pnl = pos["shares"] * (precio * (1 - c_costo) - pos["entry"] * (1 + c_costo))
        operaciones.append({
            "symbol": panel.symbols[j], "signal_date": panel.dates[pos["signal_idx"]],
            "entry_date": panel.dates[pos["entry_idx"]], "entry_price": pos["entry"], "exit_date": panel.dates[end],
            "exit_price": precio, "exit_reason": "end_of_backtest",
            "return": (precio * (1 - c_costo)) / (pos["entry"] * (1 + c_costo)) - 1, "pnl": pnl,
            "holding_days": pos["bars"], "mae": pos["min_low"] / pos["entry"] - 1, "score": pos["score"],
        })
    fechas = panel.dates[start:end + 1]
    return StrategyResult(name=name, equity=pd.Series(curva, index=fechas), exposure=pd.Series(exposicion, index=fechas),
                          trades=pd.DataFrame(operaciones), invalidated=invalidadas)


def buy_and_hold(close: pd.Series, dates: pd.DatetimeIndex, capital: float) -> pd.Series:
    serie = close.reindex(dates).ffill()
    serie = serie / serie.dropna().iloc[0] * capital
    return serie


def equal_weight(panel: Panel, start: int, end: int, capital: float) -> pd.Series:
    """Cartera equiponderada de todo el universo operable (rebalanceo diario), con
    valor `capital` en la fecha inicial. El retorno de cada día usa las acciones
    operables al cierre anterior (sin mirar al futuro)."""
    cierres = panel.close[start:end + 1]
    retornos = cierres[1:] / cierres[:-1] - 1
    validos = panel.tradable[start:end] & np.isfinite(retornos)
    suma = np.where(validos, retornos, 0.0).sum(axis=1)
    cuenta = validos.sum(axis=1)
    media = np.divide(suma, cuenta, out=np.zeros_like(suma), where=cuenta > 0)
    curva = np.concatenate([[capital], capital * np.cumprod(1 + media)])
    return pd.Series(curva, index=panel.dates[start:end + 1])
