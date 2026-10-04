"""Ejecuta el backtest completo: estrategias del sistema frente a benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..logging_utils import get_logger, summary
from ..pipeline.analysis import SymbolAnalysis
from .engine import Rules, build_panel, buy_and_hold, equal_weight, run_strategy
from .metrics import equity_metrics, full_metrics, yearly_returns
from .strategies import STRATEGY_DESCRIPTIONS, strategy_signals

log = get_logger("BACKTEST")
PANEL_COLUMNS = {"statistical_score": "statistical_score", "risk_score": "risk_score",
                 "cross_ma7_ma25": "cross_ma7_ma25", "above_ma99": "above_ma99", "ret63": "ret63"}
NOTES = [
    "Las señales se calculan con datos disponibles hasta el cierre de cada sesión y se ejecutan en la apertura "
    "siguiente; nunca se usa información futura.",
    "La probabilidad de la red neuronal procede de la validación walk-forward: cada año se predice con un modelo "
    "entrenado solo con años anteriores.",
    "Noticias y fundamentales no tienen historial punto a punto en fuentes gratuitas: no intervienen en el backtest.",
    "Sesgo de supervivencia: el universo es la composición ACTUAL del S&P 500 / Nasdaq-100; las empresas que "
    "salieron de los índices no están, lo que tiende a mejorar los resultados históricos.",
    "Incluye comisión y deslizamiento en cada entrada y salida; no incluye impuestos ni dividendos de efectivo "
    "más allá de los precios ajustados.",
    "Resultados de una simulación histórica: no garantizan resultados futuros ni son recomendaciones de inversión.",
]


def _matriz_oos(oos: pd.DataFrame, columna: str, fechas: pd.DatetimeIndex, simbolos: list[str]) -> np.ndarray:
    tabla = oos.pivot_table(index="date", columns="symbol", values=columna, aggfunc="last")
    tabla.index = pd.to_datetime(tabla.index)
    return tabla.reindex(index=fechas, columns=simbolos).to_numpy(np.float64)


def _semanal(serie: pd.Series) -> pd.Series:
    return serie.resample("W-FRI").last().dropna()


def run_backtest(settings, analyses: dict[str, SymbolAnalysis], calendar: pd.DatetimeIndex,
                 benchmark_close: pd.Series, oos: pd.DataFrame | None = None,
                 model_version: str | None = None) -> tuple[dict, pd.DataFrame]:
    reglas = Rules.from_settings(settings)
    filtros = settings.universe["filters"]
    bt_cfg = settings.strategy["backtest"]
    panel = build_panel(analyses, calendar, float(filtros["min_price"]), float(filtros["min_avg_dollar_volume"]),
                        PANEL_COLUMNS)
    hay_red = oos is not None and not oos.empty
    prob = umbral = base = None
    if hay_red:
        prob = _matriz_oos(oos, "probability", panel.dates, panel.symbols)
        umbral = _matriz_oos(oos, "threshold", panel.dates, panel.symbols)
        base = _matriz_oos(oos, "train_base_rate", panel.dates, panel.symbols)
        fechas_oos = pd.to_datetime(oos["date"])
        inicio = int(panel.dates.searchsorted(fechas_oos.min()))
        fin = int(panel.dates.searchsorted(fechas_oos.max(), side="right") - 1)
    else:
        cobertura = np.isfinite(panel.columns["statistical_score"]).mean(axis=1)
        validas = np.nonzero(cobertura >= 0.3)[0]
        if len(validas) == 0:
            raise ValueError("No hay scores suficientes para el backtest")
        inicio, fin = int(validas[0]), len(panel.dates) - 1
    if bt_cfg.get("start"):
        inicio = max(inicio, int(panel.dates.searchsorted(pd.Timestamp(bt_cfg["start"]))))
    if bt_cfg.get("end"):
        fin = min(fin, int(panel.dates.searchsorted(pd.Timestamp(bt_cfg["end"]), side="right") - 1))
    if fin - inicio < 60:
        raise ValueError("Periodo de backtest demasiado corto")
    rf = float(bt_cfg.get("risk_free_rate", 0.0))
    log.info("Backtest %s → %s (%d sesiones, %d acciones)", panel.dates[inicio].date(), panel.dates[fin].date(),
             fin - inicio + 1, len(panel.symbols))

    estrategias, curvas, operaciones = {}, {}, []
    for nombre in bt_cfg["strategies"]:
        if nombre in ("combined", "neural_only") and not hay_red:
            estrategias[nombre] = {"description": STRATEGY_DESCRIPTIONS[nombre], "available": False,
                                   "reason": "No hay predicciones fuera de muestra de la red neuronal."}
            continue
        entrada, orden = strategy_signals(nombre, panel, settings, prob, umbral, base)
        res = run_strategy(panel, entrada, orden, reglas, inicio, fin, nombre)
        metricas = full_metrics(res.equity, res.trades, res.exposure, rf)
        estrategias[nombre] = {"description": STRATEGY_DESCRIPTIONS[nombre], "available": True,
                               "metrics": metricas, "invalidated_entries": res.invalidated}
        curvas[nombre] = res.equity
        if not res.trades.empty:
            operaciones.append(res.trades.assign(strategy=nombre))
        log.info("%s: retorno %.1f%%, Sharpe %.2f, %d operaciones", nombre,
                 100 * metricas["total_return"], metricas["sharpe"], metricas.get("trades", 0))
    fechas = panel.dates[inicio:fin + 1]
    benchmarks = {}
    if benchmark_close is not None and len(benchmark_close):
        bh = buy_and_hold(benchmark_close, fechas, reglas.initial_capital)
        benchmarks["buy_hold_benchmark"] = {"description": STRATEGY_DESCRIPTIONS["buy_hold_benchmark"],
                                            "metrics": {**equity_metrics(bh, rf), "yearly": yearly_returns(bh)}}
        curvas["buy_hold_benchmark"] = bh
    ew = equal_weight(panel, inicio, fin, reglas.initial_capital)
    benchmarks["equal_weight_universe"] = {"description": STRATEGY_DESCRIPTIONS["equal_weight_universe"],
                                           "metrics": {**equity_metrics(ew, rf), "yearly": yearly_returns(ew)}}
    curvas["equal_weight_universe"] = ew

    semanales = {k: _semanal(v) for k, v in curvas.items()}
    indice = sorted(set().union(*[set(s.index) for s in semanales.values()]))
    tabla = pd.DataFrame(semanales).reindex(indice).ffill()
    caidas = tabla / tabla.cummax() - 1
    todas = pd.concat(operaciones, ignore_index=True) if operaciones else pd.DataFrame()
    resultado = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_as_of": calendar[-1].strftime("%Y-%m-%d"),
        "period": {"start": fechas[0].strftime("%Y-%m-%d"), "end": fechas[-1].strftime("%Y-%m-%d"),
                   "sessions": int(len(fechas))},
        "neural_available": hay_red,
        "model_version": model_version,
        "symbols": len(panel.symbols),
        "assumptions": {
            "execution": "señal al cierre de T, entrada en la apertura de T+1",
            "holding_period_days": reglas.holding_period_days, "stop_loss": reglas.stop_loss,
            "take_profit": reglas.take_profit, "exit_on_trend_break": reglas.exit_on_trend_break,
            "max_entry_gap": reglas.max_entry_gap, "max_positions": reglas.max_positions,
            "max_new_positions_per_day": reglas.max_new_positions_per_day,
            "position_sizing": "equiponderado: 1/max_positions del valor de la cartera",
            "cost_per_side": reglas.cost_per_side, "initial_capital": reglas.initial_capital,
            "risk_free_rate": rf, "min_final_score": settings.rules["min_final_score"],
            "min_statistical_score": settings.rules["min_statistical_score"],
            "backtest_components": ["statistical", "neural", "risk"],
        },
        "strategies": estrategias,
        "benchmarks": benchmarks,
        "curves": {"dates": [d.strftime("%Y-%m-%d") for d in tabla.index],
                   "series": {k: tabla[k].round(2).tolist() for k in tabla.columns}},
        "drawdowns": {"dates": [d.strftime("%Y-%m-%d") for d in caidas.index],
                      "series": {k: caidas[k].round(4).tolist() for k in caidas.columns}},
        "notes": NOTES,
    }
    summary.set("backtest_period", f"{resultado['period']['start']} → {resultado['period']['end']}")
    return resultado, todas
