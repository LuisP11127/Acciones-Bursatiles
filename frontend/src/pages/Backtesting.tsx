import { Fragment } from "react";
import { useJson } from "../api";
import { YearlyTable } from "../components/Panels";
import { SeriesChart, type SeriesDef } from "../components/SeriesChart";
import { Card, ErrorBox, Loading } from "../components/ui";
import { date, dateTime, EXIT_LABELS, isNum, money, moneyAxis, num, pct, signClass, STRATEGY_LABELS } from "../format";
import type { BacktestJson, Metrics, StrategyBlock } from "../types";

const ORDEN = ["combined", "statistical_only", "neural_only", "ma_crossover", "buy_hold_benchmark", "equal_weight_universe"];
const COLUMNAS: [string, string, (v: number) => string][] = [
  ["total_return", "Retorno", (v) => pct(v)], ["cagr", "CAGR", (v) => pct(v)], ["sharpe", "Sharpe", (v) => num(v, 2)],
  ["sortino", "Sortino", (v) => num(v, 2)], ["max_drawdown", "Máx. DD", (v) => pct(v)], ["volatility", "Volatilidad", (v) => pct(v, 1, false)],
  ["win_rate", "Win rate", (v) => pct(v, 1, false)], ["avg_gain", "Gan. media", (v) => pct(v, 2)], ["avg_loss", "Pérd. media", (v) => pct(v, 2)],
  ["profit_factor", "Profit factor", (v) => num(v, 2)], ["trades", "Operaciones", (v) => String(v)],
  ["avg_holding_days", "Días medios", (v) => num(v, 1)], ["exposure", "Exposición", (v) => pct(v, 0, false)],
];
const ASUNCIONES: Record<string, string> = {
  execution: "Ejecución", holding_period_days: "Días de mantenimiento", stop_loss: "Stop-loss", take_profit: "Take-profit",
  exit_on_trend_break: "Salida por ruptura de tendencia", max_entry_gap: "Gap máximo de entrada", max_positions: "Posiciones máximas",
  max_new_positions_per_day: "Entradas nuevas por día", position_sizing: "Tamaño de posición", cost_per_side: "Coste por lado",
  initial_capital: "Capital inicial", risk_free_rate: "Tipo libre de riesgo", min_final_score: "Score final mínimo",
  min_statistical_score: "Score estadístico mínimo", backtest_components: "Componentes del score en el backtest",
};

function valorAsuncion(k: string, v: unknown): string {
  if (typeof v === "boolean") return v ? "sí" : "no";
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "number") {
    if (["stop_loss", "take_profit", "max_entry_gap", "cost_per_side", "risk_free_rate"].includes(k)) return pct(v, 2, false);
    if (k === "initial_capital") return money(v, 0);
    return String(v);
  }
  return v == null ? "—" : String(v);
}

export function Backtesting() {
  const { data, error, loading } = useJson<BacktestJson>("backtest.json");
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  if (!data.available) return <Card title="Backtesting"><div className="empty">{data.message}</div></Card>;
  const bloques: [string, StrategyBlock][] = ORDEN.flatMap((k) => {
    const b = data.strategies?.[k] ?? data.benchmarks?.[k];
    return b ? [[k, b] as [string, StrategyBlock]] : [];
  });
  const series: SeriesDef[] = ORDEN.map((k, i) => ({ key: k, label: STRATEGY_LABELS[k] ?? k, slot: i }))
    .filter((s) => data.curves?.series[s.key]);
  const curva = (data.curves?.dates ?? []).map((d, i) => ({ date: d, ...Object.fromEntries(series.map((s) => [s.key, data.curves!.series[s.key][i]])) }));
  const caidas = (data.drawdowns?.dates ?? []).map((d, i) => ({ date: d, ...Object.fromEntries(series.map((s) => [s.key, data.drawdowns!.series[s.key]?.[i] ?? null])) }));
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Backtesting walk-forward</h1>
          <span className="freshness">
            Periodo {date(data.period?.start)} → {date(data.period?.end)} ({data.period?.sessions} sesiones, {data.symbols} acciones) ·
            generado {dateTime(data.generated_at)} · datos al {date(data.data_as_of)} · modelo {data.model_version ?? "sin red neuronal"}
          </span>
        </div>
      </div>
      <div className="banner banner-warning">
        Simulación histórica sin información futura. Sirve para evaluar el comportamiento del sistema, no demuestra que la estrategia
        sea válida para invertir. {data.neural_available ? "La red neuronal usa predicciones fuera de muestra (cada año, un modelo entrenado solo con años anteriores)." : "Sin predicciones de la red neuronal: solo se muestran las estrategias que no la necesitan."}
      </div>
      <Card title="Estrategia vs. benchmarks">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Serie</th>{COLUMNAS.map(([k, l]) => <th key={k} className="right">{l}</th>)}</tr></thead>
            <tbody>
              {bloques.map(([k, b]) => (
                <tr key={k}>
                  <td title={b.description}><strong>{STRATEGY_LABELS[k] ?? k}</strong>{b.available === false && <div className="muted" style={{ fontSize: "0.78rem" }}>{b.reason}</div>}</td>
                  {COLUMNAS.map(([c, , f]) => {
                    const v = (b.metrics as Metrics | undefined)?.[c];
                    return <td key={c} className={`right num ${["total_return", "cagr"].includes(c) ? signClass(v as number) : ""}`}>{isNum(v) ? f(v) : "—"}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: "0.8rem" }}>Los benchmarks no tienen métricas por operación (—). «Exposición» = fracción media del capital invertida.</p>
      </Card>
      <Card title="Curva de equity" extra="Semanal · capital inicial igual para todas las series">
        <SeriesChart data={curva} xKey="date" xFormat={date} yFormat={(v) => money(v, 0)} yTickFormat={moneyAxis} height={340} series={series} />
      </Card>
      <Card title="Drawdown">
        <SeriesChart data={caidas} xKey="date" xFormat={date} yFormat={(v) => pct(v, 0)} height={240} series={series} />
      </Card>
      <Card title="Resultados por año">
        <YearlyTable rows={bloques.filter(([, b]) => b.metrics?.yearly).map(([k, b]) => ({ label: k, years: b.metrics!.yearly! }))} />
      </Card>
      <div className="grid grid-2">
        <Card title="Supuestos">
          <dl className="kv">
            {Object.entries(data.assumptions ?? {}).map(([k, v]) => (<Fragment key={k}><dt>{ASUNCIONES[k] ?? k}</dt><dd>{valorAsuncion(k, v)}</dd></Fragment>))}
          </dl>
        </Card>
        <Card title="Limitaciones y notas">
          <ul>{(data.notes ?? []).map((n) => <li key={n} style={{ marginBottom: 6 }}>{n}</li>)}</ul>
          <h3>Estrategias</h3>
          <ul>{bloques.map(([k, b]) => <li key={k}><strong>{STRATEGY_LABELS[k] ?? k}:</strong> {b.description}</li>)}</ul>
        </Card>
      </div>
      <Card title="Muestra de operaciones del sistema combinado" extra="últimas 300">
        {data.trades_sample?.length ? (
          <div className="table-wrap" style={{ maxHeight: 420, overflowY: "auto" }}>
            <table>
              <thead><tr><th>Acción</th><th>Entrada</th><th className="right">Precio</th><th>Salida</th><th className="right">Precio</th><th>Motivo</th><th className="right">Retorno</th><th className="right">Días</th></tr></thead>
              <tbody>
                {[...data.trades_sample].reverse().map((t, i) => (
                  <tr key={i}>
                    <td>{String(t.symbol)}</td><td>{date(String(t.entry_date).slice(0, 10))}</td><td className="right num">{money(t.entry_price as number)}</td>
                    <td>{date(String(t.exit_date).slice(0, 10))}</td><td className="right num">{money(t.exit_price as number)}</td><td>{EXIT_LABELS[String(t.exit_reason)] ?? String(t.exit_reason)}</td>
                    <td className={`right num ${signClass(t.return as number)}`}>{pct(t.return as number)}</td><td className="right num">{String(t.holding_days)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty">Sin operaciones.</div>}
      </Card>
    </div>
  );
}
