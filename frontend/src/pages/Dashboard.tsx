import { Link, useNavigate } from "react-router-dom";
import { useJson } from "../api";
import { DataTable, type Column } from "../components/DataTable";
import { MetricsGrid, TradeTable, YearlyTable } from "../components/Panels";
import { Card, ErrorBox, Freshness, Loading, Meter, Stat, StatusChip } from "../components/ui";
import { date, dateTime, int, money, num, pct, signClass, STRATEGY_LABELS } from "../format";
import type { MarketJson, OpportunityRow } from "../types";

export function Dashboard() {
  const { data, error, loading } = useJson<MarketJson>("market.json");
  const navegar = useNavigate();
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  const estado = data.status.state;
  const columnas: Column<OpportunityRow>[] = [
    { key: "symbol", header: "Acción", render: (r) => <><strong>{r.symbol}</strong> <span className="muted">{r.name}</span></> },
    { key: "price", header: "Precio", align: "right", render: (r) => money(r.price) },
    { key: "final", header: "Score final", render: (r) => <Meter value={r.final_score} label="Score final" /> },
    { key: "stat", header: "Estadístico", align: "right", render: (r) => num(r.statistical_score, 0) },
    { key: "prob", header: "Prob. red", align: "right", render: (r) => pct(r.opportunity_probability, 1, false) },
    { key: "risk", header: "Riesgo", render: (r) => r.risk_level ?? "—" },
  ];
  const perf = data.system_performance;
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Dashboard</h1>
          <Freshness generated={data.status.last_update ?? data.generated_at} asOf={data.data_as_of} />
        </div>
        <StatusChip kind={estado === "ok" ? "good" : estado === "synthetic" ? "critical" : "warning"}>
          {estado === "ok" ? "Sistema operativo" : data.status.message}
        </StatusChip>
      </div>
      <div className="banner banner-info">{data.disclaimer}</div>

      <div className="stats">
        <Stat label="Estado del sistema" value={estado === "ok" ? "Operativo" : estado === "synthetic" ? "Desarrollo" : "Sin datos"}
              sub={data.status.last_run ? `${data.status.last_run.n_errors ?? 0} errores · ${data.status.last_run.n_warnings ?? 0} avisos en la última ejecución` : undefined} />
        <Stat label="Última actualización" value={dateTime(data.status.last_update)} sub={`Datos al cierre del ${date(data.data_as_of)}`} />
        <Stat label="Mercado analizado" value="EE.UU." sub={`${data.market.name} · ${data.market.timezone}`} />
        <Stat label="Acciones analizadas" value={int(data.universe.eligible)}
              sub={`de ${int(data.universe.size)} en el universo · ${int(data.universe.skipped)} omitidas por calidad/liquidez`} />
        <Stat label="Oportunidades" value={int(data.counts.opportunities)} sub="score final sobre el umbral" />
        <Stat label="Operaciones abiertas" value={int(data.counts.open_trades)} sub={`${data.counts.pending_trades} pendientes de ejecutar · modo ${data.mode}`} />
        <Stat label="Modelo neuronal" value={data.model.production_version ?? "Sin modelo"}
              sub={data.model.production_version ? `Entrenado el ${date(data.model.training_date)} · AUC fuera de muestra ${num(data.model.oos_auc, 3)}` : (data.model.reason ?? "Aún no hay un modelo promovido")} />
        <Stat label="Último entrenamiento" value={date(data.model.last_training)}
              sub={data.model.last_candidate ? `${data.model.last_candidate}: ${data.model.last_candidate_promoted ? "promovido" : "no promovido (no superó los criterios)"}` : "Aún no se ha entrenado"} />
      </div>

      <div className="grid grid-2">
        <Card title="Top oportunidades" extra={<Link to="/opportunities">Ver todas →</Link>}>
          <p className="muted" style={{ fontSize: "0.82rem" }}>
            Ordenadas por el score final, una métrica interna del sistema: no es una certeza ni una recomendación.
          </p>
          <DataTable rows={data.top_opportunities} columns={columnas} rowKey={(r) => r.symbol}
                     onRowClick={(r) => navegar(`/stock/${r.symbol}`)} empty="Hoy ninguna acción supera el umbral de oportunidad." />
        </Card>
        <Card title="Mercado" extra={`Al cierre del ${date(data.data_as_of)}`}>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Índice</th><th className="right">Precio</th><th className="right">Día</th><th className="right">1 mes</th><th className="right">3 meses</th><th>Tendencia (MA 7/25/99)</th></tr></thead>
              <tbody>
                {data.market.indices.map((i) => (
                  <tr key={i.symbol}>
                    <td><strong>{i.symbol}</strong></td>
                    <td className="right num">{money(i.price)}</td>
                    <td className={`right num ${signClass(i.change_1d)}`}>{pct(i.change_1d, 2)}</td>
                    <td className={`right num ${signClass(i.ret21)}`}>{pct(i.ret21)}</td>
                    <td className={`right num ${signClass(i.ret63)}`}>{pct(i.ret63)}</td>
                    <td>{i.trend}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <Card title="Rendimiento del sistema (paper trading)" extra={perf.start ? `${date(perf.start)} → ${date(perf.end)}` : "Sin historial todavía"}>
        <MetricsGrid metrics={{ ...perf, trades: perf.closed_trades }} />
        {perf.notes.map((n) => <p key={n} className="muted" style={{ fontSize: "0.82rem", marginTop: 8 }}>ℹ️ {n}</p>)}
        <p className="muted" style={{ fontSize: "0.82rem" }}>
          Total de operaciones simuladas: {perf.total_trades} · cerradas {perf.closed_trades} · abiertas {perf.open_trades} · pendientes {perf.pending_trades} · invalidadas {perf.invalidated_trades}.
        </p>
        <h3 style={{ marginTop: 12 }}>Resultados por año</h3>
        <YearlyTable rows={[{ label: "Paper trading", years: perf.by_year ?? [] }]} />
      </Card>

      <div className="grid grid-2">
        <Card title="Backtest histórico" extra={<Link to="/backtesting">Detalle →</Link>}>
          {data.backtest ? (
            <>
              <p className="muted" style={{ fontSize: "0.82rem" }}>
                Simulación walk-forward {date(data.backtest.period.start)} → {date(data.backtest.period.end)} · estrategia {STRATEGY_LABELS[data.backtest.strategy] ?? data.backtest.strategy}
                {data.backtest.model_version ? ` · modelo ${data.backtest.model_version}` : ""} · generado {dateTime(data.backtest.generated_at)}.
                Resultados históricos simulados: no garantizan resultados futuros.
              </p>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Métrica</th><th className="right">Estrategia</th><th className="right">Buy &amp; Hold SPY</th></tr></thead>
                  <tbody>
                    {([["total_return", "Retorno total", pct], ["cagr", "CAGR", pct], ["sharpe", "Sharpe", (v: number) => num(v, 2)],
                       ["max_drawdown", "Drawdown máximo", pct]] as [string, string, (v: number) => string][]).map(([k, l, f]) => (
                      <tr key={k}>
                        <td>{l}</td>
                        <td className="right num">{typeof data.backtest!.metrics[k] === "number" ? f(data.backtest!.metrics[k] as number) : "—"}</td>
                        <td className="right num">{typeof data.backtest!.benchmark[k] === "number" ? f(data.backtest!.benchmark[k] as number) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : <div className="empty">El backtest se genera con el workflow semanal de entrenamiento.</div>}
        </Card>
        <Card title="¿Acierta el sistema? (evaluación en vivo)" extra={<Link to="/model">Modelo IA →</Link>}>
          {data.live_evaluation.available ? (
            <div className="stats">
              <Stat label="Días evaluados" value={data.live_evaluation.evaluated_days} sub={`${int(data.live_evaluation.samples)} predicciones`} />
              <Stat label="Tasa base" value={pct(data.live_evaluation.base_rate, 1, false)} sub="acciones que fueron oportunidad" />
              <Stat label="Acierto top 10 diario" value={pct(data.live_evaluation.top10_final_hit_rate, 1, false)} sub="según el score final" />
              <Stat label="Acierto de las oportunidades" value={pct(data.live_evaluation.opportunities_hit_rate, 1, false)} />
            </div>
          ) : <div className="empty">{data.live_evaluation.note ?? "Sin evaluación todavía."}</div>}
          {data.live_evaluation.note && data.live_evaluation.available && <p className="muted" style={{ fontSize: "0.8rem", marginTop: 8 }}>{data.live_evaluation.note}</p>}
        </Card>
      </div>

      <div className="grid grid-2">
        <Card title="Mejores operaciones cerradas">
          <TradeTable trades={data.best_trades} empty="Todavía no hay operaciones cerradas." />
        </Card>
        <Card title="Peores operaciones cerradas">
          <TradeTable trades={data.worst_trades} empty="Todavía no hay operaciones cerradas." />
        </Card>
      </div>
    </div>
  );
}
