import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { useJson } from "../api";
import { ExplanationPanel, ModelFactors, NewsList, TradeTable } from "../components/Panels";
import { type ChartMarker, PriceChart } from "../components/PriceChart";
import { SeriesChart } from "../components/SeriesChart";
import { Card, ErrorBox, Freshness, Loading, Meter, Stat } from "../components/ui";
import { COMPONENT_LABELS, date, isNum, money, num, pct, signClass } from "../format";
import type { StockJson, Trade } from "../types";

function buildMarkers(stock: StockJson): ChartMarker[] {
  const marcas: ChartMarker[] = [];
  const operacion = (t: { entry_date?: unknown; entry_price?: unknown; exit_date?: unknown; exit_price?: unknown; return?: unknown; actual_return?: unknown }, origen: string) => {
    const ret = (t.actual_return ?? t.return) as number | null | undefined;
    if (t.entry_date) marcas.push({ date: String(t.entry_date).slice(0, 10), kind: "entry", text: `Entrada${origen === "paper" ? "" : " (bt)"}`,
      detail: `Entrada ${origen === "paper" ? "simulada" : "backtest"} a ${money(t.entry_price as number)}` });
    if (t.exit_date) marcas.push({ date: String(t.exit_date).slice(0, 10), kind: isNum(ret) && ret > 0 ? "exit_win" : "exit_loss",
      text: `Salida ${pct(ret)}`, detail: `Salida ${origen === "paper" ? "simulada" : "backtest"} a ${money(t.exit_price as number)} (${pct(ret)})` });
  };
  stock.trades.paper.filter((t) => t.status === "OPEN" || t.status === "CLOSED").forEach((t: Trade) => operacion(t, "paper"));
  stock.trades.backtest.forEach((t) => operacion(t as never, "backtest"));
  (stock.signals ?? []).forEach((s) => marcas.push({ date: s.date, kind: s.type.startsWith("golden") ? "golden" : "death",
    text: "", detail: s.type.startsWith("golden") ? "MA(7) cruza sobre MA(25)" : "MA(7) cruza bajo MA(25)" }));
  return marcas;
}

export function StockDetail() {
  const { symbol = "" } = useParams();
  const { data, error, loading } = useJson<StockJson>(`stocks/${encodeURIComponent(symbol.toUpperCase())}.json`);
  const marcas = useMemo(() => (data ? buildMarkers(data) : []), [data]);
  if (loading) return <Loading what={symbol} />;
  if (error || !data) return <ErrorBox message={error ? `No hay datos de ${symbol}` : "Sin datos"} />;
  const s = data.scores;
  const p = data.price;
  const ind = data.latest_indicators ?? {};
  const historial = data.score_history.map((h) => ({ date: h.date, final: h.final_score, estadistico: h.statistical_score }));
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <div className="hero-symbol">{data.symbol} <span className="muted" style={{ fontSize: "1.1rem", fontWeight: 400 }}>{data.name}</span></div>
          <div className="secondary" style={{ fontSize: "0.9rem" }}>{[data.sector, data.industry, data.indices?.replace(";", " · ")].filter(Boolean).join(" · ")}</div>
          <Freshness generated={data.generated_at} asOf={data.data_as_of} />
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="price-big num">{money(p?.last)}</div>
          <div className={signClass(p?.change_1d)}>{pct(p?.change_1d, 2)} en la sesión del {date(p?.date)}</div>
          <div style={{ marginTop: 4 }}>
            {s?.is_opportunity && <span className="chip">oportunidad</span>} {s?.signal_label && <span className="chip">{s.signal_label}</span>}
          </div>
        </div>
      </div>

      <div className="stats">
        <Stat label="Variación 1 mes" value={pct(p?.change_1m)} valueClass={signClass(p?.change_1m)} />
        <Stat label="Variación 3 meses" value={pct(p?.change_3m)} valueClass={signClass(p?.change_3m)} />
        <Stat label="Variación 1 año" value={pct(p?.change_1y)} valueClass={signClass(p?.change_1y)} />
        <Stat label="Rango 52 semanas" value={<span style={{ fontSize: "1rem" }}>{money(p?.low52)} – {money(p?.high52)}</span>} sub="ajustado" />
        <Stat label="RSI(14)" value={num(ind.rsi14, 1)} />
        <Stat label="MACD (hist.)" value={num(ind.macd_hist, 3)} />
      </div>

      <Card title="Gráfico" extra="MA(7) · MA(25) · MA(99) · volumen · RSI · MACD">
        {data.series ? <PriceChart stock={data} markers={marcas} /> : <div className="empty">No hay serie de precios en esta publicación.</div>}
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead><tr><th>Media</th><th className="right">Valor</th><th className="right">Precio frente a la media</th><th className="right">Pendiente</th></tr></thead>
            <tbody>
              {(["7", "25", "99"] as const).map((n) => (
                <tr key={n}>
                  <td>MA({n})</td>
                  <td className="right num">{money(ind[`ma${n}`])}</td>
                  <td className={`right num ${signClass(ind[`dist_ma${n}`])}`}>{pct(ind[`dist_ma${n}`])}</td>
                  <td className={`right num ${signClass(ind[`slope_ma${n}`])}`}>{pct(ind[`slope_ma${n}`], 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-2">
        <Card title="Scores" extra={s?.rank ? `Puesto ${s.rank} del ranking` : undefined}>
          {s ? (
            <div className="kv">
              <dt>Score final</dt><dd><Meter value={s.final_score} label="Final" /></dd>
              <dt>Estadístico</dt><dd><Meter value={s.statistical_score} label="Estadístico" /></dd>
              <dt>Noticias</dt><dd><Meter value={s.news_score} label="Noticias" /></dd>
              <dt>Fundamental</dt><dd><Meter value={s.fundamental_score} label="Fundamental" /></dd>
              <dt>Red neuronal</dt><dd><Meter value={s.neural_score} label="Red neuronal" /></dd>
              <dt>Riesgo</dt><dd><Meter value={s.risk_score} label="Riesgo (alto = menos riesgo)" /> <span className="muted">{s.risk_level}</span></dd>
            </div>
          ) : <div className="empty">Esta acción no se puntuó en la última ejecución (liquidez o calidad de datos insuficiente).</div>}
          <p className="muted" style={{ fontSize: "0.8rem", marginTop: 10 }}>Métricas internas de 0 a 100: no son probabilidades ni recomendaciones.</p>
        </Card>
        <Card title="Cómo llegó el sistema a este resultado">
          {data.breakdown ? (
            <>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Componente</th><th className="right">Valor</th><th className="right">Peso usado</th><th className="right">Aporte</th></tr></thead>
                  <tbody>
                    {Object.entries(data.breakdown.weights_used).map(([k, w]) => (
                      <tr key={k}>
                        <td>{COMPONENT_LABELS[k] ?? k}</td>
                        <td className="right num">{num(s?.[`${k}_score`] as number, 1)}</td>
                        <td className="right num">{pct(w, 0, false)}</td>
                        <td className="right num">{num(data.breakdown!.contributions[k], 1)}</td>
                      </tr>
                    ))}
                    <tr><td><strong>Score final</strong></td><td /><td className="right num">100 %</td><td className="right num"><strong>{num(data.breakdown.final_score, 1)}</strong></td></tr>
                  </tbody>
                </table>
              </div>
              {data.breakdown.missing.length > 0 && (
                <p className="muted" style={{ fontSize: "0.82rem" }}>
                  Sin dato: {data.breakdown.missing.map((m) => COMPONENT_LABELS[m] ?? m).join(", ")}. Su peso configurado se repartió entre los demás.
                </p>
              )}
            </>
          ) : <p className="muted">El desglose completo solo existe para las candidatas analizadas a fondo ese día.</p>}
          {data.statistical_breakdown && (
            <>
              <h3 style={{ marginTop: 12 }}>Dentro del score estadístico</h3>
              <div className="bars">
                {data.statistical_breakdown.map((c) => (
                  <div className="bar-row" key={c.component}>
                    <span>{COMPONENT_LABELS[c.component] ?? c.component} <span className="muted">({Math.round(c.weight * 100)} %)</span></span>
                    <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.max(0, Math.min(100, c.score ?? 0))}%` }} /></span>
                    <span className="num">{num(c.score, 0)}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </Card>
      </div>

      <div className="grid grid-2">
        <Card title="Razones (datos objetivos)">
          <ExplanationPanel exp={data.explanation} />
        </Card>
        <Card title="Predicción de la red neuronal" extra={data.neural.model_version ? `Modelo ${data.neural.model_version}` : undefined}>
          {data.neural.available ? (
            <>
              <div className="stats">
                <Stat label="Prob. de oportunidad" value={pct(data.neural.opportunity_probability, 1, false)}
                      sub={`tasa base ${pct(data.neural.base_rate, 1, false)} · umbral ${pct(data.neural.threshold, 1, false)}`} />
                <Stat label="Confianza" value={pct(data.neural.confidence, 0, false)} sub="estabilidad con MC dropout" />
                <Stat label="Retorno esperado" value={pct(data.neural.expected_return)} sub="de la operación simulada" />
                <Stat label="Drawdown esperado" value={pct(data.neural.expected_drawdown)} />
              </div>
              <div style={{ marginTop: 12 }}><ModelFactors exp={data.neural.explanation} /></div>
            </>
          ) : <div className="empty">{data.neural.reason}</div>}
        </Card>
      </div>

      <div className="grid grid-2">
        <Card title="Noticias" extra={data.news ? `${data.news.count} analizadas · ${data.news.label}` : undefined}>
          {data.news && <p className="secondary" style={{ fontSize: "0.9rem" }}>{data.news.explanation}</p>}
          <NewsList items={data.news?.items?.length ? data.news.items : data.news_archive}
                    empty="Sin noticias recientes (solo se consultan para las candidatas, posiciones y seguimiento)." />
        </Card>
        <Card title="Análisis fundamental">
          {data.fundamentals && Object.keys(data.fundamentals.values).length ? (
            <>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Métrica</th><th className="right">Valor</th><th className="right">Puntuación</th></tr></thead>
                  <tbody>
                    {Object.entries(data.fundamentals.values).map(([k, v]) => (
                      <tr key={k}>
                        <td>{data.fundamental_labels[k] ?? k}</td>
                        <td className="right num">{typeof v === "number" ? (["revenue_growth", "earnings_growth", "profit_margin", "roe", "roic"].includes(k) ? pct(v) : ["market_cap", "free_cash_flow"].includes(k) ? `${num(v / 1e9, 1)} mil M$` : num(v, 2)) : String(v)}</td>
                        <td className="right num">{num(data.fundamentals!.metric_scores[k], 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="muted" style={{ fontSize: "0.8rem" }}>
                Score fundamental: {num(data.fundamentals.fundamental_score, 1)} · no disponibles en el proveedor: {data.fundamentals.unavailable.map((u) => data.fundamental_labels[u] ?? u).join(", ") || "ninguna"}. {data.fundamentals.note}
              </p>
            </>
          ) : <div className="empty">Sin datos fundamentales para esta acción en la última ejecución.</div>}
        </Card>
      </div>

      <Card title="Evolución del score" extra="Snapshots diarios">
        <SeriesChart data={historial} xKey="date" xFormat={date} height={220}
                     series={[{ key: "final", label: "Score final", slot: 0 }, { key: "estadistico", label: "Estadístico", slot: 1 }]} />
      </Card>

      <Card title="Operaciones simuladas (paper trading)">
        <TradeTable trades={data.trades.paper} showSymbol={false} empty="El sistema no ha operado esta acción." />
      </Card>
      <Card title="Backtest de esta acción" extra={data.backtest_stats?.period ? `${date(data.backtest_stats.period.start)} → ${date(data.backtest_stats.period.end)}` : undefined}>
        {data.backtest_stats ? (
          <>
            <div className="stats">
              <Stat label="Operaciones" value={data.backtest_stats.trades} />
              <Stat label="Win rate" value={pct(data.backtest_stats.win_rate, 0, false)} />
              <Stat label="Retorno medio" value={pct(data.backtest_stats.avg_return, 2)} valueClass={signClass(data.backtest_stats.avg_return)} />
            </div>
            <p className="muted" style={{ fontSize: "0.8rem", marginTop: 8 }}>
              Operaciones de la estrategia combinada en la simulación walk-forward (últimas 60 marcadas en el gráfico como «bt»). <Link to="/backtesting">Ver el backtest completo</Link>.
            </p>
          </>
        ) : <div className="empty">El backtest no operó esta acción o aún no se ha ejecutado.</div>}
      </Card>
    </div>
  );
}
