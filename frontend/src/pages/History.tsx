import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useJson } from "../api";
import { MetricsGrid, TradeTable, YearlyTable } from "../components/Panels";
import { SeriesChart } from "../components/SeriesChart";
import { Card, ErrorBox, Freshness, Loading, Stat } from "../components/ui";
import { date, money, moneyAxis, num, pct, signClass } from "../format";
import type { HistoryJson } from "../types";

export function History() {
  const { data, error, loading } = useJson<HistoryJson>("history.json");
  const [simbolo, setSimbolo] = useState("");
  const [desde, setDesde] = useState("");
  const [hasta, setHasta] = useState("");
  const [estado, setEstado] = useState("");
  const [resultado, setResultado] = useState("");
  const [scoreMin, setScoreMin] = useState(0);
  const [estrategia, setEstrategia] = useState("");
  const filtradas = useMemo(() => (data?.trades ?? []).filter((t) => {
    const fecha = t.entry_date ?? t.signal_date;
    if (simbolo && !t.symbol.toLowerCase().includes(simbolo.toLowerCase())) return false;
    if (desde && fecha < desde) return false;
    if (hasta && fecha > hasta) return false;
    if (estado && t.status !== estado) return false;
    if (estrategia && t.strategy !== estrategia) return false;
    if ((t.final_score ?? 0) < scoreMin) return false;
    if (resultado === "win" && !(t.status === "CLOSED" && (t.actual_return ?? 0) > 0)) return false;
    if (resultado === "loss" && !(t.status === "CLOSED" && (t.actual_return ?? 0) <= 0)) return false;
    return true;
  }), [data, simbolo, desde, hasta, estado, resultado, scoreMin, estrategia]);
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  const cerradas = filtradas.filter((t) => t.status === "CLOSED");
  const ganadoras = cerradas.filter((t) => (t.actual_return ?? 0) > 0).length;
  const curva = data.equity.map((e) => ({ date: e.date, equity: e.equity, benchmark: e.benchmark }));
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Historial</h1>
          <Freshness generated={data.generated_at} asOf={data.data_as_of} />
        </div>
      </div>
      <div className="stats">
        <Stat label="Operaciones (filtro)" value={filtradas.length} />
        <Stat label="Ganadoras" value={ganadoras} valueClass="pos" />
        <Stat label="Perdedoras" value={cerradas.length - ganadoras} valueClass="neg" />
        <Stat label="Abiertas" value={filtradas.filter((t) => t.status === "OPEN").length} />
        <Stat label="Pendientes" value={filtradas.filter((t) => t.status === "PENDING").length} />
        <Stat label="Invalidadas" value={filtradas.filter((t) => t.status === "INVALIDATED").length} />
      </div>
      <Card title="Curva de equity simulada" extra={curva.length ? `${date(curva[0].date)} → ${date(curva[curva.length - 1].date)}` : undefined}>
        <SeriesChart data={curva} xKey="date" xFormat={date} yFormat={(v) => money(v, 0)} yTickFormat={moneyAxis}
                     series={[{ key: "equity", label: "Paper trading", slot: 0 }, { key: "benchmark", label: "SPY (mismo capital)", slot: 4 }]} />
        <MetricsGrid metrics={{ ...data.metrics, trades: data.metrics.closed_trades }}
                     keys={["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "avg_holding_days", "exposure"]} />
        {data.metrics.notes.map((n) => <p key={n} className="muted" style={{ fontSize: "0.82rem", marginTop: 8 }}>ℹ️ {n}</p>)}
        <h3 style={{ marginTop: 12 }}>Resultados por año</h3>
        <YearlyTable rows={[{ label: "Paper trading", years: data.metrics.by_year ?? [] }]} />
      </Card>
      <Card title="Todas las operaciones simuladas">
        <div className="filters">
          <input type="search" placeholder="Acción" value={simbolo} onChange={(e) => setSimbolo(e.target.value)} aria-label="Acción" style={{ width: 110 }} />
          <label>Desde <input type="date" value={desde} onChange={(e) => setDesde(e.target.value)} /></label>
          <label>Hasta <input type="date" value={hasta} onChange={(e) => setHasta(e.target.value)} /></label>
          <select value={estado} onChange={(e) => setEstado(e.target.value)} aria-label="Estado">
            <option value="">Todos los estados</option>
            <option value="OPEN">Abiertas</option><option value="PENDING">Pendientes</option>
            <option value="CLOSED">Cerradas</option><option value="INVALIDATED">Invalidadas</option>
          </select>
          <select value={resultado} onChange={(e) => setResultado(e.target.value)} aria-label="Resultado">
            <option value="">Cualquier resultado</option><option value="win">Ganadoras</option><option value="loss">Perdedoras</option>
          </select>
          <label>Score ≥ <input type="number" min={0} max={100} value={scoreMin} onChange={(e) => setScoreMin(Number(e.target.value) || 0)} style={{ width: 64 }} /></label>
          <select value={estrategia} onChange={(e) => setEstrategia(e.target.value)} aria-label="Estrategia">
            <option value="">Todas las estrategias</option>
            {data.strategies.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <TradeTable trades={filtradas} limit={50} empty="Sin operaciones que cumplan los filtros." />
      </Card>
      <Card title="¿Qué detectó el sistema cada día y qué pasó después?" extra="Snapshots diarios (data/history)">
        {data.snapshots.length ? data.snapshots.slice(0, 30).map((s) => (
          <details key={s.date} style={{ padding: "6px 0", borderBottom: "1px solid var(--grid)" }}>
            <summary>
              <strong>{date(s.date)}</strong> · {s.opportunities.length} oportunidades · modelo {s.model_version ?? "sin modelo"} ·
              SPY desde entonces <span className={signClass(s.benchmark_return_since)}>{pct(s.benchmark_return_since)}</span>
            </summary>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Acción</th><th className="right">Score</th><th className="right">Precio entonces</th><th className="right">Desde entonces</th><th className="right">A 20 sesiones</th></tr></thead>
                <tbody>
                  {s.opportunities.map((o) => (
                    <tr key={o.symbol}>
                      <td><Link to={`/stock/${o.symbol}`}>{o.symbol}</Link></td>
                      <td className="right num">{num(o.final_score, 0)}</td>
                      <td className="right num">{money(o.price)}</td>
                      <td className={`right num ${signClass(o.return_since)}`}>{pct(o.return_since)}</td>
                      <td className={`right num ${signClass(o.return_20d)}`}>{o.return_20d == null ? "aún no" : pct(o.return_20d)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )) : <div className="empty">Aún no hay snapshots diarios.</div>}
      </Card>
    </div>
  );
}
