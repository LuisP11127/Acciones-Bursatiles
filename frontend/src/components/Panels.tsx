import { Link } from "react-router-dom";
import { date, isNum, money, num, pct, signClass, STRATEGY_LABELS } from "../format";
import type { Explanation, Metrics, ModelExplanation, NewsItem, PaperMetrics, Trade, YearRow } from "../types";
import { DataTable, type Column } from "./DataTable";
import { SentimentChip, TradeStatusChip } from "./ui";

export function ExplanationPanel({ exp, compact = false }: { exp: Explanation | null | undefined; compact?: boolean }) {
  if (!exp) return <div className="empty">Sin explicación disponible.</div>;
  return (
    <div className="stack">
      <div>
        <h3>Por qué destaca (datos objetivos)</h3>
        {exp.reasons.length ? (
          <ul className="reasons">
            {exp.reasons.slice(0, compact ? 6 : undefined).map((r) => (
              <li key={r.code}>
                <div>✓ {r.text}</div>
                <div className="why">Por qué importa: {r.why_it_matters}</div>
              </li>
            ))}
          </ul>
        ) : <p className="muted">Sin señales positivas destacadas.</p>}
      </div>
      {exp.risks.length > 0 && (
        <div>
          <h3>Riesgos y señales en contra</h3>
          <ul className="reasons">
            {exp.risks.slice(0, compact ? 4 : undefined).map((r) => (
              <li key={r.code}>
                <div>⚠ {r.text}</div>
                <div className="why">{r.why_it_matters}</div>
              </li>
            ))}
          </ul>
        </div>
      )}
      {exp.news_explanation && !compact && (
        <div>
          <h3>Noticias</h3>
          <p className="secondary">{exp.news_explanation}</p>
        </div>
      )}
      {exp.fundamental_facts && exp.fundamental_facts.length > 0 && !compact && (
        <div>
          <h3>Datos fundamentales (proveedor)</h3>
          <ul>{exp.fundamental_facts.map((f) => <li key={f}>{f}</li>)}</ul>
        </div>
      )}
      <p className="muted" style={{ fontSize: "0.8rem" }}>{exp.disclaimers.facts}</p>
    </div>
  );
}

export function ModelFactors({ exp }: { exp: ModelExplanation | null | undefined }) {
  if (!exp?.factors?.length) return null;
  const max = Math.max(...exp.factors.map((f) => Math.abs(f.effect)), 1e-6);
  return (
    <div>
      <h3>Explicación del modelo</h3>
      <p className="muted" style={{ fontSize: "0.82rem" }}>{exp.disclaimer} Método: {exp.method}.</p>
      <div className="bars">
        {exp.factors.map((f) => (
          <div className="bar-row" key={f.feature} title={`Valor actual: ${num(f.value, 4)}`}>
            <span>{f.label}</span>
            <span className="bar-track"><span className={`bar-fill ${f.effect < 0 ? "neg-fill" : ""}`} style={{ width: `${(Math.abs(f.effect) / max) * 100}%` }} /></span>
            <span className={`num ${signClass(f.effect)}`}>{f.effect > 0 ? "+" : ""}{(f.effect * 100).toFixed(1)} pp</span>
          </div>
        ))}
      </div>
      <p className="muted" style={{ fontSize: "0.78rem", marginTop: 6 }}>
        pp = cambio en puntos porcentuales de la probabilidad al neutralizar cada variable (azul: la subió; naranja: la bajó).
      </p>
    </div>
  );
}

export function NewsList({ items, showSymbol = false, empty = "Sin noticias." }: { items: NewsItem[]; showSymbol?: boolean; empty?: string }) {
  if (!items.length) return <div className="empty">{empty}</div>;
  return (
    <div>
      {items.map((n, i) => (
        <div className="news-item" key={`${n.url}-${i}`}>
          <div>{n.url ? <a href={n.url} target="_blank" rel="noopener noreferrer">{n.title}</a> : n.title}</div>
          <div className="news-meta">
            {showSymbol && n.symbol && <Link to={`/stock/${n.symbol}`}><strong>{n.symbol}</strong></Link>}
            <span>{n.source}</span>
            <span>{date(n.published_at)}</span>
            <SentimentChip value={n.sentiment} />
            {(n.themes ?? []).slice(0, 2).map((t) => <span className="chip" key={t}>{t}</span>)}
          </div>
        </div>
      ))}
    </div>
  );
}

export function TradeTable({ trades, showSymbol = true, limit, empty }: { trades: Trade[]; showSymbol?: boolean; limit?: number; empty?: string }) {
  const columns: Column<Trade>[] = [
    ...(showSymbol ? [{ key: "symbol", header: "Acción", render: (t: Trade) => <Link to={`/stock/${t.symbol}`} onClick={(e) => e.stopPropagation()}><strong>{t.symbol}</strong></Link>, sortValue: (t: Trade) => t.symbol }] : []),
    { key: "status", header: "Estado", render: (t) => <TradeStatusChip status={t.status} />, sortValue: (t) => t.status },
    { key: "signal", header: "Señal", render: (t) => date(t.signal_date), sortValue: (t) => t.signal_date },
    { key: "entry", header: "Entrada", render: (t) => t.entry_date ? <>{date(t.entry_date)}<br /><span className="num">{money(t.entry_price)}</span></> : "—", sortValue: (t) => t.entry_date },
    { key: "exit", header: "Salida", render: (t) => t.exit_date ? <>{date(t.exit_date)}<br /><span className="num">{money(t.exit_price)}</span></> : "—", sortValue: (t) => t.exit_date },
    { key: "ret", header: "Resultado", align: "right", render: (t) => <span className={signClass(t.actual_return)}>{pct(t.actual_return)}</span>, sortValue: (t) => t.actual_return },
    { key: "dd", header: "Drawdown máx.", align: "right", render: (t) => pct(t.max_drawdown), sortValue: (t) => t.max_drawdown },
    { key: "days", header: "Días", align: "right", render: (t) => t.holding_days ?? "—", sortValue: (t) => t.holding_days },
    { key: "score", header: "Score final", align: "right", render: (t) => isNum(t.final_score) ? t.final_score.toFixed(0) : "—", sortValue: (t) => t.final_score },
    { key: "reason", header: "Motivo / salida", render: (t) => <span className="secondary" style={{ fontSize: "0.82rem" }}>{t.exit_reason ?? t.invalidation_reason ?? (t.reason ? t.reason.replace("Seleccionada porque: ", "") .slice(0, 90) + (t.reason.length > 110 ? "…" : "") : "—")}</span> },
  ];
  return <DataTable rows={trades} columns={columns} rowKey={(t) => t.trade_id} initialSort={{ key: "signal", dir: "desc" }} limit={limit} empty={empty} />;
}

const METRIC_ROWS: [string, string, (v: number) => string][] = [
  ["total_return", "Retorno total", (v) => pct(v)],
  ["cagr", "CAGR", (v) => pct(v)],
  ["sharpe", "Sharpe", (v) => num(v, 2)],
  ["sortino", "Sortino", (v) => num(v, 2)],
  ["max_drawdown", "Drawdown máximo", (v) => pct(v)],
  ["volatility", "Volatilidad anual", (v) => pct(v, 1, false)],
  ["win_rate", "Win rate", (v) => pct(v, 1, false)],
  ["avg_return", "Retorno medio", (v) => pct(v, 2)],
  ["median_return", "Retorno mediano", (v) => pct(v, 2)],
  ["avg_gain", "Ganancia media", (v) => pct(v, 2)],
  ["avg_loss", "Pérdida media", (v) => pct(v, 2)],
  ["profit_factor", "Profit factor", (v) => num(v, 2)],
  ["trades", "Operaciones", (v) => String(v)],
  ["avg_holding_days", "Días medios en cartera", (v) => num(v, 1)],
  ["exposure", "Exposición media", (v) => pct(v, 0, false)],
];

export function MetricsGrid({ metrics, keys }: { metrics: Metrics | PaperMetrics | undefined; keys?: string[] }) {
  if (!metrics) return null;
  const filas = METRIC_ROWS.filter(([k]) => !keys || keys.includes(k));
  return (
    <div className="stats">
      {filas.map(([k, label, fmt]) => {
        const v = metrics[k] as number | null | undefined;
        return (
          <div className="stat" key={k}>
            <div className="stat-label">{label}</div>
            <div className={`stat-value ${["total_return", "cagr", "avg_return", "median_return"].includes(k) ? signClass(v) : ""}`}>
              {isNum(v) ? fmt(v) : <span className="muted" title="Sin historial suficiente para calcularla">n/d</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function YearlyTable({ rows }: { rows: { label: string; years: YearRow[] }[] }) {
  const anios = [...new Set(rows.flatMap((r) => r.years.map((y) => y.year)))].sort();
  if (!anios.length) return <div className="empty">Sin resultados anuales todavía.</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>Serie</th>{anios.map((a) => <th key={a} className="right">{a}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td>{STRATEGY_LABELS[r.label] ?? r.label}</td>
              {anios.map((a) => {
                const y = r.years.find((x) => x.year === a);
                return <td key={a} className={`right num ${signClass(y?.return)}`}>{y ? pct(y.return) : "—"}{y?.partial ? "*" : ""}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: "0.78rem" }}>* año incompleto.</p>
    </div>
  );
}
