import { Link } from "react-router-dom";
import { useJson } from "../api";
import { ModelFactors, NewsList } from "../components/Panels";
import { Card, ErrorBox, Freshness, Loading, Stat, TradeStatusChip } from "../components/ui";
import { date, isNum, money, num, pct, signClass } from "../format";
import type { ManualPosition, Num, TrackedTrade, TrackingJson } from "../types";

const SCORE_KEYS: [string, string][] = [
  ["final_score", "Final"], ["statistical_score", "Estadístico"], ["news_score", "Noticias"],
  ["fundamental_score", "Fundamental"], ["neural_score", "Red neuronal"], ["risk_score", "Riesgo"],
];

function ScoreEvolution({ initial, current }: { initial: Record<string, Num | undefined>; current?: Record<string, unknown> | null }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Score</th><th className="right">Al seleccionarla</th><th className="right">Actual</th><th className="right">Cambio</th></tr></thead>
        <tbody>
          {SCORE_KEYS.map(([k, label]) => {
            const a = initial[k];
            const b = current?.[k] as Num | undefined;
            const delta = isNum(a) && isNum(b) ? b - a : null;
            return (
              <tr key={k}>
                <td>{label}</td>
                <td className="right num">{num(a, 0)}</td>
                <td className="right num">{num(b, 0)}</td>
                <td className={`right num ${signClass(delta)}`}>{isNum(delta) ? `${delta > 0 ? "+" : ""}${delta.toFixed(0)}` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradeCard({ t }: { t: TrackedTrade }) {
  const precioBase = t.entry_price ?? t.signal_price;
  const variacion = isNum(t.current_price) && isNum(precioBase) ? t.current_price / precioBase - 1 : null;
  return (
    <Card title={<h2><Link to={`/stock/${t.symbol}`}>{t.symbol}</Link> <span className="muted" style={{ fontWeight: 400 }}>{t.name}</span></h2>}
          extra={<><TradeStatusChip status={t.status} /> · {t.trade_id} · {t.strategy}</>}>
      <div className="stats">
        <Stat label="Fecha de entrada" value={t.entry_date ? date(t.entry_date) : "Pendiente"} sub={`Señal del ${date(t.signal_date)}`} />
        <Stat label="Precio de entrada" value={money(t.entry_price ?? null)} sub={t.entry_price ? "apertura tras la señal" : `cierre de la señal ${money(t.signal_price)}`} />
        <Stat label="Precio actual" value={money(t.current_price ?? null)} />
        <Stat label="Variación" value={pct(t.status === "OPEN" ? t.actual_return : variacion)} valueClass={signClass(t.status === "OPEN" ? t.actual_return : variacion)}
              sub={t.status === "OPEN" ? "neta de costes simulados" : "desde el precio de la señal"} />
        <Stat label="Días en seguimiento" value={t.sessions_tracked ?? "—"} sub="sesiones de mercado" />
        <Stat label="Drawdown máximo" value={pct(t.max_drawdown)} />
      </div>
      <div className="grid grid-2" style={{ marginTop: 14 }}>
        <div>
          <h3>Motivo original de la selección</h3>
          <ul className="reasons">
            {t.reasons.slice(0, 5).map((r, i) => (
              <li key={r}>
                <div>✓ {r}</div>
                {t.why_it_matters[i] && <div className="why">Por qué importa: {t.why_it_matters[i]}</div>}
              </li>
            ))}
          </ul>
          {t.reasons.length > 5 && (
            <details>
              <summary>Ver las {t.reasons.length - 5} razones restantes</summary>
              <ul className="reasons">
                {t.reasons.slice(5).map((r, i) => (
                  <li key={r}>
                    <div>✓ {r}</div>
                    {t.why_it_matters[i + 5] && <div className="why">Por qué importa: {t.why_it_matters[i + 5]}</div>}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {t.risks_at_entry.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary>Riesgos identificados al entrar ({t.risks_at_entry.length})</summary>
              <ul>{t.risks_at_entry.map((r) => <li key={r}>{r}</li>)}</ul>
            </details>
          )}
          {t.news_explanation && <p className="secondary" style={{ marginTop: 10, fontSize: "0.88rem" }}>📰 {t.news_explanation}</p>}
          <ModelFactors exp={t.model_explanation} />
        </div>
        <div className="stack">
          <div>
            <h3>Scores iniciales vs actuales</h3>
            <ScoreEvolution initial={t as unknown as Record<string, Num>} current={t.current_scores} />
            <p className="muted" style={{ fontSize: "0.8rem" }}>
              Probabilidad de la red al seleccionarla: {pct(t.opportunity_probability, 1, false)} · confianza {pct(t.confidence, 0, false)} ·
              retorno esperado {pct(t.expected_return)} · modelo {t.model_version ?? "sin modelo"}.
            </p>
          </div>
          <div>
            <h3>Noticias nuevas desde la entrada</h3>
            <NewsList items={t.news_since_entry} empty="Sin noticias nuevas en el archivo." />
          </div>
        </div>
      </div>
    </Card>
  );
}

function ManualCard({ m }: { m: ManualPosition }) {
  return (
    <Card title={<h2><Link to={`/stock/${m.symbol}`}>{m.symbol}</Link> <span className="muted" style={{ fontWeight: 400 }}>{m.name}</span></h2>}
          extra={<TradeStatusChip status="MANUAL" />}>
      <div className="stats">
        <Stat label="Fecha de compra" value={date(m.entry_date)} />
        <Stat label="Precio de compra" value={money(m.entry_price)} sub={m.shares ? `${m.shares} acciones` : undefined} />
        <Stat label="Precio actual" value={money(m.current_price)} />
        <Stat label="Variación" value={pct(m.return)} valueClass={signClass(m.return)} />
        <Stat label="Días" value={m.days ?? "—"} />
        <Stat label="Score final actual" value={num(m.current_scores?.final_score, 0)} />
      </div>
      <div className="grid grid-2" style={{ marginTop: 14 }}>
        <div>
          <h3>Razones de la compra {m.reasons_auto && <span className="chip">análisis automático</span>}</h3>
          <ul>{m.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          <h3>Por qué es importante {m.why_it_matters_auto && <span className="chip">análisis automático</span>}</h3>
          <ul>{m.why_it_matters.map((r) => <li key={r}>{r}</li>)}</ul>
          {m.risks.length > 0 && <><h3>Riesgos actuales</h3><ul>{m.risks.map((r) => <li key={r}>{r}</li>)}</ul></>}
        </div>
        <div>
          <h3>Noticias desde la compra</h3>
          <NewsList items={m.news_since_entry} empty="Sin noticias en el archivo." />
        </div>
      </div>
    </Card>
  );
}

export function Tracking() {
  const { data, error, loading } = useJson<TrackingJson>("tracking.json");
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Seguimiento</h1>
          <Freshness generated={data.generated_at} asOf={data.data_as_of} />
        </div>
        <span className="chip">Modo {data.mode}</span>
      </div>
      <p className="secondary">
        Operaciones simuladas abiertas o pendientes. Cada una conserva el motivo original de su selección y se compara a diario con
        su situación actual para ver cómo evoluciona la predicción. Nada de esto es una orden real.
      </p>
      {data.trades.length ? data.trades.map((t) => <TradeCard key={t.trade_id} t={t} />) : (
        <Card><div className="empty">No hay operaciones simuladas abiertas ni pendientes{data.mode === "RESEARCH" ? " (modo RESEARCH: no se abren operaciones)" : ""}.</div></Card>
      )}
      <h2 style={{ marginTop: 24 }}>Mis posiciones (config/mis_acciones.yaml)</h2>
      {data.manual_positions.length ? data.manual_positions.map((m) => <ManualCard key={m.symbol + m.entry_date} m={m} />) : (
        <Card><div className="empty">No hay posiciones manuales. Añádelas en <code>config/mis_acciones.yaml</code> para seguirlas aquí.</div></Card>
      )}
    </div>
  );
}
