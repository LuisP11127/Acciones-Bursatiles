import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useJson } from "../api";
import { DataTable, type Column } from "../components/DataTable";
import { Card, ErrorBox, Freshness, Loading, Meter } from "../components/ui";
import { COMPONENT_LABELS, date, money, num, pct, signClass } from "../format";
import type { OpportunitiesJson, OpportunityRow } from "../types";

export function Opportunities() {
  const { data, error, loading } = useJson<OpportunitiesJson>("opportunities.json");
  const navegar = useNavigate();
  const [texto, setTexto] = useState("");
  const [sector, setSector] = useState("");
  const [soloOportunidades, setSoloOportunidades] = useState(true);
  const [minimo, setMinimo] = useState(0);
  const filas = useMemo(() => (data?.rows ?? []).filter((r) =>
    (!soloOportunidades || r.is_opportunity) &&
    (!sector || r.sector === sector) &&
    (r.final_score ?? 0) >= minimo &&
    (!texto || `${r.symbol} ${r.name ?? ""}`.toLowerCase().includes(texto.toLowerCase()))), [data, texto, sector, soloOportunidades, minimo]);
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  const sectores = [...new Set(data.rows.map((r) => r.sector).filter(Boolean))].sort() as string[];
  const columnas: Column<OpportunityRow>[] = [
    { key: "rank", header: "#", align: "right", render: (r) => r.rank ?? "—", sortValue: (r) => (r.rank == null ? null : -r.rank) },
    { key: "symbol", header: "Acción", render: (r) => <><strong>{r.symbol}</strong><br /><span className="muted" style={{ fontSize: "0.8rem" }}>{r.name}</span></>, sortValue: (r) => r.symbol },
    { key: "price", header: "Precio", align: "right", render: (r) => <>{money(r.price)}<br /><span className={signClass(r.change_1d)} style={{ fontSize: "0.8rem" }}>{pct(r.change_1d, 2)}</span></>, sortValue: (r) => r.price },
    { key: "stat", header: "Estadístico", render: (r) => <Meter value={r.statistical_score} label="Statistical score" />, sortValue: (r) => r.statistical_score },
    { key: "news", header: "Noticias", align: "right", render: (r) => num(r.news_score, 0), sortValue: (r) => r.news_score },
    { key: "fund", header: "Fundamental", align: "right", render: (r) => num(r.fundamental_score, 0), sortValue: (r) => r.fundamental_score },
    { key: "nn", header: "Red neuronal", align: "right", render: (r) => num(r.neural_score, 0), sortValue: (r) => r.neural_score },
    { key: "final", header: "Score final", render: (r) => <Meter value={r.final_score} label="Final score" />, sortValue: (r) => r.final_score },
    { key: "prob", header: "Prob. oportunidad", align: "right", title: "Probabilidad estimada por la red neuronal", render: (r) => pct(r.opportunity_probability, 1, false), sortValue: (r) => r.opportunity_probability },
    { key: "risk", header: "Riesgo", render: (r) => r.risk_level ? <span title={`risk_score ${num(r.risk_score, 0)} (más alto = menos riesgo)`}>{r.risk_level}</span> : "—", sortValue: (r) => (r.risk_score == null ? null : -r.risk_score) },
    { key: "upd", header: "Actualización", render: (r) => date(r.last_update) },
  ];
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Oportunidades</h1>
          <Freshness generated={data.generated_at} asOf={data.data_as_of} />
        </div>
      </div>
      <div className="banner banner-info">
        El sistema ordena internamente por <strong>score final</strong> (0-100), que combina componentes con pesos de
        <code> config/model.yaml</code>: {Object.entries(data.weights).map(([k, v]) => `${COMPONENT_LABELS[k] ?? k} ${Math.round(v * 100)} %`).join(" · ")}.
        Es una métrica interna, no una certeza. {data.model_available ? "" : "La red neuronal aún no tiene un modelo en producción: su peso se reparte entre los demás componentes."}
      </div>
      <Card>
        <div className="filters">
          <input type="search" placeholder="Buscar símbolo o empresa" value={texto} onChange={(e) => setTexto(e.target.value)} aria-label="Buscar" />
          <select value={sector} onChange={(e) => setSector(e.target.value)} aria-label="Sector">
            <option value="">Todos los sectores</option>
            {sectores.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <label>Score mínimo <input type="number" min={0} max={100} value={minimo} onChange={(e) => setMinimo(Number(e.target.value) || 0)} style={{ width: 70 }} /></label>
          <label><input type="checkbox" checked={soloOportunidades} onChange={(e) => setSoloOportunidades(e.target.checked)} /> Solo oportunidades (score ≥ {data.opportunity_threshold})</label>
          <span className="muted" style={{ fontSize: "0.85rem" }}>{filas.length} acciones</span>
        </div>
        <DataTable rows={filas} columns={columnas} rowKey={(r) => r.symbol} onRowClick={(r) => navegar(`/stock/${r.symbol}`)}
                   initialSort={{ key: "final", dir: "desc" }} limit={60}
                   empty="Ninguna acción cumple los filtros. Desmarca «Solo oportunidades» para ver todo el universo analizado." />
      </Card>
    </div>
  );
}
