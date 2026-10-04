import { useMemo, useState } from "react";
import { useJson } from "../api";
import { NewsList } from "../components/Panels";
import { Card, ErrorBox, Freshness, Loading, Stat } from "../components/ui";
import type { NewsJson } from "../types";

export function News() {
  const { data, error, loading } = useJson<NewsJson>("news.json");
  const [texto, setTexto] = useState("");
  const [sentimiento, setSentimiento] = useState("");
  const [fuente, setFuente] = useState("");
  const items = useMemo(() => (data?.items ?? []).filter((n) => {
    if (texto && !`${n.symbol ?? ""} ${n.title}`.toLowerCase().includes(texto.toLowerCase())) return false;
    if (fuente && n.source !== fuente) return false;
    const s = n.sentiment ?? 0;
    if (sentimiento === "pos" && s < 0.15) return false;
    if (sentimiento === "neg" && s > -0.15) return false;
    if (sentimiento === "neu" && (s >= 0.15 || s <= -0.15)) return false;
    return true;
  }), [data, texto, sentimiento, fuente]);
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  const fuentes = [...new Set(data.items.map((n) => n.source))].sort();
  const positivas = data.items.filter((n) => (n.sentiment ?? 0) >= 0.15).length;
  const negativas = data.items.filter((n) => (n.sentiment ?? 0) <= -0.15).length;
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Noticias</h1>
          <Freshness generated={data.generated_at} asOf={data.data_as_of} />
        </div>
      </div>
      <p className="secondary">
        Titulares recientes de las acciones candidatas y en seguimiento ({data.providers.join(", ") || "sin fuentes"}). El sentimiento
        es un análisis léxico automático del titular: no es una valoración de la empresa y puede equivocarse con ironía o contexto.
      </p>
      <div className="stats">
        <Stat label="Noticias" value={data.items.length} />
        <Stat label="Positivas" value={positivas} valueClass="pos" />
        <Stat label="Negativas" value={negativas} valueClass="neg" />
        <Stat label="Neutras" value={data.items.length - positivas - negativas} />
      </div>
      <Card>
        <div className="filters">
          <input type="search" placeholder="Acción o palabra" value={texto} onChange={(e) => setTexto(e.target.value)} aria-label="Buscar" />
          <select value={sentimiento} onChange={(e) => setSentimiento(e.target.value)} aria-label="Sentimiento">
            <option value="">Cualquier sentimiento</option><option value="pos">Positivo</option><option value="neu">Neutral</option><option value="neg">Negativo</option>
          </select>
          <select value={fuente} onChange={(e) => setFuente(e.target.value)} aria-label="Fuente">
            <option value="">Todas las fuentes</option>
            {fuentes.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <span className="muted" style={{ fontSize: "0.85rem" }}>{items.length} titulares</span>
        </div>
        <NewsList items={items.slice(0, 200)} showSymbol empty="Sin noticias que cumplan los filtros." />
      </Card>
    </div>
  );
}
