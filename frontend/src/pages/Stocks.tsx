import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useJson } from "../api";
import { DataTable, type Column } from "../components/DataTable";
import { Card, ErrorBox, Freshness, Loading, Meter } from "../components/ui";
import { money, pct, signClass } from "../format";
import type { StockIndexRow, StocksIndexJson } from "../types";

export function Stocks() {
  const { data, error, loading } = useJson<StocksIndexJson>("stocks/index.json");
  const navegar = useNavigate();
  const [texto, setTexto] = useState("");
  const filas = useMemo(() => (data?.stocks ?? []).filter((s) =>
    !texto || `${s.symbol} ${s.name} ${s.sector ?? ""}`.toLowerCase().includes(texto.toLowerCase())), [data, texto]);
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  const columnas: Column<StockIndexRow>[] = [
    { key: "symbol", header: "Acción", render: (s) => <strong>{s.symbol}</strong>, sortValue: (s) => s.symbol },
    { key: "name", header: "Empresa", render: (s) => s.name, sortValue: (s) => s.name },
    { key: "sector", header: "Sector", render: (s) => s.sector ?? "—", sortValue: (s) => s.sector },
    { key: "price", header: "Precio", align: "right", render: (s) => money(s.price), sortValue: (s) => s.price },
    { key: "chg", header: "Día", align: "right", render: (s) => <span className={signClass(s.change_1d)}>{pct(s.change_1d, 2)}</span>, sortValue: (s) => s.change_1d },
    { key: "final", header: "Score final", render: (s) => <Meter value={s.final_score} />, sortValue: (s) => s.final_score },
    { key: "flags", header: "", render: (s) => <>{s.is_opportunity && <span className="chip">oportunidad</span>} {s.in_tracking && <span className="chip">en seguimiento</span>}</> },
  ];
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Acciones</h1>
          <Freshness generated={data.generated_at} asOf={data.data_as_of} />
        </div>
      </div>
      <Card>
        <div className="filters">
          <input type="search" placeholder="Buscar símbolo, empresa o sector" value={texto} onChange={(e) => setTexto(e.target.value)} aria-label="Buscar" style={{ minWidth: 260 }} />
          <span className="muted" style={{ fontSize: "0.85rem" }}>{filas.length} de {data.stocks.length}</span>
        </div>
        <DataTable rows={filas} columns={columnas} rowKey={(s) => s.symbol} onRowClick={(s) => navegar(`/stock/${s.symbol}`)}
                   initialSort={{ key: "final", dir: "desc" }} limit={100} />
      </Card>
    </div>
  );
}
