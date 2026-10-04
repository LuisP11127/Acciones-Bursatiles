import { useMemo, useState, type ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: ReactNode;
  render: (row: T) => ReactNode;
  sortValue?: (row: T) => number | string | null | undefined;
  align?: "left" | "right";
  title?: string;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  initialSort?: { key: string; dir: "asc" | "desc" };
  empty?: ReactNode;
  limit?: number;
}

export function DataTable<T>({ rows, columns, rowKey, onRowClick, initialSort, empty, limit }: Props<T>) {
  const [sort, setSort] = useState(initialSort);
  const [mostrar, setMostrar] = useState(limit ?? Infinity);
  const ordenadas = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    const valor = col.sortValue;
    return [...rows].sort((a, b) => {
      const va = valor(a);
      const vb = valor(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1; // los vacíos siempre al final
      if (vb == null) return -1;
      const cmp = va < vb ? -1 : va > vb ? 1 : 0;
      return sort.dir === "asc" ? cmp : -cmp;
    });
  }, [rows, columns, sort]);

  if (!rows.length) return <div className="empty">{empty ?? "Sin datos."}</div>;

  const pulsar = (c: Column<T>) => {
    if (!c.sortValue) return;
    setSort((s) => (s?.key === c.key ? { key: c.key, dir: s.dir === "desc" ? "asc" : "desc" } : { key: c.key, dir: "desc" }));
  };
  const visibles = ordenadas.slice(0, mostrar);
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.key} className={`${c.sortValue ? "sortable" : ""} ${c.align === "right" ? "right" : ""}`}
                    onClick={() => pulsar(c)} title={c.title}
                    aria-sort={sort?.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : undefined}>
                  {c.header}
                  {sort?.key === c.key ? (sort.dir === "desc" ? " ↓" : " ↑") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibles.map((r) => (
              <tr key={rowKey(r)} className={onRowClick ? "clickable" : ""} onClick={onRowClick ? () => onRowClick(r) : undefined}
                  tabIndex={onRowClick ? 0 : undefined}
                  onKeyDown={onRowClick ? (e) => e.key === "Enter" && onRowClick(r) : undefined}>
                {columns.map((c) => (
                  <td key={c.key} className={c.align === "right" ? "right num" : ""}>{c.render(r)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {ordenadas.length > visibles.length && (
        <div style={{ textAlign: "center", marginTop: 10 }}>
          <button className="theme-toggle" onClick={() => setMostrar((m) => m + (limit ?? 50))}>
            Mostrar más ({ordenadas.length - visibles.length} restantes)
          </button>
        </div>
      )}
    </>
  );
}
