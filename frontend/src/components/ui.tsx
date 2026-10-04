import type { ReactNode } from "react";
import { isNum } from "../format";
import type { Num } from "../types";

export function Card({ title, extra, children, id }: { title?: ReactNode; extra?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="card" id={id}>
      {(title || extra) && (
        <div className="card-title">
          {typeof title === "string" ? <h2>{title}</h2> : title}
          {extra && <div className="muted" style={{ fontSize: "0.85rem" }}>{extra}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, valueClass }: { label: string; value: ReactNode; sub?: ReactNode; valueClass?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${valueClass ?? ""}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

/** Medidor 0-100 de un score interno (no es una probabilidad ni una certeza). */
export function Meter({ value, label }: { value: Num | undefined; label?: string }) {
  if (!isNum(value)) return <span className="muted">—</span>;
  const v = Math.max(0, Math.min(100, value));
  return (
    <span className="meter" title={label ? `${label}: ${v.toFixed(1)} / 100` : `${v.toFixed(1)} / 100`}>
      <span className="meter-track"><span className="meter-fill" style={{ width: `${v}%` }} /></span>
      <span className="meter-value">{v.toFixed(0)}</span>
    </span>
  );
}

const STATUS_STYLE: Record<string, { color: string; icon: string }> = {
  good: { color: "var(--good)", icon: "●" },
  warning: { color: "var(--warning)", icon: "▲" },
  serious: { color: "var(--serious)", icon: "◆" },
  critical: { color: "var(--critical)", icon: "■" },
  neutral: { color: "var(--axis)", icon: "○" },
};

/** Estado con color + icono + texto (el color nunca va solo). */
export function StatusChip({ kind, children }: { kind: keyof typeof STATUS_STYLE; children: ReactNode }) {
  const s = STATUS_STYLE[kind] ?? STATUS_STYLE.neutral;
  return (
    <span className="chip">
      <span aria-hidden style={{ color: s.color, fontSize: "0.7rem" }}>{s.icon}</span>
      {children}
    </span>
  );
}

export function TradeStatusChip({ status }: { status: string }) {
  const mapa: Record<string, [keyof typeof STATUS_STYLE, string]> = {
    PENDING: ["warning", "Pendiente"],
    OPEN: ["good", "Abierta"],
    CLOSED: ["neutral", "Cerrada"],
    INVALIDATED: ["critical", "Invalidada"],
    MANUAL: ["neutral", "Manual"],
  };
  const [kind, texto] = mapa[status] ?? ["neutral", status];
  return <StatusChip kind={kind}>{texto}</StatusChip>;
}

export function SentimentChip({ value }: { value: Num | undefined }) {
  if (!isNum(value)) return <span className="chip">sin dato</span>;
  if (value >= 0.15) return <span className="chip chip-good">▲ positivo {value.toFixed(2)}</span>;
  if (value <= -0.15) return <span className="chip chip-bad">▼ negativo {value.toFixed(2)}</span>;
  return <span className="chip">● neutral {value.toFixed(2)}</span>;
}

export function Loading({ what = "datos" }: { what?: string }) {
  return <div className="empty">Cargando {what}…</div>;
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="empty">
      {message}. Los datos los publica GitHub Actions tras ejecutar el workflow <strong>Market update</strong>.
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Freshness({ generated, asOf }: { generated?: string | null; asOf?: string | null }) {
  const fmt = (s?: string | null) => (s ? new Date(s.length === 10 ? `${s}T12:00:00Z` : s) : null);
  const g = fmt(generated);
  return (
    <span className="freshness">
      Datos al cierre del {asOf ?? "—"}
      {g && !Number.isNaN(g.getTime()) && <> · generado {g.toLocaleString("es-ES", { dateStyle: "medium", timeStyle: "short" })}</>}
    </span>
  );
}
