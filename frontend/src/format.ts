import type { Num } from "./types";

const LOCALE = "es-ES";

export const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function pct(v: Num | undefined, digits = 1, signed = true): string {
  if (!isNum(v)) return "—";
  const s = new Intl.NumberFormat(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v * 100);
  return `${signed && v > 0 ? "+" : ""}${s} %`;
}

export function num(v: Num | undefined, digits = 2): string {
  if (!isNum(v)) return "—";
  return new Intl.NumberFormat(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v);
}

export function int(v: Num | undefined): string {
  if (!isNum(v)) return "—";
  return new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 }).format(v);
}

export function money(v: Num | undefined, digits?: number): string {
  if (!isNum(v)) return "—";
  const d = digits ?? (Math.abs(v) >= 1000 ? 0 : 2);
  return new Intl.NumberFormat(LOCALE, { style: "currency", currency: "USD", minimumFractionDigits: d, maximumFractionDigits: d }).format(v);
}

export function compact(v: Num | undefined): string {
  if (!isNum(v)) return "—";
  return new Intl.NumberFormat(LOCALE, { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

export function score(v: Num | undefined): string {
  return isNum(v) ? v.toFixed(0) : "—";
}

function toDate(s: string): Date {
  return /^\d{4}-\d{2}-\d{2}$/.test(s) ? new Date(`${s}T12:00:00Z`) : new Date(s);
}

export function date(s: string | null | undefined): string {
  if (!s) return "—";
  const d = toDate(s);
  if (Number.isNaN(d.getTime())) return s;
  return new Intl.DateTimeFormat(LOCALE, { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(d);
}

export function dateTime(s: string | null | undefined): string {
  if (!s) return "—";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return new Intl.DateTimeFormat(LOCALE, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" }).format(d);
}

export function signClass(v: Num | undefined): string {
  if (!isNum(v) || v === 0) return "";
  return v > 0 ? "pos" : "neg";
}

export const STATUS_LABELS: Record<string, string> = {
  PENDING: "Pendiente",
  OPEN: "Abierta",
  CLOSED: "Cerrada",
  INVALIDATED: "Invalidada",
  MANUAL: "Manual",
};

export const STRATEGY_LABELS: Record<string, string> = {
  combined: "Sistema combinado",
  statistical_only: "Solo estadístico",
  neural_only: "Solo red neuronal",
  ma_crossover: "Cruce MA(7)/MA(25)",
  buy_hold_benchmark: "Buy & Hold SPY",
  equal_weight_universe: "Equiponderada universo",
};

export const COMPONENT_LABELS: Record<string, string> = {
  statistical: "Estadístico",
  news: "Noticias",
  fundamental: "Fundamental",
  neural: "Red neuronal",
  risk: "Riesgo",
  trend: "Tendencia",
  momentum: "Momentum",
  technical: "Técnico",
  volume: "Volumen",
  history: "Comportamiento histórico",
};

export const EXIT_LABELS: Record<string, string> = {
  stop_loss: "Stop-loss",
  take_profit: "Take-profit",
  time: "Fin del periodo",
  trend_break: "Ruptura de tendencia",
  end_of_backtest: "Fin del backtest (cierre)",
};

export const FEATURE_GROUP_LABELS: Record<string, string> = {
  ohlcv: "OHLCV normalizado",
  momentum: "Momentum",
  moving_averages: "Medias móviles",
  oscillators: "Osciladores",
  risk: "Riesgo y volatilidad",
  volume: "Volumen",
  market: "Contexto de mercado",
};

/** Etiqueta de eje compacta para importes (100 mil US$). */
export function moneyAxis(v: number): string {
  return `${new Intl.NumberFormat(LOCALE, { notation: "compact", maximumFractionDigits: 1 }).format(v)} $`;
}
