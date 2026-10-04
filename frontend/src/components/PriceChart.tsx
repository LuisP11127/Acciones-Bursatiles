import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  type IChartApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import { compact, isNum, num } from "../format";
import type { Num, StockJson } from "../types";
import { cssVar, useThemeVersion } from "../theme";

export type MarkerKind = "entry" | "exit_win" | "exit_loss" | "golden" | "death";
export interface ChartMarker {
  date: string;
  kind: MarkerKind;
  text: string;
  detail: string;
}

const PERIODS: [string, number | null][] = [["1M", 1], ["3M", 3], ["6M", 6], ["1Y", 12], ["3Y", 36], ["5Y", 60], ["MAX", null]];

export function decodeDates(t0: string, dt: number[]): string[] {
  const base = Date.parse(`${t0}T00:00:00Z`);
  let dias = 0;
  return dt.map((d) => {
    dias += d;
    return new Date(base + dias * 86_400_000).toISOString().slice(0, 10);
  });
}

interface Hover {
  date: string;
  o?: Num; h?: Num; l?: Num; c?: Num; ma7?: Num; ma25?: Num; ma99?: Num; vol?: Num; rsi?: Num; macd?: Num;
}

export function PriceChart({ stock, markers }: { stock: StockJson; markers: ChartMarker[] }) {
  const contenedor = useRef<HTMLDivElement>(null);
  const grafico = useRef<IChartApi | null>(null);
  const tema = useThemeVersion();
  const [periodo, setPeriodo] = useState("1Y");
  const [hover, setHover] = useState<Hover | null>(null);
  const s = stock.series!;
  const fechas = useMemo(() => decodeDates(s.t0, s.dt), [s.t0, s.dt]);
  const indice = useMemo(() => new Map(fechas.map((f, i) => [f, i])), [fechas]);
  const porFecha = useMemo(() => {
    const m = new Map<string, ChartMarker[]>();
    markers.forEach((mk) => m.set(mk.date, [...(m.get(mk.date) ?? []), mk]));
    return m;
  }, [markers]);
  const ultimo: Hover = useMemo(() => valores(fechas.length - 1), [fechas]); // eslint-disable-line react-hooks/exhaustive-deps

  function valores(i: number): Hover {
    const o = stock.oscillators;
    const j = o ? i - o.start_index : -1;
    return {
      date: fechas[i], o: s.open[i], h: s.high[i], l: s.low[i], c: s.close[i], ma7: s.ma7[i], ma25: s.ma25[i], ma99: s.ma99[i],
      vol: s.volume[i], rsi: o && j >= 0 ? o.rsi14[j] : null, macd: o && j >= 0 ? o.macd[j] : null,
    };
  }

  useEffect(() => {
    if (!contenedor.current) return;
    const c = {
      surface: cssVar("--surface"), text: cssVar("--text-secondary"), grid: cssVar("--grid"), axis: cssVar("--axis"),
      ma7: cssVar("--series-1"), ma25: cssVar("--series-2"), ma99: cssVar("--series-3"),
      upBody: cssVar("--candle-up-body"), upLine: cssVar("--candle-up-line"), down: cssVar("--candle-down"),
      volume: cssVar("--volume"), good: cssVar("--good"), critical: cssVar("--critical"), muted: cssVar("--text-muted"),
      accent: cssVar("--accent"),
    };
    const chart = createChart(contenedor.current, {
      autoSize: true,
      height: 640,
      layout: { background: { type: ColorType.Solid, color: c.surface }, textColor: c.text, fontSize: 11,
                panes: { separatorColor: c.grid, enableResize: true } },
      grid: { vertLines: { visible: false }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.axis },
      timeScale: { borderColor: c.axis, rightOffset: 4, minBarSpacing: 0.5 },
      crosshair: { mode: CrosshairMode.Normal },
      localization: { locale: "es-ES" },
    });
    grafico.current = chart;
    const t = (i: number) => fechas[i] as Time;
    const velas = chart.addSeries(CandlestickSeries, {
      upColor: c.upBody, borderUpColor: c.upLine, wickUpColor: c.upLine,
      downColor: c.down, borderDownColor: c.down, wickDownColor: c.down,
      priceLineVisible: false,
    });
    velas.setData(fechas.flatMap((_, i) => (isNum(s.open[i]) && isNum(s.high[i]) && isNum(s.low[i]) && isNum(s.close[i])
      ? [{ time: t(i), open: s.open[i] as number, high: s.high[i] as number, low: s.low[i] as number, close: s.close[i] as number }] : [])));
    const linea = (valores: Num[], color: string, pane = 0, offset = 0) => {
      const serie = chart.addSeries(LineSeries, { color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
                                                 crosshairMarkerVisible: false }, pane);
      serie.setData(valores.flatMap((v, k) => (isNum(v) ? [{ time: t(k + offset), value: v }] : [])));
      return serie;
    };
    linea(s.ma7, c.ma7);
    linea(s.ma25, c.ma25);
    linea(s.ma99, c.ma99);
    const volumen = chart.addSeries(HistogramSeries, { color: c.volume, priceFormat: { type: "volume" }, priceLineVisible: false,
                                                       lastValueVisible: false }, 1);
    volumen.setData(s.volume.flatMap((v, i) => (isNum(v) ? [{ time: t(i), value: v }] : [])));
    const osc = stock.oscillators;
    if (osc) {
      const rsi = linea(osc.rsi14, c.accent, 2, osc.start_index);
      [70, 30].forEach((nivel) => rsi.createPriceLine({ price: nivel, color: c.axis, lineWidth: 1, lineStyle: LineStyle.Solid,
                                                       axisLabelVisible: true, title: "" }));
      const hist = chart.addSeries(HistogramSeries, { priceLineVisible: false, lastValueVisible: false }, 3);
      hist.setData(osc.macd_hist.flatMap((v, k) => (isNum(v) ? [{ time: t(k + osc.start_index), value: v,
                                                                  color: v >= 0 ? c.volume : c.axis }] : [])));
      linea(osc.macd, c.ma7, 3, osc.start_index);
      linea(osc.macd_signal, c.ma25, 3, osc.start_index);
    }
    const lista: SeriesMarker<Time>[] = markers
      .filter((m) => indice.has(m.date))
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((m) => ({
        time: m.date as Time,
        position: m.kind === "entry" || m.kind === "golden" ? "belowBar" : "aboveBar",
        shape: m.kind === "entry" ? "arrowUp" : m.kind.startsWith("exit") ? "arrowDown" : "circle",
        color: m.kind === "entry" || m.kind === "exit_win" ? c.good : m.kind === "exit_loss" ? c.critical : c.muted,
        text: m.text,
        size: m.kind === "golden" || m.kind === "death" ? 0.6 : 1.2,
      }));
    createSeriesMarkers(velas, lista);
    const paneles = chart.panes();
    const alturas = [380, 70, 90, 100];
    paneles.forEach((p, i) => p.setHeight(alturas[i] ?? 90));
    chart.subscribeCrosshairMove((param) => {
      if (!param.time) {
        setHover(null);
        return;
      }
      const i = indice.get(String(param.time));
      if (i !== undefined) setHover(valores(i));
    });
    aplicarPeriodo(chart, periodo);
    return () => {
      chart.remove();
      grafico.current = null;
    };
  }, [stock, markers, tema]); // eslint-disable-line react-hooks/exhaustive-deps

  function aplicarPeriodo(chart: IChartApi, p: string) {
    const meses = PERIODS.find(([k]) => k === p)?.[1] ?? null;
    if (meses === null) {
      chart.timeScale().fitContent();
      return;
    }
    const fin = fechas[fechas.length - 1];
    const d = new Date(`${fin}T00:00:00Z`);
    d.setUTCMonth(d.getUTCMonth() - meses);
    const desde = d.toISOString().slice(0, 10);
    const inicio = fechas.find((f) => f >= desde) ?? fechas[0];
    chart.timeScale().setVisibleRange({ from: inicio as Time, to: fin as Time });
  }

  const v = hover ?? ultimo;
  const eventos = porFecha.get(v.date) ?? [];
  const d = s.decimals ?? 2;
  return (
    <div>
      <div className="chart-toolbar">
        <div className="seg" role="group" aria-label="Periodo">
          {PERIODS.map(([k]) => (
            <button key={k} className={periodo === k ? "on" : ""} onClick={() => {
              setPeriodo(k);
              if (grafico.current) aplicarPeriodo(grafico.current, k);
            }}>{k}</button>
          ))}
        </div>
        <span className="muted" style={{ fontSize: "0.8rem" }}>
          Rueda o pellizco: zoom · arrastrar: desplazar · precios ajustados por dividendos y splits
        </span>
      </div>
      <div className="chart-box">
        <div className="chart-legend" aria-live="polite">
          <span className="legend-key"><strong>{v.date}</strong></span>
          <span className="legend-key">A <strong>{num(v.o, d)}</strong></span>
          <span className="legend-key">Máx <strong>{num(v.h, d)}</strong></span>
          <span className="legend-key">Mín <strong>{num(v.l, d)}</strong></span>
          <span className="legend-key">C <strong>{num(v.c, d)}</strong></span>
          <span className="legend-key"><span className="line" style={{ background: "var(--series-1)" }} />MA(7) <strong>{num(v.ma7, d)}</strong></span>
          <span className="legend-key"><span className="line" style={{ background: "var(--series-2)" }} />MA(25) <strong>{num(v.ma25, d)}</strong></span>
          <span className="legend-key"><span className="line" style={{ background: "var(--series-3)" }} />MA(99) <strong>{num(v.ma99, d)}</strong></span>
          <span className="legend-key">Vol <strong>{compact(v.vol)}</strong></span>
          {isNum(v.rsi) && <span className="legend-key">RSI <strong>{num(v.rsi, 1)}</strong></span>}
          {isNum(v.macd) && <span className="legend-key">MACD <strong>{num(v.macd, d + 1)}</strong></span>}
          {eventos.map((e) => <span className="legend-key" key={e.text + e.kind}><strong>{e.detail}</strong></span>)}
        </div>
        <div ref={contenedor} style={{ width: "100%", height: 640 }} />
      </div>
      <p className="muted" style={{ fontSize: "0.78rem", marginTop: 6 }}>
        Paneles: precio con MA(7), MA(25) y MA(99) · volumen · RSI(14) con niveles 30/70 · MACD (línea, señal e histograma).
        Velas huecas = sesión alcista; rellenas = bajista. ▲ entrada y ▼ salida de operaciones simuladas; ● cruces de la MA(7) con la MA(25).
      </p>
    </div>
  );
}
