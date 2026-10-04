import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { cssVar, SERIES_VARS, useThemeVersion } from "../theme";

export interface SeriesDef {
  key: string;
  label: string;
  slot: number; // posición fija en la paleta categórica (el color sigue a la entidad, no al orden)
}

interface Props {
  data: Record<string, string | number | null>[];
  xKey: string;
  series: SeriesDef[];
  height?: number;
  yFormat?: (v: number) => string;
  yTickFormat?: (v: number) => string;
  xFormat?: (v: string) => string;
}

/** Líneas con tooltip de todas las series en la fecha señalada y leyenda (identidad nunca solo por color). */
export function SeriesChart({ data, xKey, series, height = 300, yFormat = (v) => v.toFixed(0), yTickFormat, xFormat }: Props) {
  useThemeVersion();
  const grid = cssVar("--grid");
  const texto = cssVar("--text-muted");
  const superficie = cssVar("--surface");
  const borde = cssVar("--border");
  if (!data.length) return <div className="empty">Sin datos suficientes para el gráfico.</div>;
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis dataKey={xKey} tick={{ fill: texto, fontSize: 11 }} tickLine={false} axisLine={{ stroke: grid }}
                 minTickGap={40} tickFormatter={xFormat} />
          <YAxis tick={{ fill: texto, fontSize: 11 }} tickLine={false} axisLine={false} width={72}
                 tickFormatter={(v: number) => (yTickFormat ?? yFormat)(v)} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: superficie, border: `1px solid ${borde}`, borderRadius: 8, fontSize: 12 }}
            formatter={(v, nombre) => [typeof v === "number" ? yFormat(v) : String(v ?? "—"), nombre]}
            labelFormatter={(l) => (xFormat ? xFormat(String(l)) : String(l))}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
          {series.map((s) => (
            <Line key={s.key} type="linear" dataKey={s.key} name={s.label} stroke={cssVar(SERIES_VARS[s.slot % SERIES_VARS.length])}
                  strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
