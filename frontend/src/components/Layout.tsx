import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useJson } from "../api";
import { dateTime } from "../format";
import { applyTheme, readTheme, type ThemeChoice } from "../theme";
import type { MarketJson } from "../types";

const LINKS: [string, string][] = [
  ["/", "Dashboard"],
  ["/opportunities", "Oportunidades"],
  ["/tracking", "Seguimiento"],
  ["/history", "Historial"],
  ["/stocks", "Acciones"],
  ["/backtesting", "Backtesting"],
  ["/model", "Modelo IA"],
  ["/news", "Noticias"],
];
const TEMAS: Record<ThemeChoice, string> = { auto: "Tema: auto", light: "Tema: claro", dark: "Tema: oscuro" };

export function Layout() {
  const { data } = useJson<MarketJson>("market.json");
  const [tema, setTema] = useState<ThemeChoice>(readTheme());
  const cambiarTema = () => {
    const siguiente: ThemeChoice = tema === "auto" ? "light" : tema === "light" ? "dark" : "auto";
    applyTheme(siguiente);
    setTema(siguiente);
  };
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="brand"><span className="brand-mark" aria-hidden />Quant Research</NavLink>
          <nav className="nav" aria-label="Secciones">
            {LINKS.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>
            ))}
          </nav>
          <button className="theme-toggle" onClick={cambiarTema} aria-label="Cambiar tema">{TEMAS[tema]}</button>
        </div>
      </header>
      <div className="disclaimer" role="note">
        <div className="disclaimer-inner">
          ⚠️ Este proyecto es una herramienta experimental de análisis cuantitativo y paper trading. No ejecuta operaciones
          reales ni constituye asesoramiento financiero. Las operaciones mostradas son <strong>simulaciones</strong>.
        </div>
      </div>
      <main>
        {data?.status.state === "synthetic" && (
          <div className="banner banner-critical">
            <strong>DATOS SINTÉTICOS.</strong> Esta publicación se generó en modo desarrollo con datos inventados: no
            representan el mercado real.
          </div>
        )}
        {data?.status.state === "no_data" && (
          <div className="banner banner-warning">
            Todavía no hay datos de mercado: ejecuta el workflow <strong>Market update</strong> en GitHub Actions.
          </div>
        )}
        <Outlet />
      </main>
      <footer>
        Investigación cuantitativa y paper trading · datos de fuentes públicas gratuitas (pueden contener errores o retrasos)
        {data && <> · última actualización {dateTime(data.status.last_update ?? data.generated_at)} · datos al cierre del {data.data_as_of ?? "—"}</>}
      </footer>
    </>
  );
}
