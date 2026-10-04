import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router-dom";
import { App } from "./App";
import { applyTheme, readTheme } from "./theme";
import "./styles.css";

applyTheme(readTheme());

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* HashRouter: las rutas (/#/stock/AAPL) funcionan en GitHub Pages sin configuración del servidor */}
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
);
