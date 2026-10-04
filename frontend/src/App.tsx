import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Loading } from "./components/ui";
import { Dashboard } from "./pages/Dashboard";
import { NotFound } from "./pages/NotFound";

// Las páginas con gráficos se cargan bajo demanda para que el dashboard abra rápido
const Opportunities = lazy(() => import("./pages/Opportunities").then((m) => ({ default: m.Opportunities })));
const Tracking = lazy(() => import("./pages/Tracking").then((m) => ({ default: m.Tracking })));
const History = lazy(() => import("./pages/History").then((m) => ({ default: m.History })));
const Stocks = lazy(() => import("./pages/Stocks").then((m) => ({ default: m.Stocks })));
const StockDetail = lazy(() => import("./pages/StockDetail").then((m) => ({ default: m.StockDetail })));
const Backtesting = lazy(() => import("./pages/Backtesting").then((m) => ({ default: m.Backtesting })));
const Model = lazy(() => import("./pages/Model").then((m) => ({ default: m.Model })));
const News = lazy(() => import("./pages/News").then((m) => ({ default: m.News })));

export function App() {
  return (
    <Suspense fallback={<Loading what="la página" />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="opportunities" element={<Opportunities />} />
          <Route path="tracking" element={<Tracking />} />
          <Route path="history" element={<History />} />
          <Route path="stocks" element={<Stocks />} />
          <Route path="stock/:symbol" element={<StockDetail />} />
          <Route path="backtesting" element={<Backtesting />} />
          <Route path="model" element={<Model />} />
          <Route path="news" element={<News />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
