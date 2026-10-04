"""Exportación de datasets estáticos para la web (GitHub Pages).

La web no ejecuta Python ni llama a APIs: solo lee estos JSON. Se generan a
partir del estado versionado (snapshots, paper trading, registro de modelos,
backtest) y de la caché de precios. Todo archivo lleva su fecha de generación
y la fecha de los datos.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtesting.metrics import equity_metrics, trade_metrics, yearly_returns
from ..config import Settings, resolve
from ..data.market import load_market_data
from ..data.storage import read_json, write_json
from ..explanations.engine import explain_symbol
from ..features.builder import FEATURE_GROUPS, FEATURE_LABELS
from ..fundamentals.scoring import LABELS as FUND_LABELS
from ..logging_utils import get_logger, summary
from ..ml.metrics import roc_auc
from ..ranking.engine import signal_label
from ..trading.manual import load_manual_positions
from .analysis import analyze_universe
from .history import list_snapshots, load_details, load_news_archive, load_scores, load_snapshot

log = get_logger("EXPORT")
DISCLAIMER = ("Este proyecto es una herramienta experimental de análisis cuantitativo y paper trading. No ejecuta "
              "operaciones reales ni constituye asesoramiento financiero.")


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _f(v, nd: int | None = None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(x) or np.isinf(x):
        return None
    return round(x, nd) if nd is not None else x


def _serie(valores, nd: int = 3) -> list:
    arr = np.asarray(valores, dtype=float)
    return [None if not np.isfinite(x) else round(float(x), nd) for x in arr]


def encode_dates(index: pd.DatetimeIndex) -> dict:
    """Fechas compactas: fecha inicial + días transcurridos entre sesiones consecutivas."""
    if len(index) == 0:
        return {"t0": None, "dt": []}
    dias = np.diff(index.values.astype("datetime64[D]").astype(np.int64), prepend=index.values[:1].astype(
        "datetime64[D]").astype(np.int64))
    return {"t0": index[0].strftime("%Y-%m-%d"), "dt": [int(x) for x in dias]}


def price_decimals(close: pd.Series) -> int:
    mediana = float(np.nanmedian(close.to_numpy(dtype=float))) if len(close) else 100.0
    return 2 if mediana >= 10 else 3 if mediana >= 1 else 4


def risk_level(risk_score) -> str | None:
    v = _f(risk_score)
    if v is None:
        return None
    return "Bajo" if v >= 65 else "Medio" if v >= 45 else "Alto"


# --------------------------------------------------------------------------
# Métricas del paper trading y evaluación en vivo
# --------------------------------------------------------------------------

def paper_metrics(trades: list[dict], equity: dict) -> dict:
    cerradas = [t for t in trades if t["status"] == "CLOSED" and t.get("actual_return") is not None]
    tabla = pd.DataFrame([{"return": t["actual_return"], "pnl": t.get("pnl", 0.0),
                           "holding_days": t.get("holding_days", 0)} for t in cerradas])
    metricas = {"total_trades": len(trades), "closed_trades": len(cerradas),
                "open_trades": sum(t["status"] == "OPEN" for t in trades),
                "pending_trades": sum(t["status"] == "PENDING" for t in trades),
                "invalidated_trades": sum(t["status"] == "INVALIDATED" for t in trades)}
    tm = trade_metrics(tabla) if len(tabla) else trade_metrics(pd.DataFrame())
    metricas.update({k: _f(v) for k, v in tm.items() if k != "trades"})
    historial = equity.get("history") or []
    avisos = []
    if historial:
        serie = pd.Series([h["equity"] for h in historial], index=pd.to_datetime([h["date"] for h in historial]))
        em = equity_metrics(serie)
        metricas.update({k: _f(v) for k, v in em.items() if k not in ("start", "end", "days")})
        metricas["start"], metricas["end"], metricas["days"] = em.get("start"), em.get("end"), em.get("days")
        expos = [h["invested"] / h["equity"] for h in historial if h.get("equity")]
        metricas["exposure"] = _f(np.mean(expos)) if expos else None
        metricas["by_year"] = yearly_returns(serie)
        if len(serie) < 20:
            for k in ("sharpe", "sortino", "volatility", "cagr"):
                metricas[k] = None
            avisos.append(f"Solo {len(serie)} días de historial: Sharpe, Sortino, volatilidad y CAGR necesitan al menos 20.")
    else:
        avisos.append("El paper trading aún no tiene historial.")
    if len(cerradas) < 5:
        avisos.append(f"Solo {len(cerradas)} operaciones cerradas: las estadísticas por operación aún no son representativas.")
    metricas["notes"] = avisos
    return metricas


def live_evaluation(history_dir: Path, analyses: dict, horizon: int) -> dict:
    """Compara lo que el sistema dijo cada día con lo que pasó después (etiqueta
    realizada con la misma regla de la operación simulada, solo si ya pasaron H sesiones)."""
    filas = []
    for fecha in list_snapshots(history_dir):
        scores = load_scores(history_dir, fecha)
        if scores is None or scores.empty:
            continue
        ts = pd.Timestamp(fecha)
        for _, r in scores.iterrows():
            a = analyses.get(r["symbol"])
            if a is None or ts not in a.labels.index:
                continue
            etiqueta = a.labels.at[ts, "label"]
            if pd.isna(etiqueta):
                continue  # aún no han pasado H sesiones
            filas.append({"date": fecha, "symbol": r["symbol"], "label": float(etiqueta),
                          "trade_return": float(a.labels.at[ts, "trade_return"]),
                          "final_score": r.get("final_score"), "statistical_score": r.get("statistical_score"),
                          "probability": r.get("opportunity_probability"), "opportunity": bool(r.get("opportunity"))})
    if not filas:
        return {"available": False, "evaluated_days": 0,
                "note": f"Aún no hay predicciones evaluables: cada día se comprueba {horizon} sesiones después."}
    df = pd.DataFrame(filas)
    por_fecha = []
    for fecha, g in df.groupby("date"):
        top_final = g.nlargest(10, "final_score") if g["final_score"].notna().any() else g.iloc[0:0]
        top_prob = g.dropna(subset=["probability"]).nlargest(10, "probability")
        por_fecha.append({
            "date": fecha, "base_rate": float(g["label"].mean()), "n": int(len(g)),
            "top10_final_hit_rate": _f(top_final["label"].mean()) if len(top_final) else None,
            "top10_final_avg_return": _f(top_final["trade_return"].mean()) if len(top_final) else None,
            "top10_neural_hit_rate": _f(top_prob["label"].mean()) if len(top_prob) else None,
            "opportunities_hit_rate": _f(g[g["opportunity"]]["label"].mean()) if g["opportunity"].any() else None,
        })
    con_prob = df.dropna(subset=["probability"])
    return {
        "available": True,
        "evaluated_days": len(por_fecha),
        "samples": int(len(df)),
        "base_rate": _f(df["label"].mean()),
        "final_score_auc": _f(roc_auc(df["label"].to_numpy(), df["final_score"].to_numpy(dtype=float))),
        "statistical_score_auc": _f(roc_auc(df["label"].to_numpy(), df["statistical_score"].to_numpy(dtype=float))),
        "neural_auc": _f(roc_auc(con_prob["label"].to_numpy(), con_prob["probability"].to_numpy(dtype=float)))
        if len(con_prob) else None,
        "neural_samples": int(len(con_prob)),
        "top10_final_hit_rate": _f(np.nanmean([d["top10_final_hit_rate"] for d in por_fecha
                                               if d["top10_final_hit_rate"] is not None])) if por_fecha else None,
        "opportunities_hit_rate": _f(df[df["opportunity"]]["label"].mean()) if df["opportunity"].any() else None,
        "by_date": por_fecha[-90:],
        "note": "Etiqueta realizada: la operación simulada (apertura siguiente, stop/take-profit/plazo) superó el "
                "retorno mínimo neto de costes.",
    }


# --------------------------------------------------------------------------
# Exportación
# --------------------------------------------------------------------------

def _trade_compacto(t: dict) -> dict:
    claves = ("trade_id", "symbol", "name", "status", "strategy", "mode", "signal_date", "signal_price", "entry_date",
              "entry_price", "exit_date", "exit_price", "exit_reason", "invalidation_reason", "actual_return",
              "pnl", "max_drawdown", "holding_days", "final_score", "statistical_score", "news_score",
              "fundamental_score", "neural_score", "risk_score", "opportunity_probability", "confidence",
              "expected_return", "expected_drawdown", "reason", "model_version", "current_price")
    return {k: t.get(k) for k in claves}


def _noticias_simbolo(archivo: list[dict], symbol: str, desde: str | None = None, maximo: int = 10) -> list[dict]:
    lista = [n for n in archivo if n.get("symbol") == symbol and (desde is None or str(n.get("published_at", "")) >= desde)]
    lista.sort(key=lambda n: str(n.get("published_at", "")), reverse=True)
    return [{k: n.get(k) for k in ("title", "source", "url", "published_at", "sentiment", "themes")} for n in lista[:maximo]]


def run_export(settings: Settings, out_dir: str | Path | None = None) -> dict:
    summary.reset("export")
    paths = settings.paths
    salida = Path(out_dir or paths.frontend_data)
    if (salida / "stocks").exists():
        shutil.rmtree(salida / "stocks")
    (salida / "stocks").mkdir(parents=True, exist_ok=True)
    generado = _ahora()
    fechas = list_snapshots(paths.history)
    ultimo = fechas[-1] if fechas else None
    snap = load_snapshot(paths.history, ultimo) if ultimo else None
    scores = load_scores(paths.history, ultimo) if ultimo else None
    detalles = load_details(paths.history, ultimo) if ultimo else {}
    trades_data = read_json(paths.state / "trades.json", {"trades": []})
    trades = trades_data.get("trades", [])
    equity = read_json(paths.state / "equity.json", {"history": []})
    registro = read_json(paths.models / "registry.json", {"production": None, "versions": []})
    backtest = read_json(paths.results / "backtest_latest.json")
    entrenamiento = read_json(paths.results / "training_latest.json")
    calidad = read_json(paths.results / "data_quality.json")
    resumen_run = read_json(paths.results / "last_run_summary.json")
    archivo_noticias = load_news_archive(paths.news, months=int(settings.data.get("export", {}).get("news_archive_months", 3)))
    universo_csv = resolve(settings.universe["universe"]["snapshot_file"], settings)
    universo = pd.read_csv(universo_csv, dtype=str).fillna("") if universo_csv.exists() else pd.DataFrame(
        columns=["symbol", "name", "sector", "industry", "indices"])
    meta_u = universo.set_index("symbol").to_dict("index") if len(universo) else {}
    as_of = snap["as_of"] if snap else None
    fuente = snap.get("data_source") if snap else None
    base = {"generated_at": generado, "data_as_of": as_of, "data_source": fuente}
    horizonte = int(settings.rules["holding_period_days"])
    manuales = load_manual_positions()

    # Recalcular análisis desde la caché de precios (gráficos, explicaciones y evaluación en vivo)
    simbolos = sorted(set(scores["symbol"]) if scores is not None else set())
    simbolos = sorted(set(simbolos) | {t["symbol"] for t in trades} | {m["symbol"] for m in manuales})
    mercado, analisis = None, {}
    if simbolos:
        try:
            mercado = load_market_data(settings, simbolos, as_of=as_of)
            analisis = analyze_universe(mercado, [s for s in simbolos if s in mercado.frames], settings,
                                        with_labels=True)
        except Exception as exc:  # sin caché de precios: se exporta lo que hay, sin gráficos
            summary.warning("EXPORT", f"Sin caché de precios ({exc}); las páginas de acciones no tendrán gráficos")

    filas_scores = scores.to_dict("records") if scores is not None else []
    por_simbolo = {r["symbol"]: r for r in filas_scores}
    umbral_op = float(settings.strategy["opportunities"]["min_final_score"])
    activos = {t["symbol"] for t in trades if t["status"] in ("OPEN", "PENDING")}

    # ---- opportunities.json
    filas_op = []
    for r in sorted(filas_scores, key=lambda r: -(r.get("final_score") or -1)):
        if not r.get("eligible"):
            continue
        filas_op.append({
            "symbol": r["symbol"], "name": r.get("name"), "sector": r.get("sector"), "price": _f(r.get("price"), 4),
            "change_1d": _f(r.get("change_1d"), 5), "statistical_score": _f(r.get("statistical_score"), 1),
            "news_score": _f(r.get("news_score"), 1), "fundamental_score": _f(r.get("fundamental_score"), 1),
            "neural_score": _f(r.get("neural_score"), 1), "final_score": _f(r.get("final_score"), 1),
            "opportunity_probability": _f(r.get("opportunity_probability"), 4), "confidence": _f(r.get("confidence"), 3),
            "risk_score": _f(r.get("risk_score"), 1), "risk_level": risk_level(r.get("risk_score")),
            "rank": int(r["rank"]) if pd.notna(r.get("rank")) else None, "is_opportunity": bool(r.get("opportunity")),
            "signal_label": signal_label(_f(r.get("final_score")), umbral_op), "in_tracking": r["symbol"] in activos,
            "summary": ((detalles.get(r["symbol"]) or {}).get("explanation") or {}).get("summary"),
            "last_update": as_of,
        })
    write_json(salida / "opportunities.json", {**base, "model_available": bool(snap and snap["model"]["available"]),
                                               "weights": settings.model["ranking"]["weights"],
                                               "opportunity_threshold": umbral_op, "rows": filas_op}, compact=True)

    # ---- news.json
    recientes = sorted(archivo_noticias, key=lambda n: str(n.get("published_at", "")), reverse=True)
    limite = (pd.Timestamp(as_of) - pd.Timedelta(days=int(settings.data["news"]["lookback_days"]))).strftime("%Y-%m-%d") \
        if as_of else None
    recientes = [n for n in recientes if limite is None or str(n.get("published_at", "")) >= limite][:600]
    write_json(salida / "news.json", {**base, "providers": snap.get("news_providers", []) if snap else [],
                                      "items": [{k: n.get(k) for k in ("symbol", "title", "source", "url", "published_at",
                                                                       "sentiment", "relevance", "themes", "provider")}
                                                for n in recientes]}, compact=True)

    # ---- stocks/*.json y stocks/index.json
    bt_trades = pd.DataFrame()
    ruta_bt = paths.results / "backtest_trades.csv.gz"
    if ruta_bt.exists():
        bt_trades = pd.read_csv(ruta_bt)
    historial_scores = _historial_scores(paths.history, fechas[-120:])
    ecfg = settings.data.get("export", {})
    anios = int(ecfg.get("chart_history_years", 10))
    anios_osc = int(ecfg.get("oscillator_years", 3))
    indice = []
    for s in simbolos:
        a = analisis.get(s)
        r = por_simbolo.get(s, {})
        info = meta_u.get(s, {})
        det = detalles.get(s) or {}
        nombre = r.get("name") or info.get("name") or s
        pagina = {**base, "symbol": s, "name": nombre, "sector": r.get("sector") or info.get("sector"),
                  "industry": info.get("industry"), "indices": info.get("indices")}
        if a is not None:
            ind = a.ind
            corte = ind.index[-1] - pd.DateOffset(years=anios)
            graf = ind[ind.index >= corte]
            precio_real = mercado.raw[s]["close"] if mercado is not None and s in mercado.raw else ind["close"]
            ultimo_ind = ind.iloc[-1]
            pagina["price"] = {"last": _f(precio_real.iloc[-1], 4), "change_1d": _f(ultimo_ind["ret1"], 5),
                               "change_1m": _f(ultimo_ind["ret21"], 5), "change_3m": _f(ultimo_ind["ret63"], 5),
                               "change_1y": _f(ind["close"].iloc[-1] / ind["close"].iloc[-253] - 1, 5) if len(ind) > 253 else None,
                               "high52": _f(ultimo_ind["high252"], 4), "low52": _f(ultimo_ind["low252"], 4),
                               "date": ind.index[-1].strftime("%Y-%m-%d"), "adjusted_series": True}
            nd = price_decimals(graf["close"])
            pagina["series"] = {**encode_dates(graf.index), "decimals": nd,
                                **{c: _serie(graf[c], nd) for c in ("open", "high", "low", "close", "ma7", "ma25", "ma99")},
                                "volume": [int(v) if np.isfinite(v) else None for v in graf["volume"].to_numpy(dtype=float)]}
            inicio_osc = int(graf.index.searchsorted(graf.index[-1] - pd.DateOffset(years=anios_osc)))
            osc = graf.iloc[inicio_osc:]
            pagina["oscillators"] = {"start_index": inicio_osc, "rsi14": _serie(osc["rsi14"], 1),
                                     **{c: _serie(osc[c], nd + 1) for c in ("macd", "macd_signal", "macd_hist")}}
            pagina["latest_indicators"] = {c: _f(ultimo_ind[c], 5) for c in (
                "ma7", "ma25", "ma99", "rsi14", "macd", "macd_signal", "macd_hist", "atr_pct", "vol20", "bb_pctb",
                "dist_ma7", "dist_ma25", "dist_ma99", "slope_ma7", "slope_ma25", "slope_ma99", "rel_volume",
                "drawdown252", "rel_strength63", "ret63")}
            cruces = graf[graf["cross_ma7_ma25"] != 0]
            pagina["signals"] = [{"date": d.strftime("%Y-%m-%d"), "type": "golden_cross_7_25" if v > 0 else "death_cross_7_25",
                                  "price": _f(graf.at[d, "close"], 3)} for d, v in cruces["cross_ma7_ma25"].items()]
            noticias_det = det.get("news")
            neural_det = det.get("neural")
            pagina["explanation"] = det.get("explanation") or explain_symbol(
                ind.iloc[-1], a.scores.iloc[-1], settings, None, None, None)
            sc = a.scores.iloc[-1]
            pesos_st = settings.model["statistical"]["weights"]
            pagina["statistical_breakdown"] = [{"component": c, "score": _f(sc.get(f"{c}_score"), 1),
                                                "weight": float(pesos_st.get(c, 0))}
                                               for c in ("trend", "momentum", "technical", "volume", "risk", "history")]
        else:
            pagina["series"] = None
            pagina["explanation"] = det.get("explanation")
            noticias_det, neural_det = det.get("news"), det.get("neural")
        pagina["scores"] = {k: _f(r.get(k), 4) for k in ("final_score", "statistical_score", "news_score",
                                                         "fundamental_score", "neural_score", "risk_score",
                                                         "opportunity_probability", "confidence", "expected_return",
                                                         "expected_drawdown")} | {
            "rank": int(r["rank"]) if pd.notna(r.get("rank")) else None, "eligible": bool(r.get("eligible")),
            "is_opportunity": bool(r.get("opportunity")), "signal_label": signal_label(_f(r.get("final_score")), umbral_op),
            "risk_level": risk_level(r.get("risk_score"))} if r else None
        pagina["breakdown"] = det.get("breakdown")
        pagina["news"] = ({k: noticias_det.get(k) for k in ("news_score", "label", "explanation", "count", "positive",
                                                            "negative", "neutral", "themes", "sentiment")}
                          | {"items": [{k: n.get(k) for k in ("title", "source", "url", "published_at", "sentiment",
                                                              "relevance", "themes")} for n in noticias_det.get("items", [])]}
                          if noticias_det else None)
        pagina["news_archive"] = _noticias_simbolo(archivo_noticias, s, maximo=15)
        pagina["fundamentals"] = det.get("fundamentals")
        pagina["fundamental_labels"] = FUND_LABELS
        modelo_snap = (snap or {}).get("model", {})
        pagina["neural"] = ({"available": True, "model_version": modelo_snap.get("version"),
                             "base_rate": modelo_snap.get("base_rate"), "threshold": modelo_snap.get("threshold"),
                             **{k: _f(r.get(k), 4) for k in ("opportunity_probability", "confidence", "expected_return",
                                                             "expected_drawdown", "neural_score")},
                             "explanation": (neural_det or {}).get("explanation")}
                            if r and r.get("opportunity_probability") is not None and not pd.isna(r.get("opportunity_probability"))
                            else {"available": False, "reason": modelo_snap.get("reason") or "Sin predicción para esta acción."})
        pagina["trades"] = {
            "paper": [_trade_compacto(t) for t in trades if t["symbol"] == s],
            "backtest": (bt_trades[(bt_trades["symbol"] == s) & (bt_trades["strategy"] == "combined")]
                         .tail(60).to_dict("records") if not bt_trades.empty and "strategy" in bt_trades else []),
        }
        if not bt_trades.empty and "strategy" in bt_trades:
            propias = bt_trades[(bt_trades["symbol"] == s) & (bt_trades["strategy"] == "combined")]
            pagina["backtest_stats"] = ({"trades": int(len(propias)), "win_rate": _f((propias["return"] > 0).mean(), 4),
                                         "avg_return": _f(propias["return"].mean(), 5),
                                         "period": (backtest or {}).get("period")} if len(propias) else None)
        pagina["score_history"] = historial_scores.get(s, [])
        write_json(salida / "stocks" / f"{s}.json", pagina, compact=True, decimals=None)
        indice.append({"symbol": s, "name": nombre, "sector": pagina.get("sector"),
                       "price": (pagina.get("price") or {}).get("last"), "change_1d": (pagina.get("price") or {}).get("change_1d"),
                       "final_score": _f(r.get("final_score"), 1) if r else None,
                       "statistical_score": _f(r.get("statistical_score"), 1) if r else None,
                       "is_opportunity": bool(r.get("opportunity")) if r else False, "in_tracking": s in activos,
                       "has_chart": a is not None})
    write_json(salida / "stocks" / "index.json", {**base, "stocks": indice}, compact=True)
    summary.count("stock_pages", len(indice))

    # ---- tracking.json
    def _enriquecer(t: dict) -> dict:
        r = por_simbolo.get(t["symbol"], {})
        desde = t.get("entry_date") or t.get("signal_date")
        sesiones = None
        if mercado is not None and desde:
            sesiones = int(((mercado.calendar > pd.Timestamp(desde)) & (mercado.calendar <= mercado.as_of)).sum())
        precio = t.get("current_price") or _f(r.get("price"))
        return {**_trade_compacto(t), "reasons": t.get("reasons", []), "why_it_matters": t.get("why_it_matters", []),
                "risks_at_entry": t.get("risks_at_entry", []), "news_explanation": t.get("news_explanation"),
                "model_explanation": t.get("model_explanation"), "current_scores": t.get("current_scores"),
                "current_price": precio, "sessions_tracked": sesiones,
                "news_since_entry": _noticias_simbolo(archivo_noticias, t["symbol"], desde=desde, maximo=8)}

    seguimiento = [_enriquecer(t) for t in trades if t["status"] in ("OPEN", "PENDING")]
    manuales_out = []
    for m in manuales:
        a = analisis.get(m["symbol"])
        r = por_simbolo.get(m["symbol"], {})
        precio = _f(mercado.raw[m["symbol"]]["close"].iloc[-1]) if mercado is not None and m["symbol"] in mercado.raw else None
        exp = (detalles.get(m["symbol"]) or {}).get("explanation") or (
            explain_symbol(a.ind.iloc[-1], a.scores.iloc[-1], settings) if a is not None else None)
        manuales_out.append({
            **m, "name": r.get("name") or meta_u.get(m["symbol"], {}).get("name") or m["symbol"], "current_price": precio,
            "return": (precio / m["entry_price"] - 1) if precio else None,
            "days": (pd.Timestamp(as_of) - pd.Timestamp(m["entry_date"])).days if as_of and m.get("entry_date") else None,
            "reasons_auto": not m["reasons"], "why_it_matters_auto": not m["why_it_matters"],
            "reasons": m["reasons"] or ([x["text"] for x in exp["reasons"]] if exp else []),
            "why_it_matters": m["why_it_matters"] or ([x["why_it_matters"] for x in exp["reasons"]] if exp else []),
            "current_scores": {k: _f(r.get(k), 4) for k in ("final_score", "statistical_score", "news_score",
                                                            "fundamental_score", "neural_score", "risk_score")} if r else None,
            "risks": [x["text"] for x in exp["risks"]] if exp else [],
            "news_since_entry": _noticias_simbolo(archivo_noticias, m["symbol"], desde=m.get("entry_date"), maximo=8),
        })
    write_json(salida / "tracking.json", {**base, "mode": settings.mode, "trades": seguimiento,
                                          "manual_positions": manuales_out})

    # ---- history.json
    metricas_paper = paper_metrics(trades, equity)
    spy = mercado.frames.get(mercado.benchmark) if mercado is not None else None
    curva = []
    hist_eq = equity.get("history") or []
    if hist_eq:
        ref = spy["close"].reindex(pd.to_datetime([h["date"] for h in hist_eq])).ffill() if spy is not None else None
        for i, h in enumerate(hist_eq):
            bench = None
            if ref is not None and np.isfinite(ref.iloc[i]) and np.isfinite(ref.iloc[0]):
                bench = float(equity["initial_capital"]) * float(ref.iloc[i] / ref.iloc[0])
            curva.append({"date": h["date"], "equity": h["equity"], "benchmark": _f(bench, 2),
                          "invested": h.get("invested"), "open_positions": h.get("open_positions")})
    snapshots_out = _snapshots_con_resultados(paths.history, fechas[-120:], analisis, mercado)
    write_json(salida / "history.json", {**base, "trades": [_trade_compacto(t) for t in trades],
                                         "equity": curva, "metrics": metricas_paper, "snapshots": snapshots_out,
                                         "strategies": sorted({t.get("strategy") for t in trades if t.get("strategy")})})

    # ---- backtest.json
    if backtest:
        muestra = (bt_trades[bt_trades["strategy"] == "combined"] if not bt_trades.empty and "strategy" in bt_trades
                   else bt_trades)
        write_json(salida / "backtest.json", {**backtest, "available": True, "exported_at": generado,
                                              "trades_sample": muestra.tail(300).to_dict("records") if len(muestra) else []})
    else:
        write_json(salida / "backtest.json", {"available": False, "exported_at": generado,
                                              "message": "Todavía no se ha ejecutado el backtest (workflow model_training)."})

    # ---- model.json
    evaluacion = live_evaluation(paths.history, analisis, horizonte) if analisis else {
        "available": False, "evaluated_days": 0, "note": "Sin datos de precios para evaluar."}
    versiones = registro.get("versions", [])
    produccion = next((v for v in versiones if v["model_version"] == registro.get("production")), None)
    ultimo_candidato = versiones[-1] if versiones else None

    def _compacta(v: dict) -> dict:
        agg = (v.get("walk_forward") or {}).get("aggregate") or {}
        return {"model_version": v["model_version"], "status": v.get("status"), "training_date": v.get("training_date"),
                "training_period": v.get("training_period"), "features_version": v.get("features_version"),
                "oos_auc": agg.get("roc_auc"), "top_decile_lift": agg.get("top_decile_lift"),
                "combined_sharpe": (v.get("backtest") or {}).get("combined", {}).get("sharpe"),
                "promoted_at": v.get("promoted_at"), "checks": v.get("promotion_checks", [])}

    def _detalle(v: dict | None) -> dict | None:
        if not v:
            return None
        d = {k: v.get(k) for k in v if k not in ("scaler",)}
        d["feature_importance"] = [{**fi, "label": FEATURE_LABELS.get(fi["feature"], fi["feature"])}
                                   for fi in v.get("feature_importance", [])]
        return d

    write_json(salida / "model.json", {
        **base,
        "production_version": registro.get("production"),
        "registry": [_compacta(v) for v in versiones],
        "production": _detalle(produccion),
        "latest_candidate": _detalle(ultimo_candidato) if ultimo_candidato is not produccion else None,
        "live_evaluation": evaluacion,
        "features": [{"name": c, "label": FEATURE_LABELS.get(c, c), "group": g}
                     for g, cols in FEATURE_GROUPS.items() for c in cols],
        "architecture": {**settings.model["neural_network"], "lookback": settings.model["features"]["lookback"]},
        "walk_forward_config": settings.model["walk_forward"],
        "promotion_config": settings.model["promotion"],
        "status_reason": (snap or {}).get("model", {}).get("reason"),
    })

    # ---- market.json (dashboard)
    estado = "no_data" if not snap else "ok"
    mensaje = ("Todavía no hay datos: ejecuta el workflow «Market update» en GitHub Actions." if not snap else
               "Datos actualizados.")
    if snap and fuente == "synthetic":
        estado, mensaje = "synthetic", "DATOS SINTÉTICOS de desarrollo: no representan el mercado real."
    cerradas = [t for t in trades if t["status"] == "CLOSED" and t.get("actual_return") is not None]
    mejores = sorted(cerradas, key=lambda t: t["actual_return"], reverse=True)[:5]
    peores = sorted(cerradas, key=lambda t: t["actual_return"])[:5]
    titular_bt = None
    if backtest:
        nombre = "combined" if backtest["strategies"].get("combined", {}).get("available") else "statistical_only"
        titular_bt = {"strategy": nombre, "metrics": {k: v for k, v in (backtest["strategies"].get(nombre, {})
                                                                         .get("metrics") or {}).items() if k != "yearly"},
                      "benchmark": {k: v for k, v in ((backtest.get("benchmarks") or {}).get("buy_hold_benchmark", {})
                                                      .get("metrics") or {}).items() if k != "yearly"},
                      "period": backtest.get("period"), "generated_at": backtest.get("generated_at"),
                      "model_version": backtest.get("model_version")}
    write_json(salida / "market.json", {
        **base,
        "disclaimer": DISCLAIMER,
        "mode": settings.mode,
        "status": {"state": estado, "message": mensaje, "last_update": snap.get("generated_at") if snap else None,
                   "last_run": {k: (resumen_run or {}).get(k) for k in ("task", "finished_at", "n_errors", "n_warnings")}
                   if resumen_run else None},
        "market": {"name": "Estados Unidos (NYSE / Nasdaq)", "timezone": settings.data["market_calendar"]["timezone"],
                   "indices": (snap or {}).get("market", [])},
        "universe": {"size": (snap or {}).get("universe_size"), "analyzed": (snap or {}).get("symbols_analyzed"),
                     "eligible": (snap or {}).get("symbols_eligible"), "skipped": (snap or {}).get("symbols_skipped"),
                     "deep_analyzed": (snap or {}).get("deep_analyzed"),
                     "quality_rejected": (calidad or {}).get("rejected")},
        "counts": {"opportunities": sum(1 for r in filas_op if r["is_opportunity"]),
                   "open_trades": sum(t["status"] == "OPEN" for t in trades),
                   "pending_trades": sum(t["status"] == "PENDING" for t in trades),
                   "closed_trades": len(cerradas)},
        "top_opportunities": [r for r in filas_op if r["is_opportunity"]][:10],
        "system_performance": metricas_paper,
        "best_trades": [_trade_compacto(t) for t in mejores],
        "worst_trades": [_trade_compacto(t) for t in peores],
        "model": {"production_version": registro.get("production"),
                  "available": bool(snap and snap["model"]["available"]),
                  "reason": (snap or {}).get("model", {}).get("reason"),
                  "training_date": (produccion or {}).get("training_date"),
                  "training_period": (produccion or {}).get("training_period"),
                  "features_version": (produccion or {}).get("features_version"),
                  "oos_auc": ((produccion or {}).get("walk_forward") or {}).get("aggregate", {}).get("roc_auc"),
                  "last_training": (entrenamiento or {}).get("generated_at"),
                  "last_candidate": (entrenamiento or {}).get("model_version"),
                  "last_candidate_promoted": (entrenamiento or {}).get("promoted")},
        "backtest": titular_bt,
        "live_evaluation": {k: evaluacion.get(k) for k in ("available", "evaluated_days", "samples", "base_rate",
                                                           "top10_final_hit_rate", "opportunities_hit_rate", "note")},
        "history_days": len(fechas),
    })
    summary.set("status", "ok")
    log.info("Exportación completa en %s (%d páginas de acciones)", salida, len(indice))
    return {"status": estado, "stocks": len(indice), "out_dir": str(salida)}


def _historial_scores(history_dir: Path, fechas: list[str]) -> dict[str, list[dict]]:
    salida: dict[str, list[dict]] = {}
    for fecha in fechas:
        tabla = load_scores(history_dir, fecha)
        if tabla is None:
            continue
        for r in tabla.to_dict("records"):
            salida.setdefault(r["symbol"], []).append({
                "date": fecha, "final_score": _f(r.get("final_score"), 1),
                "statistical_score": _f(r.get("statistical_score"), 1),
                "probability": _f(r.get("opportunity_probability"), 4)})
    return salida


def _snapshots_con_resultados(history_dir: Path, fechas: list[str], analisis: dict, mercado) -> list[dict]:
    """¿Qué oportunidades detectó el sistema cada día y qué pasó después?"""
    salida = []
    spy = mercado.frames.get(mercado.benchmark)["close"] if mercado is not None and mercado.benchmark in mercado.frames else None
    for fecha in reversed(fechas):
        snap = load_snapshot(history_dir, fecha) or {}
        ts = pd.Timestamp(fecha)
        filas = []
        for op in snap.get("opportunities", [])[:15]:
            a = analisis.get(op["symbol"])
            desde = hasta = None
            if a is not None and ts in a.ind.index:
                serie = a.ind["close"]
                p0 = float(serie.loc[ts])
                desde = float(serie.iloc[-1] / p0 - 1)
                pos = serie.index.get_loc(ts)
                hasta = float(serie.iloc[pos + 20] / p0 - 1) if pos + 20 < len(serie) else None
            filas.append({"symbol": op["symbol"], "final_score": op.get("final_score"), "price": op.get("price"),
                          "return_since": _f(desde, 5), "return_20d": _f(hasta, 5)})
        bench = None
        if spy is not None and ts in spy.index:
            bench = _f(float(spy.iloc[-1] / spy.loc[ts] - 1), 5)
        salida.append({"date": fecha, "model_version": (snap.get("model") or {}).get("version"),
                       "opportunities": filas, "benchmark_return_since": bench})
    return salida
