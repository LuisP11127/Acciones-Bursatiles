"""Pipeline diario (días de mercado, después del cierre):

 1. Comprueba el calendario de la NYSE (zona horaria America/New_York).
 2. Universo -> descarga incremental -> validación de datos.
 3. Indicadores, features, etiquetas históricas y scores estadísticos.
 4. Inferencia de la red neuronal en producción (si existe y es compatible).
 5. Preselección -> noticias + fundamentales de las candidatas.
 6. Ranking final, oportunidades y explicaciones.
 7. Paper trading: entradas pendientes, salidas, nuevas señales y seguimiento.
 8. Snapshot del día, archivo de noticias, informe de calidad y resumen.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Settings
from ..data.market import load_market_data, update_market_data
from ..data.providers import create_fundamentals_provider, create_market_provider, create_news_providers
from ..data.storage import write_json
from ..data.universe import load_universe
from ..explanations.engine import explain_symbol
from ..fundamentals.scoring import fundamental_analysis
from ..logging_utils import get_logger, summary
from ..market_calendar import last_completed_session
from ..ml.inference import compatible, predict_latest
from ..ml.registry import ModelRegistry
from ..news.scoring import score_news
from ..ranking.engine import combine_scores
from ..trading.manual import load_manual_positions
from ..trading.paper import PaperTrader, neural_gate, trend_break_flags
from .analysis import analyze_universe
from .history import archive_news, snapshot_exists, write_snapshot

log = get_logger("PIPELINE")


def _f(valor):
    if valor is None:
        return None
    try:
        v = float(valor)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def market_overview(market, settings) -> list[dict]:
    from ..indicators.technical import compute_indicators

    salida = []
    for s in settings.universe["universe"]["reference_symbols"]:
        frame = market.frames.get(s)
        if frame is None or len(frame) < 100:
            continue
        ind = compute_indicators(frame).iloc[-1]
        tendencia = ("Alcista" if ind["ma7"] > ind["ma25"] > ind["ma99"] and ind["close"] > ind["ma99"]
                     else "Bajista" if ind["close"] < ind["ma99"] and ind["ma25"] < ind["ma99"] else "Lateral")
        salida.append({"symbol": s, "price": _f(market.raw[s]["close"].iloc[-1]), "change_1d": _f(ind["ret1"]),
                       "ret21": _f(ind["ret21"]), "ret63": _f(ind["ret63"]), "ma7": _f(ind["ma7"]),
                       "ma25": _f(ind["ma25"]), "ma99": _f(ind["ma99"]), "rsi14": _f(ind["rsi14"]), "trend": tendencia})
    return salida


def run_daily_update(settings: Settings, force: bool = False, now: datetime | None = None) -> dict:
    summary.reset("market_update")
    paths = settings.paths
    paths.ensure()
    cal = settings.data["market_calendar"]
    ahora = now or datetime.now(timezone.utc)
    sesion = last_completed_session(ahora, cal["close_time"], int(cal["data_ready_delay_minutes"]), cal["timezone"])
    summary.set("expected_session", sesion.isoformat())
    summary.set("mode", settings.mode)
    if not force and snapshot_exists(paths.history, sesion):
        log.info("La sesión %s ya está procesada: no hay nada nuevo que hacer", sesion)
        summary.set("status", "skipped")
        return {"status": "skipped", "as_of": sesion.isoformat()}

    universo = load_universe(settings)
    meta_universo = universo.set_index("symbol").to_dict("index")
    trader = PaperTrader(paths.state, settings)
    manuales = load_manual_positions()
    seguidos = list(dict.fromkeys([t["symbol"] for t in trader.by_status("PENDING", "OPEN")]
                                  + [m["symbol"] for m in manuales]))
    referencias = list(settings.universe["universe"]["reference_symbols"])
    simbolos = list(dict.fromkeys(universo["symbol"].tolist() + seguidos))
    proveedor = create_market_provider(settings, synthetic_end=sesion)
    update_market_data(settings, simbolos + referencias, proveedor, as_of=sesion)
    mercado = load_market_data(settings, simbolos, as_of=sesion)
    as_of = mercado.as_of
    summary.set("data_as_of", as_of.strftime("%Y-%m-%d"))
    if as_of.date() < sesion:
        summary.warning("DATA", f"El proveedor aún no publica la sesión {sesion}; se usa la última disponible "
                                f"({as_of.date()})")
    if not force and snapshot_exists(paths.history, as_of):
        log.info("Los datos al %s ya estaban procesados", as_of.date())
        summary.set("status", "skipped")
        return {"status": "skipped", "as_of": as_of.strftime("%Y-%m-%d")}

    # Indicadores, features, etiquetas históricas y scores
    utilizables = [s for s in mercado.usable_for_signals() if s not in referencias]
    analizar = list(dict.fromkeys(utilizables + [s for s in seguidos if s in mercado.frames]))
    analisis = analyze_universe(mercado, analizar, settings, with_labels=True)
    filtros = settings.universe["filters"]
    elegibles = []
    for s in utilizables:
        a = analisis.get(s)
        if a is None or a.ind.index[-1] != as_of:
            continue
        if bool(a.tradable(float(filtros["min_price"]), float(filtros["min_avg_dollar_volume"])).iloc[-1]):
            elegibles.append(s)
    summary.set("symbols_eligible", len(elegibles))
    log.info("%d acciones elegibles (liquidez y calidad) al %s", len(elegibles), as_of.date())

    # Red neuronal en producción
    registro = ModelRegistry(paths.models)
    modelo, meta = registro.load_production()
    fcfg = settings.model["features"]
    disponible, motivo = compatible(meta, fcfg["version"], int(fcfg["lookback"]))
    if modelo is None:
        disponible = False
    summary.set("model_version", meta["model_version"] if disponible else None)
    if not disponible:
        log.info("Red neuronal no disponible: %s", motivo or "no se pudo cargar")
    nn_cfg = settings.model["neural_network"]
    objetivo = list(dict.fromkeys(elegibles + [s for s in seguidos if s in analisis]))
    pred = pd.DataFrame()
    if disponible:
        pred = predict_latest(modelo, meta, {s: analisis[s].features for s in objetivo},
                              mc_passes=int(nn_cfg["mc_dropout_passes"])).set_index("symbol")
    base = float(meta["base_rate"]) if disponible else None
    umbral = neural_gate(None, base, settings.rules) if disponible else None

    pesos = settings.model["ranking"]["weights"]
    registros: dict[str, dict] = {}
    for s in objetivo:
        a = analisis[s]
        ind, sc = a.ind.iloc[-1], a.scores.iloc[-1]
        info = meta_universo.get(s, {})
        rec = {
            "symbol": s, "name": info.get("name") or s, "sector": info.get("sector") or "",
            "industry": info.get("industry") or "", "indices": info.get("indices") or "",
            "price": _f(mercado.raw[s]["close"].iloc[-1]), "change_1d": _f(ind["ret1"]),
            "data_date": a.ind.index[-1].strftime("%Y-%m-%d"), "eligible": s in elegibles,
            **{k: _f(sc[k]) for k in ("statistical_score", "trend_score", "momentum_score", "technical_score",
                                      "volume_score", "risk_score", "history_score")},
            "news_score": None, "fundamental_score": None, "neural_score": None, "opportunity_probability": None,
            "confidence": None, "expected_return": None, "expected_drawdown": None,
            "neural_threshold": umbral,
        }
        if disponible and s in pred.index:
            p = pred.loc[s]
            rec.update({"neural_score": _f(p["neural_score"]), "opportunity_probability": _f(p["probability"]),
                        "confidence": _f(p["confidence"]), "expected_return": _f(p["expected_return"]),
                        "expected_drawdown": _f(p["expected_drawdown"]), "neural_lift": _f(p["lift"])})
        rec["preliminary_score"] = combine_scores({"statistical": rec["statistical_score"],
                                                   "neural": rec["neural_score"], "risk": rec["risk_score"]},
                                                  pesos)["final_score"]
        registros[s] = rec

    # Preselección para el análisis profundo (noticias, fundamentales, explicación de la red)
    n_pre = int(settings.model["ranking"]["preliminary_candidates"])
    ordenados = sorted((s for s in elegibles if registros[s]["preliminary_score"] is not None),
                       key=lambda s: registros[s]["preliminary_score"], reverse=True)
    profundos = list(dict.fromkeys([s for s in seguidos if s in registros] + ordenados[:n_pre]))
    profundos = profundos[: max(int(settings.data["news"]["max_symbols_per_run"]), len(seguidos))]
    if disponible and profundos:
        explicadas = predict_latest(modelo, meta, {s: analisis[s].features for s in profundos}, mc_passes=0,
                                    explain=profundos).set_index("symbol")
        for s in profundos:
            if s in explicadas.index:
                registros[s]["neural_explanation"] = explicadas.loc[s, "explanation"]
    proveedores = create_news_providers(settings)
    for prov in proveedores:
        if getattr(prov, "is_synthetic", False) and getattr(prov, "now", 0) is None:
            prov.now = ahora  # las noticias sintéticas nunca son posteriores al momento de decisión
    fundamentales = create_fundamentals_provider(settings)
    ncfg = settings.data["news"]
    detalles, noticias_archivo = {}, []
    for s in profundos:
        rec = registros[s]
        items = []
        for prov in proveedores:
            try:
                items.extend(prov.fetch(s, rec["name"], int(ncfg["lookback_days"]), int(ncfg["max_items_per_symbol"])))
            except Exception as exc:  # una fuente caída no detiene nada
                summary.warning("NEWS", f"{prov.name} falló: {exc}", s)
        noticias = score_news(items, s, rec["name"], ncfg, as_of=ahora)
        summary.count("news_items_processed", noticias["count"])
        try:
            fund = fundamental_analysis(fundamentales.fetch(s), int(settings.data["fundamentals"]["min_metrics"]))
        except Exception as exc:
            summary.warning("DATA", f"fundamentales no disponibles: {exc}", s)
            fund = fundamental_analysis({})
        rec["news_score"] = noticias["news_score"]
        rec["news_label"] = noticias["label"]
        rec["fundamental_score"] = fund["fundamental_score"]
        neural = None
        if rec.get("opportunity_probability") is not None:
            neural = {"probability": rec["opportunity_probability"], "lift": rec.get("neural_lift"),
                      "explanation": rec.get("neural_explanation")}
        a = analisis[s]
        rec["explanation"] = explain_symbol(a.ind.iloc[-1], a.scores.iloc[-1], settings, noticias, fund, neural)
        detalles[s] = {"news": {k: v for k, v in noticias.items()}, "fundamentals": fund, "neural": neural,
                       "explanation": rec["explanation"]}
        for item in noticias["items"]:
            noticias_archivo.append({**item, "key": "".join(ch for ch in item["title"].lower() if ch.isalnum())[:90]})

    # Ranking final
    for rec in registros.values():
        comb = combine_scores({"statistical": rec["statistical_score"], "news": rec["news_score"],
                               "fundamental": rec["fundamental_score"], "neural": rec["neural_score"],
                               "risk": rec["risk_score"]}, pesos)
        rec["final_score"] = comb["final_score"]
        rec["breakdown"] = comb
        if rec["symbol"] in detalles:
            detalles[rec["symbol"]]["breakdown"] = comb
    ocfg = settings.strategy["opportunities"]
    candidatas = sorted((registros[s] for s in elegibles if registros[s]["final_score"] is not None),
                        key=lambda r: r["final_score"], reverse=True)
    for i, rec in enumerate(candidatas, 1):
        rec["rank"] = i
    oportunidades = [r for r in candidatas if r["final_score"] >= float(ocfg["min_final_score"])
                     and r["symbol"] in detalles][: int(ocfg["top_n"])]
    for rec in oportunidades:
        rec["opportunity"] = True
    summary.set("opportunities", len(oportunidades))

    # Paper trading
    flags = {s: trend_break_flags(a.ind) for s, a in analisis.items()}
    eventos = trader.process(as_of, mercado.raw, flags, mercado.calendar)
    if settings.mode == "PAPER_TRADING":
        eventos += trader.create_signals(as_of, [r for r in candidatas if r["symbol"] in detalles], disponible,
                                         meta["model_version"] if disponible else None)
    trader.refresh_tracking(registros, as_of)
    trader.save()
    summary.set("open_trades", len(trader.by_status("OPEN")))
    summary.set("pending_trades", len(trader.by_status("PENDING")))

    # Snapshot e historial
    nuevas = archive_news(paths.news, noticias_archivo)
    summary.count("news_archived", nuevas)
    tabla = pd.DataFrame([{**r, "opportunity": bool(r.get("opportunity", False))} for r in registros.values()])
    compactas = [{k: r.get(k) for k in ("rank", "symbol", "name", "sector", "price", "change_1d", "final_score",
                                         "statistical_score", "news_score", "fundamental_score", "neural_score",
                                         "risk_score", "opportunity_probability", "confidence", "expected_return",
                                         "expected_drawdown", "news_label")}
                 | {"summary": r["explanation"]["summary"], "breakdown": r["breakdown"]} for r in oportunidades]
    rechazados = {s: r.issues for s, r in mercado.reports.items() if not r.usable_for_signals}
    snapshot = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": settings.mode,
        "data_source": "synthetic" if proveedor.is_synthetic else settings.data["market_data"]["provider"],
        "config_hash": settings.config_hash(),
        "features_version": fcfg["version"],
        "model": {"available": disponible, "version": meta["model_version"] if disponible else None,
                  "reason": None if disponible else motivo, "base_rate": base, "threshold": umbral},
        "universe_size": int(len(universo)),
        "symbols_analyzed": len(analisis),
        "symbols_eligible": len(elegibles),
        "symbols_skipped": len(rechazados),
        "deep_analyzed": len(detalles),
        "news_providers": [p.name for p in proveedores],
        "market": market_overview(mercado, settings),
        "opportunities": compactas,
        "events": eventos,
        "ranking_weights": pesos,
    }
    write_snapshot(paths.history, as_of, snapshot, tabla, detalles)
    write_json(paths.results / "data_quality.json", {
        "as_of": as_of.strftime("%Y-%m-%d"), "generated_at": snapshot["generated_at"],
        "checked": len(mercado.reports), "rejected": len(rechazados),
        "rejected_symbols": rechazados,
        "reports": {s: r.to_dict() for s, r in mercado.reports.items() if r.issues},
    })
    write_json(paths.state / "latest.json", {k: snapshot[k] for k in ("as_of", "generated_at", "mode", "data_source",
                                                                       "model", "symbols_analyzed", "symbols_eligible")}
               | {"opportunities": len(oportunidades)})
    summary.set("status", "ok")
    summary.write(paths.results / "last_run_summary.json")
    return {"status": "ok", "as_of": snapshot["as_of"], "opportunities": len(oportunidades), "events": eventos}
