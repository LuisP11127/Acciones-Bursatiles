"""Actualización incremental de precios y carga de datos validados para el análisis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import Settings
from ..logging_utils import get_logger, summary
from .providers.base import MarketDataProvider
from .storage import load_prices, save_prices
from .validation import QualityReport, adjusted_ohlcv, validate_prices

log = get_logger("DATA")


def merge_incremental(cached: pd.DataFrame, fresh: pd.DataFrame, tolerance: float = 1e-3) -> pd.DataFrame | None:
    """Une la caché con los datos recientes. Devuelve None si los ajustes históricos
    cambiaron (dividendo o split nuevo, revisión del proveedor): hace falta descarga completa."""
    if fresh is None or fresh.empty:
        return None
    comunes = cached.index.intersection(fresh.index)
    if len(comunes) < 3:
        return None
    for col in ("adj_close", "close"):
        ratio = fresh.loc[comunes, col] / cached.loc[comunes, col]
        if not np.all(np.isfinite(ratio)) or float((ratio - 1).abs().max()) > tolerance:
            return None
    nuevos = fresh[fresh.index > cached.index[-1]]
    if (nuevos.get("splits", pd.Series(dtype=float)).fillna(0) != 0).any():
        return None
    if (nuevos.get("dividends", pd.Series(dtype=float)).fillna(0) != 0).any():
        return None
    return pd.concat([cached[cached.index < fresh.index[0]], fresh]).sort_index()


@dataclass
class UpdateResult:
    updated: list[str] = field(default_factory=list)
    full_refresh: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def update_market_data(settings: Settings, symbols: list[str], provider: MarketDataProvider,
                       as_of: date | None = None) -> UpdateResult:
    """Descarga solo lo nuevo; re-descarga todo el historial si los ajustes cambiaron.
    Si el proveedor falla para un símbolo, se conserva la caché (marcada como desactualizada)."""
    cfg = settings.data["market_data"]
    raw = settings.paths.raw
    raw.mkdir(parents=True, exist_ok=True)
    inicio_completo = date.fromisoformat(str(cfg["history_start"]))
    margen = int(cfg["incremental_lookback_days"])
    resultado = UpdateResult()
    cache = {s: load_prices(raw, s) for s in symbols}
    con_cache = [s for s in symbols if cache[s] is not None and len(cache[s]) > 10]
    completos = [s for s in symbols if s not in con_cache]

    if con_cache:
        hoy = as_of or date.today()
        ultima_minima = min(cache[s].index[-1] for s in con_cache).date()
        inicio = max(ultima_minima, hoy - timedelta(days=120)) - timedelta(days=margen)
        recientes = provider.fetch_history(con_cache, inicio, as_of)
        for s in con_cache:
            unido = merge_incremental(cache[s], recientes.get(s)) if s in recientes else None
            if unido is not None:
                save_prices(raw, s, unido)
                resultado.updated.append(s)
            elif s in recientes:
                completos.append(s)
            else:
                resultado.stale.append(s)
    if completos:
        log.info("Descarga completa del historial para %d símbolos", len(completos))
        historicos = provider.fetch_history(completos, inicio_completo, as_of)
        for s in completos:
            if s in historicos:
                save_prices(raw, s, historicos[s])
                resultado.full_refresh.append(s)
            elif cache.get(s) is not None:
                resultado.stale.append(s)
            else:
                resultado.failed.append(s)
    summary.count("symbols_updated", len(resultado.updated) + len(resultado.full_refresh))
    summary.count("symbols_stale", len(resultado.stale))
    summary.count("symbols_failed", len(resultado.failed))
    if resultado.failed:
        summary.warning("DATA", f"Sin datos para {len(resultado.failed)} símbolos: {', '.join(resultado.failed[:15])}")
    log.info("Actualizados %d (completos %d), desactualizados %d, sin datos %d",
             len(resultado.updated), len(resultado.full_refresh), len(resultado.stale), len(resultado.failed))
    return resultado


@dataclass
class MarketData:
    """Precios validados y ajustados listos para el análisis."""

    frames: dict[str, pd.DataFrame]          # OHLCV ajustado
    raw: dict[str, pd.DataFrame]             # datos originales del proveedor (validados)
    reports: dict[str, QualityReport]
    calendar: pd.DatetimeIndex               # sesiones del índice de referencia
    as_of: pd.Timestamp
    benchmark: str

    def usable_for_signals(self) -> list[str]:
        return [s for s, r in self.reports.items() if r.usable_for_signals and s in self.frames]

    def usable_for_training(self) -> list[str]:
        return [s for s, r in self.reports.items() if r.usable_for_training and s in self.frames]


def load_market_data(settings: Settings, symbols: list[str], as_of: date | None = None) -> MarketData:
    """Lee la caché, valida cada símbolo y prepara el OHLCV ajustado."""
    ucfg = settings.universe
    vcfg = settings.data["validation"]
    raw_dir = settings.paths.raw
    benchmark = ucfg["universe"]["benchmark"]
    referencia = load_prices(raw_dir, benchmark)
    if referencia is None or referencia.empty:
        raise RuntimeError(f"No hay datos del índice de referencia {benchmark}: no se puede continuar")
    calendario = pd.DatetimeIndex(referencia.index)
    if as_of is not None:
        calendario = calendario[calendario <= pd.Timestamp(as_of)]
    fecha = calendario[-1]
    frames, crudos, informes = {}, {}, {}
    minimo = int(ucfg["filters"]["min_history_days"])
    for s in dict.fromkeys(list(symbols) + list(ucfg["universe"]["reference_symbols"])):
        df = load_prices(raw_dir, s)
        if df is not None and as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        limpio, rep = validate_prices(s, df, vcfg, min_history=minimo, calendar=calendario, as_of=fecha,
                                      max_stale_sessions=int(ucfg["filters"]["max_stale_days"]))
        informes[s] = rep
        if limpio is not None and len(limpio) > 0:
            crudos[s] = limpio
            frames[s] = adjusted_ohlcv(limpio)
    rechazados = [s for s, r in informes.items() if not r.usable_for_signals]
    summary.count("symbols_rejected_quality", len(rechazados))
    log.info("Datos validados al %s: %d símbolos utilizables, %d rechazados por calidad",
             fecha.date(), len(frames) - len(rechazados), len(rechazados))
    return MarketData(frames=frames, raw=crudos, reports=informes, calendar=calendario, as_of=fecha,
                      benchmark=benchmark)
