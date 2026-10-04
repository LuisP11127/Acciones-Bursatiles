"""Registro de proveedores: aquí se elige la implementación según config/data.yaml."""

from __future__ import annotations

from datetime import date

from ...config import Settings
from .base import FundamentalsProvider, MarketDataProvider, NewsItem, NewsProvider
from .fundamentals import NullFundamentals, SyntheticFundamentals, YFinanceFundamentals
from .market_data import SyntheticMarketProvider, YFinanceProvider
from .news import (FinnhubNewsProvider, GoogleNewsRSSProvider, SyntheticNewsProvider, YahooNewsProvider,
                   news_api_key)

__all__ = ["MarketDataProvider", "NewsProvider", "FundamentalsProvider", "NewsItem",
           "create_market_provider", "create_news_providers", "create_fundamentals_provider"]


def create_market_provider(settings: Settings, synthetic_end: date | None = None) -> MarketDataProvider:
    cfg = settings.data["market_data"]
    nombre = cfg["provider"]
    if nombre == "yfinance":
        return YFinanceProvider(batch_size=int(cfg["batch_size"]), retries=int(cfg["retries"]),
                                backoff=float(cfg["retry_backoff_seconds"]), timeout=float(cfg["timeout_seconds"]))
    if nombre == "synthetic":
        return SyntheticMarketProvider(end=synthetic_end, index_symbols=settings.universe["universe"]["reference_symbols"])
    raise ValueError(f"Proveedor de mercado desconocido: {nombre}")


def create_news_providers(settings: Settings) -> list[NewsProvider]:
    cfg = settings.data["news"]
    proveedores: list[NewsProvider] = []
    for nombre in cfg["providers"]:
        if nombre == "yahoo":
            proveedores.append(YahooNewsProvider())
        elif nombre == "google_rss":
            proveedores.append(GoogleNewsRSSProvider(timeout=float(cfg["request_timeout_seconds"])))
        elif nombre == "finnhub":
            clave = news_api_key()
            if clave:  # sin NEWS_API_KEY la fuente se omite sin error
                proveedores.append(FinnhubNewsProvider(clave, timeout=float(cfg["request_timeout_seconds"])))
        elif nombre == "synthetic":
            proveedores.append(SyntheticNewsProvider())
        else:
            raise ValueError(f"Proveedor de noticias desconocido: {nombre}")
    return proveedores


def create_fundamentals_provider(settings: Settings) -> FundamentalsProvider:
    cfg = settings.data["fundamentals"]
    if not cfg.get("enabled", True):
        return NullFundamentals()
    nombre = cfg["provider"]
    if nombre == "yfinance":
        return YFinanceFundamentals()
    if nombre == "synthetic":
        return SyntheticFundamentals()
    if nombre == "none":
        return NullFundamentals()
    raise ValueError(f"Proveedor de fundamentales desconocido: {nombre}")
