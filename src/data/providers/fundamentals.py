"""Proveedores de datos fundamentales (opcionales: el pipeline sigue sin ellos)."""

from __future__ import annotations

import zlib

import numpy as np

from ...logging_utils import get_logger
from .base import FundamentalsProvider

log = get_logger("DATA")

# Métrica interna -> claves de Yahoo Finance (la primera disponible gana)
YAHOO_FIELDS = {
    "name": ("longName", "shortName"),
    "sector": ("sector",),
    "industry": ("industry",),
    "summary": ("longBusinessSummary",),
    "market_cap": ("marketCap",),
    "revenue_growth": ("revenueGrowth",),
    "earnings_growth": ("earningsGrowth", "earningsQuarterlyGrowth"),
    "profit_margin": ("profitMargins",),
    "pe": ("trailingPE",),
    "forward_pe": ("forwardPE",),
    "peg": ("pegRatio", "trailingPegRatio"),
    "debt_to_equity": ("debtToEquity",),
    "free_cash_flow": ("freeCashflow",),
    "roe": ("returnOnEquity",),
    "roic": (),  # Yahoo no lo publica: queda como "no disponible"
    "target_mean_price": ("targetMeanPrice",),
    "recommendation_mean": ("recommendationMean",),
    "analyst_count": ("numberOfAnalystOpinions",),
}


class YFinanceFundamentals(FundamentalsProvider):
    name = "yfinance"

    def fetch(self, symbol: str) -> dict:
        try:
            import yfinance as yf

            crudo = yf.Ticker(symbol).info or {}
        except Exception as exc:
            log.warning("Fundamentales no disponibles para %s: %s", symbol, exc)
            return {}
        datos = {}
        for campo, claves in YAHOO_FIELDS.items():
            for clave in claves:
                valor = crudo.get(clave)
                if valor not in (None, "", "None", "Infinity"):
                    datos[campo] = valor
                    break
        if "debt_to_equity" in datos:  # Yahoo lo da en porcentaje (150 = 1,5)
            try:
                datos["debt_to_equity"] = float(datos["debt_to_equity"]) / 100.0
            except (TypeError, ValueError):
                datos.pop("debt_to_equity")
        return datos


class SyntheticFundamentals(FundamentalsProvider):
    """Valores inventados para pruebas offline (nunca se publican como reales)."""

    name = "synthetic"
    is_synthetic = True

    def fetch(self, symbol: str) -> dict:
        rng = np.random.default_rng(zlib.crc32(f"fund{symbol}".encode()))
        return {
            "name": symbol, "market_cap": float(rng.uniform(5e9, 2e12)),
            "revenue_growth": float(rng.normal(0.08, 0.1)), "earnings_growth": float(rng.normal(0.1, 0.2)),
            "profit_margin": float(rng.uniform(-0.05, 0.35)), "pe": float(rng.uniform(8, 60)),
            "peg": float(rng.uniform(0.6, 3.5)), "debt_to_equity": float(rng.uniform(0.1, 2.5)),
            "free_cash_flow": float(rng.normal(2e9, 3e9)), "roe": float(rng.normal(0.15, 0.1)),
        }


class NullFundamentals(FundamentalsProvider):
    name = "none"

    def fetch(self, symbol: str) -> dict:
        return {}
