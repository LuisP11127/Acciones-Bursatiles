"""Interfaces de proveedores de datos.

El resto del sistema solo depende de estas clases: para cambiar de fuente
(Polygon, Alpha Vantage, Finnhub...) basta con implementar otra subclase y
registrarla en `src/data/providers/__init__.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import date, datetime

import pandas as pd

# Columnas estándar que devuelve todo proveedor de mercado (índice = fecha de la sesión)
MARKET_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]


class MarketDataProvider(ABC):
    name: str = "base"
    is_synthetic: bool = False

    @abstractmethod
    def fetch_history(self, symbols: list[str], start: date | None,
                      end: date | None = None) -> dict[str, pd.DataFrame]:
        """Devuelve {símbolo: DataFrame con MARKET_COLUMNS}. Los símbolos sin datos se omiten."""


@dataclass
class NewsItem:
    symbol: str
    source: str
    title: str
    url: str
    published_at: datetime | None
    summary: str = ""
    provider: str = ""

    def key(self) -> str:
        import re

        return re.sub(r"[^a-z0-9]", "", self.title.lower())[:90]

    def to_dict(self) -> dict:
        datos = asdict(self)
        datos["published_at"] = self.published_at.isoformat() if self.published_at else None
        return datos


class NewsProvider(ABC):
    name: str = "base"
    is_synthetic: bool = False

    @abstractmethod
    def fetch(self, symbol: str, company_name: str, lookback_days: int, max_items: int) -> list[NewsItem]:
        """Noticias recientes de un símbolo. Debe devolver [] (no lanzar) si la fuente falla."""


class FundamentalsProvider(ABC):
    name: str = "base"
    is_synthetic: bool = False

    @abstractmethod
    def fetch(self, symbol: str) -> dict:
        """Métricas fundamentales disponibles (las ausentes no se incluyen)."""
