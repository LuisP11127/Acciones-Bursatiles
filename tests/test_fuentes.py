"""Lectura de fuentes reales (Yahoo, Google News, Wikipedia) con respuestas simuladas."""

import sys
import types

import numpy as np
import pandas as pd

from bolsa import noticias, universo
from bolsa.datos import _separar_descarga


def _ohlcv(n=150, inicio=100.0):
    fechas = pd.bdate_range("2025-01-01", periods=n, tz="America/New_York")
    c = inicio + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1e6}, index=fechas)


def test_separar_descarga_multiindex_por_ticker():
    crudo = pd.concat({"AAA": _ohlcv(), "BBB": _ohlcv(inicio=50)}, axis=1)  # (Ticker, Price)
    datos = _separar_descarga(crudo, ["AAA", "BBB", "FALTA"])
    assert set(datos) == {"AAA", "BBB"}
    assert datos["BBB"]["Close"].iloc[0] == 50
    assert datos["AAA"].index.tz is None  # fechas sin zona horaria


def test_separar_descarga_multiindex_por_precio_y_simple():
    crudo = pd.concat({"AAA": _ohlcv()}, axis=1).swaplevel(axis=1)  # (Price, Ticker)
    assert "AAA" in _separar_descarga(crudo, ["AAA"])
    assert "AAA" in _separar_descarga(_ohlcv(), ["AAA"])
    assert _separar_descarga(_ohlcv(n=50), ["AAA"]) == {}  # historial demasiado corto


def test_noticias_yahoo_formatos_nuevo_y_antiguo(monkeypatch):
    crudas = [
        {"id": "1", "content": {"title": "Apple beats estimates", "summary": "Strong quarter",
                                "pubDate": "2026-10-02T14:00:00Z", "provider": {"displayName": "Reuters"},
                                "canonicalUrl": {"url": "https://example.com/a"}}},
        {"title": "Old format headline", "publisher": "AP", "link": "https://example.com/b",
         "providerPublishTime": 1790000000},
        {"id": "3", "content": {"title": ""}},
    ]
    falso = types.SimpleNamespace(Ticker=lambda t: types.SimpleNamespace(news=crudas))
    monkeypatch.setitem(sys.modules, "yfinance", falso)
    lista = noticias.noticias_yahoo("AAPL", 10)
    assert [n.titulo for n in lista] == ["Apple beats estimates", "Old format headline"]
    assert lista[0].fuente == "Reuters" and lista[0].enlace == "https://example.com/a"
    assert lista[0].fecha.year == 2026 and lista[1].fecha is not None


RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Nvidia shares surge on AI demand - Reuters</title><link>https://n.example/1</link>
<pubDate>Fri, 02 Oct 2026 13:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>Nvidia faces antitrust probe - Bloomberg</title><link>https://n.example/2</link>
<pubDate>Thu, 01 Oct 2026 10:00:00 GMT</pubDate><source url="https://bloomberg.com">Bloomberg</source></item>
</channel></rss>"""


def test_noticias_google_rss(monkeypatch):
    respuesta = types.SimpleNamespace(content=RSS, raise_for_status=lambda: None)
    monkeypatch.setattr(noticias.requests, "get", lambda *a, **k: respuesta)
    lista = noticias.noticias_google("NVDA stock Nvidia", 10)
    assert [n.titulo for n in lista] == ["Nvidia shares surge on AI demand", "Nvidia faces antitrust probe"]
    assert lista[1].fuente == "Bloomberg"
    assert lista[0].fecha.day == 2


def test_universo_desde_wikipedia(monkeypatch):
    filas = "".join(f"<tr><td>T{i}.B</td><td>Empresa {i}</td><td>Industrials</td><td>Machinery</td></tr>"
                    for i in range(60))
    html = ("<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>"
            f"{filas}</table>")
    respuesta = types.SimpleNamespace(text=html, raise_for_status=lambda: None)
    monkeypatch.setattr(universo.requests, "get", lambda *a, **k: respuesta)
    cfg = {"universo": {"fuentes": ["sp500", "nasdaq100"], "extra": ["zzz"], "max_tickers": 0,
                        "respaldo": "config/universo_respaldo.csv"}}
    resultado = universo.obtener_universo(cfg)
    assert "T0-B" in resultado and resultado["T0-B"]["sector"] == "Industrials"
    assert resultado["T0-B"]["indices"] == ["S&P 500", "Nasdaq-100"]
    assert "ZZZ" in resultado


def test_universo_respaldo_si_falla_wikipedia(monkeypatch):
    def falla(*a, **k):
        raise ConnectionError("sin red")
    monkeypatch.setattr(universo.requests, "get", falla)
    cfg = {"universo": {"fuentes": ["sp500"], "extra": [], "max_tickers": 0,
                        "respaldo": "config/universo_respaldo.csv"}}
    resultado = universo.obtener_universo(cfg)
    assert "AAPL" in resultado and "BRK-B" in resultado
