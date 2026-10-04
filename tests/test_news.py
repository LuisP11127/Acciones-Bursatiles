"""News & Sentiment Engine y proveedores (con respuestas simuladas)."""

import sys
import types
from datetime import datetime, timedelta, timezone

from src.data.providers import news as news_mod
from src.data.providers.base import NewsItem
from src.news.scoring import score_news

CFG = {"lookback_days": 14, "half_life_days": 3, "max_items_per_symbol": 15}
AHORA = datetime(2026, 10, 2, 21, 0, tzinfo=timezone.utc)


def item(title, hours=5, source="Reuters"):
    return NewsItem("AAPL", source, title, "https://x", AHORA - timedelta(hours=hours))


def test_sentiment_score_and_explanation_from_real_headlines():
    buenas = [item("Apple beats earnings estimates and raises guidance"), item("Analysts upgrade Apple", 30),
              item("Apple announces dividend increase", 50)]
    r = score_news(buenas, "AAPL", "Apple Inc.", CFG, as_of=AHORA)
    assert r["news_score"] > 55 and r["label"] in ("Positivo", "Muy positivo") and r["positive"] >= 2
    assert "resultados trimestrales" in r["themes"]
    # la explicación solo cita titulares existentes
    citados = [t for t in (n["title"] for n in r["items"]) if f"“{t}”" in r["explanation"]]
    assert citados and all(any(c == n["title"] for n in r["items"]) for c in citados)
    malas = [item("Apple misses revenue, shares plunge"), item("Regulators open fraud probe into Apple", 10)]
    r = score_news(malas, "AAPL", "Apple Inc.", CFG, as_of=AHORA)
    assert r["news_score"] < 45 and "negativo" in r["label"].lower()


def test_dedupe_recency_and_empty():
    lista = [item("Same headline"), item("Same headline!"), item("Very old", hours=24 * 40)]
    r = score_news(lista, "AAPL", "Apple Inc.", CFG, as_of=AHORA)
    assert r["count"] == 1
    vacio = score_news([], "AAPL", "Apple", CFG, as_of=AHORA)
    assert vacio["news_score"] is None and "No se encontraron" in vacio["explanation"]


def test_relevance_mentions_company():
    r = score_news([item("Apple unveils new chip"), item("Markets rally broadly", 2)], "AAPL", "Apple Inc.", CFG,
                   as_of=AHORA)
    relevancia = {n["title"]: n["relevance"] for n in r["items"]}
    assert relevancia["Apple unveils new chip"] == 1.0 and relevancia["Markets rally broadly"] < 1.0


def test_yahoo_provider_parses_both_formats(monkeypatch):
    crudas = [{"id": "1", "content": {"title": "Apple beats estimates", "pubDate": "2026-10-02T14:00:00Z",
                                      "provider": {"displayName": "Reuters"}, "canonicalUrl": {"url": "https://a"}}},
              {"title": "Old format", "publisher": "AP", "link": "https://b", "providerPublishTime": 1790000000},
              {"id": "3", "content": {"title": ""}}]
    monkeypatch.setitem(sys.modules, "yfinance",
                        types.SimpleNamespace(Ticker=lambda s: types.SimpleNamespace(news=crudas)))
    lista = news_mod.YahooNewsProvider().fetch("AAPL", "Apple", 14, 10)
    assert [n.title for n in lista] == ["Apple beats estimates", "Old format"]
    assert lista[0].source == "Reuters" and lista[0].published_at.year == 2026


def test_google_rss_and_finnhub_without_key(monkeypatch):
    rss = b"""<?xml version="1.0"?><rss><channel>
    <item><title>Nvidia shares surge - Reuters</title><link>https://n/1</link>
    <pubDate>Fri, 02 Oct 2026 13:00:00 GMT</pubDate><source url="https://r">Reuters</source></item>
    </channel></rss>"""
    monkeypatch.setattr(news_mod.requests, "get",
                        lambda *a, **k: types.SimpleNamespace(content=rss, raise_for_status=lambda: None))
    lista = news_mod.GoogleNewsRSSProvider().fetch("NVDA", "NVIDIA Corp.", 14, 10)
    assert lista[0].title == "Nvidia shares surge" and lista[0].source == "Reuters"
    assert news_mod.FinnhubNewsProvider(None).fetch("NVDA", "NVIDIA", 14, 10) == []


def test_failing_source_returns_empty(monkeypatch):
    def falla(*a, **k):
        raise TimeoutError("caída")
    monkeypatch.setattr(news_mod.requests, "get", falla)
    assert news_mod.GoogleNewsRSSProvider().fetch("AAPL", "Apple", 14, 5) == []
