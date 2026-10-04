"""News & Sentiment Engine: news_score y explicación legible.

La explicación se construye SOLO con datos presentes en los titulares: cuántos
hay, su sentimiento, los temas que aparecen literalmente y los titulares más
extremos citados tal cual. No se inventa información.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from ..data.providers.base import NewsItem
from ..data.providers.news import short_company_name
from .sentiment import analyzer

POSITIVE, NEGATIVE = 0.15, -0.15

# Temas detectados por palabras presentes en el titular (no se infieren)
THEMES = {
    "resultados trimestrales": ("earnings", "eps", "quarter", "quarterly", "results"),
    "ingresos o ventas": ("revenue", "sales"),
    "previsiones (guidance)": ("guidance", "outlook", "forecast"),
    "recomendaciones o precios objetivo de analistas": ("upgrade", "downgrade", "price target", "rating",
                                                         "analyst", "analysts", "outperform", "underperform"),
    "temas legales o regulatorios": ("lawsuit", "probe", "investigation", " sec ", "antitrust", "fined",
                                     "regulator", "regulators", "subpoena"),
    "acuerdos, adquisiciones o contratos": ("acquisition", "acquire", "merger", "deal", "partnership",
                                            "contract"),
    "productos o aprobaciones": ("launch", "launches", "unveil", "unveils", "approval", "approved", "fda"),
    "dividendos, recompras o financiación": ("buyback", "dividend", "offering", "repurchase"),
    "reestructuración o despidos": ("layoffs", "job cuts", "restructuring"),
    "inteligencia artificial": (" ai ", "artificial intelligence", " ai-"),
}


def _temas(titulo: str) -> list[str]:
    texto = f" {titulo.lower()} "
    return [tema for tema, claves in THEMES.items() if any(k in texto for k in claves)]


def _relevancia(item: NewsItem, symbol: str, company: str) -> float:
    titulo = item.title.lower()
    nombre = short_company_name(company).lower()
    if re.search(rf"\b{re.escape(symbol.lower())}\b", titulo) or (nombre and nombre in titulo):
        return 1.0
    return 0.6  # la fuente la asoció al símbolo pero no lo menciona en el titular


def dedupe(items: list[NewsItem]) -> list[NewsItem]:
    vistos, salida = set(), []
    for item in items:
        clave = item.key()
        if clave and clave not in vistos:
            vistos.add(clave)
            salida.append(item)
    return salida


def score_news(items: list[NewsItem], symbol: str, company: str, cfg: dict,
               as_of: datetime | None = None) -> dict:
    """Analiza noticias publicadas hasta `as_of` (nunca posteriores) dentro de la ventana."""
    ahora = as_of or datetime.now(timezone.utc)
    if ahora.tzinfo is None:
        ahora = ahora.replace(tzinfo=timezone.utc)
    limite = ahora - timedelta(days=int(cfg["lookback_days"]))
    vida_media = float(cfg["half_life_days"])
    analizadas = []
    for item in dedupe(items):
        fecha = item.published_at
        if fecha is not None and fecha.tzinfo is None:
            fecha = fecha.replace(tzinfo=timezone.utc)
        if fecha is None or fecha > ahora or fecha < limite:
            continue  # sin fecha fiable, futura respecto a la decisión o demasiado antigua
        texto = f"{item.title}. {item.summary[:300]}" if item.summary else item.title
        sentimiento = analyzer().score(texto)
        edad = max(0.0, (ahora - fecha).total_seconds() / 86400)
        recencia = 0.5 ** (edad / vida_media)
        relevancia = _relevancia(item, symbol, company)
        analizadas.append({
            **item.to_dict(), "sentiment": round(sentimiento, 4), "relevance": relevancia,
            "recency_weight": round(recencia, 4), "themes": _temas(item.title),
        })
    analizadas.sort(key=lambda n: n["published_at"], reverse=True)
    analizadas = analizadas[: int(cfg["max_items_per_symbol"])]
    if not analizadas:
        return {"news_score": None, "sentiment": None, "count": 0, "positive": 0, "negative": 0, "neutral": 0,
                "concentration": None, "themes": [], "items": [], "label": "Sin noticias recientes",
                "explanation": "No se encontraron noticias recientes con fecha fiable para esta acción."}
    pesos = [n["recency_weight"] * n["relevance"] for n in analizadas]
    total = sum(pesos) or 1.0
    medio = sum(n["sentiment"] * w for n, w in zip(analizadas, pesos)) / total
    efectivo = sum(n["relevance"] for n in analizadas)
    ajustado = medio * efectivo / (efectivo + 2)  # con pocas noticias se acerca a neutral
    positivas = sum(n["sentiment"] >= POSITIVE for n in analizadas)
    negativas = sum(n["sentiment"] <= NEGATIVE for n in analizadas)
    neutras = len(analizadas) - positivas - negativas
    concentracion = max(positivas, negativas) / len(analizadas)
    if ajustado >= 0.2:
        etiqueta = "Muy positivo"
    elif ajustado >= 0.06:
        etiqueta = "Positivo"
    elif ajustado <= -0.2:
        etiqueta = "Muy negativo"
    elif ajustado <= -0.06:
        etiqueta = "Negativo"
    else:
        etiqueta = "Neutral"
    temas: dict[str, int] = {}
    for n in analizadas:
        for t in n["themes"]:
            temas[t] = temas.get(t, 0) + 1
    temas_ordenados = sorted(temas, key=temas.get, reverse=True)
    return {
        "news_score": round(max(0.0, min(100.0, 50 + 50 * ajustado * 1.5)), 1),
        "sentiment": round(ajustado, 4),
        "count": len(analizadas),
        "positive": int(positivas),
        "negative": int(negativas),
        "neutral": int(neutras),
        "concentration": round(concentracion, 3),
        "themes": temas_ordenados,
        "items": analizadas,
        "label": etiqueta,
        "explanation": _explicacion(etiqueta, ajustado, analizadas, positivas, negativas, neutras,
                                    temas_ordenados, int(cfg["lookback_days"])),
    }


def _explicacion(etiqueta, sentimiento, items, pos, neg, neu, temas, dias) -> str:
    texto = (f"Sentimiento reciente {etiqueta.lower()} ({sentimiento:+.2f}) calculado sobre {len(items)} "
             f"titulares de los últimos {dias} días: {pos} positivos, {neg} negativos y {neu} neutros.")
    if temas:
        texto += f" Temas que aparecen en los titulares: {', '.join(temas[:3])}."
    mejor = max(items, key=lambda n: n["sentiment"])
    peor = min(items, key=lambda n: n["sentiment"])
    if mejor["sentiment"] >= POSITIVE:
        texto += f" Titular más positivo: “{mejor['title']}” ({mejor['source']})."
    if peor["sentiment"] <= NEGATIVE:
        texto += f" Titular más negativo: “{peor['title']}” ({peor['source']})."
    return texto + " (Análisis léxico automático de titulares, no una valoración de la empresa.)"
