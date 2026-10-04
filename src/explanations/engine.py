"""Explanation Engine.

Cada razón se construye a partir de valores calculados (precio, medias, RSI...)
y se muestra con su número real. Se separan claramente:
  - facts: datos objetivos derivados de precios, noticias o del proveedor;
  - model: explicación del MODELO (qué variables movieron su probabilidad),
    que no es una afirmación sobre la empresa.
`why_it_matters` explica qué papel tiene la señal dentro del sistema.
"""

from __future__ import annotations

import math

import pandas as pd

from ..fundamentals.scoring import LABELS as FUND_LABELS

FACTS_DISCLAIMER = ("Datos objetivos calculados con precios y volúmenes hasta la fecha indicada, titulares "
                    "publicados y datos del proveedor. No son predicciones.")
MODEL_DISCLAIMER = ("Explicación del modelo: describe cómo llegó la red neuronal a su probabilidad. No es una "
                    "afirmación objetiva sobre la empresa.")


def _ok(*valores) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in valores)


def _pct(x: float, signo: bool = True) -> str:
    return f"{x * 100:{'+' if signo else ''}.1f}%"


def _peso(settings, componente: str) -> str:
    pesos = settings.model["statistical"]["weights"]
    total = sum(float(v) for v in pesos.values()) or 1.0
    return f"{float(pesos.get(componente, 0)) / total * 100:.0f}%"


def explain_symbol(ind: pd.Series, scores: pd.Series, settings, news: dict | None = None,
                   fundamentals: dict | None = None, neural: dict | None = None) -> dict:
    """`ind` y `scores` son la última fila (sesión de decisión) del análisis."""
    g = lambda k: (None if k not in ind or pd.isna(ind[k]) else float(ind[k]))  # noqa: E731
    sc = lambda k: (None if k not in scores or pd.isna(scores[k]) else float(scores[k]))  # noqa: E731
    pos: list[dict] = []
    neg: list[dict] = []

    def add(lista, code, category, text, why):
        lista.append({"code": code, "category": category, "text": text, "why_it_matters": why})

    c, ma7, ma25, ma99 = g("close"), g("ma7"), g("ma25"), g("ma99")
    w_trend, w_mom, w_tec = _peso(settings, "trend"), _peso(settings, "momentum"), _peso(settings, "technical")
    w_vol, w_risk, w_hist = _peso(settings, "volume"), _peso(settings, "risk"), _peso(settings, "history")

    if _ok(c, ma99):
        if c > ma99:
            add(pos, "price_above_ma99", "tendencia", f"Precio {_pct(c / ma99 - 1)} sobre la MA(99) ({c:.2f} vs {ma99:.2f})",
                f"La MA(99) es la referencia de tendencia de largo plazo (~5 meses). Estar por encima suma al "
                f"trend_score (peso {w_trend} del statistical_score).")
        else:
            add(neg, "price_below_ma99", "tendencia", f"Precio {_pct(c / ma99 - 1)} bajo la MA(99) ({c:.2f} vs {ma99:.2f})",
                f"Indica tendencia de largo plazo débil y resta en el trend_score (peso {w_trend}).")
    if _ok(c, ma25):
        if c > ma25:
            add(pos, "price_above_ma25", "tendencia", f"Precio {_pct(c / ma25 - 1)} sobre la MA(25)",
                "La MA(25) resume la tendencia de medio plazo (~5 semanas); cotizar por encima indica fortaleza reciente.")
        else:
            add(neg, "price_below_ma25", "tendencia", f"Precio {_pct(c / ma25 - 1)} bajo la MA(25)",
                "Cotizar bajo la MA(25) indica debilidad de medio plazo.")
    if _ok(ma7, ma25) and ma7 > ma25:
        add(pos, "ma7_above_ma25", "tendencia", "La MA(7) está por encima de la MA(25)",
            f"Impulso de corto plazo superior al de medio plazo; suma al trend_score (peso {w_trend}).")
    if _ok(ma25, ma99) and ma25 > ma99:
        add(pos, "ma25_above_ma99", "tendencia", "La MA(25) está por encima de la MA(99)",
            f"Estructura de tendencia alcista de medio frente a largo plazo; suma al trend_score (peso {w_trend}).")
    barras = g("bars_since_cross_7_25")
    if _ok(barras, ma7, ma25) and barras <= 5:
        if ma7 > ma25:
            add(pos, "golden_cross_7_25", "tendencia", f"La MA(7) cruzó por encima de la MA(25) hace {int(barras)} sesiones",
                f"Un cruce alcista reciente suma al technical_score (peso {w_tec}); marca un posible inicio de impulso.")
        else:
            add(neg, "death_cross_7_25", "tendencia", f"La MA(7) cruzó por debajo de la MA(25) hace {int(barras)} sesiones",
                f"Un cruce bajista reciente resta en el technical_score (peso {w_tec}).")
    barras99 = g("bars_since_golden_25_99")
    if _ok(barras99, ma25, ma99) and barras99 <= 20 and ma25 > ma99:
        add(pos, "golden_cross_25_99", "tendencia", f"La MA(25) cruzó por encima de la MA(99) hace {int(barras99)} sesiones",
            "Cruce de medias de medio y largo plazo: cambio de tendencia hacia alcista en el modelo estadístico.")
    s99 = g("slope_ma99")
    if _ok(s99):
        if s99 > 0:
            add(pos, "ma99_rising", "tendencia", f"La MA(99) sube ({_pct(s99)} en 10 sesiones)",
                f"Pendiente positiva de la tendencia de largo plazo; suma al trend_score (peso {w_trend}).")
        elif s99 < -0.01:
            add(neg, "ma99_falling", "tendencia", f"La MA(99) baja ({_pct(s99)} en 10 sesiones)",
                "Pendiente negativa de la tendencia de largo plazo.")
    r63 = g("ret63")
    if _ok(r63):
        if r63 > 0:
            add(pos, "positive_momentum", "momentum", f"Rentabilidad de 3 meses {_pct(r63)}",
                f"Momentum positivo; suma al momentum_score (peso {w_mom}).")
        elif r63 < -0.10:
            add(neg, "negative_momentum", "momentum", f"Rentabilidad de 3 meses {_pct(r63)}",
                f"Momentum negativo; resta en el momentum_score (peso {w_mom}).")
    rs = g("rel_strength63")
    if _ok(rs):
        if rs > 0:
            add(pos, "relative_strength", "momentum", f"Supera al S&P 500 en {rs * 100:+.1f} puntos en 3 meses",
                "La fuerza relativa frente al índice forma parte del momentum_score.")
        elif rs < -0.10:
            add(neg, "relative_weakness", "momentum", f"Queda {rs * 100:+.1f} puntos por detrás del S&P 500 en 3 meses",
                "Debilidad relativa frente al índice; resta en el momentum_score.")
    hist = g("macd_hist")
    if _ok(hist):
        if hist > 0:
            add(pos, "macd_positive", "momentum", "MACD por encima de su línea de señal (histograma positivo)",
                "Aceleración alcista del precio según el MACD; suma al momentum_score.")
        else:
            add(neg, "macd_negative", "momentum", "MACD por debajo de su línea de señal",
                "Pérdida de impulso según el MACD; resta en el momentum_score.")
    rsi = g("rsi14")
    if _ok(rsi):
        if 45 <= rsi <= 65:
            add(pos, "rsi_healthy", "técnico", f"RSI(14) en {rsi:.0f}: fuerza sin sobrecompra",
                f"El technical_score (peso {w_tec}) favorece un RSI intermedio.")
        elif rsi > 75:
            add(neg, "rsi_overbought", "técnico", f"RSI(14) en {rsi:.0f}: sobrecompra",
                "Un RSI muy alto resta en el technical_score: aumenta el riesgo de corrección a corto plazo.")
        elif rsi < 30:
            add(neg, "rsi_oversold", "técnico", f"RSI(14) en {rsi:.0f}: sobreventa",
                "Un RSI muy bajo indica presión vendedora intensa.")
    d25 = g("dist_ma25")
    if _ok(d25, ma25, ma99) and 0 <= d25 <= 0.03 and ma25 > ma99:
        add(pos, "pullback_ma25", "técnico", f"Precio a {_pct(d25, False)} de la MA(25) dentro de tendencia alcista",
            "Un retroceso cerca de la MA(25) en tendencia alcista puntúa alto en el technical_score.")
    if _ok(d25) and d25 > 0.12:
        add(neg, "overextended", "técnico", f"Precio {_pct(d25)} sobre la MA(25): muy extendido",
            "Alejarse mucho de la MA(25) resta en el technical_score (riesgo de corrección).")
    dh = g("dist_high252")
    if _ok(dh, c, ma99) and dh >= -0.05 and c > ma99:
        add(pos, "near_52w_high", "técnico", f"A {abs(dh) * 100:.1f}% del máximo de 52 semanas",
            "Cotizar cerca de máximos en tendencia alcista indica fortaleza.")
    rv, r1, vt, r5 = g("rel_volume"), g("ret1"), g("vol_trend"), g("ret5")
    if _ok(rv, r1) and rv >= 1.3 and r1 > 0:
        add(pos, "volume_up_day", "volumen", f"Volumen de la sesión {rv:.1f}× su media de 20 sesiones, con subida",
            f"Subidas con volumen alto suman al volume_score (peso {w_vol}).")
    if _ok(vt, r5):
        if vt >= 1.2 and r5 > 0:
            add(pos, "volume_trend_up", "volumen", f"Volumen medio de 5 sesiones {vt:.1f}× el de 50, con precio al alza",
                f"Volumen creciente acompañando la subida; suma al volume_score (peso {w_vol}).")
        elif vt >= 1.2 and r5 < 0:
            add(neg, "volume_on_decline", "volumen", f"Volumen creciente ({vt:.1f}× la media) con el precio a la baja",
                f"Volumen alto en caídas resta en el volume_score (peso {w_vol}).")
    v20 = g("vol20")
    if _ok(v20):
        if v20 > 0.5:
            add(neg, "high_volatility", "riesgo", f"Volatilidad anualizada de {v20 * 100:.0f}%",
                f"Más volatilidad = más riesgo; reduce el risk_score (peso {w_risk}).")
        elif v20 < 0.25:
            add(pos, "low_volatility", "riesgo", f"Volatilidad anualizada moderada ({v20 * 100:.0f}%)",
                f"Menor volatilidad mejora el risk_score (peso {w_risk}).")
    dd = g("drawdown252")
    if _ok(dd) and dd < -0.25:
        add(neg, "deep_drawdown", "riesgo", f"{abs(dd) * 100:.0f}% por debajo de su máximo de 52 semanas",
            "Un drawdown profundo reduce el risk_score.")
    hs = sc("history_score")
    if _ok(hs):
        if hs >= 55:
            add(pos, "history_good", "histórico", f"El {hs:.0f}% de sus señales pasadas en tendencia alcista fueron oportunidades",
                f"Calculado solo con operaciones simuladas ya cerradas (history_score, peso {w_hist}).")
        elif hs <= 35:
            add(neg, "history_weak", "histórico", f"Solo el {hs:.0f}% de sus señales pasadas en tendencia alcista fueron oportunidades",
                f"Comportamiento histórico débil de la acción en este contexto (history_score, peso {w_hist}).")
    if news and news.get("news_score") is not None:
        etiqueta = news.get("label", "")
        if "positivo" in etiqueta.lower():
            add(pos, "news_positive", "noticias", f"Noticias recientes con sentimiento {etiqueta.lower()} "
                f"({news['count']} titulares)", "El sentimiento de los titulares alimenta el news_score del ranking.")
        elif "negativo" in etiqueta.lower():
            add(neg, "news_negative", "noticias", f"Noticias recientes con sentimiento {etiqueta.lower()} "
                f"({news['count']} titulares)", "Un sentimiento negativo reduce el news_score del ranking.")
    if neural and _ok(neural.get("probability"), neural.get("lift")):
        lift = float(neural["lift"])
        texto = (f"El modelo neuronal estima {neural['probability'] * 100:.1f}% de probabilidad de oportunidad "
                 f"({lift:.2f}× su tasa base histórica)")
        if lift >= 1.15:
            add(pos, "neural_high", "modelo", texto, "Situación parecida a otras que históricamente fueron buenas "
                "oportunidades según la red (alimenta el neural_score). " + MODEL_DISCLAIMER)
        elif lift < 0.9:
            add(neg, "neural_low", "modelo", texto, "La red ve la situación menos favorable que la media. "
                + MODEL_DISCLAIMER)

    hechos_fund = []
    if fundamentals:
        for clave, valor in (fundamentals.get("values") or {}).items():
            if clave not in FUND_LABELS or valor is None:
                continue
            if clave in ("revenue_growth", "earnings_growth", "profit_margin", "roe", "roic"):
                texto = _pct(float(valor))
            elif clave in ("market_cap", "free_cash_flow"):
                texto = f"${float(valor) / 1e9:,.1f} mil millones"
            else:
                texto = f"{float(valor):.2f}"
            hechos_fund.append(f"{FUND_LABELS[clave]}: {texto} (dato del proveedor)")

    resumen = "; ".join(r["text"] for r in pos[:6])
    return {
        "summary": f"Seleccionada porque: {resumen}." if pos else "Sin señales positivas destacadas.",
        "reasons": pos,
        "risks": neg,
        "news_explanation": (news or {}).get("explanation"),
        "fundamental_facts": hechos_fund,
        "model_explanation": (neural or {}).get("explanation"),
        "disclaimers": {"facts": FACTS_DISCLAIMER, "model": MODEL_DISCLAIMER},
    }
