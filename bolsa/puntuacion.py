"""Puntuación del potencial de subida de cada acción.

Cada componente da un puntaje de 0 a 100:
- técnico: estructura de las MA(7), MA(25), MA(99) y posición del precio
- momentum: RSI, MACD, fuerza relativa frente al S&P 500 y volumen
- noticias: sentimiento ponderado de los titulares recientes
- analistas: precio objetivo y recomendación de consenso
- red neuronal: probabilidad de punto de compra (solo cuando la red está lista)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .indicadores import cruce_reciente


def _pct(valor: float) -> str:
    return f"{valor * 100:+.1f}%"


def _limitar(valor: float, minimo: float = 0.0, maximo: float = 100.0) -> float:
    return float(max(minimo, min(maximo, valor)))


def analisis_tecnico(df: pd.DataFrame, mercado: pd.DataFrame | None = None) -> dict | None:
    """Analiza la última vela de un DataFrame con indicadores (ver agregar_indicadores)."""
    if len(df) < 100 or pd.isna(df["MA99"].iloc[-1]):
        return None
    u = df.iloc[-1]
    c = float(u["Close"])
    ma7, ma25, ma99 = float(u["MA7"]), float(u["MA25"]), float(u["MA99"])
    positivas: list[str] = []
    negativas: list[str] = []

    def ret(n: int) -> float:
        return float(c / df["Close"].iloc[-n - 1] - 1) if len(df) > n else 0.0

    ret_1, ret_5, ret_21, ret_63 = ret(1), ret(5), ret(21), ret(63)
    atr_pct = float(u["ATR14"] / c) if pd.notna(u["ATR14"]) else 0.0
    rsi = float(u["RSI14"]) if pd.notna(u["RSI14"]) else 50.0
    max52 = float(u["Max252"]) if pd.notna(u["Max252"]) else float(df["High"].max())
    min52 = float(u["Min252"]) if pd.notna(u["Min252"]) else float(df["Low"].min())
    dist_max = c / max52 - 1
    ma99_sube = len(df) > 120 and ma99 > float(df["MA99"].iloc[-21])
    cruce_7_25 = cruce_reciente(df["MA7"], df["MA25"], 5)
    cruce_25_99 = cruce_reciente(df["MA25"], df["MA99"], 15)
    alcista = c > ma99 and ma25 > ma99

    # ---------------- Técnico (estructura de medias) ----------------
    tec = 0.0
    if c > ma99:
        tec += 20
        positivas.append(f"Precio por encima de la MA(99) ({_pct(c / ma99 - 1)}): tendencia de largo plazo alcista")
    else:
        negativas.append(f"Precio por debajo de la MA(99) ({_pct(c / ma99 - 1)})")
    if ma25 > ma99:
        tec += 15
    if ma7 > ma25:
        tec += 15
    if ma7 > ma25 > ma99:
        positivas.append("Medias alineadas al alza: MA(7) > MA(25) > MA(99)")
    if ma99_sube:
        tec += 10
    if cruce_7_25 > 0:
        tec += 15
        positivas.append("Cruce alcista reciente de la MA(7) sobre la MA(25)")
    elif cruce_7_25 < 0:
        tec -= 10
        negativas.append("Cruce bajista reciente de la MA(7) bajo la MA(25)")
    if cruce_25_99 > 0:
        tec += 10
        positivas.append("Cruce dorado: la MA(25) cruzó por encima de la MA(99)")
    elif cruce_25_99 < 0:
        negativas.append("Cruce de la muerte: la MA(25) cruzó por debajo de la MA(99)")
    toco_ma25 = float(df["Low"].iloc[-3:].min()) <= ma25 * 1.01
    if alcista and c >= ma25 and (c / ma25 - 1 <= 0.03 or toco_ma25):
        tec += 15
        positivas.append("Retroceso a la MA(25) dentro de tendencia alcista: buen punto de entrada")
    if alcista and dist_max >= -0.05:
        tec += 10
        positivas.append(f"A {abs(dist_max) * 100:.1f}% del máximo de 52 semanas: posible ruptura")
    if c / ma25 - 1 > 0.12:
        tec -= 15
        negativas.append(f"Muy extendida sobre la MA(25) ({_pct(c / ma25 - 1)}): riesgo de corrección")
    if c < ma99 and ma25 < ma99:
        tec -= 10
    tec = _limitar(tec)

    # ---------------- Momentum ----------------
    mom = 40.0
    if 50 <= rsi <= 65:
        mom += 15
        positivas.append(f"RSI saludable ({rsi:.0f}): fuerza sin sobrecompra")
    elif 40 <= rsi < 50:
        mom += 5
    elif 65 < rsi <= 72:
        mom += 5
    elif rsi > 72:
        mom -= 15
        negativas.append(f"RSI en sobrecompra ({rsi:.0f})")
    elif rsi < 30:
        if c > ma99:
            mom += 5
            positivas.append(f"RSI en sobreventa ({rsi:.0f}) con tendencia de fondo alcista: posible rebote")
        else:
            mom -= 5
            negativas.append(f"RSI en sobreventa ({rsi:.0f}) en tendencia bajista")
    hist = df["MACD_hist"].dropna()
    if len(hist) >= 4:
        if hist.iloc[-1] > 0:
            mom += 10
        if hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]:
            mom += 5
        cruce_macd = cruce_reciente(df["MACD"], df["MACD_senal"], 5)
        if cruce_macd > 0:
            mom += 10
            positivas.append("Cruce alcista del MACD sobre su señal")
        elif cruce_macd < 0:
            mom -= 5
            negativas.append("Cruce bajista del MACD")
    if 0 < ret_21 <= 0.15:
        mom += 10
    elif ret_21 > 0.25:
        mom -= 5
        negativas.append(f"Subió {_pct(ret_21)} en un mes: puede necesitar consolidar")
    fuerza_relativa = None
    if mercado is not None and len(mercado) > 64:
        ret_mercado = float(mercado["Close"].iloc[-1] / mercado["Close"].iloc[-64] - 1)
        fuerza_relativa = ret_63 - ret_mercado
        if fuerza_relativa > 0:
            mom += 10
            positivas.append(f"Más fuerte que el S&P 500 en 3 meses ({fuerza_relativa * 100:+.1f} pp)")
        elif fuerza_relativa < -0.10:
            mom -= 10
            negativas.append(f"Más débil que el S&P 500 en 3 meses ({fuerza_relativa * 100:+.1f} pp)")
    vol_ratio = float(u["VolMA5"] / u["VolMA50"]) if pd.notna(u["VolMA50"]) and u["VolMA50"] > 0 else 1.0
    if vol_ratio > 1.3 and ret_5 > 0:
        mom += 10
        positivas.append(f"Volumen {vol_ratio:.1f}× la media con subida: acumulación")
    elif vol_ratio > 1.3 and ret_5 < 0:
        mom -= 10
        negativas.append(f"Volumen {vol_ratio:.1f}× la media con caída: distribución")
    if atr_pct > 0.05:
        mom -= 5
        negativas.append(f"Volatilidad alta (ATR {atr_pct * 100:.1f}% diario)")
    mom = _limitar(mom)

    # ---------------- Tendencia y potencial técnico ----------------
    if ma7 > ma25 > ma99 and c > ma7:
        tendencia = "Alcista fuerte"
    elif alcista:
        tendencia = "Alcista"
    elif c < ma99 and ma25 < ma99:
        tendencia = "Bajista"
    else:
        tendencia = "Lateral"
    if dist_max < -0.03:
        potencial, base_potencial = max52 / c - 1, "hasta el máximo de 52 semanas"
    else:
        potencial, base_potencial = min(0.30, max(0.05, 4 * atr_pct)), "proyección por volatilidad (4×ATR)"

    return {
        "fecha": df.index[-1].strftime("%Y-%m-%d"),
        "precio": round(c, 4),
        "cambio_1d": ret_1,
        "ret_5": ret_5,
        "ret_21": ret_21,
        "ret_63": ret_63,
        "rsi": rsi,
        "ma7": ma7,
        "ma25": ma25,
        "ma99": ma99,
        "atr_pct": atr_pct,
        "vol_ratio": vol_ratio,
        "max_52s": max52,
        "min_52s": min52,
        "dist_max_52s": dist_max,
        "fuerza_relativa": fuerza_relativa,
        "tendencia": tendencia,
        "puntaje_tecnico": round(tec, 1),
        "puntaje_momentum": round(mom, 1),
        "potencial_tecnico": potencial,
        "base_potencial": base_potencial,
        "senales_positivas": positivas,
        "senales_negativas": negativas,
    }


def puntaje_analistas(info: dict, precio: float) -> dict | None:
    objetivo = info.get("objetivo_medio")
    if not objetivo or not precio:
        return None
    potencial = float(objetivo) / precio - 1
    puntaje = 50 + potencial * 150
    recomendacion = info.get("recomendacion")
    if recomendacion:
        puntaje += (3 - float(recomendacion)) * 8  # 1 = compra fuerte ... 5 = venta
    analistas = int(info.get("num_analistas") or 0)
    if analistas < 3:
        puntaje = 50 + (puntaje - 50) * 0.5
    return {
        "puntaje": round(_limitar(puntaje, 5, 95), 1),
        "potencial": potencial,
        "objetivo": float(objetivo),
        "recomendacion": float(recomendacion) if recomendacion else None,
        "num_analistas": analistas,
    }


def puntaje_red(probabilidad: float | None, tasa_base: float | None) -> float | None:
    if probabilidad is None or not tasa_base:
        return None
    lift = max(probabilidad, 1e-4) / tasa_base
    return round(_limitar(50 + 20 * math.log2(lift)), 1)


def combinar(componentes: dict[str, float | None], pesos: dict[str, float]) -> float:
    """Media ponderada de los componentes disponibles (los ausentes no cuentan)."""
    suma = peso_total = 0.0
    for nombre, valor in componentes.items():
        peso = float(pesos.get(nombre, 0))
        if valor is None or peso <= 0 or np.isnan(valor):
            continue
        suma += peso * valor
        peso_total += peso
    return round(suma / peso_total, 1) if peso_total else 0.0


def recomendacion_texto(puntaje: float) -> str:
    if puntaje >= 75:
        return "Compra fuerte"
    if puntaje >= 65:
        return "Compra"
    if puntaje >= 50:
        return "Vigilar"
    if puntaje >= 35:
        return "Neutral"
    return "Evitar"
