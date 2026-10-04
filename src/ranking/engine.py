"""Ranking Engine: combina los componentes con los pesos de config/model.yaml.

final_score es una métrica interna para ordenar candidatas, no una certeza.
Si un componente no está disponible (sin noticias, sin fundamentales o sin
modelo neuronal en producción), su peso se reparte entre los disponibles y el
desglose lo deja indicado.
"""

from __future__ import annotations

import math

RANKING_COMPONENTS = ("statistical", "news", "fundamental", "neural", "risk")


def combine_scores(components: dict[str, float | None], weights: dict[str, float]) -> dict:
    disponibles = {}
    for nombre in RANKING_COMPONENTS:
        valor = components.get(nombre)
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            continue
        w = float(weights.get(nombre, 0))
        if w > 0:
            disponibles[nombre] = (float(valor), w)
    total = sum(w for _, w in disponibles.values())
    if total <= 0:
        return {"final_score": None, "weights_used": {}, "contributions": {},
                "missing": [c for c in RANKING_COMPONENTS if c not in disponibles]}
    usados = {n: w / total for n, (_, w) in disponibles.items()}
    aportes = {n: round(v * usados[n], 2) for n, (v, _) in disponibles.items()}
    return {
        "final_score": round(sum(aportes.values()), 1),
        "weights_used": {n: round(w, 4) for n, w in usados.items()},
        "configured_weights": {n: float(weights.get(n, 0)) for n in RANKING_COMPONENTS},
        "contributions": aportes,
        "missing": [c for c in RANKING_COMPONENTS if c not in disponibles],
    }


def signal_label(final_score: float | None, opportunity_threshold: float) -> str:
    if final_score is None:
        return "Sin datos"
    if final_score >= opportunity_threshold + 15:
        return "Oportunidad destacada"
    if final_score >= opportunity_threshold:
        return "Oportunidad"
    if final_score >= opportunity_threshold - 10:
        return "Vigilar"
    return "Sin señal"
