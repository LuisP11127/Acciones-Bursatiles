"""Posiciones reales declaradas por el usuario en config/mis_acciones.yaml.

Solo se siguen (precio, variación, scores, noticias, explicación); el sistema no
opera con ellas.
"""

from __future__ import annotations

from datetime import date, datetime

import yaml

from ..config import resolve
from ..logging_utils import get_logger

log = get_logger("TRADING")
MANUAL_FILE = "config/mis_acciones.yaml"


def load_manual_positions(path: str = MANUAL_FILE) -> list[dict]:
    archivo = resolve(path)
    if not archivo.exists():
        return []
    with open(archivo, encoding="utf-8") as fh:
        datos = yaml.safe_load(fh) or {}
    posiciones = []
    for item in datos.get("posiciones") or datos.get("positions") or []:
        try:
            fecha = item.get("fecha_compra") or item.get("entry_date")
            if isinstance(fecha, (date, datetime)):
                fecha = fecha.strftime("%Y-%m-%d")
            razones = item.get("razones") or item.get("reasons") or []
            razones = [razones] if isinstance(razones, str) else list(razones)
            importancia = item.get("importancia") or item.get("why_it_matters") or []
            importancia = [importancia] if isinstance(importancia, str) else list(importancia)
            posiciones.append({
                "symbol": str(item["ticker"] if "ticker" in item else item["symbol"]).strip().upper().replace(".", "-"),
                "entry_date": str(fecha) if fecha else None,
                "entry_price": float(item.get("precio_compra") or item.get("entry_price")),
                "shares": float(item.get("acciones") or item.get("shares") or 0),
                "reasons": [str(r) for r in razones],
                "why_it_matters": [str(i) for i in importancia],
                "status": "MANUAL",
            })
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Posición manual inválida %s: %s", item, exc)
    return posiciones
