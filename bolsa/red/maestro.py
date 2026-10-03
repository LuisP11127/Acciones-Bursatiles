"""El "maestro" que enseña a la red: comprar en el mínimo y vender en el máximo.

Mirando el pasado (cuando ya se conoce lo que pasó después), cada día se
etiqueta como buen punto de compra si:

  1. comprando al cierre de ese día, el precio llegó en los siguientes
     `horizonte` días a un máximo con al menos `ganancia_minima` de subida, y
  2. antes de ese máximo el precio no cayó más de `caida_maxima` bajo la compra
     (o sea, se compró muy cerca del precio más bajo).

La red neuronal solo ve la información disponible ese día y aprende a
reconocer estos momentos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def etiquetas_maestro(df: pd.DataFrame, horizonte: int = 20, ganancia_minima: float = 0.08,
                      caida_maxima: float = 0.03) -> pd.DataFrame:
    """Devuelve por fecha: etiqueta (1/0), ganancia máxima, caída previa y días al máximo.
    Los últimos `horizonte` días quedan sin etiqueta (NaN): su futuro aún no existe."""
    n = len(df)
    resultado = pd.DataFrame(np.nan, index=df.index, columns=["etiqueta", "ganancia", "caida", "dias_al_maximo"])
    if n <= horizonte:
        return resultado
    cierre = df["Close"].to_numpy(dtype=float)
    maximos = sliding_window_view(df["High"].to_numpy(dtype=float)[1:], horizonte)
    minimos = sliding_window_view(df["Low"].to_numpy(dtype=float)[1:], horizonte)
    m = n - horizonte
    pos_max = maximos.argmax(axis=1)
    maximo = maximos[np.arange(m), pos_max]
    antes_del_maximo = np.arange(horizonte)[None, :] <= pos_max[:, None]
    minimo_previo = np.where(antes_del_maximo, minimos, np.inf).min(axis=1)
    ganancia = maximo / cierre[:m] - 1
    caida = minimo_previo / cierre[:m] - 1
    etiqueta = ((ganancia >= ganancia_minima) & (caida >= -caida_maxima)).astype(float)
    resultado.iloc[:m] = np.column_stack([etiqueta, ganancia, caida, pos_max + 1])
    return resultado


def operaciones_maestro(df: pd.DataFrame, etiquetas: pd.DataFrame) -> list[dict]:
    """Operaciones ideales del maestro (sin solaparse): en cada racha de días
    etiquetados compra en el cierre más bajo y vende en el máximo posterior."""
    operaciones = []
    et = etiquetas["etiqueta"].to_numpy()
    cierre = df["Close"].to_numpy()
    maximo = df["High"].to_numpy()
    i, n = 0, len(df)
    while i < n:
        if et[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and et[j + 1] == 1:
            j += 1
        compra = i + int(np.argmin(cierre[i:j + 1]))
        venta = compra + int(etiquetas["dias_al_maximo"].iloc[compra])
        venta = min(venta, n - 1)
        operaciones.append({
            "fecha_compra": df.index[compra],
            "precio_compra": float(cierre[compra]),
            "fecha_venta": df.index[venta],
            "precio_venta": float(maximo[venta]),
            "ganancia": float(maximo[venta] / cierre[compra] - 1),
        })
        i = max(j, venta) + 1
    return operaciones
