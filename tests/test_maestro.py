import numpy as np

from bolsa.red.maestro import etiquetas_maestro, operaciones_maestro
from conftest import serie_desde_cierres


def test_compra_en_el_minimo_y_vende_en_el_maximo():
    # Baja hasta 80, luego sube hasta 100: el mejor punto de compra es el mínimo
    cierres = np.concatenate([np.linspace(100, 80, 21), np.linspace(80, 100, 21)[1:], np.full(30, 100.0)])
    df = serie_desde_cierres(cierres)
    et = etiquetas_maestro(df, horizonte=20, ganancia_minima=0.10, caida_maxima=0.02)
    minimo = int(np.argmin(cierres))
    assert et["etiqueta"].iloc[minimo] == 1
    assert et["etiqueta"].iloc[0] == 0  # comprar arriba implica caer >2% antes de subir
    assert et["ganancia"].iloc[minimo] > 0.24
    # El futuro de los últimos días aún no existe
    assert et["etiqueta"].iloc[-20:].isna().all()
    operaciones = operaciones_maestro(df, et)
    assert operaciones[0]["fecha_compra"] == df.index[minimo]
    assert operaciones[0]["ganancia"] > 0.24


def test_sin_subida_no_hay_puntos_de_compra():
    df = serie_desde_cierres(np.full(80, 50.0))
    et = etiquetas_maestro(df, horizonte=10)
    assert et["etiqueta"].dropna().sum() == 0
    assert operaciones_maestro(df, et) == []
