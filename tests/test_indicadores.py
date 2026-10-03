import numpy as np
import pandas as pd

from bolsa.indicadores import agregar_indicadores, cruce_reciente, rsi, sma
from conftest import serie_desde_cierres


def test_medias_moviles_7_25_99():
    df = agregar_indicadores(serie_desde_cierres(np.arange(1, 201)))
    # Media de una progresión aritmética = valor central de la ventana
    assert df["MA7"].iloc[-1] == np.mean(np.arange(194, 201))
    assert df["MA25"].iloc[-1] == np.mean(np.arange(176, 201))
    assert df["MA99"].iloc[-1] == np.mean(np.arange(102, 201))
    assert df["MA99"].iloc[:98].isna().all() and pd.notna(df["MA99"].iloc[98])


def test_rsi_extremos():
    subida = pd.Series(np.arange(1, 60, dtype=float))
    bajada = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(subida).iloc[-1] == 100
    assert rsi(bajada).iloc[-1] < 1


def test_cruce_reciente():
    rapida = pd.Series([1, 1, 1, 3, 3], dtype=float)
    lenta = pd.Series([2, 2, 2, 2, 2], dtype=float)
    assert cruce_reciente(rapida, lenta, 3) == 1
    assert cruce_reciente(lenta, rapida, 3) == -1
    assert cruce_reciente(rapida, lenta + 10, 3) == 0


def test_sma_requiere_ventana_completa():
    s = sma(pd.Series([1.0, 2.0, 3.0]), 5)
    assert s.isna().all()
