import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bolsa.datos import serie_sintetica  # noqa: E402
from bolsa.indicadores import agregar_indicadores  # noqa: E402


@pytest.fixture
def df_sintetico() -> pd.DataFrame:
    return agregar_indicadores(serie_sintetica("TEST", dias=600, fin=pd.Timestamp("2026-06-30")))


def serie_desde_cierres(cierres, volumen=1_000_000) -> pd.DataFrame:
    cierres = np.asarray(cierres, dtype=float)
    fechas = pd.bdate_range("2024-01-01", periods=len(cierres))
    return pd.DataFrame({
        "Open": cierres, "High": cierres * 1.001, "Low": cierres * 0.999, "Close": cierres,
        "Volume": np.full(len(cierres), volumen, dtype=float),
    }, index=fechas)
