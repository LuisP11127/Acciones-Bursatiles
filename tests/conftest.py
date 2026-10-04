import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.data.providers.market_data import synthetic_series  # noqa: E402
from src.data.validation import adjusted_ohlcv  # noqa: E402

END = date(2026, 9, 30)


@pytest.fixture
def settings(tmp_path):
    return load_settings(data_dir=tmp_path / "data", frontend_data_dir=tmp_path / "web",
                         overrides={"universe": {"universe": {"max_symbols": 12}}}, offline=True)


@pytest.fixture
def frame():
    """OHLCV ajustado sintético de una acción."""
    return adjusted_ohlcv(synthetic_series("TEST", END))


@pytest.fixture
def market_frame():
    return adjusted_ohlcv(synthetic_series("SPY", END, is_index=True))


def ohlc_from_closes(closes, volume=2_000_000, start="2024-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    fechas = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"open": closes, "high": closes * 1.001, "low": closes * 0.999, "close": closes,
                         "volume": np.full(len(closes), volume, dtype=float), "raw_close": closes}, index=fechas)
