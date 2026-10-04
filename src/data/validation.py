"""Control de calidad de los precios.

Verifica fechas duplicadas, valores faltantes, precios y volúmenes inválidos,
incoherencias OHLC, cambios extremos, orden cronológico, huecos y datos
desactualizados. Las acciones que no superan los controles no se usan para
entrenar ni para generar señales.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close", "adj_close", "volume"]


@dataclass
class QualityReport:
    symbol: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    ohlc_fixed: int = 0
    spikes_removed: int = 0
    suspicious_moves: int = 0
    zero_volume_days: int = 0
    missing_ratio: float = 0.0
    max_gap_sessions: int = 0
    first_date: str | None = None
    last_date: str | None = None
    usable_for_training: bool = True
    usable_for_signals: bool = True
    issues: list[str] = field(default_factory=list)

    def reject(self, motivo: str, entrenamiento: bool = True, senales: bool = True) -> None:
        self.issues.append(motivo)
        if entrenamiento:
            self.usable_for_training = False
        if senales:
            self.usable_for_signals = False

    def to_dict(self) -> dict:
        return asdict(self)


def validate_prices(symbol: str, df: pd.DataFrame | None, cfg: dict, min_history: int = 300,
                    calendar: pd.DatetimeIndex | None = None, as_of: pd.Timestamp | None = None,
                    max_stale_sessions: int = 5) -> tuple[pd.DataFrame | None, QualityReport]:
    """Devuelve (datos limpios, informe). `cfg` es la sección `validation` de data.yaml."""
    rep = QualityReport(symbol=symbol)
    if df is None or df.empty:
        rep.reject("sin datos")
        return None, rep
    rep.rows_in = len(df)
    faltan = [c for c in REQUIRED if c not in df.columns]
    if faltan:
        rep.reject(f"faltan columnas: {', '.join(faltan)}")
        return None, rep
    df = df.copy()
    for col in ("dividends", "splits"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0

    # Orden cronológico y duplicados
    df.index = pd.to_datetime(df.index).normalize()
    if not df.index.is_monotonic_increasing:
        rep.issues.append("fechas desordenadas (corregido)")
        df = df.sort_index()
    duplicados = df.index.duplicated(keep="last")
    rep.duplicates_removed = int(duplicados.sum())
    df = df[~duplicados]

    # Valores faltantes o inválidos
    precios = ["open", "high", "low", "close", "adj_close"]
    df[REQUIRED] = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    invalida = df[precios].isna().any(axis=1) | (df[precios] <= 0).any(axis=1)
    invalida |= df["volume"].isna() | (df["volume"] < 0)
    invalida |= ~np.isfinite(df[precios + ["volume"]]).all(axis=1)

    # Coherencia OHLC: pequeñas diferencias se corrigen, las grandes invalidan la fila
    tol = float(cfg.get("ohlc_tolerance", 0.01))
    maximo_oc = df[["open", "close"]].max(axis=1)
    minimo_oc = df[["open", "close"]].min(axis=1)
    alto_mal = df["high"] < maximo_oc
    bajo_mal = df["low"] > minimo_oc
    grave = (alto_mal & (maximo_oc / df["high"] - 1 > tol)) | (bajo_mal & (1 - minimo_oc / df["low"] > tol))
    grave |= df["high"] < df["low"] * (1 - tol)
    invalida |= grave
    corregibles = (alto_mal | bajo_mal) & ~grave & ~invalida
    rep.ohlc_fixed = int(corregibles.sum())
    df.loc[corregibles, "high"] = np.maximum(df.loc[corregibles, "high"], maximo_oc[corregibles])
    df.loc[corregibles, "low"] = np.minimum(df.loc[corregibles, "low"], minimo_oc[corregibles])
    rep.invalid_rows_removed = int(invalida.sum())
    df = df[~invalida]
    if df.empty:
        rep.reject("todas las filas eran inválidas")
        return None, rep

    # Picos de un día que se revierten (error típico de dato): se eliminan
    ret = df["adj_close"].pct_change()
    pico = float(cfg.get("spike_reversal_move", 0.35))
    siguiente = ret.shift(-1)
    reversion = ((ret > pico) & (siguiente < -pico / (1 + pico))) | ((ret < -pico) & (siguiente > pico / (1 - pico)))
    reversion &= df["splits"].fillna(0).eq(0) & df["splits"].shift(-1).fillna(0).eq(0)
    rep.spikes_removed = int(reversion.sum())
    df = df[~reversion]

    # Cambios extremos sin split: sospechosos
    ret = df["adj_close"].pct_change().abs()
    sospechosos = (ret > float(cfg.get("max_daily_move", 0.6))) & df["splits"].eq(0)
    rep.suspicious_moves = int(sospechosos.sum())
    if rep.suspicious_moves > int(cfg.get("max_suspicious_days", 3)):
        rep.reject(f"{rep.suspicious_moves} variaciones diarias extremas sin split")

    rep.zero_volume_days = int((df["volume"] == 0).sum())
    rep.first_date = df.index[0].strftime("%Y-%m-%d")
    rep.last_date = df.index[-1].strftime("%Y-%m-%d")

    # Huecos respecto al calendario de referencia (sesiones del índice)
    if calendar is not None and len(calendar):
        sesiones = calendar[(calendar >= df.index[0]) & (calendar <= df.index[-1])]
        if len(sesiones):
            presentes = sesiones.isin(df.index)
            rep.missing_ratio = float(1 - presentes.mean())
            faltantes = (~presentes).astype(int)
            grupos = (presentes.astype(int)).cumsum()
            rachas = pd.Series(faltantes).groupby(grupos).sum()
            rep.max_gap_sessions = int(rachas.max()) if len(rachas) else 0
            if rep.missing_ratio > float(cfg.get("max_missing_ratio", 0.03)):
                rep.reject(f"faltan {rep.missing_ratio:.1%} de las sesiones")
            elif rep.max_gap_sessions > int(cfg.get("max_gap_sessions", 10)):
                rep.reject(f"hueco de {rep.max_gap_sessions} sesiones seguidas")

    if len(df) < min_history:
        rep.reject(f"historial insuficiente ({len(df)} sesiones < {min_history})")
    if as_of is not None and calendar is not None and len(calendar):
        sesiones_desde = int(((calendar > df.index[-1]) & (calendar <= as_of)).sum())
        if sesiones_desde > max_stale_sessions:
            rep.reject(f"datos desactualizados ({sesiones_desde} sesiones sin datos)", entrenamiento=False)
    rep.rows_out = len(df)
    return df, rep


def adjusted_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC ajustado por splits y dividendos (factor = adj_close / close).

    Todas las features son relativas (retornos, ratios), por lo que el ajuste
    hacia atrás no introduce información futura en ellas."""
    factor = (df["adj_close"] / df["close"]).where(lambda s: np.isfinite(s) & (s > 0), 1.0)
    salida = pd.DataFrame({
        "open": df["open"] * factor,
        "high": df["high"] * factor,
        "low": df["low"] * factor,
        "close": df["adj_close"],
        "volume": df["volume"].astype(float),
        "raw_close": df["close"],
    }, index=df.index)
    return salida
