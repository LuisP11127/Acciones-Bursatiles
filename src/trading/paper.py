"""Paper Trading / Tracking Engine. Operaciones SIMULADAS: nunca se envían órdenes reales.

Ciclo de una operación:
  PENDING      señal al cierre de T; se simula la compra en la apertura de T+1.
  OPEN         comprada (precio de apertura real de T+1 más costes).
  CLOSED       vendida por stop-loss, take-profit, ruptura de tendencia o fin del periodo.
  INVALIDATED  no llegó a abrirse (gap excesivo, sin datos, cartera completa) o sus datos
               dejaron de ser válidos.
Las reglas de salida son las mismas que en el backtesting (src/backtesting/engine.check_exit).
Se usan precios negociados reales (sin ajuste por dividendos); los splits se aplican a la posición.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..backtesting.engine import Rules, check_exit
from ..data.storage import read_json, write_json
from ..logging_utils import get_logger, summary

log = get_logger("TRADING")
STATUSES = ("PENDING", "OPEN", "CLOSED", "INVALIDATED")
EXIT_LABELS = {
    "stop_loss": "Stop-loss",
    "take_profit": "Take-profit",
    "time": "Fin del periodo de mantenimiento",
    "trend_break": "Ruptura de tendencia (MA7 < MA25 y precio < MA99)",
}


def neural_gate(probability: float | None, base_rate: float | None, rules_cfg: dict) -> float | None:
    if base_rate is None:
        return None
    valor = rules_cfg.get("min_neural_probability", "auto")
    if valor in (None, "auto"):
        return float(base_rate) * float(rules_cfg.get("min_neural_lift", 1.15))
    return float(valor)


def entry_decision(record: dict, rules_cfg: dict, model_available: bool) -> tuple[bool, list[str]]:
    """¿Cumple la candidata las reglas de entrada de strategy.yaml?"""
    fallos = []
    final = record.get("final_score")
    if final is None or final < float(rules_cfg["min_final_score"]):
        fallos.append(f"final_score {final} < {rules_cfg['min_final_score']}")
    stat = record.get("statistical_score")
    if stat is None or stat < float(rules_cfg["min_statistical_score"]):
        fallos.append(f"statistical_score {stat} < {rules_cfg['min_statistical_score']}")
    if model_available:
        umbral = record.get("neural_threshold")
        prob = record.get("opportunity_probability")
        if prob is None or umbral is None or prob < umbral:
            fallos.append(f"probabilidad neuronal {prob} < umbral {umbral}")
    elif rules_cfg.get("require_neural_model"):
        fallos.append("no hay modelo neuronal en producción (require_neural_model: true)")
    return not fallos, fallos


class PaperTrader:
    def __init__(self, state_dir: Path, settings):
        self.settings = settings
        self.rules = Rules.from_settings(settings)
        self.trades_path = Path(state_dir) / "trades.json"
        self.equity_path = Path(state_dir) / "equity.json"
        self.data = read_json(self.trades_path, {"trades": [], "next_id": 1})
        capital = self.rules.initial_capital
        self.equity = read_json(self.equity_path, {"initial_capital": capital, "cash": capital, "history": []})

    # ------------------------------------------------------------------
    @property
    def trades(self) -> list[dict]:
        return self.data["trades"]

    def by_status(self, *estados: str) -> list[dict]:
        return [t for t in self.trades if t["status"] in estados]

    def _last_equity(self) -> float:
        historial = self.equity.get("history") or []
        return float(historial[-1]["equity"]) if historial else float(self.equity["initial_capital"])

    def _apply_splits(self, trade: dict, df: pd.DataFrame, as_of: pd.Timestamp) -> None:
        if "splits" not in df.columns or trade.get("entry_date") is None:
            return
        aplicados = set(trade.get("splits_applied", []))
        eventos = df[(df.index > pd.Timestamp(trade["entry_date"])) & (df.index <= as_of) & (df["splits"] > 0)]
        for fecha, fila in eventos.iterrows():
            clave = fecha.strftime("%Y-%m-%d")
            if clave in aplicados or fila["splits"] in (0, 1):
                continue
            ratio = float(fila["splits"])
            for campo in ("entry_price", "signal_price", "stop_price", "take_profit_price", "min_low"):
                if trade.get(campo):
                    trade[campo] = trade[campo] / ratio
            trade["shares"] = trade["shares"] * ratio
            aplicados.add(clave)
            log.info("Split %.2f:1 aplicado a %s (%s)", ratio, trade["symbol"], trade["trade_id"])
        trade["splits_applied"] = sorted(aplicados)

    def _close(self, trade: dict, fecha: pd.Timestamp, precio: float, motivo: str) -> dict:
        c = self.rules.cost_per_side
        trade["status"] = "CLOSED"
        trade["exit_date"] = fecha.strftime("%Y-%m-%d")
        trade["exit_price"] = round(float(precio), 4)
        trade["exit_reason"] = EXIT_LABELS.get(motivo, motivo)
        trade["exit_code"] = motivo
        trade["actual_return"] = round((precio * (1 - c)) / (trade["entry_price"] * (1 + c)) - 1, 6)
        trade["pnl"] = round(trade["shares"] * (precio * (1 - c) - trade["entry_price"] * (1 + c)), 2)
        trade["current_price"] = trade["exit_price"]
        self.equity["cash"] = round(self.equity["cash"] + trade["shares"] * precio * (1 - c), 2)
        return {"type": "closed", "trade_id": trade["trade_id"], "symbol": trade["symbol"],
                "date": trade["exit_date"], "price": trade["exit_price"], "reason": trade["exit_reason"],
                "return": trade["actual_return"]}

    def _invalidate(self, trade: dict, fecha: pd.Timestamp, motivo: str) -> dict:
        trade["status"] = "INVALIDATED"
        trade["invalidated_date"] = fecha.strftime("%Y-%m-%d")
        trade["invalidation_reason"] = motivo
        return {"type": "invalidated", "trade_id": trade["trade_id"], "symbol": trade["symbol"],
                "date": trade["invalidated_date"], "reason": motivo}

    # ------------------------------------------------------------------
    def process(self, as_of: pd.Timestamp, raw: dict[str, pd.DataFrame], trend_flags: dict[str, pd.Series],
                calendar: pd.DatetimeIndex) -> list[dict]:
        """Ejecuta entradas pendientes y revisa salidas con las sesiones hasta `as_of`."""
        eventos: list[dict] = []
        c = self.rules.cost_per_side
        for trade in self.by_status("PENDING"):
            df = raw.get(trade["symbol"])
            senal = pd.Timestamp(trade["signal_date"])
            sesiones = int(((calendar > senal) & (calendar <= as_of)).sum())
            siguientes = df[(df.index > senal) & (df.index <= as_of)] if df is not None else pd.DataFrame()
            if siguientes.empty:
                if sesiones >= 3:
                    eventos.append(self._invalidate(trade, as_of, "Sin datos de precio en las sesiones posteriores a la señal"))
                continue
            fecha = siguientes.index[0]
            apertura = float(siguientes["open"].iloc[0])
            gap = apertura / float(trade["signal_price"]) - 1
            if self.rules.max_entry_gap is not None and gap > self.rules.max_entry_gap:
                eventos.append(self._invalidate(trade, fecha, f"Abrió {gap * 100:+.1f}% sobre el cierre de la señal "
                                                              f"(máximo {self.rules.max_entry_gap * 100:.0f}%)"))
                continue
            if len(self.by_status("OPEN")) >= self.rules.max_positions:
                eventos.append(self._invalidate(trade, fecha, "Cartera simulada completa (max_positions)"))
                continue
            asignacion = min(self._last_equity() / self.rules.max_positions, float(self.equity["cash"]))
            if asignacion <= 1:
                eventos.append(self._invalidate(trade, fecha, "Sin efectivo simulado disponible"))
                continue
            trade.update({
                "status": "OPEN", "entry_date": fecha.strftime("%Y-%m-%d"), "entry_price": round(apertura, 4),
                "shares": asignacion / (apertura * (1 + c)), "stop_price": apertura * (1 - self.rules.stop_loss),
                "take_profit_price": apertura * (1 + self.rules.take_profit), "bars_held": 0, "min_low": apertura,
                "pending_trend_exit": False, "last_bar_date": None, "entry_gap": round(gap, 4),
            })
            self.equity["cash"] = round(self.equity["cash"] - trade["shares"] * apertura * (1 + c), 2)
            eventos.append({"type": "opened", "trade_id": trade["trade_id"], "symbol": trade["symbol"],
                            "date": trade["entry_date"], "price": trade["entry_price"]})
        for trade in self.by_status("OPEN"):
            df = raw.get(trade["symbol"])
            if df is None:
                continue
            self._apply_splits(trade, df, as_of)
            desde = pd.Timestamp(trade["last_bar_date"]) if trade.get("last_bar_date") else None
            barras = df[(df.index > desde) if desde is not None else (df.index >= pd.Timestamp(trade["entry_date"]))]
            barras = barras[barras.index <= as_of]
            flags = trend_flags.get(trade["symbol"])
            for fecha, fila in barras.iterrows():
                trade["bars_held"] += 1
                trade["min_low"] = min(float(trade["min_low"]), float(fila["low"]))
                salida = check_exit(float(fila["open"]), float(fila["high"]), float(fila["low"]), float(fila["close"]),
                                    float(trade["stop_price"]), float(trade["take_profit_price"]), trade["bars_held"],
                                    fecha == pd.Timestamp(trade["entry_date"]), bool(trade["pending_trend_exit"]),
                                    self.rules)
                trade["last_bar_date"] = fecha.strftime("%Y-%m-%d")
                trade["current_price"] = round(float(fila["close"]), 4)
                if salida is not None:
                    eventos.append(self._close(trade, fecha, salida[0], salida[1]))
                    break
                if self.rules.exit_on_trend_break and flags is not None and bool(flags.get(fecha, False)):
                    trade["pending_trend_exit"] = True
            if trade["status"] == "OPEN":
                trade["actual_return"] = round((trade["current_price"] * (1 - c)) / (trade["entry_price"] * (1 + c)) - 1, 6)
            trade["max_drawdown"] = round(float(trade["min_low"]) / float(trade["entry_price"]) - 1, 6)
            trade["holding_days"] = int(trade["bars_held"])
        self._snapshot_equity(as_of, raw)
        for e in eventos:
            summary.count(f"paper_{e['type']}")
        return eventos

    def _snapshot_equity(self, as_of: pd.Timestamp, raw: dict[str, pd.DataFrame]) -> None:
        invertido = 0.0
        for trade in self.by_status("OPEN"):
            df = raw.get(trade["symbol"])
            precio = trade.get("current_price") or trade["entry_price"]
            if df is not None:
                previos = df[df.index <= as_of]
                if len(previos):
                    precio = float(previos["close"].iloc[-1])
            invertido += trade["shares"] * precio
        fecha = as_of.strftime("%Y-%m-%d")
        equity = round(float(self.equity["cash"]) + invertido, 2)
        historial = [h for h in self.equity.get("history", []) if h["date"] != fecha]
        historial.append({"date": fecha, "equity": equity, "cash": round(float(self.equity["cash"]), 2),
                          "invested": round(invertido, 2), "open_positions": len(self.by_status("OPEN"))})
        self.equity["history"] = sorted(historial, key=lambda h: h["date"])

    # ------------------------------------------------------------------
    def create_signals(self, as_of: pd.Timestamp, candidates: list[dict], model_available: bool,
                       model_version: str | None) -> list[dict]:
        """Crea operaciones PENDING para las candidatas que cumplen las reglas (solo PAPER_TRADING)."""
        reglas = self.settings.rules
        activos = {t["symbol"] for t in self.by_status("PENDING", "OPEN")}
        ya_hoy = {t["symbol"] for t in self.trades if t["signal_date"] == as_of.strftime("%Y-%m-%d")}
        # volver a ejecutar la misma sesión no crea más señales que el límite diario
        huecos = min(int(reglas["max_new_positions_per_day"]) - len(ya_hoy),
                     int(reglas["max_positions"]) - len(self.by_status("PENDING", "OPEN")))
        eventos = []
        for rec in candidates:
            if huecos <= 0:
                break
            if rec["symbol"] in activos or rec["symbol"] in ya_hoy:
                continue
            ok, _ = entry_decision(rec, reglas, model_available)
            if not ok:
                continue
            exp = rec.get("explanation") or {}
            trade = {
                "trade_id": f"T{self.data['next_id']:05d}",
                "symbol": rec["symbol"],
                "name": rec.get("name"),
                "strategy": reglas["name"],
                "mode": "PAPER_TRADING",
                "status": "PENDING",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "signal_date": as_of.strftime("%Y-%m-%d"),
                "signal_price": round(float(rec["price"]), 4),
                "entry_date": None, "entry_price": None, "exit_date": None, "exit_price": None,
                "reason": exp.get("summary"),
                "reasons": [r["text"] for r in exp.get("reasons", [])],
                "why_it_matters": [r["why_it_matters"] for r in exp.get("reasons", [])],
                "risks_at_entry": [r["text"] for r in exp.get("risks", [])],
                "news_explanation": exp.get("news_explanation"),
                "model_explanation": exp.get("model_explanation"),
                "statistical_score": rec.get("statistical_score"),
                "news_score": rec.get("news_score"),
                "fundamental_score": rec.get("fundamental_score"),
                "neural_score": rec.get("neural_score"),
                "risk_score": rec.get("risk_score"),
                "final_score": rec.get("final_score"),
                "opportunity_probability": rec.get("opportunity_probability"),
                "confidence": rec.get("confidence"),
                "expected_return": rec.get("expected_return"),
                "expected_drawdown": rec.get("expected_drawdown"),
                "model_version": model_version,
                "actual_return": None, "max_drawdown": None, "holding_days": 0,
            }
            self.data["next_id"] += 1
            self.trades.append(trade)
            activos.add(rec["symbol"])
            huecos -= 1
            eventos.append({"type": "signal", "trade_id": trade["trade_id"], "symbol": trade["symbol"],
                            "date": trade["signal_date"], "price": trade["signal_price"],
                            "final_score": trade["final_score"]})
            summary.count("paper_signal")
        return eventos

    def refresh_tracking(self, latest: dict[str, dict], as_of: pd.Timestamp) -> None:
        """Guarda los scores actuales de cada operación activa para seguir su evolución."""
        for trade in self.by_status("PENDING", "OPEN"):
            rec = latest.get(trade["symbol"])
            if not rec:
                continue
            trade["current_scores"] = {k: rec.get(k) for k in ("final_score", "statistical_score", "news_score",
                                                                 "fundamental_score", "neural_score", "risk_score",
                                                                 "opportunity_probability")}
            trade["current_scores"]["date"] = as_of.strftime("%Y-%m-%d")
            if trade["status"] == "PENDING":
                trade["current_price"] = rec.get("price")

    def save(self) -> None:
        write_json(self.trades_path, self.data)
        write_json(self.equity_path, self.equity)


def trend_break_flags(ind: pd.DataFrame) -> pd.Series:
    """MA7 < MA25 y cierre < MA99 en cada sesión (datos hasta esa sesión)."""
    return ((ind["ma7"] < ind["ma25"]) & (ind["close"] < ind["ma99"])).fillna(False)
