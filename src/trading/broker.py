"""Interfaz de broker preparada para el futuro, SIN ejecución real.

El proyecto funciona exclusivamente como investigación y paper trading. Esta
interfaz existe para que una futura integración tenga un punto claro de
extensión, pero el único broker incluido es el simulado y cualquier intento de
operar en real lanza un error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Broker(ABC):
    is_real: bool = False

    @abstractmethod
    def submit_order(self, symbol: str, side: str, quantity: float, order_type: str = "market") -> dict:
        ...


class PaperBroker(Broker):
    """Registra la intención de orden; la ejecución se simula en src/trading/paper.py."""

    def submit_order(self, symbol: str, side: str, quantity: float, order_type: str = "market") -> dict:
        return {"symbol": symbol, "side": side, "quantity": quantity, "type": order_type, "simulated": True}


class RealExecutionDisabled(RuntimeError):
    pass


def get_broker(mode: str) -> Broker:
    if mode not in ("RESEARCH", "PAPER_TRADING"):
        raise RealExecutionDisabled(f"Modo {mode!r} no permitido: este sistema nunca ejecuta operaciones reales.")
    return PaperBroker()
