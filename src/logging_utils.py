"""Logs con etiquetas ([DATA], [NEWS], [MODEL]...) y resumen final de cada ejecución."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_CONFIGURADO = False


def setup_logging(verbose: bool = False) -> None:
    global _CONFIGURADO
    if _CONFIGURADO:
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for ruidoso in ("yfinance", "urllib3", "peewee", "matplotlib", "curl_cffi"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    _CONFIGURADO = True


class _TagAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{self.extra['tag']}] {msg}", kwargs


def get_logger(tag: str) -> logging.LoggerAdapter:
    return _TagAdapter(logging.getLogger(f"quant.{tag.lower()}"), {"tag": tag.upper()})


class RunSummary:
    """Acumula contadores, avisos y errores para el resumen final."""

    def __init__(self):
        self.reset()

    def reset(self, task: str = "") -> None:
        self.task = task
        self.started_at = datetime.now(timezone.utc)
        self.counters: dict[str, int | float | str] = defaultdict(int)
        self.info: dict[str, object] = {}
        self.errors: list[dict] = []
        self.warnings: list[dict] = []

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def set(self, key: str, value) -> None:
        self.info[key] = value

    def error(self, tag: str, message: str, symbol: str | None = None) -> None:
        self.errors.append({"tag": tag, "symbol": symbol, "message": str(message)[:500]})
        get_logger(tag).error("%s%s", f"{symbol}: " if symbol else "", message)

    def warning(self, tag: str, message: str, symbol: str | None = None) -> None:
        self.warnings.append({"tag": tag, "symbol": symbol, "message": str(message)[:500]})
        get_logger(tag).warning("%s%s", f"{symbol}: " if symbol else "", message)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counters": dict(self.counters),
            "info": self.info,
            "errors": self.errors[-200:],
            "warnings": self.warnings[-200:],
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
        }

    def to_markdown(self) -> str:
        datos = self.to_dict()
        lineas = [f"## Resumen de la ejecución: {self.task}", "",
                  f"- Inicio: {datos['started_at']} · Fin: {datos['finished_at']}"]
        for clave, valor in {**datos["info"], **datos["counters"]}.items():
            lineas.append(f"- {clave}: {valor}")
        lineas.append(f"- Errores: {datos['n_errors']} · Avisos: {datos['n_warnings']}")
        for err in datos["errors"][:20]:
            lineas.append(f"  - [{err['tag']}] {err['symbol'] or ''} {err['message']}")
        return "\n".join(lineas) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        resumen_gh = os.environ.get("GITHUB_STEP_SUMMARY")
        if resumen_gh:
            with open(resumen_gh, "a", encoding="utf-8") as fh:
                fh.write(self.to_markdown())
        log = get_logger("PIPELINE")
        for linea in self.to_markdown().splitlines():
            if linea.strip():
                log.info(linea)


summary = RunSummary()
