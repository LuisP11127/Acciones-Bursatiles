"""Utilidades comunes de los scripts (argumentos, configuración y salida para GitHub Actions)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import load_settings  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402


def parser(descripcion: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=descripcion)
    p.add_argument("--data-dir", help="Carpeta de datos (por defecto ./data o $QUANT_DATA_DIR)")
    p.add_argument("--config-dir", help="Carpeta de configuración (por defecto ./config)")
    p.add_argument("--max-symbols", type=int, default=None, help="Limita el universo (pruebas rápidas)")
    p.add_argument("--offline", action="store_true",
                   help="Datos SINTÉTICOS para desarrollo sin conexión (nunca usar en producción)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def settings_from(args, extra_overrides: dict | None = None):
    setup_logging(args.verbose)
    overrides = dict(extra_overrides or {})
    if args.max_symbols is not None:
        overrides.setdefault("universe", {}).setdefault("universe", {})["max_symbols"] = args.max_symbols
    return load_settings(config_dir=args.config_dir, data_dir=args.data_dir, overrides=overrides,
                         offline=args.offline)


def github_output(**valores) -> None:
    destino = os.environ.get("GITHUB_OUTPUT")
    if destino:
        with open(destino, "a", encoding="utf-8") as fh:
            for clave, valor in valores.items():
                fh.write(f"{clave}={valor}\n")
