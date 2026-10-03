"""Línea de comandos.

    python -m bolsa escanear                 # escaneo diario completo
    python -m bolsa entrenar --minutos 45    # sesión larga de la red neuronal
    python -m bolsa escanear --offline       # prueba con datos sintéticos, sin internet
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import cargar_config
from .principal import entrenar, escanear


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bolsa", description="Escáner de acciones de EE.UU.")
    parser.add_argument("comando", choices=["escanear", "entrenar"])
    parser.add_argument("--salida", default="salida", help="Carpeta de reportes y estado (por defecto: salida)")
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    parser.add_argument("--offline", action="store_true", help="Usar datos sintéticos (sin conexión)")
    parser.add_argument("--minutos", type=float, default=None, help="Minutos de entrenamiento de la red")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limitar el universo (pruebas rápidas)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for ruidoso in ("yfinance", "urllib3", "peewee", "matplotlib"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    cfg = cargar_config(args.config)
    if args.max_tickers is not None:
        cfg["universo"]["max_tickers"] = args.max_tickers
    if args.comando == "escanear":
        escanear(cfg, args.salida, offline=args.offline, minutos=args.minutos)
    else:
        entrenar(cfg, args.salida, offline=args.offline, minutos=args.minutos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
