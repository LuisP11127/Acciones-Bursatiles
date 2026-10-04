"""Carga de configuración (config/*.yaml) y rutas de datos.

La configuración vive solo en los YAML; el código no define reglas ni pesos propios.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILES = ("universe", "data", "model", "strategy")
MODES = ("RESEARCH", "PAPER_TRADING")


def deep_merge(base: dict, extra: dict | None) -> dict:
    result = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass
class Paths:
    """Estructura de almacenamiento (ver README, sección "Estructura de datos")."""

    data: Path
    frontend_data: Path
    config: Path

    def __post_init__(self):
        self.data = Path(self.data)
        self.frontend_data = Path(self.frontend_data)
        self.config = Path(self.config)

    @property
    def raw(self) -> Path:            # precios por acción (caché, no versionado)
        return self.data / "raw"

    @property
    def processed(self) -> Path:      # datasets y predicciones fuera de muestra (no versionado)
        return self.data / "processed"

    @property
    def universe(self) -> Path:
        return self.data / "universe"

    @property
    def state(self) -> Path:          # paper trading y equity (versionado)
        return self.data / "state"

    @property
    def history(self) -> Path:        # snapshots diarios (versionado)
        return self.data / "history"

    @property
    def news(self) -> Path:           # archivo de noticias (versionado)
        return self.data / "news"

    @property
    def models(self) -> Path:         # modelos y registro de versiones (versionado)
        return self.data / "models"

    @property
    def results(self) -> Path:        # backtests, calidad de datos, resumen de ejecución
        return self.data / "results"

    def ensure(self) -> None:
        for path in (self.raw, self.processed, self.universe, self.state, self.history, self.news,
                     self.models, self.results):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    universe: dict
    data: dict
    model: dict
    strategy: dict
    paths: Paths
    offline: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        mode = str(self.strategy.get("mode", "PAPER_TRADING")).upper()
        if mode not in MODES:
            raise ValueError(f"Modo desconocido {mode!r}; usa uno de {MODES}")
        return mode

    @property
    def rules(self) -> dict:
        return self.strategy["strategy"]

    def as_dict(self) -> dict:
        return {"universe": self.universe, "data": self.data, "model": self.model, "strategy": self.strategy}

    def config_hash(self) -> str:
        texto = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(texto.encode()).hexdigest()[:12]


def load_settings(config_dir: str | Path | None = None, data_dir: str | Path | None = None,
                  frontend_data_dir: str | Path | None = None, overrides: dict | None = None,
                  offline: bool = False) -> Settings:
    """Lee config/*.yaml. `overrides` permite ajustar valores (pruebas, CLI)."""
    config_dir = Path(config_dir or os.environ.get("QUANT_CONFIG_DIR") or ROOT / "config")
    cargados = {}
    for nombre in CONFIG_FILES:
        ruta = config_dir / f"{nombre}.yaml"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta el archivo de configuración {ruta}")
        with open(ruta, encoding="utf-8") as fh:
            cargados[nombre] = yaml.safe_load(fh) or {}
    cargados = deep_merge(cargados, overrides or {})
    paths = Paths(
        data=Path(data_dir or os.environ.get("QUANT_DATA_DIR") or ROOT / "data"),
        frontend_data=Path(frontend_data_dir or os.environ.get("QUANT_FRONTEND_DATA_DIR")
                           or ROOT / "frontend" / "public" / "data"),
        config=config_dir,
    )
    if offline:
        cargados["data"] = deep_merge(cargados["data"], {
            "market_data": {"provider": "synthetic"},
            "fundamentals": {"provider": "synthetic"},
            "news": {"providers": ["synthetic"]},
        })
    settings = Settings(paths=paths, offline=offline, **cargados)
    settings.mode  # valida el modo al cargar
    return settings


def resolve(path: str | Path, settings: Settings | None = None) -> Path:
    """Rutas relativas: data/... respecto a la carpeta de datos, el resto respecto al repo."""
    path = Path(path)
    if path.is_absolute():
        return path
    if settings is not None and path.parts and path.parts[0] == "data":
        return settings.paths.data.joinpath(*path.parts[1:])
    return ROOT / path
