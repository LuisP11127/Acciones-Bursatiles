"""Carga de la configuración y rutas de salida."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "universo": {
        "fuentes": ["sp500", "nasdaq100"],
        "extra": [],
        "max_tickers": 0,
        "respaldo": "config/universo_respaldo.csv",
    },
    "datos": {
        "periodo": "5y",
        "indices": ["SPY", "QQQ", "DIA", "IWM"],
    },
    "escaneo": {
        "precio_minimo": 5,
        "volumen_medio_minimo": 500_000,
        "candidatos_fase2": 40,
        "top_oportunidades": 15,
        "dias_grafico": 180,
    },
    "noticias": {
        "max_por_accion": 12,
        "dias_recientes": 14,
        "vida_media_dias": 3,
        "google_news": True,
    },
    "pesos": {
        "tecnico": 0.35,
        "momentum": 0.20,
        "noticias": 0.20,
        "analistas": 0.25,
        "red_neuronal": 0.25,
    },
    "cartera": {
        "capital_inicial": 100_000,
        "max_posiciones": 10,
        "max_compras_por_dia": 3,
        "puntaje_minimo_compra": 65,
        "stop_loss_pct": 0.08,
        "stop_atr": 2.5,
        "objetivo_pct": 0.25,
        "activar_stop_dinamico": 0.10,
        "stop_dinamico_pct": 0.08,
        "max_dias_sin_avance": 60,
    },
    "red_neuronal": {
        "horizonte": 20,
        "ganancia_minima": 0.08,
        "caida_maxima": 0.03,
        "capas": [128, 64, 32],
        "tasa_aprendizaje": 0.001,
        "l2": 0.0001,
        "batch": 512,
        "muestras_por_epoca": 250_000,
        "max_epocas_por_sesion": 15,
        "paciencia": 3,
        "minutos_por_sesion": 8,
        "minutos_sesion_profunda": 45,
        "dias_validacion": 126,
        "top_predicciones": 10,
        "madurez": {
            "sesiones_minimas": 20,
            "auc_minimo": 0.60,
            "lift_minimo": 1.4,
            "ventaja_sobre_referencia": 0.02,
            "predicciones_evaluadas_minimas": 60,
            "ventaja_vivo_minima": 1.2,
        },
    },
}


def _fusionar(base: dict, extra: dict) -> dict:
    resultado = copy.deepcopy(base)
    for clave, valor in (extra or {}).items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _fusionar(resultado[clave], valor)
        else:
            resultado[clave] = valor
    return resultado


def cargar_config(ruta: str | Path | None = None) -> dict:
    """Devuelve la configuración por defecto combinada con el YAML del usuario."""
    ruta = Path(ruta) if ruta else RAIZ / "config" / "config.yaml"
    usuario = {}
    if ruta.exists():
        with open(ruta, encoding="utf-8") as fh:
            usuario = yaml.safe_load(fh) or {}
    return _fusionar(DEFAULTS, usuario)


def ruta_proyecto(relativa: str | Path) -> Path:
    ruta = Path(relativa)
    return ruta if ruta.is_absolute() else RAIZ / ruta


@dataclass
class Rutas:
    """Estructura de la carpeta de salida (lo que se publica en la rama `reportes`)."""

    base: Path

    def __post_init__(self):
        self.base = Path(self.base)
        for carpeta in (self.graficos, self.estado, self.datos):
            carpeta.mkdir(parents=True, exist_ok=True)

    @property
    def graficos(self) -> Path:
        return self.base / "graficos"

    @property
    def estado(self) -> Path:
        return self.base / "estado"

    @property
    def datos(self) -> Path:
        return self.base / "datos"

    @property
    def cartera(self) -> Path:
        return self.estado / "cartera.json"

    @property
    def historial_escaneos(self) -> Path:
        return self.estado / "historial_escaneos.json"
