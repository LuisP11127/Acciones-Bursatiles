"""Ejecuta el escaneo completo varios días seguidos con datos sintéticos."""

import json

import pandas as pd

from bolsa.config import cargar_config
from bolsa.datos import ProveedorSintetico
from bolsa.principal import entrenar, escanear
from bolsa.universo import obtener_universo


def _cfg():
    cfg = cargar_config()
    cfg["universo"]["max_tickers"] = 25
    cfg["escaneo"]["top_oportunidades"] = 5
    cfg["escaneo"]["candidatos_fase2"] = 10
    cfg["red_neuronal"]["horizonte"] = 3  # para que la evaluación en vivo llegue en la prueba
    cfg["red_neuronal"]["max_epocas_por_sesion"] = 1
    cfg["red_neuronal"]["muestras_por_epoca"] = 5000
    return cfg


def test_escaneo_varios_dias(tmp_path):
    cfg = _cfg()
    universo = obtener_universo(cfg, offline=True)
    fechas = pd.bdate_range(end=pd.Timestamp("2026-09-30"), periods=6)
    for fecha in fechas:
        proveedor = ProveedorSintetico(universo, cfg["datos"]["indices"], fin=fecha)
        ctx = escanear(cfg, tmp_path, offline=True, minutos=0.05, proveedor=proveedor)
        assert ctx["fecha_datos"] == fecha.strftime("%Y-%m-%d")

    for pagina in ("README.md", "oportunidades.md", "seguimiento.md", "red_neuronal.md"):
        texto = (tmp_path / pagina).read_text(encoding="utf-8")
        assert "SINTÉTICOS" in texto
    oportunidades = (tmp_path / "oportunidades.md").read_text(encoding="utf-8")
    assert "MA(7)" in oportunidades and "MA(25)" in oportunidades and "MA(99)" in oportunidades
    for r in ctx["top"]:
        assert (tmp_path / r["grafico"]).exists()

    estado = json.loads((tmp_path / "estado" / "red_neuronal.json").read_text())
    assert estado["sesiones"] == len(fechas)
    assert estado["vivo"]["evaluadas"] > 0  # horizonte 3: las predicciones de los primeros días ya se evaluaron
    assert (tmp_path / "estado" / "red_neuronal.npz").exists()

    cartera = json.loads((tmp_path / "estado" / "cartera.json").read_text())
    assert len(cartera["historial"]) == len(fechas)
    for pos in cartera["posiciones"]:
        assert pos["razones"] and pos["importancia"]
    seguimiento = (tmp_path / "seguimiento.md").read_text(encoding="utf-8")
    if cartera["posiciones"]:
        assert "Razones de la compra" in seguimiento and "Por qué es importante" in seguimiento

    assert (tmp_path / "datos" / "escaneo.csv").exists()
    historial = json.loads((tmp_path / "estado" / "historial_escaneos.json").read_text())
    assert [h["fecha"] for h in historial] == [f.strftime("%Y-%m-%d") for f in fechas]


def test_entrenamiento_profundo(tmp_path):
    cfg = _cfg()
    universo = obtener_universo(cfg, offline=True)
    proveedor = ProveedorSintetico(universo, cfg["datos"]["indices"], fin=pd.Timestamp("2026-09-30"))
    ctx = entrenar(cfg, tmp_path, offline=True, minutos=0.05, proveedor=proveedor)
    assert ctx["red"]["meta"]["sesiones"] == 1
    assert (tmp_path / "red_neuronal.md").exists()
