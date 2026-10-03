import numpy as np

from bolsa.cartera import Cartera, cargar_posiciones_manuales
from bolsa.config import cargar_config
from bolsa.indicadores import agregar_indicadores
from conftest import serie_desde_cierres


def _registro(ticker, precio, puntaje=80):
    return {
        "ticker": ticker, "nombre": ticker, "sector": "Information Technology", "puntaje": puntaje,
        "recomendacion": "Compra fuerte", "potencial": 0.2, "base_potencial": "test",
        "componentes": {"tecnico": 80, "momentum": 70, "noticias": None, "analistas": None, "red_neuronal": None},
        "tecnico": {"precio": precio, "atr_pct": 0.02, "senales_positivas": ["Medias alineadas al alza"],
                    "senales_negativas": [], "tendencia": "Alcista", "ma25": precio, "ma99": precio * 0.9,
                    "rsi": 55},
        "info": {"capitalizacion": 3e11, "resumen": "Makes things. And more."},
        "meta": {"indices": ["S&P 500"]},
        "noticias": None, "analistas": None,
    }


def test_compra_con_razones_e_importancia(tmp_path):
    cfg = cargar_config()
    cartera = Cartera.cargar(tmp_path / "cartera.json", cfg)
    eventos = cartera.comprar("2026-01-05", [_registro("AAA", 100), _registro("BBB", 50, puntaje=40)], False)
    assert [e["ticker"] for e in eventos] == ["AAA"]  # BBB no alcanza el puntaje mínimo
    pos = cartera.posiciones[0]
    assert pos["acciones"] == 100  # 100k / 10 posiciones / $100
    assert any("Puntaje total" in r for r in pos["razones"])
    assert any("Medias alineadas" in r for r in pos["razones"])
    assert any("Tecnología" in i for i in pos["importancia"])
    assert any("S&P 500" in i for i in pos["importancia"])
    assert pos["stop_loss"] < 100 < pos["objetivo"]
    # Volver a ejecutar el mismo día no duplica la compra
    assert cartera.comprar("2026-01-05", [_registro("AAA", 100)], False) == []
    cartera.guardar(tmp_path / "cartera.json")
    assert Cartera.cargar(tmp_path / "cartera.json", cfg).tickers() == ["AAA"]


def test_stop_loss_y_objetivo(tmp_path):
    cfg = cargar_config()
    cartera = Cartera.cargar(tmp_path / "c.json", cfg)
    cartera.comprar("2024-06-03", [_registro("CAE", 100), _registro("SUBE", 100)], False)
    cae = agregar_indicadores(serie_desde_cierres(np.r_[np.full(150, 100.0), [85.0]]))
    sube = agregar_indicadores(serie_desde_cierres(np.r_[np.full(150, 100.0), [130.0]]))
    eventos = cartera.revisar_ventas("2024-08-05", {"CAE": cae, "SUBE": sube})
    motivos = {e["ticker"]: e["motivo_venta"] for e in eventos}
    assert motivos["CAE"] == "Stop-loss alcanzado"
    assert motivos["SUBE"] == "Objetivo de ganancia alcanzado"
    assert cartera.posiciones == []
    assert abs(cartera.valor_total() - (100_000 + 100 * 100 * (-0.15 + 0.30))) < 1e-6


def test_posiciones_manuales(tmp_path, monkeypatch):
    archivo = tmp_path / "mis.yaml"
    archivo.write_text(
        "posiciones:\n"
        "  - ticker: brk.b\n    fecha_compra: 2026-01-02\n    precio_compra: 400\n    acciones: 2\n"
        "    razones: Valor a largo plazo\n    importancia: Diversificación\n"
        "  - ticker: MALA\n", encoding="utf-8")
    posiciones = cargar_posiciones_manuales(str(archivo))
    assert len(posiciones) == 1
    assert posiciones[0]["ticker"] == "BRK-B"
    assert posiciones[0]["fecha_compra"] == "2026-01-02"
    assert posiciones[0]["razones"] == ["Valor a largo plazo"]
