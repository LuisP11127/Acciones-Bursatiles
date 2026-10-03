import numpy as np

from bolsa.indicadores import agregar_indicadores
from bolsa.puntuacion import analisis_tecnico, combinar, puntaje_analistas, recomendacion_texto
from conftest import serie_desde_cierres


def test_tendencia_alcista_puntua_mejor_que_bajista():
    rng = np.random.default_rng(0)
    ruido = rng.normal(0, 0.004, 300)
    alcista = agregar_indicadores(serie_desde_cierres(100 * np.exp(np.cumsum(0.002 + ruido))))
    bajista = agregar_indicadores(serie_desde_cierres(100 * np.exp(np.cumsum(-0.002 + ruido))))
    a, b = analisis_tecnico(alcista), analisis_tecnico(bajista)
    assert a["puntaje_tecnico"] > b["puntaje_tecnico"]
    assert a["tendencia"].startswith("Alcista")
    assert b["tendencia"] == "Bajista"
    assert any("MA(99)" in s for s in a["senales_positivas"])
    assert any("MA(99)" in s for s in b["senales_negativas"])
    assert 0 <= a["puntaje_momentum"] <= 100


def test_historial_insuficiente():
    df = agregar_indicadores(serie_desde_cierres(np.linspace(10, 20, 60)))
    assert analisis_tecnico(df) is None


def test_combinar_ignora_componentes_ausentes():
    pesos = {"tecnico": 1, "noticias": 1, "red_neuronal": 1}
    assert combinar({"tecnico": 80, "noticias": None, "red_neuronal": None}, pesos) == 80
    assert combinar({"tecnico": 80, "noticias": 40}, pesos) == 60


def test_analistas():
    alto = puntaje_analistas({"objetivo_medio": 130, "recomendacion": 1.8, "num_analistas": 20}, 100)
    bajo = puntaje_analistas({"objetivo_medio": 90, "recomendacion": 3.5, "num_analistas": 20}, 100)
    assert alto["puntaje"] > bajo["puntaje"]
    assert abs(alto["potencial"] - 0.30) < 1e-9
    assert puntaje_analistas({}, 100) is None
    assert recomendacion_texto(80) == "Compra fuerte"
