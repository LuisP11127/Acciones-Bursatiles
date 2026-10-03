import numpy as np

from bolsa.red.entrenador import auc_roc
from bolsa.red.neuronal import RedNeuronal


def _datos(n=4000, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.standard_normal((n, 6)).astype(np.float32)
    # Patrón no lineal: positivo si x0*x1 > 0 y x2 > -0.5
    y = ((X[:, 0] * X[:, 1] > 0) & (X[:, 2] > -0.5)).astype(np.float32)
    return X, y


def test_la_red_aprende_un_patron_no_lineal():
    X, y = _datos()
    red = RedNeuronal(6, (32, 16), semilla=1)
    red.normalizador.ajustar(X)
    Xn = red.normalizador.transformar(X)
    rng = np.random.default_rng(0)
    perdidas = []
    for _ in range(40):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), 128):
            b = idx[i:i + 128]
            perdidas.append(red.entrenar_lote(Xn[b], y[b], tasa=3e-3, l2=0))
    X_test, y_test = _datos(2000, semilla=9)
    auc = auc_roc(y_test, red.predecir(red.normalizador.transformar(X_test)))
    assert np.mean(perdidas[-30:]) < np.mean(perdidas[:30])
    assert auc > 0.9


def test_guardar_y_cargar_continua_igual(tmp_path):
    X, y = _datos(500)
    red = RedNeuronal(6, (8, 4), semilla=3)
    red.normalizador.ajustar(X)
    red.entrenar_lote(red.normalizador.transformar(X), y)
    ruta = tmp_path / "red.npz"
    red.guardar(ruta)
    copia = RedNeuronal.cargar(ruta)
    Xn = red.normalizador.transformar(X)
    assert np.allclose(red.predecir(Xn), copia.predecir(copia.normalizador.transformar(X)))
    assert copia.t == red.t
    # El estado de Adam también se conserva: un paso más da el mismo resultado
    red.entrenar_lote(Xn, y)
    copia.entrenar_lote(Xn, y)
    assert np.allclose(red.predecir(Xn), copia.predecir(Xn), atol=1e-6)


def test_auc():
    assert auc_roc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_roc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc_roc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
