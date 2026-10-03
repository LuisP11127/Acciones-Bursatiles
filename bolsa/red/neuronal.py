"""Perceptrón multicapa escrito con NumPy (sin dependencias pesadas).

Se entrena por mini-lotes con Adam y entropía cruzada binaria. Su estado
(pesos, momentos de Adam y normalización de entradas) se guarda en un .npz,
así cada ejecución de GitHub Actions continúa el aprendizaje donde quedó.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PENDIENTE_RELU = 0.01  # Leaky ReLU


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def entropia_cruzada(logits: np.ndarray, y: np.ndarray) -> float:
    """BCE numéricamente estable a partir de logits."""
    return float(np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))))


class Normalizador:
    """Estandariza las entradas con la media/desviación del primer entrenamiento."""

    def __init__(self, media: np.ndarray | None = None, desv: np.ndarray | None = None):
        self.media = media
        self.desv = desv

    @property
    def ajustado(self) -> bool:
        return self.media is not None

    def ajustar(self, X: np.ndarray) -> None:
        self.media = X.mean(axis=0).astype(np.float32)
        desv = X.std(axis=0).astype(np.float32)
        self.desv = np.where(desv < 1e-8, 1.0, desv).astype(np.float32)

    def transformar(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.media) / self.desv, -5, 5).astype(np.float32)


class RedNeuronal:
    def __init__(self, n_entradas: int, capas: tuple[int, ...] | list[int] = (128, 64, 32), semilla: int = 0):
        self.n_entradas = int(n_entradas)
        self.capas = [int(c) for c in capas]
        rng = np.random.default_rng(semilla)
        dims = [self.n_entradas, *self.capas, 1]
        self.W = [
            (rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])).astype(np.float32)
            for i in range(len(dims) - 1)
        ]
        self.W[-1] *= 0.1
        self.b = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)]
        self._reiniciar_adam()
        self.normalizador = Normalizador()

    def _reiniciar_adam(self) -> None:
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    @property
    def n_parametros(self) -> int:
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))

    # ------------------------------------------------------------------
    def _propagar(self, X: np.ndarray):
        entradas, previas = [X], []
        h = X
        ultima = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            previas.append(z)
            h = z if i == ultima else np.where(z > 0, z, PENDIENTE_RELU * z)
            entradas.append(h)
        return h[:, 0], previas, entradas

    def logits(self, X: np.ndarray, lote: int = 16384) -> np.ndarray:
        salida = [self._propagar(X[i:i + lote])[0] for i in range(0, len(X), lote)]
        return np.concatenate(salida) if salida else np.zeros(0, dtype=np.float32)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Probabilidad de que cada fila sea un buen punto de compra.
        `X` debe estar ya normalizada."""
        return sigmoide(self.logits(X))

    def entrenar_lote(self, X: np.ndarray, y: np.ndarray, tasa: float = 1e-3, l2: float = 1e-4,
                      beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                      max_norma: float = 5.0) -> float:
        logits, previas, entradas = self._propagar(X)
        perdida = entropia_cruzada(logits, y)
        grad = ((sigmoide(logits) - y) / len(y)).astype(np.float32)[:, None]
        gW: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        gb: list[np.ndarray] = [None] * len(self.b)  # type: ignore[list-item]
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = entradas[i].T @ grad + l2 * self.W[i]
            gb[i] = grad.sum(axis=0)
            if i > 0:
                grad = (grad @ self.W[i].T) * np.where(previas[i - 1] > 0, 1.0, PENDIENTE_RELU).astype(np.float32)
        norma = float(np.sqrt(sum(float((g * g).sum()) for g in gW + gb)))
        if norma > max_norma:
            gW = [g * (max_norma / norma) for g in gW]
            gb = [g * (max_norma / norma) for g in gb]
        self.t += 1
        corr1 = 1 - beta1 ** self.t
        corr2 = 1 - beta2 ** self.t
        for params, grads, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
            for j in range(len(params)):
                m[j] = beta1 * m[j] + (1 - beta1) * grads[j]
                v[j] = beta2 * v[j] + (1 - beta2) * grads[j] * grads[j]
                params[j] -= (tasa * (m[j] / corr1) / (np.sqrt(v[j] / corr2) + eps)).astype(np.float32)
        return perdida

    # ------------------------------------------------------------------
    def copiar_pesos(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def restaurar_pesos(self, pesos: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        self.W = [w.copy() for w in pesos[0]]
        self.b = [b.copy() for b in pesos[1]]

    def guardar(self, ruta: str | Path) -> None:
        arrays = {
            "n_entradas": np.array(self.n_entradas),
            "capas": np.array(self.capas),
            "t": np.array(self.t),
        }
        for nombre, lista in (("W", self.W), ("b", self.b), ("mW", self.mW), ("vW", self.vW),
                              ("mb", self.mb), ("vb", self.vb)):
            for i, arr in enumerate(lista):
                arrays[f"{nombre}{i}"] = arr
        if self.normalizador.ajustado:
            arrays["media"] = self.normalizador.media
            arrays["desv"] = self.normalizador.desv
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ruta, **arrays)

    @classmethod
    def cargar(cls, ruta: str | Path) -> "RedNeuronal":
        with np.load(ruta, allow_pickle=False) as datos:
            red = cls(int(datos["n_entradas"]), [int(c) for c in datos["capas"]])
            n = len(red.W)
            red.W = [datos[f"W{i}"] for i in range(n)]
            red.b = [datos[f"b{i}"] for i in range(n)]
            red.mW = [datos[f"mW{i}"] for i in range(n)]
            red.vW = [datos[f"vW{i}"] for i in range(n)]
            red.mb = [datos[f"mb{i}"] for i in range(n)]
            red.vb = [datos[f"vb{i}"] for i in range(n)]
            red.t = int(datos["t"])
            if "media" in datos:
                red.normalizador = Normalizador(datos["media"], datos["desv"])
        return red
