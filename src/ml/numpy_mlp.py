"""Perceptrón multicapa en NumPy (procede de la primera versión del proyecto).

Se usa como benchmark sencillo: sin capas ocultas es una regresión logística
sobre las features del último día, entrenada solo con datos de entrenamiento.
"""

from __future__ import annotations

import numpy as np

LEAKY = 0.01


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class NumpyMLP:
    def __init__(self, n_inputs: int, hidden: tuple[int, ...] | list[int] = (), seed: int = 0):
        rng = np.random.default_rng(seed)
        dims = [int(n_inputs), *[int(h) for h in hidden], 1]
        self.W = [(rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])).astype(np.float32)
                  for i in range(len(dims) - 1)]
        self.W[-1] *= 0.1
        self.b = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _forward(self, X):
        entradas, previas, h = [X], [], X
        ultima = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            previas.append(z)
            h = z if i == ultima else np.where(z > 0, z, LEAKY * z)
            entradas.append(h)
        return h[:, 0], previas, entradas

    def predict_proba(self, X: np.ndarray, batch: int = 65536) -> np.ndarray:
        partes = [sigmoid(self._forward(X[i:i + batch])[0]) for i in range(0, len(X), batch)]
        return np.concatenate(partes) if partes else np.zeros(0, np.float32)

    def train_batch(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-3, l2: float = 1e-4) -> float:
        logits, previas, entradas = self._forward(X)
        perdida = float(np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))))
        grad = ((sigmoid(logits) - y) / len(y)).astype(np.float32)[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = entradas[i].T @ grad + l2 * self.W[i]
            gb[i] = grad.sum(axis=0)
            if i > 0:
                grad = (grad @ self.W[i].T) * np.where(previas[i - 1] > 0, 1.0, LEAKY).astype(np.float32)
        self.t += 1
        c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for params, grads, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
            for j in range(len(params)):
                m[j] = 0.9 * m[j] + 0.1 * grads[j]
                v[j] = 0.999 * v[j] + 0.001 * grads[j] * grads[j]
                params[j] -= (lr * (m[j] / c1) / (np.sqrt(v[j] / c2) + 1e-8)).astype(np.float32)
        return perdida

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 3, batch: int = 1024, lr: float = 3e-3,
            seed: int = 0) -> "NumpyMLP":
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            orden = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                b = orden[i:i + batch]
                self.train_batch(X[b], y[b], lr=lr)
        return self
