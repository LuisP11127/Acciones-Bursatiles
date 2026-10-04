"""Red neuronal temporal (GRU o LSTM) con tres salidas:

- probabilidad de oportunidad P(oportunidad | datos hasta T)
- retorno esperado de la operación simulada
- drawdown esperado durante la operación
"""

from __future__ import annotations

import torch
from torch import nn

TARGET_SCALE = 0.1  # los objetivos de retorno/drawdown se escalan para equilibrar la pérdida


class OpportunityNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 48, num_layers: int = 1, dropout: float = 0.25,
                 architecture: str = "gru"):
        super().__init__()
        if architecture not in ("gru", "lstm"):
            raise ValueError(f"Arquitectura desconocida: {architecture}")
        celda = nn.GRU if architecture == "gru" else nn.LSTM
        self.rnn = celda(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 32)
        self.drop2 = nn.Dropout(dropout)
        self.head_prob = nn.Linear(32, 1)
        self.head_return = nn.Linear(32, 1)
        self.head_drawdown = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        salida, _ = self.rnn(x)
        h = self.norm(salida[:, -1, :])
        z = self.drop2(torch.relu(self.fc(self.drop1(h))))
        return (self.head_prob(z).squeeze(-1), self.head_return(z).squeeze(-1),
                self.head_drawdown(z).squeeze(-1))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
