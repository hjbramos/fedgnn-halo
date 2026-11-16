"""Encoder–decoder recorrente por nó (GRU ou LSTM), sem estrutura espacial.

Cada nó é uma sequência independente com pesos compartilhados (batch efetivo B·N), como o
"local model" GRU das especificações de CNFGNN / ST-PFLA: 1 camada, 64 unidades.
Entrada por passo = [features padronizadas, hora-do-dia]; decoder autoregressivo com
scheduled sampling (mesmo esquema do DCRNN) e entrada [previsão anterior, hora-do-dia].
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RNNSeq2Seq(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1,
                 horizon: int = 12, cell: str = "gru", cl_decay_steps: int = 2000, use_curriculum_learning: bool = True):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell deve ser 'gru' ou 'lstm'")
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.cell, self.horizon, self.output_dim = cell, int(horizon), int(output_dim)
        self.cl_decay_steps, self.use_cl = cl_decay_steps, use_curriculum_learning
        self.encoder = rnn(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = rnn(output_dim + 1, hidden_dim, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def _teacher_forcing_prob(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, x, x_tod, y=None, y_tod=None, batches_seen: int = 0):
        """x (B, W, N, F), x_tod (B, W); y (B, H, N, 1) só no treino; y_tod (B, H). Retorna (B, H, N, 1)."""
        B, W, N, F = x.shape
        tod = x_tod[:, :, None, None].expand(B, W, N, 1)
        seq = torch.cat([x, tod], dim=-1).permute(0, 2, 1, 3).reshape(B * N, W, F + 1)
        _, state = self.encoder(seq)

        prev = seq.new_zeros(B * N, 1, self.output_dim)  # símbolo GO
        outputs = []
        for t in range(self.horizon):
            step_tod = y_tod[:, t, None, None].expand(B, N, 1).reshape(B * N, 1, 1)
            out, state = self.decoder(torch.cat([prev, step_tod], dim=-1), state)
            out = self.projection(out)                     # (B·N, 1, output_dim)
            outputs.append(out)
            prev = out
            if self.training and self.use_cl and y is not None:
                if np.random.uniform(0, 1) < self._teacher_forcing_prob(batches_seen):
                    prev = y[:, t].reshape(B * N, 1, self.output_dim)
        pred = torch.cat(outputs, dim=1)                   # (B·N, H, output_dim)
        return pred.reshape(B, N, self.horizon, self.output_dim).permute(0, 2, 1, 3)
