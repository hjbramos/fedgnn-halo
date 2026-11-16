"""DCRNN — Li, Yu, Shahabi & Liu (2017), "Diffusion Convolutional Recurrent Neural Network".

Encoder–decoder de DCGRUs com convolução de difusão bidirecional (random walk direto e
reverso até K passos) e scheduled sampling no decoder. Segue a implementação de referência
(liyaguang/DCRNN, dcrnn_cell.py / dcrnn_model.py) com os hiperparâmetros do paper.

Opções (padrão = comportamento original):
  input_masks   concatena as máscaras de observação às features (ausência ≠ zero), como a LEConv;
  time_of_day   hora do dia na entrada do encoder e do decoder (original: sim);
  no_messages   controle sem comunicação: as matrizes de transição viram N×N vazias. A parametrização é a
                mesma, mas a recursão T_k deixa de misturar nós: P x = 0 e T_2 = 2P(Px) − x = −x, logo os
                termos k=1 ficam inativos (sem gradiente) e os k=2 viram pesos locais redundantes (−x).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def empty_supports(nodes: int, n: int = 2) -> list[torch.Tensor]:
    """Matrizes de transição sem nenhuma entrada (controle no_messages), no mesmo formato esparso."""
    return [torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), (nodes, nodes)).coalesce()
            for _ in range(n)]


def random_walk_supports(adj: np.ndarray, bidirectional: bool = True) -> list[torch.Tensor]:
    """Matrizes de transição esparsas: D_O^-1 W (direto) e D_I^-1 W^T (reverso), transpostas para uso como P^T @ X."""
    adj = np.asarray(adj, dtype=np.float32)

    def rw(a):
        d = a.sum(1)
        d_inv = np.where(d > 0, 1.0 / d, 0.0)
        return (d_inv[:, None] * a).T  # (D^-1 W)^T, para calcular P @ X como (P^T)^T X

    mats = [rw(adj)]
    if bidirectional:
        mats.append(rw(adj.T))
    return [torch.from_numpy(m).to_sparse_coo() for m in mats]


class DiffusionGraphConv(nn.Module):
    """Σ_k Σ_p θ_{k,p} (P_p)^k X — Chebyshev-like recursão da implementação de referência."""

    def __init__(self, supports: list[torch.Tensor], input_dim: int, hidden_dim: int, output_dim: int, K: int, bias_start: float = 0.0):
        super().__init__()
        self.K = int(K)
        for i, s in enumerate(supports):
            self.register_buffer(f"support_{i}", s, persistent=False)
        self.n_supports = len(supports)
        n_matrices = self.n_supports * self.K + 1
        self.weight = nn.Parameter(torch.empty((input_dim + hidden_dim) * n_matrices, output_dim))
        self.bias = nn.Parameter(torch.full((output_dim,), float(bias_start)))
        nn.init.xavier_normal_(self.weight)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor, nodes: int) -> torch.Tensor:
        # inputs (B, N, input_dim), state (B, N, hidden_dim) -> (B, N, output_dim)
        B = inputs.shape[0]
        x = torch.cat([inputs, state], dim=-1)                       # (B, N, C)
        C = x.shape[-1]
        x0 = x.permute(1, 2, 0).reshape(nodes, C * B)                # (N, C·B)
        xs = [x0]
        for i in range(self.n_supports):
            if self.K < 1:      # K=0: só o termo identidade, coerente com n_matrices = n_supports·K + 1
                break
            support = getattr(self, f"support_{i}")
            x1 = torch.sparse.mm(support, x0)
            xs.append(x1)
            for _ in range(2, self.K + 1):
                x2 = 2 * torch.sparse.mm(support, x1) - x0
                xs.append(x2)
                x1, x0 = x2, x1
            x0 = xs[0]
        x = torch.stack(xs, 0)                                       # (M, N, C·B)
        x = x.reshape(len(xs), nodes, C, B).permute(3, 1, 2, 0).reshape(B * nodes, C * len(xs))
        return (x @ self.weight + self.bias).reshape(B, nodes, -1)


class DCGRUCell(nn.Module):
    def __init__(self, supports, input_dim, hidden_dim, K):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = DiffusionGraphConv(supports, input_dim, hidden_dim, 2 * hidden_dim, K, bias_start=1.0)
        self.candidate = DiffusionGraphConv(supports, input_dim, hidden_dim, hidden_dim, K)

    def forward(self, inputs, state, nodes):
        r, u = torch.sigmoid(self.gate(inputs, state, nodes)).chunk(2, dim=-1)
        c = torch.tanh(self.candidate(inputs, r * state, nodes))
        return u * state + (1.0 - u) * c


class DCRNN(nn.Module):
    """Seq2seq: encoder (W passos) -> decoder (H passos) com scheduled sampling durante o treino."""

    def __init__(self, adj: np.ndarray, input_dim: int = 2, output_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, max_diffusion_step: int = 2, horizon: int = 12,
                 cl_decay_steps: int = 2000, use_curriculum_learning: bool = True,
                 gradient_checkpointing: bool = False, input_masks: bool = False, time_of_day: bool = True,
                 no_messages: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing  # recomputa cada passo no backward (menos memória)
        self.input_masks, self.time_of_day, self.no_messages = bool(input_masks), bool(time_of_day), bool(no_messages)
        supports = empty_supports(adj.shape[0]) if self.no_messages else random_walk_supports(adj)
        self.nodes = adj.shape[0]
        self.hidden_dim, self.num_layers, self.horizon = hidden_dim, num_layers, int(horizon)
        self.input_dim, self.output_dim = input_dim, output_dim
        self.cl_decay_steps, self.use_cl = cl_decay_steps, use_curriculum_learning
        self.encoder = nn.ModuleList(
            [DCGRUCell(supports, input_dim if l == 0 else hidden_dim, hidden_dim, max_diffusion_step) for l in range(num_layers)])
        # o decoder recebe [alvo anterior (output_dim), hora-do-dia (se time_of_day)]
        dec_in = output_dim + int(self.time_of_day)
        self.decoder = nn.ModuleList(
            [DCGRUCell(supports, dec_in if l == 0 else hidden_dim, hidden_dim, max_diffusion_step) for l in range(num_layers)])
        self.projection = nn.Linear(hidden_dim, output_dim)

    def _teacher_forcing_prob(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def _run(self, cells, inputs, states):
        new_states = []
        for cell, state in zip(cells, states):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                inputs = checkpoint(cell, inputs, state, self.nodes, use_reentrant=False)
            else:
                inputs = cell(inputs, state, self.nodes)
            new_states.append(inputs)
        return inputs, new_states

    def forward(self, x, x_tod, y=None, y_tod=None, batches_seen: int = 0, x_mask=None):
        """x (B, W, N, F) padronizado, x_tod (B, W); y (B, H, N, 1) padronizado só no treino; y_tod (B, H);
        x_mask (B, W, N, F) obrigatório com input_masks. Retorna (B, H, N, 1) padronizado."""
        if self.input_masks:
            if x_mask is None:
                raise ValueError("DCRNN com input_masks requer x_mask")
            x = torch.cat([x, x_mask.to(x.dtype)], dim=-1)
        B, W, N, _ = x.shape
        states = [x.new_zeros(B, N, self.hidden_dim) for _ in range(self.num_layers)]
        for t in range(W):
            step = torch.cat([x[:, t], x_tod[:, t, None, None].expand(B, N, 1)], dim=-1) if self.time_of_day else x[:, t]
            _, states = self._run(self.encoder, step, states)

        go = x.new_zeros(B, N, self.output_dim)
        prev = go
        outputs = []
        for t in range(self.horizon):
            step = torch.cat([prev, y_tod[:, t, None, None].expand(B, N, 1)], dim=-1) if self.time_of_day else prev
            out, states = self._run(self.decoder, step, states)
            out = self.projection(out)
            outputs.append(out)
            prev = out
            if self.training and self.use_cl and y is not None:
                if np.random.uniform(0, 1) < self._teacher_forcing_prob(batches_seen):
                    prev = y[:, t]
        return torch.stack(outputs, dim=1)
