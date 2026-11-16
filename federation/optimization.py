"""Treino local com objetivo injetado: nenhum contrato de batch ou métrica de tarefa."""
from dataclasses import dataclass
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    metric_sum: float
    count: int


def proximal_penalty(parameters: Sequence[torch.Tensor], anchors: Sequence[torch.Tensor]):
    """Distância quadrática à referência fixa; o chamador escolhe o coeficiente."""
    if len(parameters) != len(anchors):
        raise ValueError("parameters e anchors devem ter o mesmo tamanho")
    if any(p.shape != a.shape for p, a in zip(parameters, anchors)):
        raise ValueError("parameters e anchors devem ter shapes compatíveis")
    return sum(((p - a.detach()) ** 2).sum() for p, a in zip(parameters, anchors))


def train_local_epoch(model: nn.Module, loader: Iterable, optimizer, objective: Callable,
                      clip: float, batches_seen: int = 0, regularizer: Callable | None = None):
    """objective(model, batch, step) retorna ObjectiveResult ou None para ignorar o batch.

    regularizer(model), quando fornecido, altera a perda mas não a métrica reportada.
    O contador é explícito para permitir scheduled sampling e retomada.
    """
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        result = objective(model, batch, batches_seen)
        if result is None:
            continue
        loss = result.loss
        if regularizer is not None:
            loss = loss + regularizer(model)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += result.metric_sum
        count += result.count
        batches_seen += 1
    if count == 0:
        raise ValueError("época local sem alvos observados; verifique o objetivo e os dados do cliente")
    return total / count, batches_seen
