"""Operações sobre estados compatíveis; grafos locais não devem integrar o estado compartilhado."""
import torch
from torch import nn


def aggregate_states(states: list[dict], weights: list[float]) -> dict:
    total = float(sum(weights))
    out = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float64)
        for st, w in zip(states, weights):
            acc += st[key].to(torch.float64) * (float(w) / total)
        out[key] = acc.to(states[0][key].dtype)
    return out


def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


