"""Entradas explícitas para construir modelos, sem acesso a datasets ou configuração de runs."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class EdgeAttributes:
    static: torch.Tensor
    profiles: torch.Tensor
    mask: torch.Tensor
    scaler_mean: torch.Tensor
    scaler_std: torch.Tensor
    has_attributes: torch.Tensor


@dataclass(frozen=True)
class ModelSpec:
    nodes: int
    input_steps: int
    horizon: int
    n_features: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    adj: np.ndarray | None = None
    edge_attributes: EdgeAttributes | None = None
