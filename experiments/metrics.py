"""RegressionAccumulator — cópia de sunt-fedstgnn/src/fedstgnn/metrics.py — e NodeErrorAccumulator (erro por nó)."""
from __future__ import annotations

import math

import numpy as np
import torch


class RegressionAccumulator:
    def __init__(self):
        self.absolute_error = self.squared_error = self.absolute_target = 0.0
        self.target_sum = self.target_squared_sum = self.absolute_percentage_error = 0.0
        self.observations = self.positive_targets = 0

    def update(self, prediction, target, mask):
        observed = target[mask].double()
        error = prediction[mask].double() - observed
        if not torch.isfinite(observed).all() or not torch.isfinite(error).all():
            raise ValueError("Non-finite observed target or prediction")
        positive = observed > 0
        self.absolute_error += error.abs().sum().item()
        self.squared_error += error.square().sum().item()
        self.absolute_target += observed.abs().sum().item()
        self.target_sum += observed.sum().item()
        self.target_squared_sum += observed.square().sum().item()
        self.absolute_percentage_error += (error[positive].abs() / observed[positive]).sum().item()
        self.observations += observed.numel()
        self.positive_targets += int(positive.sum().item())

    def compute(self) -> dict[str, float]:
        if self.observations <= 0:
            raise ValueError("No valid observations available for metrics")
        variation = self.target_squared_sum - self.target_sum**2 / self.observations
        mse = self.squared_error / self.observations
        return {
            "MSE": mse,
            "MAE": self.absolute_error / self.observations,
            "RMSE": math.sqrt(mse),
            "MAPE_percent": 100.0 * self.absolute_percentage_error / self.positive_targets if self.positive_targets else math.nan,
            "MAPE_coverage_percent": 100.0 * self.positive_targets / self.observations,
            "WAPE_percent": 100.0 * self.absolute_error / self.absolute_target if self.absolute_target > 0 else math.nan,
            "R2": 1.0 - self.squared_error / variation if variation > 0 else math.nan,
            "observations": int(self.observations),
            "positive_targets": int(self.positive_targets),
        }


class NodeErrorAccumulator:
    """Soma do erro absoluto e nº de alvos observados por nó GLOBAL (para a análise estratificada: fronteira/interior etc.).

    `prediction`/`target`/`mask` têm forma (B, H, n, 1) com n nós locais; `node_indices` mapeia-os para os índices globais.
    """

    def __init__(self, num_nodes: int):
        self.abs_error = np.zeros(num_nodes, dtype=np.float64)
        self.count = np.zeros(num_nodes, dtype=np.int64)

    def update(self, prediction, target, mask, node_indices):
        err = ((prediction - target).abs().double() * mask).sum(dim=(0, 1, 3)).cpu().numpy()
        cnt = mask.sum(dim=(0, 1, 3)).cpu().numpy()
        idx = np.asarray(node_indices, dtype=np.int64)
        self.abs_error[idx] += err
        self.count[idx] += cnt

    def compute(self) -> dict:
        return {"abs_error": self.abs_error.tolist(), "count": self.count.tolist()}
