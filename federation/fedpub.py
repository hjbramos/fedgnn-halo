"""Similaridade, agregação personalizada e máscaras locais do FED-PUB."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from .aggregation import cpu_state


def similarity_and_weights(embeddings: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """S = cosseno entre embeddings; α_ij = exp(τ S_ij) / Σ_k exp(τ S_ik) (Eq. 3)."""
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    unit = embeddings / norm
    S = unit @ unit.T
    logits = tau * S
    logits -= logits.max(axis=1, keepdims=True)
    alpha = np.exp(logits)
    alpha /= alpha.sum(axis=1, keepdims=True)
    return S, alpha


def personalized_aggregate(sent: list[dict], alpha_row: np.ndarray) -> dict:
    """θ̄_i = Σ_j α_ij θ_j sobre os estados enviados (pesos zerados pelas máscaras contam como zero)."""
    out = {}
    for key in sent[0]:
        acc = torch.zeros_like(sent[0][key], dtype=torch.float64)
        for st, a in zip(sent, alpha_row):
            acc += st[key].to(torch.float64) * float(a)
        out[key] = acc.to(sent[0][key].dtype)
    return out


# ----------------------------------------------------------------------------------------------
# modelo com máscara local θ ⊙ μ
# ----------------------------------------------------------------------------------------------
def maskable(name: str, param: torch.Tensor) -> bool:
    return param.dim() >= 2


class MaskedModel(nn.Module):
    """Aplica a máscara local μ (Sec. 4.2) aos pesos do modelo base: forward com θ ⊙ μ via functional_call.

    Em treino usa μ contínua; em avaliação zera as entradas com |μ| <= threshold (Ap. B.3).
    """

    def __init__(self, model: nn.Module, masks: dict[str, torch.Tensor], threshold: float):
        super().__init__()
        self.model = model
        self.threshold = float(threshold)
        self.mask_names = list(masks)
        self.masks = nn.ParameterList([nn.Parameter(masks[n].clone()) for n in self.mask_names])

    def _effective_params(self) -> dict[str, torch.Tensor]:
        params = dict(self.model.named_parameters())
        eff = {}
        for name, mask in zip(self.mask_names, self.masks):
            m = mask if self.training else mask * (mask.abs() > self.threshold)
            eff[name] = params[name] * m
        return eff

    def forward(self, *args, **kwargs):
        return functional_call(self.model, self._effective_params(), args, kwargs)

    def mask_l1(self) -> torch.Tensor:
        return sum(m.abs().sum() for m in self.masks)

    def masks_cpu(self) -> dict[str, torch.Tensor]:
        return {n: m.detach().cpu().clone() for n, m in zip(self.mask_names, self.masks)}

    @torch.no_grad()
    def effective_state(self) -> dict[str, torch.Tensor]:
        """state_dict com θ ⊙ μ (μ limiarizada) — o modelo que o cliente usa para prever."""
        state = cpu_state(self.model)
        for name, mask in zip(self.mask_names, self.masks):
            m = mask.detach().cpu()
            state[name] = state[name] * (m * (m.abs() > self.threshold))
        return state

    @torch.no_grad()
    def sent_state(self) -> tuple[dict[str, torch.Tensor], int]:
        """Estado enviado ao servidor: θ com as posições esparsificadas a zero; devolve também o nº de valores transmitidos."""
        state = cpu_state(self.model)
        count = 0
        for key, value in state.items():
            if key in self.mask_names:
                keep = self.masks[self.mask_names.index(key)].detach().cpu().abs() > self.threshold
                state[key] = value * keep
                count += int(keep.sum())
            else:
                count += value.numel()
        return state, count


def mask_sparsity(masks: dict[str, torch.Tensor], threshold: float) -> float:
    total = sum(m.numel() for m in masks.values())
    zeros = sum(int((m.abs() <= threshold).sum()) for m in masks.values())
    return zeros / total if total else 0.0



@dataclass
class FedPubRound:
    received: list[dict]
    masks: list[dict]
    trained: list[dict]
    effective: list[dict]
    similarity: np.ndarray
    alpha: np.ndarray
    metric: float
    batches_seen: int
    sent_counts: list[int]


def fedpub_round(received, masks, clients, weights, make_model, make_optimizer, train_epoch,
                 embedding, threshold, tau, local_epochs, batches_seen=0):
    """Uma rodada FED-PUB; embedding(state) define o comportamento observado no proxy.

    train_epoch(masked, client, optimizer, anchors, step) é fornecido pela tarefa.
    sent_counts permite ao experimento contabilizar o payload conforme seu protocolo.
    """
    if (not clients or len(clients) != len(weights) or len(received) != len(clients)
            or len(masks) != len(clients) or local_epochs < 1 or sum(weights) <= 0):
        raise ValueError("rodada FED-PUB requer estados, máscaras, clientes, pesos e épocas válidos")
    sent, effective, embeddings, updated_masks, trained, counts = [], [], [], [], [], []
    error, observations = 0.0, 0
    for state, mask, client, weight in zip(received, masks, clients, weights):
        model = make_model(client)
        model.load_state_dict(state, strict=True)
        # Máscaras persistidas em CPU acompanham os parâmetros locais na retomada.
        params = dict(model.named_parameters())
        masked = MaskedModel(model, {key: value.to(params[key].device) for key, value in mask.items()}, threshold)
        anchors = [p.detach().clone() for p in model.parameters()]
        optimizer = make_optimizer(masked)
        for _ in range(local_epochs):
            metric, batches_seen = train_epoch(masked, client, optimizer, anchors, batches_seen)
            error += metric * weight
            observations += weight
        updated_masks.append(masked.masks_cpu())
        trained.append(cpu_state(model))
        effective.append(masked.effective_state())
        embeddings.append(embedding(effective[-1]))
        state, count = masked.sent_state()
        sent.append(state)
        counts.append(count)
        del model, masked, optimizer, anchors
    similarity, alpha = similarity_and_weights(np.stack(embeddings), tau)
    received = [personalized_aggregate(sent, row) for row in alpha]
    return FedPubRound(received, updated_masks, trained, effective, similarity, alpha,
                       error / observations, batches_seen, counts)
