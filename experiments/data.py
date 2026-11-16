"""Monta os tensores de experimento a partir dos loaders (série, máscara, grafo, origens, scaler).

Convenções (== sunt-fedstgnn):
  - series (T, N, F) float32, mask (T, N, F) bool (True = observado); alvo = feature target_feature_index
  - origin = índice do 1º alvo; entrada = [origin-W, origin), alvo = [origin, origin+H)
  - scaler = média/desvio dos valores observados do treino (padronização global)
  - edge_index/edge_weight = entradas não nulas da adjacência
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import data_loader as dl
from data_loader import sunt
from data_loader.common import hour_of_week, split_by_dates, time_of_day


@dataclass
class ExperimentData:
    name: str
    series: np.ndarray            # (T, N, F)
    mask: np.ndarray              # (T, N, F) bool
    feature_names: list[str]
    target_feature_index: int
    timestamps: pd.DatetimeIndex
    node_ids: list[str]
    adj: np.ndarray               # (N, N) adjacência original
    edge_index: torch.Tensor      # (2, E)
    edge_weight: torch.Tensor     # (E,)
    origins: dict[str, np.ndarray]
    input_steps: int
    horizon: int
    scaler_mean: np.ndarray       # (F,)
    scaler_std: np.ndarray        # (F,)
    graph: str = "default"        # SUNT: transit (simétrico) | transit_directed; demais: grafo do dataset
    origin_steps: int | None = None   # origens elegíveis calculadas com esta janela (None = input_steps)

    @property
    def nodes(self) -> int:
        return self.series.shape[1]

    @property
    def n_features(self) -> int:
        return self.series.shape[2]

    @property
    def target_mean(self) -> float:
        return float(self.scaler_mean[self.target_feature_index])

    @property
    def target_std(self) -> float:
        return float(self.scaler_std[self.target_feature_index])


def _graph_tensors(adj: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = np.nonzero(adj)
    return (
        torch.from_numpy(np.stack([src, dst]).astype(np.int64)),
        torch.from_numpy(adj[src, dst].astype(np.float32)),
    )


def _origins_by_ratio(T: int, W: int, H: int, train: float = 0.7, val: float = 0.1) -> dict[str, np.ndarray]:
    """Split cronológico 70/10/20 sobre as janelas (padrão DCRNN); origin = 1º alvo."""
    n = T - W - H + 1
    n_train, n_val = int(n * train), int(n * val)
    o = np.arange(n) + W
    return {"train": o[:n_train], "validation": o[n_train : n_train + n_val], "test": o[n_train + n_val :]}


def _fit_scaler(series: np.ndarray, mask: np.ndarray, train_end_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Média/desvio por feature sobre os valores observados do treino."""
    mean, std = [], []
    for k in range(series.shape[2]):
        values = series[:train_end_idx, :, k][mask[:train_end_idx, :, k]]
        mean.append(float(values.mean(dtype=np.float64)))
        std.append(float(values.std(dtype=np.float64)))
    mean, std = np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    return mean, std


def load_experiment(name: str, input_steps: int, horizon: int, features: list[str] | None = None,
                    target: str | None = None, graph: str | None = None, origin_steps: int | None = None) -> ExperimentData:
    """features/target/graph só se aplicam ao SUNT (METR-LA/PEMS-BAY têm uma única feature: velocidade).

    graph=None mantém o grafo padrão do dataset (no SUNT, o 'transit' simétrico).
    origin_steps (só SUNT): calcula as origens elegíveis como se a janela tivesse `origin_steps` passos, de modo que
    uma janela mais curta use exatamente as origens (e os alvos) da mais longa, sem exemplos adicionais no início do
    treino. None = input_steps (comportamento de sempre)."""
    origin_steps = int(origin_steps or input_steps)
    if origin_steps < input_steps:
        raise ValueError("origin_steps não pode ser menor que input_steps (a janela sairia da série)")
    if origin_steps != input_steps and name != "sunt":
        raise ValueError("origin_steps só é suportado no SUNT")
    df = dl.load(name)
    adj, ids, df = dl.load_adjacency(name, df, graph)   # colunas de df na ordem do grafo
    feats, mask_all, timestamps, feat_ids = dl.load_features(name)
    if name == "sunt" and graph and graph.startswith(sunt.ITINERARY_PREFIX):
        # recorte de paradas (SUNT/Montevideo): séries e máscaras nas colunas do grafo, na ordem do grafo
        col = {s: i for i, s in enumerate(feat_ids)}
        sel = np.array([col[s] for s in ids])
        feats, mask_all, feat_ids = feats[:, sel], mask_all[:, sel], [feat_ids[i] for i in sel]
    assert feat_ids == ids, "features e grafo com ordens de nós diferentes"
    all_names = dl.feature_names(name)
    feature_names = list(features or all_names)
    cols = [all_names.index(f) for f in feature_names]
    series, mask = feats[:, :, cols], mask_all[:, :, cols]
    target_index = feature_names.index(target or all_names[0])
    if name in ("metr-la", "pems-bay"):
        origins = _origins_by_ratio(len(timestamps), input_steps, horizon)
    elif name == "sunt":
        tr, va, te = split_by_dates(timestamps, sunt.TRAIN_END_EXCLUSIVE, sunt.VALIDATION_END_EXCLUSIVE, origin_steps, horizon)
        origins = {"train": tr + origin_steps, "validation": va + origin_steps, "test": te + origin_steps}
    else:
        raise ValueError(f"dataset desconhecido: {name}")

    edge_index, edge_weight = _graph_tensors(adj)
    train_end_idx = int(origins["validation"][0]) if len(origins["validation"]) else len(series)
    mean, std = _fit_scaler(series, mask, train_end_idx)
    return ExperimentData(
        name=name, series=np.ascontiguousarray(series), mask=np.ascontiguousarray(mask),
        feature_names=feature_names, target_feature_index=target_index,
        timestamps=timestamps, node_ids=ids, adj=np.asarray(adj, dtype=np.float32), edge_index=edge_index, edge_weight=edge_weight,
        origins=origins, input_steps=input_steps, horizon=horizon, origin_steps=origin_steps, scaler_mean=mean, scaler_std=std,
        graph=graph or ("transit" if name == "sunt" else "default"),
    )


class WindowDataset(Dataset):
    """Janelas causais construídas sob demanda (== fedstgnn.data.WindowDataset, F features, 1 alvo)."""

    def __init__(self, data: ExperimentData, split: str, node_indices: np.ndarray | None = None):
        self.series, self.mask = data.series, data.mask
        self.nodes = None if node_indices is None else np.asarray(node_indices, dtype=np.int64)
        self.origins = data.origins[split]
        self.W, self.H = data.input_steps, data.horizon
        self.mean, self.std = data.scaler_mean, data.scaler_std
        self.k = data.target_feature_index
        self.tmean, self.tstd = np.float32(data.target_mean), np.float32(data.target_std)
        self.tod = time_of_day(data.timestamps)  # (T,) fração do dia — feature extra do DCRNN
        self.how = hour_of_week(data.timestamps)  # (T,) 0..167 — usado pelos pesos de aresta dinâmicos

    def __len__(self):
        return len(self.origins)

    def _slice(self, a, b):
        if self.nodes is None:
            return self.series[a:b], self.mask[a:b]
        return self.series[a:b, self.nodes], self.mask[a:b, self.nodes]

    def __getitem__(self, i):
        o = int(self.origins[i])
        x_raw, x_mask = self._slice(o - self.W, o)                      # (W, N, F)
        y_all, y_mask_all = self._slice(o, o + self.H)
        y_raw = y_all[:, :, self.k : self.k + 1]                          # (H, N, 1) — só o alvo
        y_mask = y_mask_all[:, :, self.k : self.k + 1]
        x = np.where(x_mask, (x_raw - self.mean) / self.std, 0.0).astype(np.float32)
        y = np.where(y_mask, (y_raw - self.tmean) / self.tstd, 0.0).astype(np.float32)
        return {
            "x": torch.from_numpy(x), "x_mask": torch.from_numpy(x_mask.copy()),
            "y": torch.from_numpy(y), "y_mask": torch.from_numpy(y_mask.copy()),
            "y_raw": torch.from_numpy(y_raw.copy()),  # métricas usam o alvo bruto exato
            "x_tod": torch.from_numpy(self.tod[o - self.W : o].copy()),
            "y_tod": torch.from_numpy(self.tod[o : o + self.H].copy()),
            "y_how": torch.from_numpy(self.how[o : o + self.H].copy()),
        }
