"""Série sintética pequena com separação temporal de treino, validação e teste."""
import numpy as np
import pandas as pd
import torch
from experiments.data import ExperimentData


def synthetic(seed=7, nodes=12, steps=120, input_steps=6, horizon=2):
    if nodes < 2 or input_steps < 1 or horizon < 1:
        raise ValueError("Use ao menos dois nós e janelas positivas")
    train_end, val_end = int(steps * .6), int(steps * .8)
    bounds = [(input_steps, train_end), (train_end, val_end), (val_end, steps)]
    origins = {key: np.arange(start, end - horizon + 1) for key, (start, end) in
               zip(('train', 'validation', 'test'), bounds)}
    if any(len(v) == 0 for v in origins.values()):
        raise ValueError("Série insuficiente para os três splits")
    rng = np.random.default_rng(seed)
    time = np.arange(steps)[:, None]
    series = (3 + np.sin(time / 6 + np.arange(nodes)[None, :] / 4)
              + rng.normal(0, .05, (steps, nodes))).astype(np.float32)[..., None]
    adj = np.zeros((nodes, nodes), np.float32)
    for i in range(nodes):
        j = (i + 1) % nodes
        adj[i, j] = adj[j, i] = 1
    src, dst = np.nonzero(adj)
    return ExperimentData(
        name='synthetic', series=series, mask=np.ones_like(series, dtype=bool),
        feature_names=['signal'], target_feature_index=0,
        timestamps=pd.date_range('2024-01-01', periods=steps, freq='h'),
        node_ids=[str(i) for i in range(nodes)], adj=adj,
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_weight=torch.tensor(adj[src, dst]), origins=origins,
        input_steps=input_steps, horizon=horizon,
        scaler_mean=series[:train_end].mean(axis=(0, 1)),
        scaler_std=series[:train_end].std(axis=(0, 1)))
