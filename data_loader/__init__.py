"""Loaders dos datasets de tráfego/transporte — API geral por nome do dataset.

    import data_loader as dl
    dl.download("pems-bay")                 # baixa o bruto para data/pems-bay/raw (idempotente)
    df = dl.load("pems-bay")                # DataFrame (T x N): índice = timestamp, colunas = nós
    adj, ids, df = dl.load_adjacency("pems-bay", df)   # adj (N x N) e df alinhado à ordem do grafo
    feats, mask, ts, ids = dl.load_features("sunt")    # (T, N, F) multi-feature + máscara (só SUNT)

    python -m data_loader --dataset all     # baixa e valida

Os módulos por dataset (metr_la, pems_bay, sunt) continuam acessíveis para detalhes específicos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metr_la, pems_bay, sunt
from .common import DATA_DIR, make_windows, split_by_dates, split_temporal, time_of_day

DATASETS = ("metr-la", "pems-bay", "sunt")

_MODULES = {"metr-la": metr_la, "pems-bay": pems_bay, "sunt": sunt}
_LOADERS = {"metr-la": metr_la.load_metr_la, "pems-bay": pems_bay.load_pems_bay, "sunt": sunt.load_sunt}


def _module(name: str):
    if name not in _MODULES:
        raise ValueError(f"dataset desconhecido: {name!r}; opções: {DATASETS}")
    return _MODULES[name]


def download(name: str) -> None:
    """Baixa os arquivos brutos do dataset (pula o que já existe)."""
    _module(name).download_raw()


def load(name: str, force: bool = False) -> pd.DataFrame:
    """Série principal (T x N) do dataset: velocidade (mph) no METR-LA/PEMS-BAY, embarques/h no SUNT."""
    _module(name)
    return _LOADERS[name](force=force)


def load_adjacency(name: str, df: pd.DataFrame, graph: str | None = None) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """(adj (N x N), ids, df com colunas na ordem da adjacência). graph só se aplica ao SUNT."""
    if graph is not None:
        if name != "sunt":
            raise ValueError("a escolha de grafo só existe no SUNT")
        return _module(name).load_adjacency(df, graph)
    return _module(name).load_adjacency(df)


def load_features(name: str, force: bool = False):
    """(features (T, N, F), mask (T, N, F), timestamps, ids). Só o SUNT tem F > 1."""
    if name == "sunt":
        return sunt.load_sunt_features(force)
    df = load(name, force)
    values = df.to_numpy(dtype=np.float32)[..., None]
    mask = values != 0 if name == "metr-la" else np.ones_like(values, dtype=bool)  # METR-LA: 0 = faltante
    return values, mask, df.index, [str(c) for c in df.columns]


def feature_names(name: str) -> list[str]:
    return list(sunt.FEATURE_NAMES) if name == "sunt" else ["speed"]


__all__ = ["DATASETS", "DATA_DIR", "download", "load", "load_adjacency", "load_features", "feature_names",
           "metr_la", "pems_bay", "sunt", "make_windows", "split_temporal", "split_by_dates", "time_of_day"]
