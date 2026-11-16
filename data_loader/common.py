"""Utilitários compartilhados pelos loaders (download, janelas, split)."""
from __future__ import annotations

from pathlib import Path
import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FEDGNN_DATA_DIR", str(Path.cwd() / "data"))).expanduser().resolve()


def _report(count: int, block: int, total: int) -> None:
    done = min(count * block, total)
    if done % (10 * block * 1024) < block or done == total:  # imprime a cada ~10 MB
        print(f"  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", flush=True)


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Baixa `url` para `dest` se ainda não existir."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"baixando {dest.name}")
    urlretrieve(url, dest, reporthook=_report)
    return dest


def download_hf(repo: str, files: list[str], dest_dir: Path, force: bool = False) -> None:
    """Baixa arquivos de um dataset do Hugging Face para `dest_dir` (mantendo subpastas)."""
    missing = [f for f in files if not (dest_dir / f).exists()]
    if not missing and not force:
        return
    from huggingface_hub import hf_hub_download

    print(f"baixando {len(missing)} arquivo(s) de {repo}")
    for f in missing if not force else files:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=dest_dir)


def align_to_graph(df: pd.DataFrame, sensor_ids: list[str]) -> pd.DataFrame:
    """Reordena as colunas de df para a ordem de sensor_ids.

    A ordem é crucial: coluna i da série == linha/coluna i da adjacência.
    """
    df_ids = [str(c) for c in df.columns]
    assert set(df_ids) == set(sensor_ids), "conjunto de sensores diferente entre série e grafo"
    if df_ids != sensor_ids:
        df = df[sensor_ids]
    df.columns = sensor_ids
    return df


def time_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Fração do dia em [0, 1) para cada timestamp."""
    vals = index.values
    return ((vals - vals.astype("datetime64[D]")) / np.timedelta64(1, "D")).astype(np.float32)


def hour_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Índice 0..167 da hora da semana (segunda 00h = 0) para cada timestamp."""
    return (index.dayofweek.values * 24 + index.hour.values).astype(np.int64)


def make_windows(
    df: pd.DataFrame, seq_len: int = 12, horizon: int = 12, add_time_of_day: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Janelas deslizantes no estilo DCRNN.

    Retorna x (S, seq_len, N, F) e y (S, horizon, N, F), F = 1 (+1 com time-of-day).
    """
    values = df.values.astype(np.float32)[..., None]  # (T, N, 1)
    if add_time_of_day:
        tod = np.tile(time_of_day(df.index)[:, None, None], (1, df.shape[1], 1))
        values = np.concatenate([values, tod], axis=-1)  # (T, N, 2)

    T = values.shape[0]
    n = T - seq_len - horizon + 1
    x = np.stack([values[i : i + seq_len] for i in range(n)])
    y = np.stack([values[i + seq_len : i + seq_len + horizon] for i in range(n)])
    return x, y


def split_temporal(x: np.ndarray, y: np.ndarray, train: float = 0.7, val: float = 0.1):
    """Divisão cronológica (padrão DCRNN: 70/10/20)."""
    n = x.shape[0]
    n_train, n_val = int(n * train), int(n * val)
    return (
        (x[:n_train], y[:n_train]),
        (x[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        (x[n_train + n_val :], y[n_train + n_val :]),
    )


def split_by_dates(
    index: pd.DatetimeIndex, train_end: str, val_end: str, seq_len: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Índices de janela por split, com cortes temporais fixos (regra do sunt-fedstgnn).

    Uma janela pertence a um split se TODOS os seus alvos (t+1 .. t+horizon) caem dentro dele;
    janelas que atravessam uma fronteira são descartadas. Retorna (train, val, test) como
    índices da 1ª dimensão de x/y produzidos por make_windows(seq_len, horizon).
    """
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    n = len(index) - seq_len - horizon + 1
    first_target = index[seq_len : seq_len + n]
    last_target = index[seq_len + horizon - 1 : seq_len + horizon - 1 + n]
    origins = np.arange(n)
    return (
        origins[last_target < train_end],
        origins[(first_target >= train_end) & (last_target < val_end)],
        origins[first_target >= val_end],
    )
