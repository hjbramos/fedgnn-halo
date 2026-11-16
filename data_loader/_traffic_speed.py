"""Base comum para os datasets de velocidade de tráfego publicados com o DCRNN (METR-LA e PEMS-BAY).

Ambos têm o mesmo layout:
  data/<name>/raw/
      <NAME>.csv             série bruta (timestamp x sensor, mph)  - Zenodo 5146275
      <NAME>-META.csv        metadados dos sensores (se existir)     - Zenodo 5146275
      adj_mx*.pkl            adjacência original do DCRNN (se existir)
      sensor_graph/          adj_mx.npy, adj_mx_mapping.json, distances.csv, sensor_locations.csv
                                                                     - HF witgaw/<NAME>
  data/<name>/processed/<name>.parquet   cache da série
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA_DIR, align_to_graph, download, download_hf

ZENODO = "https://zenodo.org/api/records/5146275/files/{}/content"
HF_FILES = [
    "sensor_graph/distances.csv",
    "sensor_graph/adj_mx_mapping.json",
    "sensor_graph/adj_mx.npy",
    "sensor_graph/sensor_locations.csv",
]


class TrafficSpeedDataset:
    def __init__(self, name: str, zenodo_files: list[str], hf_repo: str, adj_pkl: str | None):
        self.name = name  # ex.: "pems-bay"
        self.zenodo_files = zenodo_files
        self.hf_repo = hf_repo
        self.adj_pkl = adj_pkl
        self.raw_dir = DATA_DIR / name / "raw"
        self.graph_dir = self.raw_dir / "sensor_graph"
        self.processed_dir = DATA_DIR / name / "processed"
        self.parquet_path = self.processed_dir / f"{name.replace('-', '_')}.parquet"

    # ---------------------------------------------------------------- download
    def download_raw(self, force: bool = False) -> None:
        for f in self.zenodo_files:
            download(ZENODO.format(f), self.raw_dir / f, force)
        download_hf(self.hf_repo, HF_FILES, self.raw_dir, force)

    # ---------------------------------------------------------------- série
    def load_csv(self) -> pd.DataFrame:
        """Lê o CSV bruto -> DataFrame (timestamp x sensor), velocidade em mph."""
        df = pd.read_csv(self.raw_dir / self.zenodo_files[0], index_col=0, parse_dates=True)
        df.columns = [str(c) for c in df.columns]
        df.index.name = "timestamp"
        return df.astype(np.float32)

    def load(self, force: bool = False) -> pd.DataFrame:
        """DataFrame (T x N), baixando o bruto e cacheando em parquet."""
        if self.parquet_path.exists() and not force:
            return pd.read_parquet(self.parquet_path)
        self.download_raw()
        df = self.load_csv()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.parquet_path, engine="pyarrow", index=True)
        return df

    def load_meta(self) -> pd.DataFrame:
        self.download_raw()
        meta = [f for f in self.zenodo_files if f.endswith("META.csv")]
        if not meta:
            return pd.read_csv(self.graph_dir / "sensor_locations.csv")
        return pd.read_csv(self.raw_dir / meta[0])

    # ---------------------------------------------------------------- grafo
    def load_adjacency(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
        """sensor_graph/adj_mx.npy + df alinhado à ordem da adjacência."""
        self.download_raw()
        with open(self.graph_dir / "adj_mx_mapping.json") as f:
            mapping = json.load(f)
        sensor_ids = [str(s) for s in mapping["sensor_ids"]]
        adj = np.load(self.graph_dir / "adj_mx.npy")
        assert adj.shape == (len(sensor_ids), len(sensor_ids))
        return adj, sensor_ids, align_to_graph(df, sensor_ids)

    def load_adjacency_dcrnn(self) -> tuple[list[str], dict[str, int], np.ndarray]:
        """Adjacência original do DCRNN (.pkl): (sensor_ids, sensor_id_to_ind, adj_mx)."""
        if self.adj_pkl is None:
            raise FileNotFoundError(f"{self.name} não tem adj .pkl no Zenodo; use load_adjacency()")
        self.download_raw()
        with open(self.raw_dir / self.adj_pkl, "rb") as f:
            sensor_ids, sensor_to_ind, adj = pickle.load(f, encoding="latin1")
        return [str(s) for s in sensor_ids], {str(k): v for k, v in sensor_to_ind.items()}, adj

    def load_distances(self) -> pd.DataFrame:
        self.download_raw()
        return pd.read_csv(self.graph_dir / "distances.csv", dtype={"from": str, "to": str})
