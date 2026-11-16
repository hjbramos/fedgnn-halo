"""METR-LA: 207 sensores, 5 min, 2012-03-01 a 2012-06-27 (34272 x 207).

Ao contrário do PEMS-BAY, zeros na série indicam leitura faltante.
"""
from ._traffic_speed import TrafficSpeedDataset

_ds = TrafficSpeedDataset(
    name="metr-la",
    zenodo_files=["METR-LA.csv"],
    hf_repo="witgaw/METR-LA",
    adj_pkl=None,
)

download_raw = _ds.download_raw
load_metr_la = _ds.load
load_meta = _ds.load_meta
load_adjacency = _ds.load_adjacency
load_distances = _ds.load_distances
