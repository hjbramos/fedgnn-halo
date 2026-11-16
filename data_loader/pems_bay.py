"""PEMS-BAY: 325 sensores, 5 min, 2017-01-01 a 2017-06-30 (52116 x 325)."""
from ._traffic_speed import TrafficSpeedDataset

_ds = TrafficSpeedDataset(
    name="pems-bay",
    zenodo_files=["PEMS-BAY.csv", "PEMS-BAY-META.csv", "adj_mx_bay.pkl"],
    hf_repo="witgaw/PEMS-BAY",
    adj_pkl="adj_mx_bay.pkl",
)

download_raw = _ds.download_raw
load_pems_bay = _ds.load
load_meta = _ds.load_meta
load_adjacency = _ds.load_adjacency
load_adjacency_dcrnn = _ds.load_adjacency_dcrnn
load_distances = _ds.load_distances
