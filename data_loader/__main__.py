"""python -m data_loader --dataset [metr-la|pems-bay|sunt|all]  — baixa e valida os datasets."""
import argparse

import data_loader as dl

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="all", choices=[*dl.DATASETS, "all"])
a = p.parse_args()

for name in dl.DATASETS if a.dataset == "all" else [a.dataset]:
    dl.download(name)
    df = dl.load(name)
    adj, ids, df = dl.load_adjacency(name, df)
    feats, mask, ts, _ = dl.load_features(name)
    print(f"{name:9s} série {df.shape}  {df.index.min()} -> {df.index.max()}  adj {adj.shape}  "
          f"features {feats.shape} {dl.feature_names(name)}  observado {mask.mean():.1%}")
