"""Particionamento federado único (== sunt-fedstgnn/src/fedstgnn/partitioning.py): Leiden sobre o grafo.

    python -m experiments.partition --dataset all --clients 5 10 20 --seed 42

Método: Leiden (RBConfigurationVertexPartition, ponderado, seed fixo) sobre o grafo não-direcionado;
busca de resolução numa grade geométrica [0.01, 10] com 61 pontos, escolhendo a que dá o número de
comunidades mais próximo do pedido (desempate: menor CV do tamanho dos clientes). Para pedidos grandes
(ex.: SUNT k=163, como na campanha do sunt-fedstgnn) a grade é prolongada com o mesmo passo até 100
(20 pontos extra); os 61 pontos originais ficam inalterados, logo as partições k<=20 não mudam.

Grafos direcionados (METR-LA / PEMS-BAY, kernel gaussiano do DCRNN com auto-laços): o Leiden usa
max(W, Wᵀ) sem diagonal; sensores isolados (sem vizinho acima do limiar) são atribuídos ao cliente do
sensor mais próximo pela distância viária (distances.csv). O subgrafo de cada cliente guarda as arestas
ORIGINAIS internas (direcionadas, com auto-laço), que são as usadas no treino.

Saída: data/<dataset>/processed/partitions/leiden/k<NNN>/partition_seed_<SSS>/
    client_assignments.npy  (N,) cliente de cada nó      node_ids.npy  (N,)
    resolution_search.csv   audit.csv   metadata.json
    client_<CCC>/{node_indices, node_ids, edge_index, edge_weight, global_edge_indices}.npy
e um índice data/<dataset>/processed/partitions/leiden/partitions.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd

import data_loader as dl
from data_loader import metr_la, pems_bay

from .data import load_experiment
from .train import load_config

RESOLUTION_MIN, RESOLUTION_MAX, RESOLUTION_STEPS = 0.01, 10.0, 61
RESOLUTION_EXTENDED_MAX = 100.0  # prolongamento com o mesmo passo geométrico (1000^(1/60))


def resolution_grid() -> np.ndarray:
    base = np.geomspace(RESOLUTION_MIN, RESOLUTION_MAX, RESOLUTION_STEPS)
    extra = np.geomspace(RESOLUTION_MAX, RESOLUTION_EXTENDED_MAX, 21)[1:]
    return np.concatenate([base, extra])


def partition_dir(dataset: str, num_clients: int, seed: int) -> Path:
    return dl.DATA_DIR / dataset / "processed" / "partitions" / "leiden" / f"k{num_clients:03d}" / f"partition_seed_{seed:03d}"


def canonicalize(membership) -> np.ndarray:
    """Renumera comunidades pela ordem do primeiro nó (== sunt-fedstgnn)."""
    membership = np.asarray(membership, dtype=np.int64)
    order = sorted(np.unique(membership), key=lambda c: np.flatnonzero(membership == c)[0])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[v] for v in membership], dtype=np.int64)


def _undirected_graph(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arestas (2, E) com u < v e pesos de max(W, Wᵀ) sem diagonal."""
    a = np.asarray(adj, dtype=np.float64).copy()
    np.fill_diagonal(a, 0.0)
    sym = np.maximum(a, a.T)
    u, v = np.nonzero(np.triu(sym, 1))
    return np.stack([u, v]), sym[u, v]


def _nearest_neighbor_by_distance(dataset: str, ids: list[str], isolated: list[int]) -> dict[int, int]:
    """Para cada nó isolado, o índice do nó não-isolado mais próximo: distância viária (distances.csv)
    ou, se o sensor não consta lá, distância Haversine pelas coordenadas (sensor_locations.csv)."""
    mod = {"metr-la": metr_la, "pems-bay": pems_bay}[dataset]
    idx = {s: i for i, s in enumerate(ids)}
    iso = set(isolated)
    dist = mod.load_distances()
    dist = dist[dist["from"].isin(idx) & dist["to"].isin(idx) & (dist["from"] != dist["to"])]
    dist = dist[~dist["to"].map(idx).isin(iso)]
    nearest = {}
    for s, grp in dist.groupby("from"):
        nearest[idx[s]] = idx[grp.sort_values("cost").iloc[0]["to"]]
    missing = [i for i in isolated if i not in nearest]
    if missing:
        loc = pd.read_csv(mod._ds.graph_dir / "sensor_locations.csv")
        loc["sensor_id"] = loc["sensor_id"].astype(str)
        loc = loc.set_index("sensor_id").reindex(ids)
        lat, lon = np.radians(loc["latitude"].to_numpy()), np.radians(loc["longitude"].to_numpy())
        for i in missing:
            d = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
            d[i] = np.inf
            d[list(iso)] = np.inf
            nearest[i] = int(np.nanargmin(d))
    return nearest


def build_partition(dataset: str, requested: int, seed: int, cfg: dict, force: bool = False) -> dict:
    win = cfg["windows"][dataset]
    data = load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
    N, ids = data.nodes, data.node_ids
    if not 2 <= requested < N:
        raise ValueError("clientes pedidos deve estar entre 2 e N-1")

    edges, weights = _undirected_graph(data.adj)
    g = ig.Graph(n=N, edges=list(map(tuple, edges.T.tolist())), directed=False)
    g.es["weight"] = weights.tolist()
    components = g.connected_components()
    isolated = [c[0] for c in components if len(c) == 1]
    if len(components) - len(isolated) != 1:
        raise ValueError(f"{dataset}: grafo com {len(components) - len(isolated)} componentes não triviais; Leiden espera 1")
    core = np.array(sorted(set(range(N)) - set(isolated)), dtype=np.int64)
    sub = g.induced_subgraph(core.tolist())  # Leiden só no núcleo conexo

    rows, candidates = [], []
    for res in resolution_grid():
        part = la.find_partition(sub, la.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=float(res), n_iterations=-1, seed=seed)
        m = canonicalize(part.membership)
        sizes = np.bincount(m)
        row = {"requested_clients": requested, "resolution": float(res), "num_clients": int(len(sizes)),
               "distance_to_request": int(abs(len(sizes) - requested)),
               "node_size_cv": float(sizes.std() / sizes.mean()), "quality": float(part.quality())}
        rows.append(row)
        candidates.append((row, m))
    selected, core_membership = min(candidates, key=lambda c: (c[0]["distance_to_request"], c[0]["node_size_cv"]))

    membership = np.full(N, -1, dtype=np.int64)
    membership[core] = core_membership
    if isolated:
        nearest = _nearest_neighbor_by_distance(dataset, ids, isolated)
        for i in isolated:
            membership[i] = membership[nearest[i]]
    membership = canonicalize(membership)
    num_clients = int(membership.max() + 1)

    out = partition_dir(dataset, num_clients, seed)
    if (out / "metadata.json").exists() and not force:
        print(f"{dataset}: k={num_clients} (pedido {requested}) já existe em {out.relative_to(dl.DATA_DIR.parent)}")
        return json.loads((out / "metadata.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "client_assignments.npy", membership)
    np.save(out / "node_ids.npy", np.array(ids))
    pd.DataFrame(rows).to_csv(out / "resolution_search.csv", index=False)

    # subgrafos locais com as arestas ORIGINAIS (direcionadas; auto-laços incluídos no tráfego)
    src, dst = data.edge_index.numpy()
    ew = data.edge_weight.numpy()
    train_targets = data.origins["train"][:, None] + np.arange(data.horizon)[None, :]
    tmask = data.mask[:, :, data.target_feature_index]
    tseries = data.series[:, :, data.target_feature_index]
    audit = []
    for c in range(num_clients):
        nodes = np.flatnonzero(membership == c)
        internal = (membership[src] == c) & (membership[dst] == c)
        local_map = np.full(N, -1, dtype=np.int64)
        local_map[nodes] = np.arange(len(nodes))
        local_edges = local_map[np.stack([src[internal], dst[internal]])]
        incoming = np.bincount(local_edges[1], minlength=len(nodes))
        if np.any(incoming == 0):
            raise ValueError(f"{dataset} cliente {c}: nó sem aresta interna de entrada")
        pairs = {(min(u, v), max(u, v)) for u, v in local_edges.T.tolist() if u != v}  # simetriza p/ conexidade
        und = np.array(sorted(pairs), dtype=np.int64).T.reshape(2, -1)
        lg = nx.Graph(); lg.add_nodes_from(range(len(nodes))); lg.add_edges_from(pairs)
        comps = nx.number_connected_components(lg)
        obs = tmask[train_targets[:, :, None], nodes[None, None, :]]
        vals = tseries[train_targets[:, :, None], nodes[None, None, :]][obs]
        cdir = out / f"client_{c:03d}"
        cdir.mkdir(exist_ok=True)
        np.save(cdir / "node_indices.npy", nodes)
        np.save(cdir / "node_ids.npy", np.array(ids)[nodes])
        np.save(cdir / "edge_index.npy", local_edges)
        np.save(cdir / "edge_weight.npy", ew[internal].astype(np.float32))
        np.save(cdir / "global_edge_indices.npy", np.flatnonzero(internal))
        possible = len(nodes) * (len(nodes) - 1) / 2
        audit.append({"client_id": c, "nodes": int(len(nodes)), "undirected_internal_edges": int(und.shape[1]),
                      "connected_components": int(comps), "density": float(und.shape[1] / possible) if possible else 0.0,
                      "train_observations": int(len(vals)), "train_target_mean": float(vals.mean()) if len(vals) else float("nan")})
    audit = pd.DataFrame(audit)
    audit.to_csv(out / "audit.csv", index=False)
    assert audit["nodes"].sum() == N

    same = membership[edges[0]] == membership[edges[1]]
    sizes = np.bincount(membership)
    meta = {"dataset": dataset, "method": "leiden", "partition_seed": seed, "requested_clients": requested,
            "num_clients": num_clients, "resolution": selected["resolution"], "nodes": N,
            "undirected_edges": int(edges.shape[1]), "cut_edges": int((~same).sum()),
            "cut_weight_fraction": float(weights[~same].sum() / weights.sum()),
            "isolated_nodes_attached_by_distance": [ids[i] for i in isolated],
            "client_size_min": int(sizes.min()), "client_size_max": int(sizes.max()),
            "client_size_cv": float(sizes.std() / sizes.mean()),
            "clients_with_multiple_components": int((audit["connected_components"] > 1).sum()),
            "graph": "max(W, Wᵀ) sem diagonal; subgrafos locais com arestas originais"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    index_path = out.parent.parent / "partitions.csv"
    row = {"method": "leiden", "requested_clients": requested, "num_clients": num_clients, "partition_seed": seed,
           "resolution": selected["resolution"], "path": str(out.relative_to(dl.DATA_DIR.parent))}
    index = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
    if len(index):
        index = index[~((index["num_clients"] == num_clients) & (index["partition_seed"] == seed))]
    pd.concat([index, pd.DataFrame([row])], ignore_index=True).sort_values(["num_clients", "partition_seed"]).to_csv(index_path, index=False)
    print(f"{dataset}: pedido {requested} -> k={num_clients} (res={selected['resolution']:.3g}) tamanhos {sizes.min()}..{sizes.max()} "
          f"CV={meta['client_size_cv']:.2f} corte={meta['cut_weight_fraction']:.1%} "
          f"isolados={len(isolated)} clientes desconexos={meta['clients_with_multiple_components']}")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", choices=[*dl.DATASETS, "all"])
    p.add_argument("--clients", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None, help="YAML com as janelas (default: config/centralized.yaml)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    for ds in (dl.DATASETS if a.dataset == "all" else [a.dataset]):
        for k in a.clients:
            build_partition(ds, k, a.seed, cfg, a.force)


if __name__ == "__main__":
    main()
