"""FedProx (Li, Sahu, Zaheer, Sanjabi, Talwalkar & Smith, MLSys 2020) sobre as partições Leiden.

Diferença para o FedAvg (experiments/federated.py): cada cliente minimiza
    F_k(w) + (mu/2) * ||w - w_t||^2
onde w_t é o estado global recebido na rodada (termo proximal, Eq. 2 do paper). O resto do
protocolo é o mesmo do fedavg_0hop e da campanha trip_load do sunt-fedstgnn: participação total,
Adam reiniciado a cada rodada, `local_epochs` épocas locais, agregação ponderada por observações,
early stopping pelo MAE micro de validação. mu = 0 recupera exatamente o FedAvg.

Nota: o paper usa SGD e participação parcial com solução local inexata (gamma); aqui, por
comparabilidade com o FedAvg deste repositório e com o sunt-fedstgnn, mantém-se Adam e participação
total — o termo proximal é a única mudança.

    python -m experiments.fedprox --dataset metr-la --clients 5 --mu 0.01 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import data_loader as dl

from federation import train_local_epoch, proximal_penalty, fedavg_round
from .task import masked_mae_objective

from .data import load_experiment
from .federated import evaluate_federated, load_clients, make_client_model, partition_dir
from federation import cpu_state
from .train import ROOT, _save_atomic, gpu_name, forward, load_config, set_seed


def validate_mu(mu: float) -> float:
    mu = float(mu)
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("mu deve ser finito e >= 0 (0 = FedAvg)")
    return 0.0 if mu == 0 else mu


def mu_token(mu: float) -> str:
    mu = validate_mu(mu)
    # Preserve legacy paths when :g round-trips; otherwise retain all float precision.
    token = f"{mu:g}"
    if float(token) != mu:
        token = repr(mu)
    return f"mu{token}".replace(".", "p").replace("-", "m")


def train_epoch_proximal(model: nn.Module, anchors: list[torch.Tensor], mu: float, loader, optimizer, device, clip,
                         batches_seen: int = 0) -> tuple[float, int]:
    mu = validate_mu(mu)
    params = list(model.parameters())
    regularizer = (lambda _: 0.5 * mu * proximal_penalty(params, anchors)) if mu > 0 else None
    return train_local_epoch(model, loader, optimizer, masked_mae_objective(device, forward),
                             clip, batches_seen, regularizer)



def train_fedprox(dataset: str, requested_clients: int, mu: float, seed: int, cfg: dict, device,
                  max_rounds: int | None = None, output: str | Path | None = None, resume: bool = True, *, data=None, partition_path: str | Path | None = None) -> dict:
    fp = cfg["fedprox"]
    win = cfg["windows"][dataset]
    mu = validate_mu(mu)
    device = torch.device(device)
    set_seed(seed)
    data = data if data is not None else load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
    pdir = Path(partition_path) if partition_path is not None else partition_dir(dataset, requested_clients, int(fp["partition_seed"]))
    num_clients = int(json.loads((pdir / "metadata.json").read_text())["num_clients"])
    out = ROOT / (output or cfg["output"]) / dataset / f"k{num_clients:03d}" / mu_token(mu) / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
    clients = load_clients(data, pdir, batch_size, cfg["data"]["num_workers"], seed)
    empty_clients = [c.client_id for c in clients if c.train_observations <= 0]
    if not clients or empty_clients:
        raise ValueError(f"FedProx exige clientes com alvos observados no treino; clientes inválidos: {empty_clients}")
    weights = [c.train_observations for c in clients]
    max_rounds = int(max_rounds or fp.get("max_rounds_per_dataset", {}).get(dataset, fp["max_rounds"]))
    local_epochs, clip, lr = int(fp["local_epochs"]), float(fp["gradient_clip_norm"]), float(fp["learning_rate"])
    tag = f"{dataset}/k{num_clients:03d}/{mu_token(mu)}/seed_{seed:03d}"

    global_state = cpu_state(make_client_model(cfg, clients[0], device))
    state_bytes = sum(v.numel() * v.element_size() for v in global_state.values())
    history, best_mae, ref, stale, batches_seen = [], float("inf"), float("inf"), 0, 0
    ckpt, last = out / "best_model.pt", out / "last.pt"
    t0, start_round = time.time(), 1
    if resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        if "proximal_mu" not in st or validate_mu(st["proximal_mu"]) != mu:
            raise ValueError("checkpoint sem mu verificável ou com mu diferente; use outro output ou --no-resume")
        global_state, history, best_mae, ref, stale, batches_seen = (st["global_state"], st["history"], st["best_mae"],
                                                                     st["stopping_reference"], st["without_improvement"], st["batches_seen"])
        start_round = st["round"] + 1
        t0 -= st["elapsed_s"]
        torch.set_rng_state(st["rng"]["torch"]); np.random.set_state(st["rng"]["numpy"])
        for c, g in zip(clients, st["rng"]["loaders"]):
            c.loaders["train"].generator.set_state(g)
        print(f"{tag} | retomando da rodada {st['round']} (best val_MAE={best_mae:.4f})", flush=True)
        if st.get("stopped_early"):
            start_round = max_rounds + 1

    for rnd in range(start_round, max_rounds + 1):
        global_state, train_mae, batches_seen = fedavg_round(
            global_state, clients, weights,
            lambda client: make_client_model(cfg, client, device),
            lambda model: torch.optim.Adam(model.parameters(), lr=lr),
            lambda model, client, optimizer, anchors, seen: train_epoch_proximal(model, anchors, mu, client.loaders["train"], optimizer, device, clip, seen),
            local_epochs, batches_seen)
        val = evaluate_federated(global_state, clients, data, cfg, "validation", device)
        history.append({"round": rnd, "train_scaled_MAE": train_mae, **{f"validation_{k}": v for k, v in val.items()},
                        "elapsed_s": time.time() - t0})
        print(f"{tag} | round {rnd:03d}/{max_rounds} | train={train_mae:.4f} | val_MAE={val['MAE']:.4f} "
              f"| macro={val['macro_client_MAE']:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            _save_atomic({"model_state_dict": global_state, "best_round": rnd, "validation_MAE": best_mae}, ckpt)
        if val["MAE"] < ref - float(fp["early_stopping_min_delta"]):
            ref, stale = val["MAE"], 0
        else:
            stale += 1
        stopped_early = stale >= int(fp["patience"])
        _save_atomic({"round": rnd, "proximal_mu": mu, "global_state": global_state, "history": history, "best_mae": best_mae,
                      "stopping_reference": ref, "without_improvement": stale, "batches_seen": batches_seen,
                      "elapsed_s": time.time() - t0, "stopped_early": stopped_early,
                      "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                              "loaders": [c.loaders["train"].generator.get_state() for c in clients]}}, last)
        if stopped_early:
            break

    best = torch.load(ckpt, map_location="cpu", weights_only=True)
    rounds = len(history)
    result = {
        "dataset": dataset, "seed": seed, "method": "fedprox_0hop", "proximal_mu": mu,
        "model": cfg["model"].get("name", "leconv"),
        "requested_clients": requested_clients, "num_clients": num_clients, "partition_seed": int(fp["partition_seed"]),
        "partition": str(pdir.relative_to(ROOT)) if pdir.is_relative_to(ROOT) else str(pdir), "nodes": data.nodes,
        "client_nodes": [int(c.nodes) for c in clients], "client_train_observations": weights,
        "input_steps": data.input_steps, "horizon": data.horizon,
        "features": data.feature_names, "target": data.feature_names[data.target_feature_index], "graph": getattr(data, "graph", "default"),
        "local_epochs": local_epochs, "best_round": int(best["best_round"]), "rounds_completed": rounds,
        "parameter_count": int(sum(v.numel() for v in global_state.values())), "model_state_bytes": int(state_bytes),
        "model_communication_bytes": int(2 * state_bytes * num_clients * rounds),
        "training_seconds": time.time() - t0, "device": str(device), "gpu": gpu_name(device),
        "validation": evaluate_federated(best["model_state_dict"], clients, data, cfg, "validation", device, per_node=True),
        "test": evaluate_federated(best["model_state_dict"], clients, data, cfg, "test", device, per_node=True),
    }
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    last.unlink(missing_ok=True)
    print(f"{tag} | done | best_round={result['best_round']} | test_MAE={result['test']['MAE']:.4f} "
          f"| macro={result['test']['macro_client_MAE']:.4f} | {result['training_seconds'] / 60:.1f} min", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(dl.DATASETS))
    p.add_argument("--clients", type=int, required=True)
    p.add_argument("--mu", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(ROOT / "config" / "fedprox_leconv.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    train_fedprox(a.dataset, a.clients, a.mu, a.seed, load_config(a.config), a.device, a.max_rounds, a.output, not a.no_resume)


if __name__ == "__main__":
    main()
