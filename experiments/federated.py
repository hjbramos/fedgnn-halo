"""FedAvg 0-hop sobre as partições Leiden — replica fedstgnn.training.train_federated (método fedavg_0hop).

Protocolo (== sunt-fedstgnn/config/experiments/sunt/client_scaling.yaml, seção `federated`):
  - participação total; cada cliente treina no SEU subgrafo (arestas entre clientes ignoradas);
  - otimizador local Adam reiniciado a cada rodada; `local_epochs` épocas locais;
  - agregação FedAvg ponderada pelo nº de alvos observados no treino de cada cliente;
  - a cada rodada o modelo global é avaliado na validação de todos os clientes (métricas micro
    + MAE macro por cliente); early stopping e checkpoint pelo MAE micro de validação;
  - retomada por rodada em last.pt.

Opções opt-in (padrão = comportamento acima, usado pelas campanhas anteriores):
  - `strict_determinism: true`: o mesmo determinismo estrito do centralizado, ativado antes de qualquer uso de CUDA;
  - `lr_decay_rounds: [..]` e `lr_decay_ratio`: lr da rodada r = lr · ratio^{#{m : r > m}}. É o mesmo calendário do
    MultiStepLR centralizado, em que o passo do agendador ao fim da época m faz a época m + 1 usar o lr reduzido.
    Com marcos [30, 40], as rodadas 1–30 usam lr, as 31–40 lr·ratio e as 41–50 lr·ratio². Só com o Adam reiniciado
    a cada rodada, que é criado com o lr da rodada;
  - `instrumentation: true`: instrumentação por passo de cada cliente (experiments/instrumentacao.py, com a rodada no
    lugar da época) em instrumentacao/cliente_XXX/, mais por cliente e rodada (clientes_rodadas.csv) a perda local,
    a norma da mudança local ‖w_local − w_global‖ e as métricas de validação do modelo global no cliente; e por
    rodada a mudança do modelo global após a agregação.

    python -m experiments.federated --dataset metr-la --clients 5 --seed 42 [--config config/fedavg_leconv.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import data_loader as dl
from federation import aggregate_states, cpu_state, fedavg_round
from federation.rounds import optimizer_states_sha256
from .task import make_model

from .data import ExperimentData, WindowDataset, load_experiment
from .metrics import NodeErrorAccumulator, RegressionAccumulator
from .train import (ROOT, _save_atomic, determinism_state, enable_strict_determinism, gpu_name, forward, load_config,
                    make_loader, set_seed, train_epoch)


def _state_dict_sha256(state: dict) -> str:
    """== experiments.train.state_sha256, sobre um state_dict já extraído (nomes ordenados)."""
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class ClientView:
    """O que build_model precisa saber do subgrafo de um cliente (mesma interface de ExperimentData)."""
    name: str
    client_id: int
    node_indices: np.ndarray
    adj: np.ndarray
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    input_steps: int
    horizon: int
    n_features: int
    train_observations: int
    loaders: dict[str, DataLoader]

    @property
    def nodes(self) -> int:
        return len(self.node_indices)


CLIENT_OPTIMIZERS = ("adam_reset_each_round", "adam_persistent_per_client")


def stopping_state(history: list[dict], min_delta: float) -> tuple[float, int]:
    """(referência, rodadas sem melhora relevante) ao fim de `history`, pela mesma regra do laço de treino: a referência
    só muda quando o MAE de validação fica abaixo dela por mais de `min_delta`."""
    ref, stale = float("inf"), 0
    for row in history:
        if row["validation_MAE"] < ref - min_delta:
            ref, stale = row["validation_MAE"], 0
        else:
            stale += 1
    return ref, stale


def round_learning_rate(lr: float, rnd: int, milestones: list[int], ratio: float) -> float:
    """lr usado no treino local da rodada `rnd` (1, 2, ...): o mesmo calendário do MultiStepLR centralizado, cujo passo
    ao fim da época m faz a época m + 1 usar o lr reduzido. Rodada r: lr · ratio^{#{m : r > m}}."""
    return lr * ratio ** sum(rnd > m for m in milestones)


def partition_dir(dataset: str, requested_clients: int, seed: int) -> Path:
    """Resolve o k pedido para o k obtido via partitions.csv (ex.: SUNT 5 -> 6)."""
    base = dl.DATA_DIR / dataset / "processed" / "partitions" / "leiden"
    index = pd.read_csv(base / "partitions.csv")
    row = index[(index["requested_clients"] == requested_clients) & (index["partition_seed"] == seed)]
    if row.empty:
        row = index[(index["num_clients"] == requested_clients) & (index["partition_seed"] == seed)]
    if row.empty:
        raise FileNotFoundError(f"partição {dataset} k={requested_clients} seed={seed} não existe; rode experiments.partition")
    path = Path(row.iloc[0]["path"])
    return dl.DATA_DIR / path.relative_to("data") if not path.is_absolute() and path.parts[0] == "data" else ROOT / path


def client_edges(data: ExperimentData, cdir: Path, nodes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Arestas internas do cliente (índices locais).

    Grafo padrão: as gravadas na partição (iguais à submatriz da adjacência, conferido em 30/09). Outro grafo
    sobre a mesma partição (SUNT `transit_directed`): derivadas da submatriz da adjacência carregada, porque
    os arquivos da partição guardam o grafo com que ela foi feita. A partição (nós de cada cliente) é a mesma.
    """
    if data.graph in ("default", "transit"):
        return (torch.from_numpy(np.load(cdir / "edge_index.npy")).long(),
                torch.from_numpy(np.load(cdir / "edge_weight.npy")).float())
    sub = np.asarray(data.adj, dtype=np.float32)[np.ix_(nodes, nodes)]
    src, dst = np.nonzero(sub)
    return torch.from_numpy(np.stack([src, dst]).astype(np.int64)), torch.from_numpy(sub[src, dst].astype(np.float32))


def load_clients(data: ExperimentData, directory: Path, batch_size: int, num_workers: int, seed: int,
                 splits: tuple[str, ...] = ("train", "validation", "test")) -> list[ClientView]:
    meta = json.loads((directory / "metadata.json").read_text())
    assignments = np.load(directory / "client_assignments.npy")
    ids = np.load(directory / "node_ids.npy", allow_pickle=True).astype(str).tolist()
    if ids != list(data.node_ids):
        raise ValueError("ordem dos nós da partição difere da série")
    train_targets = data.origins["train"][:, None] + np.arange(data.horizon)[None, :]
    tmask = data.mask[:, :, data.target_feature_index]
    clients = []
    for c in range(int(meta["num_clients"])):
        cdir = directory / f"client_{c:03d}"
        nodes = np.load(cdir / "node_indices.npy")
        assert np.array_equal(nodes, np.flatnonzero(assignments == c))
        loaders = {s: make_loader(WindowDataset(data, s, nodes), batch_size, s == "train", seed + c, num_workers)
                   for s in splits}
        edge_index, edge_weight = client_edges(data, cdir, nodes)
        clients.append(ClientView(
            name=data.name, client_id=c, node_indices=nodes, adj=data.adj[np.ix_(nodes, nodes)],
            edge_index=edge_index, edge_weight=edge_weight,
            input_steps=data.input_steps, horizon=data.horizon, n_features=data.n_features,
            train_observations=int(tmask[train_targets[:, :, None], nodes[None, None, :]].sum()), loaders=loaders))
    return clients


def make_client_model(cfg: dict, client: ClientView, device) -> nn.Module:
    return make_model(cfg, client, device)


@torch.no_grad()
def evaluate_federated(global_state: dict, clients: list[ClientView], data: ExperimentData, cfg: dict, split: str, device,
                       per_node: bool = False, per_client: bool = False) -> dict:
    """Modelo global avaliado no split de cada cliente: micro sobre todos os alvos + MAE macro por cliente.

    per_node=True acrescenta "per_node" (erro absoluto e contagem por nó global) para a análise estratificada.
    per_client=True acrescenta "per_client" (MAE, RMSE e alvos de cada cliente).
    """
    total, client_maes, client_rows = RegressionAccumulator(), [], []
    nodes = NodeErrorAccumulator(data.nodes) if per_node else None
    for client in clients:
        model = make_client_model(cfg, client, device)
        model.load_state_dict(global_state, strict=True)
        model.eval()
        local = RegressionAccumulator()
        for batch in client.loaders[split]:
            pred = (forward(model, batch, device) * data.target_std + data.target_mean).clamp_min(0.0)
            y, m = batch["y_raw"].to(device), batch["y_mask"].to(device)
            local.update(pred, y, m)
            total.update(pred, y, m)
            if nodes is not None:
                nodes.update(pred, y, m, client.node_indices)
        lm = local.compute()
        client_maes.append(lm["MAE"])
        client_rows.append({"client": client.client_id, "MAE": lm["MAE"], "RMSE": lm["RMSE"], "observations": lm["observations"]})
        del model
    metrics = total.compute()
    metrics.update(macro_client_MAE=float(np.mean(client_maes)), min_client_MAE=float(np.min(client_maes)),
                   max_client_MAE=float(np.max(client_maes)))
    if nodes is not None:
        metrics["per_node"] = nodes.compute()
    if per_client:
        metrics["per_client"] = client_rows
    return metrics


def train_fedavg(dataset: str, requested_clients: int, seed: int, cfg: dict, device, max_rounds: int | None = None,
                 output: str | Path | None = None, resume: bool = True, *, data=None, partition_path: str | Path | None = None) -> dict:
    fed = cfg["federated"]
    win = cfg["windows"][dataset]
    device = torch.device(device)
    # Mesmas opções do treino centralizado: `evaluate_test: false` não cria, não avalia nem grava o teste
    # (sondagem cega); `keep_last_checkpoint: true` preserva o last.pt para estender aumentando max_rounds.
    evaluate_test = bool(fed.get("evaluate_test", True))
    keep_last = bool(fed.get("keep_last_checkpoint", False))
    t_setup = time.time()
    strict = bool(fed.get("strict_determinism", False))
    determinism = enable_strict_determinism() if strict else determinism_state()   # antes de qualquer uso de CUDA
    set_seed(seed)
    data = data if data is not None else load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
    # partição explícita (ex.: SUNT recorte L2, results/sunt_ampliado/particoes/...): não passa pelo partitions.csv
    # global, que indexa as partições da rede completa por (k, semente)
    pdir = Path(partition_path) if partition_path is not None else ROOT / fed["partition_path"] if fed.get("partition_path") else partition_dir(dataset, requested_clients,
                                                                                         int(fed["partition_seed"]))
    num_clients = int(json.loads((pdir / "metadata.json").read_text())["num_clients"])
    out = ROOT / (output or cfg["output"]) / dataset / f"k{num_clients:03d}" / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
    clients = load_clients(data, pdir, batch_size, cfg["data"]["num_workers"], seed,
                           ("train", "validation", "test") if evaluate_test else ("train", "validation"))
    weights = [c.train_observations for c in clients]
    max_rounds = int(max_rounds or fed.get("max_rounds_per_dataset", {}).get(dataset, fed["max_rounds"]))
    local_epochs, clip, lr = int(fed["local_epochs"]), float(fed["gradient_clip_norm"]), float(fed["learning_rate"])
    # Mesmas opções do centralizado (padrões = comportamento original): perda padronizada e eps do Adam do torch
    eps = float(fed.get("epsilon", 1e-8))
    loss_scale = fed.get("loss_scale", "standardized")
    if loss_scale not in ("standardized", "original"):
        raise ValueError(f"federated.loss_scale desconhecido: {loss_scale}")
    target_scale = (data.target_mean, data.target_std) if loss_scale == "original" else None
    tag = f"{dataset}/k{num_clients:03d}/seed_{seed:03d}"
    # Otimizador do cliente: reiniciado a cada rodada (padrão, comportamento original) ou, só para o piloto do
    # diagnóstico do SUNT, com o estado do Adam de cada cliente preservado entre rodadas.
    client_optimizer = fed.get("optimizer", "adam_reset_each_round")
    if client_optimizer not in CLIENT_OPTIMIZERS:
        raise ValueError(f"federated.optimizer desconhecido: {client_optimizer!r} (opções: {CLIENT_OPTIMIZERS})")
    optimizer_states = [None] * len(clients) if client_optimizer == "adam_persistent_per_client" else None
    milestones = [int(m) for m in fed.get("lr_decay_rounds", [])]
    ratio = float(fed.get("lr_decay_ratio", 0.1))
    if milestones and optimizer_states is not None:
        raise ValueError("lr_decay_rounds só com adam_reset_each_round (o estado persistente traria o lr antigo)")
    round_lr = lambda r: round_learning_rate(lr, r, milestones, ratio)
    # Parada: o melhor checkpoint é sempre o menor MAE observado; a paciência conta rodadas sem melhora RELEVANTE
    # (> min_delta) e só pode interromper a partir de `early_stopping_min_round` (padrão 0 = comportamento original).
    stopping_rule = {"patience": int(fed["patience"]), "min_delta": float(fed["early_stopping_min_delta"]),
                     "min_round": int(fed.get("early_stopping_min_round", 0))}
    instrument = bool(fed.get("instrumentation", False))
    if instrument:
        from .instrumentacao import Instrumentacao

    global_state = cpu_state(make_client_model(cfg, clients[0], device))  # inicialização comum (seed)
    init_sha = _state_dict_sha256(global_state)   # comparável ao init_state_sha256 do centralizado (mesma semente)
    state_bytes = sum(v.numel() * v.element_size() for v in global_state.values())
    history, best_mae, ref, stale, batches_seen = [], float("inf"), float("inf"), 0, 0
    ckpt, last = out / "best_model.pt", out / "last.pt"
    setup_seconds = time.time() - t_setup  # carga dos dados e dos clientes, fora do tempo de treino
    t0, start_round, resumes, runs = time.time(), 1, [], []
    if resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        global_state, history, best_mae, ref, stale, batches_seen = (st["global_state"], st["history"], st["best_mae"],
                                                                     st["stopping_reference"], st["without_improvement"], st["batches_seen"])
        start_round = st["round"] + 1
        # `retomadas` (histórico) só registra a restauração de estados do Adam; `retomadas_execucao` registra toda
        # retomada. Se a regra de parada da config difere da gravada (ou o last.pt é anterior a ela), a referência e a
        # contagem da paciência são recalculadas sobre o histórico com a regra atual.
        recalc = st.get("stopping_rule") != stopping_rule
        if recalc:
            ref, stale = stopping_state(history, stopping_rule["min_delta"])
        runs = st.get("retomadas_execucao", []) + [{"rodada_inicial": st["round"] + 1, "rodada_last_pt": int(st["round"]),
                                                     "regra_parada_gravada": st.get("stopping_rule"),
                                                     "regra_parada_recalculada": recalc,
                                                     "inicio_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
        if st.get("client_optimizer", "adam_reset_each_round") != client_optimizer:
            raise ValueError(f"last.pt foi gerado com federated.optimizer={st.get('client_optimizer')!r}; "
                             f"a config pede {client_optimizer!r}")
        resumes = st.get("retomadas", [])
        if optimizer_states is not None:
            optimizer_states = st["optimizer_states"]
            resumes = resumes + [{"rodada_inicial": st["round"] + 1,
                                  "optimizer_states_sha256": optimizer_states_sha256(optimizer_states)}]
        t0 -= st["elapsed_s"]
        torch.set_rng_state(st["rng"]["torch"]); np.random.set_state(st["rng"]["numpy"])
        if st["rng"].get("python") is not None:
            random.setstate(st["rng"]["python"])
        if torch.cuda.is_available() and st["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        for c, g in zip(clients, st["rng"]["loaders"]):
            c.loaders["train"].generator.set_state(g)
        print(f"{tag} | retomando da rodada {st['round']} (best val_MAE={best_mae:.4f})", flush=True)
        if st.get("stopped_early") and not recalc:
            start_round = max_rounds + 1

    for rnd in range(start_round, max_rounds + 1):
        t_round = time.time()
        lr_r, local_rows = round_lr(rnd), []

        def local_epoch(model, client, optimizer, anchors, seen):
            if instrument:
                inst = Instrumentacao(model, out / "instrumentacao" / f"cliente_{client.client_id:03d}", rnd)
                metric, seen = inst.epoca(rnd, model, client.loaders["train"], optimizer, device, clip, seen, target_scale)
            else:
                metric, seen = train_epoch(model, client.loaders["train"], optimizer, device, clip, seen, target_scale)
            with torch.no_grad():
                drift = float(torch.sqrt(sum((p.double() - a.double()).square().sum() for p, a in zip(model.parameters(), anchors))))
                pn = float(torch.sqrt(sum(p.double().square().sum() for p in model.parameters())))
            local_rows.append({"round": rnd, "client": client.client_id, "train_MAE_local": metric, "lr": lr_r,
                               "norma_mudanca_local": drift, "norma_parametros_local": pn,
                               "nao_finito": not (np.isfinite(metric) and np.isfinite(drift) and np.isfinite(pn))})
            return metric, seen

        previous = global_state
        global_state, train_mae, batches_seen = fedavg_round(
            global_state, clients, weights,
            lambda client: make_client_model(cfg, client, device),
            lambda model: torch.optim.Adam(model.parameters(), lr=lr_r, eps=eps),
            local_epoch, local_epochs, batches_seen, optimizer_states)
        t_eval = time.time()
        val = evaluate_federated(global_state, clients, data, cfg, "validation", device, per_client=instrument)
        per_client = val.pop("per_client", None)
        delta = float(np.sqrt(sum((global_state[k].double() - previous[k].double()).square().sum().item() for k in global_state)))
        gnorm = float(np.sqrt(sum(v.double().square().sum().item() for v in global_state.values())))
        history.append({"round": rnd, ("train_scaled_MAE" if target_scale is None else "train_MAE"): train_mae, **{f"validation_{k}": v for k, v in val.items()},
                        "lr": lr_r, "norma_mudanca_global": delta, "norma_global": gnorm,
                        "train_s": t_eval - t_round, "eval_s": time.time() - t_eval, "elapsed_s": time.time() - t0,
                        **({"cuda_max_allocated_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30}
                           if device.type == "cuda" else {})})
        if instrument:
            vc = {r["client"]: r for r in per_client}
            rows = pd.DataFrame([{**r, **{f"validation_{k}": v for k, v in vc[r["client"]].items() if k != "client"}}
                                 for r in local_rows])
            f = out / "clientes_rodadas.csv"
            if rnd == start_round and f.exists():          # retomada: descarta rodadas que serão refeitas
                old = pd.read_csv(f)
                old[old["round"] < rnd].to_csv(f, index=False)
            rows.to_csv(f, mode="a", header=not f.exists(), index=False)
        if not (np.isfinite(train_mae) and np.isfinite(val["MAE"]) and np.isfinite(delta)):
            raise RuntimeError(f"valor não finito na rodada {rnd}; last.pt da rodada anterior preservado")
        print(f"{tag} | round {rnd:03d}/{max_rounds} | train={train_mae:.4f} | val_MAE={val['MAE']:.4f} "
              f"| macro={val['macro_client_MAE']:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            _save_atomic({"model_state_dict": global_state, "best_round": rnd, "validation_MAE": best_mae}, ckpt)
        if val["MAE"] < ref - float(fed["early_stopping_min_delta"]):
            ref, stale = val["MAE"], 0
        else:
            stale += 1
        stopped_early = stale >= stopping_rule["patience"] and rnd >= stopping_rule["min_round"]
        _save_atomic({"round": rnd, "global_state": global_state, "history": history, "best_mae": best_mae,
                      "stopping_reference": ref, "without_improvement": stale, "batches_seen": batches_seen,
                      "elapsed_s": time.time() - t0, "stopped_early": stopped_early,
                      "client_optimizer": client_optimizer, "optimizer_states": optimizer_states, "retomadas": resumes,
                      "retomadas_execucao": runs, "stopping_rule": stopping_rule,
                      "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                              "loaders": [c.loaders["train"].generator.get_state() for c in clients]}}, last)
        if stopped_early:
            break

    best = torch.load(ckpt, map_location="cpu", weights_only=True)
    rounds = len(history)
    result = {
        "dataset": dataset, "seed": seed, "method": "fedavg_0hop", "model": cfg["model"].get("name", "leconv"),
        "requested_clients": requested_clients, "num_clients": num_clients, "partition_seed": int(fed["partition_seed"]),
        "partition": str(pdir.relative_to(ROOT)) if pdir.is_relative_to(ROOT) else str(pdir), "nodes": data.nodes,
        "client_nodes": [int(c.nodes) for c in clients], "client_train_observations": weights,
        "aggregation_weights": [w / sum(weights) for w in weights],
        "lr_decay_rounds": milestones, "lr_decay_ratio": ratio if milestones else None, "instrumentation": instrument,
        "init_state_sha256": init_sha, "determinism": determinism,
        "adj_sha256": hashlib.sha256(np.ascontiguousarray(data.adj, dtype=np.float32).tobytes()).hexdigest(),
        "client_edges": [int(c.edge_index.shape[1]) for c in clients],
        "input_steps": data.input_steps, "horizon": data.horizon,
        "features": data.feature_names, "target": data.feature_names[data.target_feature_index], "graph": data.graph,
        "local_epochs": local_epochs, "loss_scale": loss_scale, "epsilon": eps, "client_optimizer": client_optimizer, "retomadas": resumes, "best_round": int(best["best_round"]), "rounds_completed": rounds,
        "retomadas_execucao": runs, "stopping_rule": stopping_rule, "stopped_early": bool(history and stale >= stopping_rule["patience"]
                                                                                       and rounds >= stopping_rule["min_round"]),
        "learning_rate": lr, "lr_schedule": {"inicial": lr, "decay_rounds": milestones, "ratio": ratio if milestones else None,
                                             "fronteiras": {str(r): round_lr(r) for r in sorted({1, *milestones, *(m + 1 for m in milestones), rounds}) if r >= 1}},
        "parameter_count": int(sum(v.numel() for v in global_state.values())), "model_state_bytes": int(state_bytes),
        "model_communication_bytes": int(2 * state_bytes * num_clients * rounds),  # download + upload por rodada
        "training_seconds": time.time() - t0, "setup_seconds": setup_seconds, "device": str(device), "gpu": gpu_name(device),
        "torch_threads": torch.get_num_threads(),
        "test_evaluated": evaluate_test,
        "validation": evaluate_federated(best["model_state_dict"], clients, data, cfg, "validation", device, per_node=True),
    }
    if evaluate_test:
        result["test"] = evaluate_federated(best["model_state_dict"], clients, data, cfg, "test", device, per_node=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if not keep_last:
        last.unlink(missing_ok=True)
    scores = (f"test_MAE={result['test']['MAE']:.4f} | macro={result['test']['macro_client_MAE']:.4f}" if evaluate_test
              else f"val_MAE={result['validation']['MAE']:.4f} | teste não avaliado")
    print(f"{tag} | done | best_round={result['best_round']} | {scores} | {result['training_seconds'] / 60:.1f} min", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(dl.DATASETS))
    p.add_argument("--clients", type=int, required=True, help="k pedido (resolvido pelo partitions.csv)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(ROOT / "config" / "fedavg_leconv.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--threads", type=int, default=None,
                   help="limita os threads de CPU do PyTorch (o run_grid usa execution.cpus_per_run)")
    a = p.parse_args()
    if a.threads is not None:
        torch.set_num_threads(a.threads)
    print(f"threads de CPU do PyTorch: {torch.get_num_threads()}", flush=True)
    train_fedavg(a.dataset, a.clients, a.seed, load_config(a.config), a.device, a.max_rounds, a.output, not a.no_resume)


if __name__ == "__main__":
    main()
