"""FED-PUB (Baek, Jeong, Jin, Yoon & Hwang, ICML 2023) sobre as partições Leiden — subgraph FL personalizado.

Personalized Subgraph Federated Learning. Cada cliente k tem um subgrafo (aqui, uma comunidade
Leiden) e recebe do servidor um modelo agregado SÓ PARA ELE, ponderado pela semelhança funcional
com os outros clientes; localmente aplica uma máscara esparsa aos pesos. Três componentes (Sec. 4):

  1. Embedding funcional (Sec. 4.1, Ap. B.3): o servidor gera uma vez um grafo aleatório por
     stochastic block model (5 blocos × 100 nós, p_in = 0.1, p_out = 0.01, atributos N(0,1)) e o
     distribui; cada cliente devolve h_k = média sobre os nós da saída do seu modelo nesse grafo.
     Semelhança S(i,j) = cos(h_i, h_j).
  2. Agregação personalizada (Eq. 3): θ̄_i = Σ_j α_ij θ_j, α_ij = softmax_j(τ·S(i,j)); τ = 10.
     Todos os clientes contribuem (comunidades implícitas, Ap. C.8), mas os semelhantes dominam.
  3. Máscaras esparsas (Sec. 4.2, Eq. 4): pesos efetivos θ_i ⊙ μ_i, com μ_i local (nunca enviada),
     inicializada a 1. Objetivo local
        L(G_i; θ_i, μ_i) + λ1·||μ_i||_1 + λ2·||θ_i − θ̄_i||²_2
     (λ1 = λ2 = 0.001 no paper). Em avaliação, entradas de μ com |μ| ≤ limiar são zeradas
     (Ap. B.3) e só os pesos não zerados são enviados ao servidor (comunicação esparsa).

Escala de λ1. No paper cada época local é UM passo (GCN full-batch, 1–3 épocas, 100 rodadas);
aqui uma época local tem centenas de mini-batches e há até 400 rodadas. Como o Adam normaliza o
gradiente, o termo L1 empurra cada entrada de μ ~lr por passo sempre que o gradiente da tarefa é
fraco, e com λ1 = 0.001 as máscaras colapsam (>95 % de zeros em 2 rodadas no METR-LA). Por isso λ1
é um hiperparâmetro de run (`--l1`), com grade que inclui o valor do paper e um valor adaptado.

Observação empírica neste setup: com `similarity_source: representation` (o do paper) a semelhança
entre clientes no grafo SBM fica em 0.92–0.98 e, com tau = 3, os pesos de agregação quase colapsam
para uniformes (self-alpha 0.22 contra 0.20 do FedAvg) — a LEConv de uma camada com ReLU representa
quase o mesmo em todos os clientes. Com `prediction` a semelhança varia de -0.52 a 0.97 e discrimina
mais, mas afasta-se da referência. O default segue o paper; a alternativa fica disponível para
ablação.

Divergências face ao código oficial (github.com/JinheonBaek/FED-PUB), documentadas e não resolvidas:
o Adam é reiniciado a cada rodada aqui e preservado lá; o estado enviado ao servidor leva os pesos
base com as posições esparsificadas a zero, enquanto a referência envia o estado do modelo (as duas
separam pesos e máscaras, mas a dinâmica não é idêntica). Substituir o envio por `effective_state()`
sem rever a reaplicação das máscaras produziria um terceiro algoritmo, não uma correção.

Adaptação a este repositório: as máscaras aplicam-se a todos os tensores de peso com ≥ 2 dims do
modelo (LEConv: as três projeções lineares da convolução + cabeça; DCRNN: pesos das DCGRUs e da
projeção), não aos vieses; o embedding funcional é a saída do modelo (H passos de previsão) numa
janela aleatória sobre o grafo SBM. Protocolo comum: participação total, Adam reiniciado a cada
rodada (sobre θ_i e μ_i), `local_epochs` épocas locais. Avaliação/early stopping/checkpoint pelos
modelos personalizados θ_i ⊙ μ_i de cada cliente na sua validação (micro + macro), como no Ditto.
Não existe modelo global. Comunicação contabilizada rodada a rodada: upload = pesos não zerados +
embedding; download = θ̄_i denso; `dense_communication_bytes` guarda o equivalente FedAvg.

    python -m experiments.fedpub --dataset metr-la --clients 5 --l1 0.00001 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from federation.fedpub import MaskedModel, maskable, similarity_and_weights, personalized_aggregate, mask_sparsity, fedpub_round
from federation import train_local_epoch, proximal_penalty
from .task import masked_mae_objective

import data_loader as dl
from models import DCRNN, RNNSeq2Seq

from .data import ExperimentData, load_experiment
from .federated import ClientView, load_clients, make_client_model, partition_dir
from federation import cpu_state
from .metrics import NodeErrorAccumulator, RegressionAccumulator
from .train import ROOT, _save_atomic, gpu_name, forward, load_config, set_seed


def l1_token(l1: float) -> str:
    return f"l1_{l1:g}".replace(".", "p").replace("-", "m")


SIMILARITY_SOURCES = ("representation", "representation_pre_activation", "prediction")


def validate_similarity_source(source: str) -> str:
    if source not in SIMILARITY_SOURCES:
        raise ValueError(f"similarity_source desconhecido: {source!r}; use um de {SIMILARITY_SOURCES}")
    return source


def run_subdir(l1: float, source: str) -> Path:
    """Subdiretório por hiperparâmetro. O default mantém o layout histórico `l1_<…>`; as outras fontes
    de embedding ganham `sim_<fonte>`, para que trocar só o YAML não reaproveite resultados alheios."""
    source = validate_similarity_source(source)
    base = Path(l1_token(l1))
    return base if source == "representation" else base / f"sim_{source}"


def check_similarity_source(recorded: str | None, requested: str, what: str) -> None:
    """Cache e retomada só valem para a mesma fonte; registros sem o campo não são aceitos."""
    if recorded is None or validate_similarity_source(recorded) != validate_similarity_source(requested):
        raise ValueError(f"{what} com similarity_source {recorded!r}, pedido {requested!r}; "
                         "use outro output ou reinicie explicitamente")


# ----------------------------------------------------------------------------------------------
# grafo aleatório (SBM) para os embeddings funcionais
# ----------------------------------------------------------------------------------------------
@dataclass
class ProxyGraph:
    """Grafo SBM + janela aleatória gerados uma vez pelo servidor (mesma interface de ClientView para build_model)."""
    name: str
    node_indices: np.ndarray
    adj: np.ndarray
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    input_steps: int
    horizon: int
    n_features: int
    batch: dict[str, torch.Tensor]

    @property
    def nodes(self) -> int:
        return len(self.node_indices)


def make_proxy_graph(data: ExperimentData, blocks: int, nodes_per_block: int, p_in: float, p_out: float, seed: int) -> ProxyGraph:
    rng = np.random.default_rng(seed)
    n = blocks * nodes_per_block
    block = np.repeat(np.arange(blocks), nodes_per_block)
    same = block[:, None] == block[None, :]
    prob = np.where(same, p_in, p_out)
    upper = np.triu(rng.random((n, n)) < prob, k=1)
    adj = (upper | upper.T).astype(np.float32)
    np.fill_diagonal(adj, 1.0)  # auto-laço, como nos subgrafos dos clientes (garante entrada em todos os nós)
    src, dst = np.nonzero(adj)
    gen = torch.Generator().manual_seed(seed)
    W, H, F = data.input_steps, data.horizon, data.n_features
    batch = {
        "x": torch.randn(1, W, n, F, generator=gen),           # atributos N(0, 1) (Ap. B.3)
        "x_mask": torch.ones(1, W, n, F, dtype=torch.bool),
        "x_tod": torch.linspace(0, 1, W + H)[:W].unsqueeze(0),  # só o DCRNN usa
        "y_tod": torch.linspace(0, 1, W + H)[W:].unsqueeze(0),
        "y": torch.zeros(1, H, n, 1),
    }
    return ProxyGraph(name=data.name, node_indices=np.arange(n), adj=adj,
                      edge_index=torch.from_numpy(np.stack([src, dst])).long(), edge_weight=torch.ones(len(src)),
                      input_steps=W, horizon=H, n_features=F, batch=batch)


@torch.no_grad()
def functional_embedding(state: dict, proxy: ProxyGraph, cfg: dict, device, source: str = "representation") -> np.ndarray:
    """h = AVG sobre os nós da REPRESENTAÇÃO do modelo no grafo aleatório (Sec. 4.1).

    O FED-PUB oficial lê a saída da última convolução, antes do classificador — não a previsão.
    Usamos a representação da LEConv pelo mesmo motivo: a semelhança deve comparar o que o
    modelo representa, não o que ele prevê (a cabeça colapsa `hidden` dimensões em `horizon`).
    `representation` lê depois da ReLU (default histórico); `representation_pre_activation` lê
    antes, que é o ponto exato do oficial. O caminho de previsão é o mesmo nas duas.
    """
    model = make_client_model(cfg, proxy, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    b = proxy.batch
    if source == "representation":
        h = model.spatial_embedding(b["x"].to(device), b["x_mask"].to(device))   # (1, Ñ, hidden), pós-ReLU
        h = h.mean(dim=1)
    elif source == "representation_pre_activation":
        h = model.spatial_preactivation(b["x"].to(device), b["x_mask"].to(device))  # antes da ReLU, como o oficial
        h = h.mean(dim=1)
    elif source == "prediction":
        h = forward(model, b, device).mean(dim=2).reshape(1, -1)                 # (1, H)
    else:
        raise ValueError(f"similarity_source desconhecido: {source!r}")
    del model
    return h.reshape(-1).cpu().numpy().astype(np.float64)


def masked_forward(masked: MaskedModel, batch: dict, device, teacher: bool = False, batches_seen: int = 0):
    x, x_mask = batch["x"].to(device, non_blocking=True), batch["x_mask"].to(device, non_blocking=True)
    if isinstance(masked.model, (DCRNN, RNNSeq2Seq)):
        return masked(x, batch["x_tod"].to(device), batch["y"].to(device) if teacher else None,
                      batch["y_tod"].to(device), batches_seen)
    return masked(x, x_mask)


def train_epoch_fedpub(masked: MaskedModel, anchors: list[torch.Tensor], l1: float, prox: float, loader, optimizer,
                       device, clip, batches_seen: int = 0) -> tuple[float, int]:
    params = list(masked.model.parameters())
    def regularizer(model):
        loss = l1 * model.mask_l1() if l1 > 0 else 0.0
        if prox > 0:
            loss = loss + prox * proximal_penalty(params, anchors)
        return loss
    return train_local_epoch(masked, loader, optimizer, masked_mae_objective(device, masked_forward),
                             clip, batches_seen, regularizer)


@torch.no_grad()
def evaluate_personalized(states: list[dict], clients: list[ClientView], data: ExperimentData, cfg: dict, split: str, device,
                          per_node: bool = False) -> dict:
    """Avalia o modelo efetivo de cada cliente no split do SEU cliente (micro sobre todos os alvos + MAE macro)."""
    total, client_maes = RegressionAccumulator(), []
    nodes = NodeErrorAccumulator(data.nodes) if per_node else None
    for state, client in zip(states, clients):
        model = make_client_model(cfg, client, device)
        model.load_state_dict(state, strict=True)
        model.eval()
        local = RegressionAccumulator()
        for batch in client.loaders[split]:
            pred = (forward(model, batch, device) * data.target_std + data.target_mean).clamp_min(0.0)
            y, m = batch["y_raw"].to(device), batch["y_mask"].to(device)
            local.update(pred, y, m)
            total.update(pred, y, m)
            if nodes is not None:
                nodes.update(pred, y, m, client.node_indices)
        client_maes.append(local.compute()["MAE"])
        del model
    metrics = total.compute()
    metrics.update(macro_client_MAE=float(np.mean(client_maes)), min_client_MAE=float(np.min(client_maes)),
                   max_client_MAE=float(np.max(client_maes)))
    if nodes is not None:
        metrics["per_node"] = nodes.compute()
    return metrics


# ----------------------------------------------------------------------------------------------
# treino
# ----------------------------------------------------------------------------------------------
def train_fedpub(dataset: str, requested_clients: int, l1: float, seed: int, cfg: dict, device, max_rounds: int | None = None,
                 output: str | Path | None = None, resume: bool = True) -> dict:
    fp = cfg["fedpub"]
    win = cfg["windows"][dataset]
    # Como no FedAvg: `evaluate_test: false` não cria, não avalia nem grava o teste (smoke/sondagem cega).
    evaluate_test = bool(fp.get("evaluate_test", True))
    l1 = float(l1)
    if l1 < 0:
        raise ValueError("lambda_l1 deve ser >= 0 (0 = sem esparsificação)")
    device = torch.device(device)
    set_seed(seed)
    data = load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
    pdir = partition_dir(dataset, requested_clients, int(fp["partition_seed"]))
    num_clients = int(json.loads((pdir / "metadata.json").read_text())["num_clients"])
    sim_source = validate_similarity_source(fp.get("similarity_source", "representation"))
    sub = run_subdir(l1, sim_source)
    out = ROOT / (output or cfg["output"]) / dataset / f"k{num_clients:03d}" / sub / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
    clients = load_clients(data, pdir, batch_size, cfg["data"]["num_workers"], seed,
                           ("train", "validation", "test") if evaluate_test else ("train", "validation"))
    max_rounds = int(max_rounds or fp.get("max_rounds_per_dataset", {}).get(dataset, fp["max_rounds"]))
    local_epochs, clip, lr = int(fp["local_epochs"]), float(fp["gradient_clip_norm"]), float(fp["learning_rate"])
    tau, prox, thr = float(fp["tau"]), float(fp["lambda_prox"]), float(fp["mask_threshold"])
    pg = fp["proxy_graph"]
    proxy = make_proxy_graph(data, int(pg["blocks"]), int(pg["nodes_per_block"]), float(pg["p_in"]), float(pg["p_out"]), int(pg["seed"]))
    tag = f"{dataset}/k{num_clients:03d}/{sub.as_posix()}/seed_{seed:03d}"

    init = make_client_model(cfg, clients[0], device)
    init_state = cpu_state(init)  # θ_0 comum (seed)
    mask_names = [n for n, p in init.named_parameters() if maskable(n, p)]
    del init
    state_bytes = sum(v.numel() * v.element_size() for v in init_state.values())
    elem_bytes = next(iter(init_state.values())).element_size()
    # Embedding funcional enviado por cliente e rodada: o tamanho EFETIVO do vetor que functional_embedding
    # devolve (float64; hidden dimensões na representação, horizon na previsão). Antes contava horizon × 8 bytes
    # em qualquer fonte. Medido sobre θ_0, que tem a mesma forma de qualquer estado posterior.
    embedding_bytes = int(functional_embedding(init_state, proxy, cfg, device, sim_source).nbytes)
    # estado por cliente: θ̄_i recebido para a rodada, θ_i treinado, μ_i (local, nunca enviada)
    received = [{k: v.clone() for k, v in init_state.items()} for _ in clients]
    masks = [{n: torch.ones_like(init_state[n]) for n in mask_names} for _ in clients]  # μ_i = 1 (Sec. 4.2)
    trained = [None] * num_clients
    similarity = alpha = np.eye(num_clients)
    history, best_mae, ref, stale, batches_seen, comm_bytes = [], float("inf"), float("inf"), 0, 0, 0
    ckpt, last = out / "best_model.pt", out / "last.pt"
    t0, start_round = time.time(), 1
    if resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        check_similarity_source(st.get("similarity_source"), sim_source, "checkpoint de retomada")
        received, masks, trained, history = st["received"], st["masks"], st["trained"], st["history"]
        similarity, alpha = st["similarity"], st["alpha"]
        best_mae, ref, stale, batches_seen, comm_bytes = (st["best_mae"], st["stopping_reference"], st["without_improvement"],
                                                          st["batches_seen"], st["communication_bytes"])
        start_round = st["round"] + 1
        t0 -= st["elapsed_s"]
        torch.set_rng_state(st["rng"]["torch"]); np.random.set_state(st["rng"]["numpy"])
        for c, g in zip(clients, st["rng"]["loaders"]):
            c.loaders["train"].generator.set_state(g)
        print(f"{tag} | retomando da rodada {st['round']} (best val_MAE={best_mae:.4f})", flush=True)
        if st.get("stopped_early"):
            start_round = max_rounds + 1

    for rnd in range(start_round, max_rounds + 1):
        update = fedpub_round(
            received, masks, clients, [c.train_observations for c in clients],
            lambda client: make_client_model(cfg, client, device),
            lambda model: torch.optim.Adam(model.parameters(), lr=lr),
            lambda masked, client, optimizer, anchors, seen:
                train_epoch_fedpub(masked, anchors, l1, prox, client.loaders["train"], optimizer, device, clip, seen),
            lambda state: functional_embedding(state, proxy, cfg, device, sim_source),
            thr, tau, local_epochs, batches_seen)
        received, masks, trained, effective = update.received, update.masks, update.trained, update.effective
        similarity, alpha = update.similarity, update.alpha
        train_mae, batches_seen = update.metric, update.batches_seen
        for count in update.sent_counts:
            comm_bytes += state_bytes + count * elem_bytes + embedding_bytes
        val = evaluate_personalized(effective, clients, data, cfg, "validation", device)
        sparsity = float(np.mean([mask_sparsity(m, thr) for m in masks]))
        history.append({"round": rnd, "train_scaled_MAE": train_mae, **{f"validation_{k}": v for k, v in val.items()},
                        "mask_sparsity": sparsity, "mean_self_alpha": float(np.mean(np.diag(alpha))),
                        "communication_MB": comm_bytes / 1e6, "elapsed_s": time.time() - t0})
        print(f"{tag} | round {rnd:03d}/{max_rounds} | train={train_mae:.4f} | val_MAE={val['MAE']:.4f} "
              f"| macro={val['macro_client_MAE']:.4f} | sparsity={sparsity:.3f} | self-α={history[-1]['mean_self_alpha']:.2f} "
              f"| {(time.time() - t0) / 60:.1f} min", flush=True)
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            _save_atomic({"effective_states": effective, "trained": trained, "masks": masks, "similarity": similarity,
                          "alpha": alpha, "best_round": rnd, "validation_MAE": best_mae}, ckpt)
        if val["MAE"] < ref - float(fp["early_stopping_min_delta"]):
            ref, stale = val["MAE"], 0
        else:
            stale += 1
        stopped_early = stale >= int(fp["patience"])
        _save_atomic({"round": rnd, "similarity_source": sim_source, "received": received, "masks": masks,
                      "trained": trained, "history": history,
                      "similarity": similarity, "alpha": alpha, "best_mae": best_mae, "stopping_reference": ref,
                      "without_improvement": stale, "batches_seen": batches_seen, "communication_bytes": comm_bytes,
                      "elapsed_s": time.time() - t0, "stopped_early": stopped_early,
                      "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                              "loaders": [c.loaders["train"].generator.get_state() for c in clients]}}, last)
        if stopped_early:
            break

    best = torch.load(ckpt, map_location="cpu", weights_only=False)
    rounds = len(history)
    result = {
        "dataset": dataset, "seed": seed, "method": "fedpub_0hop", "model": cfg["model"].get("name", "leconv"),
        "tau": tau, "lambda_l1": l1, "lambda_prox": prox, "mask_threshold": thr, "proxy_graph": dict(pg),
        "similarity_source": sim_source,
        "requested_clients": requested_clients, "num_clients": num_clients, "partition_seed": int(fp["partition_seed"]),
        "partition": str(pdir.relative_to(ROOT)) if pdir.is_relative_to(ROOT) else str(pdir), "nodes": data.nodes,
        "client_nodes": [int(c.nodes) for c in clients], "client_train_observations": [c.train_observations for c in clients],
        "input_steps": data.input_steps, "horizon": data.horizon,
        "features": data.feature_names, "target": data.feature_names[data.target_feature_index], "graph": getattr(data, "graph", "default"),
        "local_epochs": local_epochs, "best_round": int(best["best_round"]), "rounds_completed": rounds,
        "parameter_count": int(sum(v.numel() for v in init_state.values())), "model_state_bytes": int(state_bytes),
        "masked_parameter_count": int(sum(init_state[n].numel() for n in mask_names)), "masked_tensors": mask_names,
        # ESTIMATIVA: download denso + valores não podados do upload + embedding; não conta índices nem formato
        # esparso (os estados manipulados são tensores densos) e não é tráfego medido.
        "model_communication_bytes": int(comm_bytes),
        "communication_accounting": "estimativa: download denso + valores do upload sem índices + embedding (tamanho efetivo, float64)",
        "embedding_bytes": embedding_bytes,
        "dense_communication_bytes": int(2 * state_bytes * num_clients * rounds),  # equivalente FedAvg
        "mask_sparsity": float(np.mean([mask_sparsity(m, thr) for m in best["masks"]])),
        "client_mask_sparsity": [mask_sparsity(m, thr) for m in best["masks"]],
        "similarity": np.asarray(best["similarity"]).round(6).tolist(), "alpha": np.asarray(best["alpha"]).round(6).tolist(),
        "training_seconds": time.time() - t0, "device": str(device), "gpu": gpu_name(device),
        "test_evaluated": evaluate_test,
        "validation": evaluate_personalized(best["effective_states"], clients, data, cfg, "validation", device, per_node=True),
    }
    if evaluate_test:
        result["test"] = evaluate_personalized(best["effective_states"], clients, data, cfg, "test", device, per_node=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    last.unlink(missing_ok=True)
    scores = (f"test_MAE={result['test']['MAE']:.4f} | macro={result['test']['macro_client_MAE']:.4f}" if evaluate_test
              else f"val_MAE={result['validation']['MAE']:.4f} | teste não avaliado")
    print(f"{tag} | done | best_round={result['best_round']} | {scores} | sparsity={result['mask_sparsity']:.3f} "
          f"| {result['training_seconds'] / 60:.1f} min", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(dl.DATASETS))
    p.add_argument("--clients", type=int, required=True)
    p.add_argument("--l1", type=float, required=True, help="lambda_1 da esparsidade das máscaras (0 = sem L1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(ROOT / "config" / "fedpub_leconv.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    train_fedpub(a.dataset, a.clients, a.l1, a.seed, load_config(a.config), a.device, a.max_rounds, a.output, not a.no_resume)


if __name__ == "__main__":
    main()
