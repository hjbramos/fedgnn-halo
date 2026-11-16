"""FedGNNHet — FedAvg 0-hop + halo de fronteira (nós virtuais com o agregado dos vizinhos externos).

Problema: no fedavg_0hop (experiments/federated.py) as arestas entre clientes são descartadas, e um nó
de fronteira v perde os vizinhos u que vivem noutros clientes. Aqui cada cliente recebe, para cada nó
de fronteira v, UM nó virtual ("halo") que carrega o agregado ponderado dos vizinhos externos,

    a_v(t) = Σ_{u ∉ cliente, u→v} w_uv · x_u(t) / W_v,      W_v = Σ_{u ∉ cliente, u→v} w_uv ,

ligado a v por uma aresta de peso W_v. Como a LEConv é linear nos vizinhos e normaliza os pesos de
entrada por nó destino,

    x'_v = W1 x_v + Σ_j e_vj (W2 x_v − W3 x_j),   e_vj = w_vj / Σ_j w_vj ,

a soma dos pesos de entrada de v volta a ser a do grafo completo e o termo Σ_j e_vj x_j é reconstruído
EXATAMENTE: com uma camada LEConv, a saída do cliente nos seus nós é igual à do modelo centralizado
com os mesmos pesos (verificado em `check_exactness`). O que se troca é o agregado, não a série de cada
vizinho — uma soma ponderada de ≥ 1 séries (quando só há 1 vizinho externo é a série dele; ver
`halo_single_neighbour_pct` no metrics.json).

Variantes (`--halo`):
  none    sem halo — equivalente ao fedavg_0hop (mesmo grafo, mesmas janelas, mesmo protocolo);
          em CPU reproduz o fedavg_0hop bit a bit. Em GPU não: o scatter-add da LEConv usa atomics
          e não é determinístico (~2e-6 no forward, ~3e-5 no gradiente por passo), pelo que dois
          runs idênticos — mesmo do MESMO script — divergem ~1e-3 na MAE final ao fim de ~50
          rodadas. Esse ruído de hardware está incluído no desvio entre seeds;
  static  a_v constante no tempo = média de a_v(t) sobre o período de treino. É o papel da
          BoundaryTable/EMA do FedGNNHet original (resumo estático do vizinho, que não depende da
          janela atual); serve de ablação "estático vs dinâmico";
  1hop    a_v(t) por janela (dinâmico; equivalente ao FedGCN 1-hop de Yao et al. 2023 adaptado a
          séries temporais). É a contribuição.
  1hop_min2  ablação do `1hop`: cria o nó virtual apenas quando o nó de fronteira tem DOIS OU MAIS
          vizinhos externos. Os nós de grau externo 1 ficam sem halo (comportam-se como `none`),
          porque nesses casos o agregado reproduz a série padronizada e a máscara do único vizinho,
          idênticas. Mede o CUSTO PREDITIVO de suprimir essas mensagens — não é mecanismo de
          privacidade, e não há garantia de proteção nos nós que restam (ver README, contrato de
          comunicação). Nada se assume sobre a utilidade relativa das mensagens suprimidas: é isso
          que a ablação mede.

O agregado é função só dos dados brutos, por isso é trocado UMA vez antes do treino (`halo_bytes` no
metrics.json: valores + máscara, float32); o resto do protocolo — participação total, Adam reiniciado,
agregação FedAvg ponderada, early stopping — é o do fedavg_0hop. Só nós locais entram na perda, nas
métricas e no `per_node`. Modelo: LEConv (o DCRNN difunde K passos, a exatidão exigiria halos K-hop).

    python -m experiments.fedgnnhet --dataset metr-la --clients 5 --halo 1hop --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
try:
    import resource
except ImportError:
    resource = None
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import data_loader as dl
from models import build_model
from .task import model_spec
from federation import fedavg_round
from federation.halo import Halo, aggregate_halo

from .data import ExperimentData, WindowDataset, load_experiment
from .federated import ClientView, client_edges, partition_dir
from federation import cpu_state
from .metrics import NodeErrorAccumulator, RegressionAccumulator
from .train import ROOT, _save_atomic, gpu_name, forward, load_config, make_loader, set_seed, train_epoch

HALOS = ("none", "static", "1hop", "1hop_min2")


def halo_token(halo: str) -> str:
    return f"halo_{halo}"


def format_exactness(exact: dict) -> str:
    """Linha de log da verificação de exatidão.

    Existe como função para ser testável: em 24/09 esta formatação recebeu um dict onde antes vinha
    um float, `{exact:.2e}` levantou TypeError e derrubou todas as execuções com halo `1hop`. Um
    teste de string sobre o código-fonte não teria apanhado isso; um teste desta função apanha.
    """
    sem = exact.get("sem_halo")
    extra = "" if sem is None else f" | nós sem halo: {sem:.2e}"
    return f"exatidão vs centralizado, nós com halo: erro máx {exact['com_halo']:.2e}{extra}"


# ----------------------------------------------------------------------------------------------
# halo: agregado dos vizinhos externos por nó de fronteira
# ----------------------------------------------------------------------------------------------
def build_halo(data: ExperimentData, nodes: np.ndarray, halo: str) -> Halo:
    values = np.where(data.mask, (data.series - data.scaler_mean) / data.scaler_std, 0.0).astype(np.float32)
    cutoff = None
    if halo == "static":
        cutoff = (int(data.origins["validation"].min()) if len(data.origins["validation"])
                  else int(data.origins["train"].max() + data.horizon))
    return aggregate_halo(data.adj, nodes, values, data.mask,
                          min_neighbors=2 if halo == "1hop_min2" else 1, static_until=cutoff)


class HaloWindowDataset(WindowDataset):
    """WindowDataset do cliente + nós virtuais no fim: x = [locais ‖ halo], x_mask float, alvo só nos locais."""

    def __init__(self, data: ExperimentData, split: str, node_indices: np.ndarray, halo: Halo):
        super().__init__(data, split, node_indices)
        self.halo = halo

    def __getitem__(self, i):
        b = super().__getitem__(i)
        o = int(self.origins[i])
        h = self.halo.size
        hx = torch.from_numpy(self.halo.values[o - self.W : o])           # (W, h, F)
        hm = torch.from_numpy(self.halo.mask[o - self.W : o])
        b["x"] = torch.cat([b["x"], hx], dim=1)
        b["x_mask"] = torch.cat([b["x_mask"].float(), hm], dim=1)
        pad = torch.zeros(self.H, h, 1)
        b["y"] = torch.cat([b["y"], pad], dim=1)
        b["y_raw"] = torch.cat([b["y_raw"], pad], dim=1)
        b["y_mask"] = torch.cat([b["y_mask"], torch.zeros(self.H, h, 1, dtype=torch.bool)], dim=1)
        return b


@dataclass
class HaloClientView(ClientView):
    """ClientView com o grafo aumentado (locais + virtuais); `nodes` inclui os virtuais, `n_local` não."""
    halo: Halo | None = None

    @property
    def nodes(self) -> int:
        return len(self.node_indices) + (self.halo.size if self.halo else 0)

    @property
    def n_local(self) -> int:
        return len(self.node_indices)


def load_halo_clients(data: ExperimentData, directory: Path, batch_size: int, num_workers: int, seed: int,
                      halo: str, splits: tuple[str, ...] = ("train", "validation", "test")) -> list[HaloClientView]:
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
        edge_index, edge_weight = client_edges(data, cdir, nodes)                # arestas ORIGINAIS internas
        h = None
        if halo != "none":
            h = build_halo(data, nodes, halo)
            n = len(nodes)
            virt = n + np.arange(h.size)
            # virtual_v -> v com peso W_v (repõe a soma de pesos de entrada do grafo completo) + auto-laço no virtual
            extra_src = np.concatenate([virt, virt])
            extra_dst = np.concatenate([h.boundary_local, virt])
            extra_w = np.concatenate([h.weight, np.ones(h.size, dtype=np.float32)])
            edge_index = torch.cat([edge_index, torch.from_numpy(np.stack([extra_src, extra_dst])).long()], dim=1)
            edge_weight = torch.cat([edge_weight, torch.from_numpy(extra_w).float()])
        ds = {s: (HaloWindowDataset(data, s, nodes, h) if h else WindowDataset(data, s, nodes)) for s in splits}
        loaders = {s: make_loader(ds[s], batch_size, s == "train", seed + c, num_workers) for s in ds}
        clients.append(HaloClientView(
            name=data.name, client_id=c, node_indices=nodes, adj=data.adj[np.ix_(nodes, nodes)],
            edge_index=edge_index, edge_weight=edge_weight,
            input_steps=data.input_steps, horizon=data.horizon, n_features=data.n_features,
            train_observations=int(tmask[train_targets[:, :, None], nodes[None, None, :]].sum()), loaders=loaders, halo=h))
    return clients


def make_client_model(cfg: dict, client: HaloClientView, device) -> nn.Module:
    m = dict(cfg["model"])
    name = m.pop("name", "leconv")
    if name == "leconv_edge":
        raise ValueError(
            "FedGNNHet não suporta leconv_edge: com pesos de aresta por janela o agregado externo teria "
            "de ser ponderado pelos mesmos pesos (a_v^(o) = Σ w_uv^(o)·x_u / Σ w_uv^(o)), com máscara e "
            "peso da aresta virtual a acompanhar. É pré-computável, mas não está implementado.")
    if name != "leconv":
        raise ValueError("FedGNNHet: o halo 1-hop é exato só para a LEConv (1 camada); use model.name: leconv")
    m.pop("gradient_checkpointing_per_dataset", None)
    return build_model(name, model_spec(client, m, name), **m).to(device)


@torch.no_grad()
def evaluate_halo(global_state: dict, clients: list[HaloClientView], data: ExperimentData, cfg: dict, split: str, device,
                  per_node: bool = False) -> dict:
    """Como federated.evaluate_federated, mas só os nós locais (os virtuais são descartados)."""
    total, client_maes = RegressionAccumulator(), []
    nodes = NodeErrorAccumulator(data.nodes) if per_node else None
    for client in clients:
        model = make_client_model(cfg, client, device)
        model.load_state_dict(global_state, strict=True)
        model.eval()
        local, n = RegressionAccumulator(), client.n_local
        for batch in client.loaders[split]:
            pred = (forward(model, batch, device) * data.target_std + data.target_mean).clamp_min(0.0)[:, :, :n]
            y, m = batch["y_raw"].to(device)[:, :, :n], batch["y_mask"].to(device)[:, :, :n]
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


@torch.no_grad()
def check_exactness(data: ExperimentData, clients: list[HaloClientView], cfg: dict, device, n_windows: int = 4) -> dict:
    """Erro máximo |saída do cliente com halo − saída do centralizado| nos nós locais, MESMOS PESOS.

    Devolve `{"todos", "com_halo", "sem_halo"}`. A separação existe para o `1hop_min2`, onde os nós de
    grau externo 1 perdem o halo e deixam de reproduzir o centralizado — só faz sentido exigir
    exatidão onde a mensagem foi de facto reposta.

    **Limite desta verificação:** é equivalência da OPERAÇÃO com pesos fixos e a mesma normalização.
    Não implica equivalência depois do treino: suprimir mensagens muda os parâmetros compartilhados e
    pode alterar previsões em TODOS os nós, inclusive nos que mantiveram halo.
    """
    from .train import make_model
    central = make_model(cfg, data, device).eval()
    state = cpu_state(central)
    full = WindowDataset(data, "train")
    pior = {"todos": 0.0, "com_halo": 0.0, "sem_halo": 0.0}
    viu_sem_halo = False
    for i in np.linspace(0, len(full) - 1, n_windows).astype(int):
        fb = full[int(i)]
        ref = central(fb["x"].unsqueeze(0).to(device), fb["x_mask"].unsqueeze(0).to(device))[0, :, :, 0]  # (H, N)
        for client in clients:
            model = make_client_model(cfg, client, device).eval()
            model.load_state_dict(state, strict=True)
            b = client.loaders["train"].dataset[int(i)]
            out = model(b["x"].unsqueeze(0).to(device), b["x_mask"].unsqueeze(0).to(device))[0, :, : client.n_local, 0]
            err = (out - ref[:, client.node_indices]).abs()                      # (H, n_local)
            pior["todos"] = max(pior["todos"], float(err.max()))
            sup = client.halo.suppressed_local if client.halo is not None else None
            if sup is not None and len(sup):
                viu_sem_halo = True
                m = torch.zeros(client.n_local, dtype=torch.bool, device=err.device)
                m[torch.from_numpy(np.asarray(sup)).long().to(err.device)] = True
                pior["sem_halo"] = max(pior["sem_halo"], float(err[:, m].max()))
                # um cliente pode ter TODOS os nós locais suprimidos: `err[:, ~m]` fica vazio e
                # `.max()` levanta RuntimeError. Nesse caso não há nó com halo para verificar.
                if bool((~m).any()):
                    pior["com_halo"] = max(pior["com_halo"], float(err[:, ~m].max()))
            else:
                pior["com_halo"] = max(pior["com_halo"], float(err.max()))
    if not viu_sem_halo:
        pior["sem_halo"] = None
    return pior


# ----------------------------------------------------------------------------------------------
# treino (== fedavg_0hop com clientes aumentados)
# ----------------------------------------------------------------------------------------------
def train_fedgnnhet(dataset: str, requested_clients: int, halo: str, seed: int, cfg: dict, device,
                    max_rounds: int | None = None, output: str | Path | None = None, resume: bool = True, *, data=None, partition_path: str | Path | None = None) -> dict:
    if halo not in HALOS:
        raise ValueError(f"halo deve ser um de {HALOS}")
    fg = cfg["fedgnnhet"]
    win = cfg["windows"][dataset]
    device = torch.device(device)
    # Mesmas opções do FedAvg e do centralizado: `evaluate_test: false` não cria, não avalia nem grava o
    # teste (smoke/sondagem); `keep_last_checkpoint: true` preserva o last.pt para retomar aumentando o teto.
    evaluate_test = bool(fg.get("evaluate_test", True))
    keep_last = bool(fg.get("keep_last_checkpoint", False))
    t_setup = time.time()
    set_seed(seed)
    data = data if data is not None else load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
    pdir = Path(partition_path) if partition_path is not None else partition_dir(dataset, requested_clients, int(fg["partition_seed"]))
    num_clients = int(json.loads((pdir / "metadata.json").read_text())["num_clients"])
    out = ROOT / (output or cfg["output"]) / dataset / f"k{num_clients:03d}" / halo_token(halo) / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
    clients = load_halo_clients(data, pdir, batch_size, cfg["data"]["num_workers"], seed, halo,
                                ("train", "validation", "test") if evaluate_test else ("train", "validation"))
    weights = [c.train_observations for c in clients]
    max_rounds = int(max_rounds or fg.get("max_rounds_per_dataset", {}).get(dataset, fg["max_rounds"]))
    local_epochs, clip, lr = int(fg["local_epochs"]), float(fg["gradient_clip_norm"]), float(fg["learning_rate"])
    tag = f"{dataset}/k{num_clients:03d}/{halo_token(halo)}/seed_{seed:03d}"

    halo_nodes = [int(c.halo.size) if c.halo else 0 for c in clients]
    suprimidos = [int(c.halo.n_suppressed) if c.halo else 0 for c in clients]
    # bytes realmente transmitidos, trocados uma vez: o `static` é constante no tempo, logo basta uma
    # cópia por nó virtual; o `1hop` e o `1hop_min2` precisam da série inteira. (Contar o array
    # expandido do `static` mediria memória, não comunicação.)
    def _halo_payload(h) -> int:
        per_step = h.values[0].nbytes + h.mask[0].nbytes
        return per_step if halo == "static" else h.values.nbytes + h.mask.nbytes
    halo_bytes = int(sum(_halo_payload(c.halo) for c in clients if c.halo))
    single = [int((c.halo.n_external == 1).sum()) for c in clients if c.halo]
    exact = None
    if halo in ("1hop", "1hop_min2"):   # o min2 é justamente onde a equivalência precisa ser delimitada
        with torch.random.fork_rng(devices=[]):  # não altera a sequência de RNG do treino
            exact = check_exactness(data, clients, cfg, device)
    if exact is not None:
        print(f"{tag} | {format_exactness(exact)}", flush=True)

    global_state = cpu_state(make_client_model(cfg, clients[0], device))  # inicialização comum (seed)
    state_bytes = sum(v.numel() * v.element_size() for v in global_state.values())
    history, best_mae, ref, stale, batches_seen = [], float("inf"), float("inf"), 0, 0
    ckpt, last = out / "best_model.pt", out / "last.pt"
    setup_seconds = time.time() - t_setup  # carga, agregados do halo e verificação de exatidão
    t0, start_round = time.time(), 1
    if resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
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
        t_round = time.time()
        global_state, train_mae, batches_seen = fedavg_round(
            global_state, clients, weights,
            lambda client: make_client_model(cfg, client, device),
            lambda model: torch.optim.Adam(model.parameters(), lr=lr),
            lambda model, client, optimizer, anchors, seen: train_epoch(model, client.loaders["train"], optimizer, device, clip, seen),
            local_epochs, batches_seen)
        t_eval = time.time()
        val = evaluate_halo(global_state, clients, data, cfg, "validation", device)
        history.append({"round": rnd, "train_scaled_MAE": train_mae, **{f"validation_{k}": v for k, v in val.items()},
                        "train_s": t_eval - t_round, "eval_s": time.time() - t_eval, "elapsed_s": time.time() - t0})
        print(f"{tag} | round {rnd:03d}/{max_rounds} | train={train_mae:.4f} | val_MAE={val['MAE']:.4f} "
              f"| macro={val['macro_client_MAE']:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            _save_atomic({"model_state_dict": global_state, "best_round": rnd, "validation_MAE": best_mae}, ckpt)
        if val["MAE"] < ref - float(fg["early_stopping_min_delta"]):
            ref, stale = val["MAE"], 0
        else:
            stale += 1
        stopped_early = stale >= int(fg["patience"])
        _save_atomic({"round": rnd, "global_state": global_state, "history": history, "best_mae": best_mae,
                      "stopping_reference": ref, "without_improvement": stale, "batches_seen": batches_seen,
                      "elapsed_s": time.time() - t0, "stopped_early": stopped_early,
                      "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                              "loaders": [c.loaders["train"].generator.get_state() for c in clients]}}, last)
        if stopped_early:
            break

    best = torch.load(ckpt, map_location="cpu", weights_only=True)
    rounds = len(history)
    result = {
        "dataset": dataset, "seed": seed, "method": f"fedgnnhet_{halo}", "halo": halo, "model": cfg["model"].get("name", "leconv"),
        "requested_clients": requested_clients, "num_clients": num_clients, "partition_seed": int(fg["partition_seed"]),
        "partition": str(pdir.relative_to(ROOT)) if pdir.is_relative_to(ROOT) else str(pdir), "nodes": data.nodes,
        "client_nodes": [int(c.n_local) for c in clients], "client_train_observations": weights,
        "client_halo_nodes": halo_nodes, "halo_nodes_total": int(sum(halo_nodes)),
        "halo_single_neighbour_pct": (100.0 * sum(single) / max(sum(halo_nodes), 1)) if halo != "none" else None,
        "halo_suppressed_nodes": int(sum(c.halo.n_suppressed for c in clients if c.halo)) if halo != "none" else None,
        # dict {todos, com_halo, sem_halo}: equivalência da OPERAÇÃO com pesos fixos, não pós-treino
        "halo_exactness_max_abs_error": exact,
        "input_steps": data.input_steps, "horizon": data.horizon,
        "features": data.feature_names, "target": data.feature_names[data.target_feature_index], "graph": getattr(data, "graph", "default"),
        "local_epochs": local_epochs, "best_round": int(best["best_round"]), "rounds_completed": rounds,
        "parameter_count": int(sum(v.numel() for v in global_state.values())), "model_state_bytes": int(state_bytes),
        "model_communication_bytes": int(2 * state_bytes * num_clients * rounds),
        "halo_bytes": halo_bytes,                                                 # agregados de fronteira, uma vez
        "total_communication_bytes": int(2 * state_bytes * num_clients * rounds + halo_bytes),
        "training_seconds": time.time() - t0, "setup_seconds": setup_seconds, "device": str(device), "gpu": gpu_name(device),
        "peak_rss_gb": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                        (1e9 if sys.platform == "darwin" else 1e6)) if resource else None,
        "peak_gpu_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None,
        "test_evaluated": evaluate_test,
        "validation": evaluate_halo(best["model_state_dict"], clients, data, cfg, "validation", device, per_node=True),
    }
    if evaluate_test:
        result["test"] = evaluate_halo(best["model_state_dict"], clients, data, cfg, "test", device, per_node=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if not keep_last:
        last.unlink(missing_ok=True)
    scores = (f"test_MAE={result['test']['MAE']:.4f} | macro={result['test']['macro_client_MAE']:.4f}" if evaluate_test
              else f"val_MAE={result['validation']['MAE']:.4f} | teste não avaliado")
    print(f"{tag} | done | best_round={result['best_round']} | {scores} | halo={sum(halo_nodes)} nós, "
          f"{halo_bytes / 1e6:.1f} MB | {result['training_seconds'] / 60:.1f} min", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(dl.DATASETS))
    p.add_argument("--clients", type=int, required=True)
    p.add_argument("--halo", required=True, choices=HALOS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(ROOT / "config" / "fedgnnhet_leconv.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    train_fedgnnhet(a.dataset, a.clients, a.halo, a.seed, load_config(a.config), a.device, a.max_rounds, a.output, not a.no_resume)


if __name__ == "__main__":
    main()
