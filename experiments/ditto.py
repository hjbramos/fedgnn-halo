"""Ditto (Li, Hu, Beirami & Smith, ICML 2021) sobre as partições Leiden — FL personalizado.

Cada cliente k mantém DOIS modelos (Alg. 1 do paper):
  - o modelo GLOBAL w, atualizado exatamente como no FedAvg (experiments/federated.py): o cliente
    recebe w_t, treina `local_epochs` épocas em F_k e devolve o estado; o servidor agrega com pesos
    proporcionais às observações de treino;
  - o modelo PERSONALIZADO v_k, que nunca sai do cliente e persiste entre rodadas, atualizado na
    mesma rodada a partir do w_t recebido minimizando (Eq. 2 do paper)
        h_k(v_k; w_t) = F_k(v_k) + (lambda/2) * ||v_k - w_t||^2 .
  lambda = 0 recupera o treino puramente local (com w_t ignorado); lambda -> inf recupera o FedAvg.

Seleção de lambda (Sec. 4 e Ap. D.2 do paper): cada cliente escolhe o SEU lambda de um conjunto
candidato pelo desempenho na sua validação local. Com `--lam per_client`, cada cliente mantém um
v_k^lambda por candidato (todos ancorados no mesmo w_t) e, a cada avaliação, usa o de menor MAE de
validação local; o teste usa o lambda escolhido na validação da melhor rodada. Com `--lam <float>`,
o lambda é fixo e igual para todos (o "best fixed lambda" da Tabela 5 do paper).

A diferença para o FedProx (experiments/fedprox.py) é estrutural, não só de sinal: no FedProx o
termo proximal regula o modelo que É agregado (um único modelo global); no Ditto a agregação é
FedAvg pura e o termo proximal ancora um modelo local separado, que é o entregável do método.

Protocolo (comum ao repositório): participação total, Adam reiniciado a cada rodada em ambos os
modelos, `local_epochs` épocas locais para cada um (o paper usa o mesmo nº de passos para os dois),
agregação ponderada por observações. Avaliação: o conjunto {v_k} é avaliado na validação/teste de
cada cliente (micro + macro por cliente) — é isso que decide early stopping e checkpoint; o modelo
global w é avaliado da mesma forma e guardado como referência (`*_global` no metrics.json) — é a
mesma regra de atualização do FedAvg, mas não a mesma trajetória, porque os dois treinos locais
partilham o gerador de embaralhamento do DataLoader.
Comunicação = 2 × estado × clientes × rodadas, igual ao FedAvg (v_k não é transmitido). Custo local
≈ 2× o do FedAvg. Retomada por rodada em last.pt (inclui todos os v_k).

Nota: o paper usa SGD e participação parcial; aqui mantém-se Adam/participação total por
comparabilidade com o FedAvg/FedProx deste repositório.

Local (`ditto.local_only: true`, só com lambda = 0): mesma trajetória dos modelos pessoais do Ditto com
lambda = 0, sem treinar, agregar nem avaliar o modelo global. Com lambda = 0 o termo proximal não existe e o w_t
recebido não entra na atualização pessoal; o único acoplamento restante é o gerador de embaralhamento do DataLoader,
que o treino global consome antes do pessoal em cada cliente. O caminho Local consome esse gerador exatamente como
uma época do treino global consumiria (`consume_like_epoch`), sem carregar lotes, e preserva a ordem dos lotes
pessoais. A equivalência com o caminho original é conferida em tests/test_local_ditto.py e no ensaio do registro
(results/lote6_sunt_local/). `ditto.evaluate_test: false` adia o teste para `evaluate_test_from_checkpoint`.

    python -m experiments.ditto --dataset metr-la --clients 5 --lam 0.1 --seed 42
    python -m experiments.ditto --dataset metr-la --clients 5 --lam per_client --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import data_loader as dl

from federation import train_local_epoch, proximal_penalty, ditto_round
from .task import masked_mae_objective

from .data import ExperimentData, load_experiment
from .federated import ClientView, evaluate_federated, load_clients, make_client_model, partition_dir
from federation import cpu_state
from .metrics import NodeErrorAccumulator, RegressionAccumulator
from .train import ROOT, _save_atomic, gpu_name, forward, load_config, set_seed, train_epoch


PER_CLIENT = "per_client"


def parse_lambda(value: str | float) -> float | str:
    """Aceita um float >= 0 ou a string 'per_client'."""
    if isinstance(value, str) and value.strip().lower() == PER_CLIENT:
        return PER_CLIENT
    lam = float(value)
    if lam < 0:
        raise ValueError("lambda deve ser >= 0 (0 = treino local puro) ou 'per_client'")
    return lam


def lambda_token(lam: float | str) -> str:
    if lam == PER_CLIENT:
        return "lam_per_client"
    return f"lam{lam:g}".replace(".", "p").replace("-", "m")


def lambda_label(lam: float | str) -> str:
    return PER_CLIENT if lam == PER_CLIENT else f"{lam:g}"


def consume_like_epoch(loader) -> None:
    """Consome do gerador do DataLoader o mesmo que uma época completa consumiria, sem montar lotes.

    Numa época, o iterador sorteia a semente base (e a compartilhada) e o RandomSampler sorteia uma permutação no
    início e outra ao se esgotar. Aqui o iterador é criado (sementes) e só o sampler é esgotado (permutações).
    tests/test_local_ditto.py confere, na versão instalada do torch, que o estado do gerador fica idêntico ao de uma
    época real.
    """
    it = iter(loader)
    for _ in it._sampler_iter:
        pass
    del it


def local_round(personal, clients, weights, make_model, make_optimizer, train_personal, local_epochs, personal_seen=0):
    """Rodada Local: só os modelos pessoais (lambda = 0), na mesma ordem e com o mesmo consumo de RNG do ditto_round.

    Para cada cliente, consome o gerador como as `local_epochs` épocas do treino global consumiriam e depois treina o
    modelo pessoal a partir do PRÓPRIO estado anterior, com otimizador novo. Nada vem de outros clientes.
    """
    if not clients or len(clients) != len(weights) or len(personal) != len(clients) or local_epochs < 1:
        raise ValueError("rodada Local requer clientes, estados pessoais, pesos e épocas válidos")
    updated, err_p, observations = [], 0.0, 0
    for client, weight, candidates in zip(clients, weights, personal):
        if list(candidates) != [0.0]:
            raise ValueError("rodada Local só com lambda = 0")
        for _ in range(local_epochs):
            consume_like_epoch(client.loaders["train"])
        model = make_model(client)
        model.load_state_dict(candidates[0.0], strict=True)
        optimizer = make_optimizer(model)
        for _ in range(local_epochs):
            metric, personal_seen = train_personal(model, client, optimizer, None, 0.0, personal_seen)
            err_p += metric * weight
        updated.append({0.0: cpu_state(model)})
        del model, optimizer
        observations += weight * local_epochs
    return updated, err_p / observations, personal_seen


def train_epoch_personalized(model: nn.Module, anchors: list[torch.Tensor], lam: float, loader, optimizer, device, clip,
                             batches_seen: int = 0) -> tuple[float, int]:
    params = list(model.parameters())
    regularizer = (lambda _: 0.5 * lam * proximal_penalty(params, anchors)) if lam > 0 else None
    return train_local_epoch(model, loader, optimizer, masked_mae_objective(device, forward),
                             clip, batches_seen, regularizer)



@torch.no_grad()
def _evaluate_client(state: dict, client: ClientView, data: ExperimentData, cfg: dict, split: str, device,
                     total: RegressionAccumulator | None = None, nodes: NodeErrorAccumulator | None = None,
                     full: bool = False) -> float | dict:
    """MAE local de um estado no split do cliente (`full`: MAE, RMSE e alvos); acumula opcionalmente em `total`
    (micro) e `nodes` (por nó)."""
    model = make_client_model(cfg, client, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    local = RegressionAccumulator()
    for batch in client.loaders[split]:
        pred = (forward(model, batch, device) * data.target_std + data.target_mean).clamp_min(0.0)
        y, m = batch["y_raw"].to(device), batch["y_mask"].to(device)
        local.update(pred, y, m)
        if total is not None:
            total.update(pred, y, m)
        if nodes is not None:
            nodes.update(pred, y, m, client.node_indices)
    del model
    m = local.compute()
    return {"MAE": m["MAE"], "RMSE": m["RMSE"], "observations": m["observations"]} if full else m["MAE"]


@torch.no_grad()
def evaluate_personalized(personal: list[dict[float, dict]], clients: list[ClientView], data: ExperimentData, cfg: dict,
                          split: str, device, chosen: list[float] | None = None, per_node: bool = False,
                          per_client: bool = False) -> tuple[dict, list[float]]:
    """Avalia os modelos personalizados no split do SEU cliente (micro sobre todos os alvos + MAE macro).

    `personal[k]` mapeia lambda -> estado de v_k^lambda. Se `chosen` é dado, usa v_k^{chosen[k]}; senão
    (seleção do paper) cada cliente escolhe o lambda de menor MAE neste split — só faz sentido na validação.
    Devolve (métricas, lambda escolhido por cliente). `per_client` acrescenta MAE, RMSE e alvos de cada cliente.
    """
    total, client_maes, picked, rows = RegressionAccumulator(), [], [], []
    nodes = NodeErrorAccumulator(data.nodes) if per_node else None
    for k, (cands, client) in enumerate(zip(personal, clients)):
        if chosen is not None:
            lam = chosen[k]
        elif len(cands) == 1:
            lam = next(iter(cands))
        else:
            lam = min(cands, key=lambda l: _evaluate_client(cands[l], client, data, cfg, split, device))
        r = _evaluate_client(cands[lam], client, data, cfg, split, device, total, nodes, full=True)
        client_maes.append(r["MAE"])
        rows.append({"client": client.client_id, **r})
        picked.append(lam)
    metrics = total.compute()
    metrics.update(macro_client_MAE=float(np.mean(client_maes)), min_client_MAE=float(np.min(client_maes)),
                   max_client_MAE=float(np.max(client_maes)))
    if nodes is not None:
        metrics["per_node"] = nodes.compute()
    if per_client:
        metrics["per_client"] = rows
    return metrics, picked


def train_ditto(dataset: str, requested_clients: int, lam: float | str, seed: int, cfg: dict, device,
                max_rounds: int | None = None, output: str | Path | None = None, resume: bool = True, *, data=None, partition_path: str | Path | None = None) -> dict:
    dt = cfg["ditto"]
    win = cfg["windows"][dataset]
    lam = parse_lambda(lam)
    # conjunto candidato de lambdas: um só (fixo) ou os candidatos para seleção por cliente
    candidates = [float(l) for l in dt["lambda_candidates"]] if lam == PER_CLIENT else [lam]
    if lam == PER_CLIENT and (len(candidates) < 2 or any(l < 0 for l in candidates)):
        raise ValueError("ditto.lambda_candidates deve ter >= 2 valores >= 0")
    local_only, evaluate_test = bool(dt.get("local_only", False)), bool(dt.get("evaluate_test", True))
    if local_only and candidates != [0.0]:
        raise ValueError("ditto.local_only exige lambda = 0")
    device = torch.device(device)
    set_seed(seed)
    data = data if data is not None else load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
    pdir = Path(partition_path) if partition_path is not None else partition_dir(dataset, requested_clients, int(dt["partition_seed"]))
    num_clients = int(json.loads((pdir / "metadata.json").read_text())["num_clients"])
    out = ROOT / (output or cfg["output"]) / dataset / f"k{num_clients:03d}" / lambda_token(lam) / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
    clients = load_clients(data, pdir, batch_size, cfg["data"]["num_workers"], seed,
                           ("train", "validation", "test") if evaluate_test else ("train", "validation"))
    weights = [c.train_observations for c in clients]
    max_rounds = int(max_rounds or dt.get("max_rounds_per_dataset", {}).get(dataset, dt["max_rounds"]))
    local_epochs, clip, lr = int(dt["local_epochs"]), float(dt["gradient_clip_norm"]), float(dt["learning_rate"])
    tag = f"{dataset}/k{num_clients:03d}/{lambda_token(lam)}/seed_{seed:03d}"

    global_state = cpu_state(make_client_model(cfg, clients[0], device))  # w_0 comum (seed)
    init_sha = hashlib.sha256(b"".join(v.cpu().contiguous().numpy().tobytes() for _, v in sorted(global_state.items()))).hexdigest()
    # personal[k][lambda] = estado de v_k^lambda; v_k^0 = w_0 para todos os candidatos
    personal = [{l: {k: v.clone() for k, v in global_state.items()} for l in candidates} for _ in clients]
    state_bytes = sum(v.numel() * v.element_size() for v in global_state.values())
    history, best_mae, ref, stale = [], float("inf"), float("inf"), 0
    batches_seen, batches_seen_p = 0, 0  # contadores separados (scheduled sampling do DCRNN) para w e v_k
    ckpt, last = out / "best_model.pt", out / "last.pt"
    t0, start_round = time.time(), 1
    if resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        if bool(st.get("local_only", False)) != local_only:
            raise ValueError(f"last.pt gerado com local_only={st.get('local_only', False)}; a config pede {local_only}")
        global_state, personal, history, best_mae, ref, stale = (st["global_state"] if st["global_state"] is not None else global_state,
                                                                 st["personal"], st["history"],
                                                                 st["best_mae"], st["stopping_reference"], st["without_improvement"])
        batches_seen, batches_seen_p = st["batches_seen"], st["batches_seen_personalized"]
        start_round = st["round"] + 1
        t0 -= st["elapsed_s"]
        torch.set_rng_state(st["rng"]["torch"]); np.random.set_state(st["rng"]["numpy"])
        for c, g in zip(clients, st["rng"]["loaders"]):
            c.loaders["train"].generator.set_state(g)
        print(f"{tag} | retomando da rodada {st['round']} (best val_MAE={best_mae:.4f})", flush=True)
        if st.get("stopped_early"):
            start_round = max_rounds + 1

    for rnd in range(start_round, max_rounds + 1):
        if local_only:
            personal, train_mae_p, batches_seen_p = local_round(
                personal, clients, weights, lambda client: make_client_model(cfg, client, device),
                lambda model: torch.optim.Adam(model.parameters(), lr=lr),
                lambda model, client, optimizer, anchors, candidate, seen:
                    train_epoch_personalized(model, anchors, candidate, client.loaders["train"], optimizer, device, clip, seen),
                local_epochs, batches_seen_p)
            val, chosen = evaluate_personalized(personal, clients, data, cfg, "validation", device)
            history.append({"round": rnd, "train_scaled_MAE": train_mae_p, **{f"validation_{k}": v for k, v in val.items()},
                            "elapsed_s": time.time() - t0})
            print(f"{tag} | local | round {rnd:03d}/{max_rounds} | train={train_mae_p:.4f} | val_MAE={val['MAE']:.4f} "
                  f"| macro={val['macro_client_MAE']:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        else:
            global_state, personal, train_mae_g, train_mae_p, batches_seen, batches_seen_p = ditto_round(
                global_state, personal, clients, weights,
                lambda client: make_client_model(cfg, client, device),
                lambda model: torch.optim.Adam(model.parameters(), lr=lr),
                lambda model, client, optimizer, anchors, seen:
                    train_epoch(model, client.loaders["train"], optimizer, device, clip, seen),
                lambda model, client, optimizer, anchors, candidate, seen:
                    train_epoch_personalized(model, anchors, candidate, client.loaders["train"], optimizer, device, clip, seen),
                local_epochs, batches_seen, batches_seen_p)
            val, chosen = evaluate_personalized(personal, clients, data, cfg, "validation", device)
            val_g = evaluate_federated(global_state, clients, data, cfg, "validation", device)
            history.append({"round": rnd, "train_scaled_MAE": train_mae_p, "train_scaled_MAE_global": train_mae_g,
                            **{f"validation_{k}": v for k, v in val.items()},
                            **{f"validation_global_{k}": v for k, v in val_g.items()},
                            "client_lambda": " ".join(f"{l:g}" for l in chosen), "elapsed_s": time.time() - t0})
            print(f"{tag} | round {rnd:03d}/{max_rounds} | train={train_mae_p:.4f} | val_MAE={val['MAE']:.4f} "
                  f"| macro={val['macro_client_MAE']:.4f} | global val_MAE={val_g['MAE']:.4f}"
                  + (f" | lambdas={history[-1]['client_lambda']}" if lam == PER_CLIENT else "")
                  + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            _save_atomic({"personal": personal, "client_lambda": chosen, "global_state": None if local_only else global_state,
                          "best_round": rnd, "validation_MAE": best_mae}, ckpt)
        if val["MAE"] < ref - float(dt["early_stopping_min_delta"]):
            ref, stale = val["MAE"], 0
        else:
            stale += 1
        stopped_early = stale >= int(dt["patience"])
        _save_atomic({"round": rnd, "global_state": None if local_only else global_state, "personal": personal, "history": history,
                      "best_mae": best_mae, "stopping_reference": ref, "without_improvement": stale,
                      "batches_seen": batches_seen, "batches_seen_personalized": batches_seen_p,
                      "elapsed_s": time.time() - t0, "stopped_early": stopped_early, "local_only": local_only,
                      "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                              "loaders": [c.loaders["train"].generator.get_state() for c in clients]}}, last)
        if stopped_early:
            break

    best = torch.load(ckpt, map_location="cpu", weights_only=True)
    rounds = len(history)
    chosen = list(best["client_lambda"])
    result = {
        "dataset": dataset, "seed": seed, "method": "local" if local_only else "ditto_0hop", "ditto_lambda": lambda_label(lam),
        "local_only": local_only, "test_evaluated": evaluate_test, "init_state_sha256": init_sha,
        "lambda_candidates": candidates, "client_lambda": chosen,  # lambda usado por cliente (escolhido na validação)
        "model": cfg["model"].get("name", "leconv"),
        "requested_clients": requested_clients, "num_clients": num_clients, "partition_seed": int(dt["partition_seed"]),
        "partition": str(pdir.relative_to(ROOT)) if pdir.is_relative_to(ROOT) else str(pdir), "nodes": data.nodes,
        "client_nodes": [int(c.nodes) for c in clients], "client_train_observations": weights,
        "input_steps": data.input_steps, "horizon": data.horizon,
        "features": data.feature_names, "target": data.feature_names[data.target_feature_index], "graph": getattr(data, "graph", "default"),
        "local_epochs": local_epochs, "best_round": int(best["best_round"]), "rounds_completed": rounds,
        "parameter_count": int(sum(v.numel() for v in global_state.values())), "model_state_bytes": int(state_bytes),
        # só w circula; v_k fica no cliente. No Local nada circula durante o treino (só estatísticas de normalização,
        # comuns a todos os nós e calculadas no treino, antes dele).
        "model_communication_bytes": 0 if local_only else int(2 * state_bytes * num_clients * rounds),
        "training_seconds": time.time() - t0, "device": str(device), "gpu": gpu_name(device),
        # entregável do Ditto: modelos personalizados {v_k} com o lambda escolhido por cada cliente na validação
        "validation": evaluate_personalized(best["personal"], clients, data, cfg, "validation", device, chosen, per_node=True,
                                            per_client=local_only)[0],
    }
    if evaluate_test:
        result["test"] = evaluate_personalized(best["personal"], clients, data, cfg, "test", device, chosen, per_node=True,
                                               per_client=local_only)[0]
    if not local_only:
        # referência: modelo global w da mesma rodada. Usa a MESMA regra de atualização do FedAvg,
        # mas não reproduz uma execução de FedAvg com a mesma semente: os dois treinos locais (w e
        # v_k) consomem o mesmo gerador de embaralhamento do DataLoader, logo a ordem dos batches
        # difere da de um FedAvg isolado.
        result["validation_global"] = evaluate_federated(best["global_state"], clients, data, cfg, "validation", device, per_node=True)
        if evaluate_test:
            result["test_global"] = evaluate_federated(best["global_state"], clients, data, cfg, "test", device, per_node=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if not dt.get("keep_last_checkpoint", False):
        last.unlink(missing_ok=True)
    if evaluate_test and not local_only:
        print(f"{tag} | done | best_round={result['best_round']} | test_MAE={result['test']['MAE']:.4f} "
              f"| macro={result['test']['macro_client_MAE']:.4f} | global test_MAE={result['test_global']['MAE']:.4f} "
              f"| {result['training_seconds'] / 60:.1f} min", flush=True)
    else:
        print(f"{tag} | done | best_round={result['best_round']} | val_MAE={result['validation']['MAE']:.4f}"
              + (f" | test_MAE={result['test']['MAE']:.4f}" if evaluate_test else " | teste não avaliado")
              + f" | {result['training_seconds'] / 60:.1f} min", flush=True)
    return result


def evaluate_test_from_checkpoint(dataset: str, requested_clients: int, lam: float | str, seed: int, cfg: dict, device,
                                  output: str | Path | None = None, tolerance: float = 1e-5) -> dict:
    """Avalia o teste UMA vez no best_model.pt de uma execução treinada com `ditto.evaluate_test: false`.

    Recalcula a validação com os lambdas gravados e exige que reproduza a do metrics.json (tolerância relativa); só
    então avalia o teste e regrava o metrics.json com `test` e `test_evaluated: true`. Recusa execução já avaliada.
    """
    dt, win, lam = cfg["ditto"], cfg["windows"][dataset], parse_lambda(lam)
    device = torch.device(device)
    pdir = partition_dir(dataset, requested_clients, int(dt["partition_seed"]))
    num_clients = int(json.loads((pdir / "metadata.json").read_text())["num_clients"])
    out = ROOT / (output or cfg["output"]) / dataset / f"k{num_clients:03d}" / lambda_token(lam) / f"seed_{seed:03d}"
    m = json.loads((out / "metrics.json").read_text())
    if m.get("test_evaluated") or "test" in m:
        raise ValueError(f"{out}: teste já avaliado")
    set_seed(seed)
    data = load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
    batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
    clients = load_clients(data, pdir, batch_size, cfg["data"]["num_workers"], seed)
    if [int(c.nodes) for c in clients] != m["client_nodes"]:
        raise ValueError("partição difere da execução")
    best = torch.load(out / "best_model.pt", map_location="cpu", weights_only=True)
    if int(best["best_round"]) != m["best_round"] or list(best["client_lambda"]) != m["client_lambda"]:
        raise ValueError("checkpoint difere do metrics.json")
    per_client = bool(m.get("local_only", False))
    val = evaluate_personalized(best["personal"], clients, data, cfg, "validation", device, m["client_lambda"], per_node=True,
                                per_client=per_client)[0]
    keys = [k for k in ("MAE", "RMSE") if k in m["validation"]]
    for key in keys:
        if abs(val[key] - m["validation"][key]) > tolerance * abs(m["validation"][key]):
            raise ValueError(f"validação recalculada difere: {key} {val[key]} != {m['validation'][key]}")
    if val.get("observations") != m["validation"].get("observations"):
        raise ValueError("alvos de validação diferentes")
    m["test"] = evaluate_personalized(best["personal"], clients, data, cfg, "test", device, m["client_lambda"], per_node=True,
                                      per_client=per_client)[0]
    m.update(test_evaluated=True, test_evaluated_from="best_model.pt", validation_recomputed={k: val[k] for k in (*keys, "observations") if k in val},
             test_device=str(device), test_gpu=gpu_name(device))
    (out / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    print(f"{out.relative_to(ROOT)} | teste avaliado | best_round={m['best_round']} | test_MAE={m['test']['MAE']:.4f} "
          f"| test_RMSE={m['test'].get('RMSE', float('nan')):.4f}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(dl.DATASETS))
    p.add_argument("--clients", type=int, required=True)
    p.add_argument("--lam", type=parse_lambda, required=True,
                   help="lambda do termo proximal de v_k (0 = treino local) ou 'per_client' (seleção local entre ditto.lambda_candidates)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(ROOT / "config" / "ditto_leconv.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--evaluate-test-from-checkpoint", action="store_true",
                   help="só avalia o teste no best_model.pt de uma execução com ditto.evaluate_test: false")
    a = p.parse_args()
    if a.evaluate_test_from_checkpoint:
        evaluate_test_from_checkpoint(a.dataset, a.clients, a.lam, a.seed, load_config(a.config), a.device, a.output)
    else:
        train_ditto(a.dataset, a.clients, a.lam, a.seed, load_config(a.config), a.device, a.max_rounds, a.output, not a.no_resume)


if __name__ == "__main__":
    main()
