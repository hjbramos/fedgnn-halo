"""Treino centralizado de um (dataset, seed) — laço idêntico a fedstgnn.training.train_centralized.

    python -m experiments.train --dataset pems-bay --seed 42 [--config config/centralized.yaml] [--max-epochs 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from federation import train_local_epoch
from .task import forward, make_model, masked_mae_objective
from .data import ExperimentData, WindowDataset, load_experiment
from .metrics import NodeErrorAccumulator, RegressionAccumulator

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config" / "centralized.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def gpu_name(device) -> str | None:
    """Nome da GPU física da execução. O Ray pede 'uma GPU' e não fixa qual, e o scatter-add da LEConv
    não é determinístico em CUDA — sem isto não se distingue variação entre placas de outras causas."""
    d = torch.device(device)
    if d.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(d if d.index is not None else torch.cuda.current_device())


STRICT_CUBLAS = (":4096:8", ":16:8")       # exigido pelo cuBLAS para determinismo; definido ANTES do processo


def determinism_state() -> dict:
    """Opções de determinismo e precisão efetivas no processo (gravadas no metrics.json)."""
    m = torch.backends.cuda.matmul
    return {"use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cuda_matmul_allow_tf32": m.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "num_threads": torch.get_num_threads(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def enable_strict_determinism() -> dict:
    """Determinismo ESTRITO (opt-in, `training.strict_determinism: true`): operação sem versão determinística levanta
    ERRO. Exige CUBLAS_WORKSPACE_CONFIG antes do processo e CUDA ainda não inicializada; desliga TF32 e o benchmark."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in STRICT_CUBLAS:
        raise RuntimeError(f"determinismo estrito exige CUBLAS_WORKSPACE_CONFIG em {STRICT_CUBLAS} antes do processo")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA já inicializada antes de configurar o determinismo estrito")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    st = determinism_state()
    if not st["use_deterministic_algorithms"] or st["warn_only"]:
        raise RuntimeError("determinismo estrito não ficou ativo")
    return st


def state_sha256(model: nn.Module) -> str:
    """SHA-256 dos tensores do estado do modelo (nomes ordenados), independente de dispositivo."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate_detail(model, loader, data: ExperimentData, device) -> tuple[dict, dict]:
    """Validação detalhada (opt-in): métricas por horizonte e somas de erro absoluto/quadrático e contagens por
    (horizonte, nó), na escala original. Devolve (métricas por horizonte, arrays para .npz)."""
    model.eval()
    H, N = data.horizon, data.nodes
    accs = [RegressionAccumulator() for _ in range(H)]
    abs_e, sq_e, cnt = (np.zeros((H, N), dtype=np.float64) for _ in range(3))
    for batch in loader:
        prediction = (forward(model, batch, device) * data.target_std + data.target_mean).clamp_min(0.0)
        y, m = batch["y_raw"].to(device), batch["y_mask"].to(device)
        for h in range(H):
            accs[h].update(prediction[:, h], y[:, h], m[:, h])
        err = (prediction.double() - y.double()) * m          # mesma ordem de operações do RegressionAccumulator
        abs_e += err.abs().sum(dim=(0, 3)).cpu().numpy()
        sq_e += err.square().sum(dim=(0, 3)).cpu().numpy()
        cnt += m.sum(dim=(0, 3)).cpu().numpy()
    por_h = [{"horizonte": h + 1, **{k: v for k, v in accs[h].compute().items() if k in ("MAE", "RMSE", "observations")}}
             for h in range(H)]
    return {"por_horizonte": por_h}, {"abs_error": abs_e, "squared_error": sq_e, "count": cnt}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=int(num_workers), pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_model(model, loader, data: ExperimentData, device, per_node: bool = False) -> dict:
    """Métricas no loader; com per_node=True inclui "per_node" (erro absoluto e contagem por nó) para a análise estratificada."""
    model.eval()
    acc = RegressionAccumulator()
    nodes = NodeErrorAccumulator(data.nodes) if per_node else None
    for batch in loader:
        prediction = forward(model, batch, device)
        prediction = (prediction * data.target_std + data.target_mean).clamp_min(0.0)
        y, m = batch["y_raw"].to(device), batch["y_mask"].to(device)
        acc.update(prediction, y, m)
        if nodes is not None:
            nodes.update(prediction, y, m, np.arange(data.nodes))
    metrics = acc.compute()
    if nodes is not None:
        metrics["per_node"] = nodes.compute()
    return metrics


def train_epoch(model, loader, optimizer, device, clip, batches_seen=0, target_scale=None) -> tuple[float, int]:
    return train_local_epoch(model, loader, optimizer, masked_mae_objective(device, forward, target_scale), clip, batches_seen)


def train_centralized(dataset: str, seed: int, cfg: dict, device: str | torch.device, max_epochs: int | None = None,
                      output: str | Path | None = None, resume: bool = True, *, data=None, partition_path: str | Path | None = None) -> dict:
    """Treina, salva history.csv / metrics.json / best_model.pt em <output>/<dataset>/seed_XXX e devolve as métricas."""
    out = ROOT / (output or cfg["output"]) / dataset / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    win = cfg["windows"][dataset]
    tr = cfg["training"]
    max_epochs = int(max_epochs or tr.get("max_epochs_per_dataset", {}).get(dataset, tr["max_epochs"]))
    device = torch.device(device)
    # Sondagens que calibram o protocolo não podem ver o teste: com `evaluate_test: false` o split de
    # teste não ganha loader, não é avaliado e o metrics.json não traz a chave "test".
    evaluate_test = bool(tr.get("evaluate_test", True))
    # Com `keep_last_checkpoint: true` o last.pt sobrevive ao fim do treino, e aumentar `max_epochs`
    # retoma da época seguinte em vez de recomeçar (ver `centralized_can_extend` em run_grid).
    keep_last = bool(tr.get("keep_last_checkpoint", False))
    strict = bool(tr.get("strict_determinism", False))
    determinism = enable_strict_determinism() if strict else determinism_state()   # antes de qualquer uso de CUDA
    detail = bool(tr.get("validation_detail", False))
    instrument = bool(tr.get("instrumentation", False))   # só observa: ver experiments/instrumentacao.py

    set_seed(seed)
    data = data if data is not None else load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"),
                           win.get("origin_steps"))
    batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
    loaders = {
        s: make_loader(WindowDataset(data, s), batch_size, s == "train", seed, cfg["data"]["num_workers"])
        for s in (("train", "validation", "test") if evaluate_test else ("train", "validation"))
    }
    model = make_model(cfg, data, device)
    init_sha = state_sha256(model)              # mesma semente ⇒ mesma inicialização, qualquer que seja o grafo
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]), eps=float(tr.get("epsilon", 1e-8)))
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tr["lr_decay_epochs"], gamma=float(tr["lr_decay_ratio"]))
                 if tr.get("lr_decay_epochs") else None)

    # loss_scale: "standardized" (padrão de sempre) ou "original" (perda contra y_raw, como a DCRNN de referência)
    loss_scale = tr.get("loss_scale", "standardized")
    if loss_scale not in ("standardized", "original"):
        raise ValueError(f"loss_scale desconhecido: {loss_scale}")
    target_scale = (data.target_mean, data.target_std) if loss_scale == "original" else None
    train_key = "train_scaled_MAE" if target_scale is None else "train_MAE"

    tag = f"{dataset}/seed_{seed:03d}"
    history, best_mae, ref, stale, batches_seen = [], float("inf"), float("inf"), 0, 0
    ckpt = out / "best_model.pt"
    last = out / "last.pt"  # estado completo da última época, para retomada após cancelamento
    t0 = time.time()
    start_epoch, elapsed_before = 1, 0.0
    if resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        history, best_mae, ref, stale, batches_seen = (state["history"], state["best_mae"], state["stopping_reference"],
                                                      state["without_improvement"], state["batches_seen"])
        start_epoch, elapsed_before = state["epoch"] + 1, state["elapsed_s"]
        if loaders["train"].generator is not None:
            loaders["train"].generator.set_state(state["rng"]["loader"])
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        if torch.cuda.is_available() and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        t0 -= elapsed_before
        print(f"{tag} | retomando da época {state['epoch']} (best val_MAE={best_mae:.4f})", flush=True)
        if state.get("stopped_early"):
            start_epoch = max_epochs + 1  # já tinha parado por early stopping: só reavalia

    if instrument:
        from .instrumentacao import Instrumentacao
        inst = Instrumentacao(model, out, start_epoch)
    for epoch in range(start_epoch, max_epochs + 1):
        if instrument:
            train_mae, batches_seen = inst.epoca(epoch, model, loaders["train"], optimizer, device,
                                                 float(tr["gradient_clip_norm"]), batches_seen, target_scale)
        else:
            train_mae, batches_seen = train_epoch(model, loaders["train"], optimizer, device, float(tr["gradient_clip_norm"]),
                                                  batches_seen, target_scale)
        if scheduler is not None:
            scheduler.step()
        val = evaluate_model(model, loaders["validation"], data, device)
        history.append({"epoch": epoch, train_key: train_mae, **{f"validation_{k}": v for k, v in val.items()},
                        "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - t0,
                        **({"cuda_max_allocated_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30}
                           if device.type == "cuda" else {})})
        print(f"{tag} | epoch {epoch:03d}/{max_epochs} | train={train_mae:.4f} | val_MAE={val['MAE']:.4f} "
              f"| lr={optimizer.param_groups[0]['lr']:.1e} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "best_epoch": epoch,
                        "validation_MAE": best_mae}, ckpt)
        if val["MAE"] < ref - float(tr["early_stopping_min_delta"]):
            ref, stale = val["MAE"], 0
        else:
            stale += 1
        stopped_early = stale >= int(tr["patience"])
        _save_atomic({
            "epoch": epoch, "history": history, "best_mae": best_mae, "stopping_reference": ref,
            "without_improvement": stale, "batches_seen": batches_seen, "elapsed_s": time.time() - t0,
            "stopped_early": stopped_early,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                    "loader": loaders["train"].generator.get_state() if loaders["train"].generator is not None else None,
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }, last)
        if stopped_early:
            break

    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = {
        "dataset": dataset, "seed": seed, "method": "centralized", "model": cfg["model"].get("name", "leconv"),
        "batch_size": int(batch_size),
        "nodes": data.nodes, "edges": int(data.edge_index.shape[1]),
        # arestas efetivas do modelo (difere de `edges` nos controles self_loops_only / cut_partition)
        "model_edges": int(model.edge_index.shape[1]) if hasattr(model, "edge_index") else None,
        "model_options": dict(cfg["model"]), "loss_scale": loss_scale,
        "input_steps": data.input_steps, "horizon": data.horizon,
        "features": data.feature_names, "target": data.feature_names[data.target_feature_index], "graph": data.graph,
        "samples": {k: int(len(v)) for k, v in data.origins.items()},
        "origin_steps": data.origin_steps or data.input_steps,
        "origins_sha256": {k: hashlib.sha256(np.ascontiguousarray(v, dtype=np.int64).tobytes()).hexdigest()
                           for k, v in data.origins.items()},
        "scaler": {"mean": data.scaler_mean.tolist(), "std": data.scaler_std.tolist()},
        "best_epoch": int(checkpoint["best_epoch"]), "epochs_completed": len(history),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": time.time() - t0, "device": str(device), "gpu": gpu_name(device),
        "test_evaluated": evaluate_test, "instrumentation": instrument,
        "validation": evaluate_model(model, loaders["validation"], data, device, per_node=True),
        "init_state_sha256": init_sha, "adj_sha256": hashlib.sha256(np.ascontiguousarray(data.adj, dtype=np.float32).tobytes()).hexdigest(),
        "determinism": {**determinism, "strict": strict, "after_run": determinism_state()},
    }
    if detail:
        por_h, arrays = evaluate_detail(model, loaders["validation"], data, device)
        result["validation_detail"] = por_h
        np.savez_compressed(out / "validation_detail.npz", **arrays)
        result["validation_detail_sha256"] = hashlib.sha256((out / "validation_detail.npz").read_bytes()).hexdigest()
    if evaluate_test:
        result["test"] = evaluate_model(model, loaders["test"], data, device, per_node=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if not keep_last:
        last.unlink(missing_ok=True)
    scores = (f"test_MAE={result['test']['MAE']:.4f} | test_RMSE={result['test']['RMSE']:.4f}" if evaluate_test
              else f"val_MAE={result['validation']['MAE']:.4f} | teste não avaliado")
    print(f"{tag} | done | best_epoch={result['best_epoch']} | {scores} | {result['training_seconds']/60:.1f} min", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["metr-la", "pems-bay", "sunt"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(ROOT / "config" / "centralized.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--output", default=None, help="diretório alternativo (ex.: /tmp/smoke) para não poluir results/")
    p.add_argument("--no-resume", action="store_true", help="ignora last.pt e treina do zero")
    a = p.parse_args()
    train_centralized(a.dataset, a.seed, load_config(a.config), a.device, a.max_epochs, a.output, resume=not a.no_resume)


if __name__ == "__main__":
    main()
