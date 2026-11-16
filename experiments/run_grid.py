"""Executa a grade datasets × seeds com Ray (1 GPU por run) e consolida média ± desvio.

    python -m experiments.run_grid                       # tudo do config
    python -m experiments.run_grid --datasets sunt --seeds 7 42
    python -m experiments.run_grid --backend local       # sequencial, sem Ray
    python -m experiments.run_grid --summarize-only      # só reagrega os metrics.json existentes
    python -m experiments.run_grid --config config/fedavg_leconv.yaml [--clients 5 10]   # federado (method: fedavg)
    python -m experiments.run_grid --config config/fedprox_leconv.yaml [--mu 0.01]       # FedProx (method: fedprox)
    python -m experiments.run_grid --config config/ditto_leconv.yaml [--lam 0.1]         # Ditto (method: ditto)
    python -m experiments.run_grid --config config/fedpub_leconv.yaml [--l1 0.00001]     # FED-PUB (method: fedpub)
    python -m experiments.run_grid --config config/fedgnnhet_leconv.yaml [--halo 1hop]   # FedGNNHet (method: fedgnnhet)

Runs cancelados retomam de results/.../seed_XXX/last.pt (estado completo da última época);
runs concluídos (metrics.json) são pulados. --force apaga o last.pt e treina do zero.
"""
from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path

import pandas as pd

from .train import ROOT, load_config, train_centralized

METRICS = ["MAE", "RMSE", "MAPE_percent", "WAPE_percent", "R2"]


# método -> (seção do YAML, nome do hiperparâmetro no YAML, prefixo do diretório de saída, chave no metrics.json)
FEDERATED = {"fedavg": ("federated", None, None, None),
             "fedprox": ("fedprox", "proximal_mu", "mu", "proximal_mu"),
             "ditto": ("ditto", "lambda", "lam", "ditto_lambda"),
             "fedpub": ("fedpub", "lambda_l1", "l1_", "lambda_l1"),
             "fedgnnhet": ("fedgnnhet", "halo", "halo_", "halo")}
# colunas extra do sumário por método: nome -> (chave de topo no metrics.json, subchave ou None)
EXTRA = {"ditto": {"global_MAE": ("test_global", "MAE"), "global_macro_client_MAE": ("test_global", "macro_client_MAE")},
         "fedpub": {"mask_sparsity": ("mask_sparsity", None), "mean_self_alpha": ("alpha", "diag")},
         "fedgnnhet": {"halo_nodes": ("halo_nodes_total", None), "halo_MB": ("halo_bytes", "MB")}}


def _method(cfg: dict) -> str:
    return cfg.get("method", "centralized")


def _label(spec: tuple, cfg: dict) -> str:
    dataset, seed, k, hp = spec
    if k is None:
        return f"{dataset}/seed_{seed:03d}"
    if hp is None:
        return f"{dataset}/k{k}/seed_{seed:03d}"
    return f"{dataset}/k{k}/{FEDERATED[_method(cfg)][2]}{hp if isinstance(hp, str) else f'{hp:g}'}/seed_{seed:03d}"


def can_extend(run_dir: Path, target_epochs: int, counter: str = "epoch") -> bool:
    """Execução concluída que ainda pode seguir: guardou o last.pt, não parou por early stopping e
    completou menos épocas (ou rodadas, `counter="round"`, no FedAvg) que o orçamento pedido. Nesse caso
    não conta como cache."""
    last, metrics = run_dir / "last.pt", run_dir / "metrics.json"
    if not (last.exists() and metrics.exists()):
        return False
    import torch
    state = torch.load(last, map_location="cpu", weights_only=False)
    return not state.get("stopped_early") and int(state[counter]) < int(target_epochs)


def execute(spec: tuple, cfg: dict, force: bool, max_epochs: int | None) -> dict:
    """spec = (dataset, seed, k, hp) — k = None no centralizado; hp = mu (FedProx) ou lambda (Ditto)."""
    dataset, seed, k, hp = spec
    try:
        import torch
        # Limita o pool de threads às CPUs reservadas por execução.
        torch.set_num_threads(int(cfg.get("execution", {}).get("cpus_per_run", 4)))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if k is None:
            run_dir = ROOT / cfg["output"] / dataset / f"seed_{seed:03d}"
        else:
            from .federated import partition_dir
            method = _method(cfg)
            section = cfg[FEDERATED[method][0]]
            num_clients = json.loads((partition_dir(dataset, k, int(section["partition_seed"])) / "metadata.json").read_text())["num_clients"]
            run_dir = ROOT / cfg["output"] / dataset / f"k{num_clients:03d}"
            if method == "fedprox":
                from .fedprox import mu_token
                run_dir = run_dir / mu_token(hp)
            elif method == "ditto":
                from .ditto import lambda_token
                run_dir = run_dir / lambda_token(hp)
            elif method == "fedpub":
                from .fedpub import run_subdir
                run_dir = run_dir / run_subdir(hp, cfg["fedpub"].get("similarity_source", "representation"))
            elif method == "fedgnnhet":
                from .fedgnnhet import halo_token
                run_dir = run_dir / halo_token(hp)
            run_dir = run_dir / f"seed_{seed:03d}"
        extend = False
        if (run_dir / "metrics.json").exists() and not force:
            if k is None:
                tr = cfg["training"]
                extend = can_extend(run_dir, int(max_epochs or tr.get("max_epochs_per_dataset", {}).get(dataset, tr["max_epochs"])))
            elif method == "fedavg":
                fed = cfg["federated"]
                extend = can_extend(run_dir, int(max_epochs or fed.get("max_rounds_per_dataset", {}).get(dataset, fed["max_rounds"])),
                                                counter="round")
        if (run_dir / "metrics.json").exists() and not force and not extend:
            if k is not None and method == "fedprox":
                from .fedprox import validate_mu
                cached = json.loads((run_dir / "metrics.json").read_text())
                if "proximal_mu" not in cached or validate_mu(cached["proximal_mu"]) != validate_mu(hp):
                    raise ValueError("cache FedProx sem mu verificável ou com mu diferente; use outro output ou --force")
            if k is not None and method == "fedpub":
                from .fedpub import check_similarity_source
                cached = json.loads((run_dir / "metrics.json").read_text())
                check_similarity_source(cached.get("similarity_source"), cfg["fedpub"].get("similarity_source", "representation"),
                                        "cache FED-PUB")
            return {"spec": spec, "status": "cached"}
        if force:
            (run_dir / "last.pt").unlink(missing_ok=True)
        if k is None:
            train_centralized(dataset, seed, cfg, device, max_epochs)
        elif method == "fedavg":
            from .federated import train_fedavg
            train_fedavg(dataset, k, seed, cfg, device, max_epochs)
        elif method == "fedprox":
            from .fedprox import train_fedprox
            train_fedprox(dataset, k, hp, seed, cfg, device, max_epochs)
        elif method == "ditto":
            from .ditto import train_ditto
            train_ditto(dataset, k, hp, seed, cfg, device, max_epochs)
        elif method == "fedpub":
            from .fedpub import train_fedpub
            train_fedpub(dataset, k, hp, seed, cfg, device, max_epochs)
        elif method == "fedgnnhet":
            from .fedgnnhet import train_fedgnnhet
            train_fedgnnhet(dataset, k, hp, seed, cfg, device, max_epochs)
        else:
            raise ValueError(f"método desconhecido: {method}")
        return {"spec": spec, "status": "ok"}
    except Exception:
        return {"spec": spec, "status": "failed", "error": traceback.format_exc()}


def run_ray(specs, cfg, force, max_epochs):
    import os

    import ray

    ex = cfg["execution"]
    if ex.get("gpus"):  # restringe o Ray a estas GPUs (índices do nvidia-smi); ex.: [0] = só a L40S
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in ex["gpus"])
    if "worker_niceness" in ex:  # o Ray roda os workers com nice 15; 0 = prioridade igual à dos outros usuários
        os.environ["RAY_worker_niceness"] = str(int(ex["worker_niceness"]))
    tmp = (ROOT / ex.get("ray_temp_dir", ".ray")).resolve()
    if len(str(tmp)) > 50:  # sockets AF_UNIX do Ray têm limite de ~103 bytes no caminho
        tmp = Path("/tmp/ray-fedstgnn")
    (tmp / "spill").mkdir(parents=True, exist_ok=True)
    print(f"Ray temp dir: {tmp} ({shutil.disk_usage(tmp).free / 1024**3:.1f} GiB free)", flush=True)
    node_ip = {"_node_ip_address": ex["ray_node_ip"]} if ex.get("ray_node_ip") else {}  # 127.0.0.1: imune a quedas de rede
    ray.init(_temp_dir=str(tmp), object_spilling_directory=str(tmp / "spill"), include_dashboard=False,
             runtime_env={"env_vars": {"PYTHONPATH": str(ROOT)}}, **node_ip)  # workers importam do projeto real, não de uma cópia
    if ex.get("gpus"):
        print(f"GPUs visíveis ao Ray: {ex['gpus']}", flush=True)
    print(f"Ray resources: {ray.cluster_resources()}", flush=True)
    gpus = float(ex["gpus_per_run"]) if ray.cluster_resources().get("GPU", 0) > 0 else 0.0
    remote = ray.remote(num_cpus=int(ex["cpus_per_run"]), num_gpus=gpus, max_retries=1)(execute)

    pending, running, results = list(specs), [], []
    while pending or running:
        while pending and len(running) < int(ex["max_parallel"]):
            running.append(remote.remote(pending.pop(0), cfg, force, max_epochs))
        done, running = ray.wait(running, num_returns=1)
        r = ray.get(done[0])
        results.append(r)
        print(f"[{len(results)}/{len(specs)}] {_label(r['spec'], cfg)}: {r['status']}", flush=True)
        if r["status"] == "failed":
            print(r["error"], flush=True)
    ray.shutdown()
    return results


def summarize(cfg: dict) -> pd.DataFrame:
    """Lê todos os metrics.json e escreve summary_runs.csv, summary.csv e summary.md (média ± desvio)."""
    base = ROOT / cfg["output"]
    method = _method(cfg)
    federated = method in FEDERATED
    hp_name, hp_prefix, hp_key = FEDERATED[method][1:] if federated else (None, None, None)
    hp_col = {"fedprox": "mu", "ditto": "lambda", "fedpub": "lambda_l1", "fedgnnhet": "halo"}.get(method)
    hp_value = (lambda v: v) if method != "ditto" else (lambda v: str(v))  # ditto mistura floats e 'per_client'
    metrics = METRICS + (["macro_client_MAE"] if federated else [])
    extra = list(EXTRA.get(method, {}))

    def _extra(m: dict, name: str):
        key, sub = EXTRA[method][name]
        if key not in m:  # Local (ditto.local_only) não tem modelo global
            return float("nan")
        v = m[key]
        if sub == "diag":
            return float(sum(v[i][i] for i in range(len(v))) / len(v))
        if sub == "MB":
            return v / 1e6
        return v if sub is None else v[sub]
    pattern = "*/seed_*/metrics.json" if not federated else ("*/k*/seed_*/metrics.json" if hp_prefix is None
                                                             else f"*/k*/{hp_prefix}*/seed_*/metrics.json")
    if method == "fedpub":  # fontes de embedding fora do default ficam em l1_<…>/sim_<fonte>/
        pattern = f"*/k*/{hp_prefix}*/**/seed_*/metrics.json"
    rows = []
    for f in sorted(base.glob(pattern)):
        m = json.loads(f.read_text())
        rows.append({"dataset": m["dataset"], **({"k": m["num_clients"]} if federated else {}),
                     **({hp_col: hp_value(m[hp_key])} if hp_col else {}),
                     **({"similarity_source": m["similarity_source"]} if method == "fedpub" else {}), "seed": m["seed"],
                     "best_unit": m["best_round"] if federated else m["best_epoch"],
                     "units": m["rounds_completed"] if federated else m["epochs_completed"],
                     "minutes": m["training_seconds"] / 60,
                     **({"comm_MB": m["model_communication_bytes"] / 1e6} if federated else {}),
                     **{f"val_{k}": m["validation"][k] for k in metrics},
                     # sondagens com `evaluate_test: false` não gravam "test": colunas ficam NaN
                     **{f"test_{k}": m["test"][k] if "test" in m else float("nan") for k in metrics},
                     **{f"test_{e}": _extra(m, e) if "test" in m else float("nan") for e in extra}})
    if not rows:
        print("nenhum metrics.json encontrado")
        return pd.DataFrame()
    keys = ["dataset"] + (["k"] if federated else []) + ([hp_col] if hp_col else []) + (["similarity_source"] if method == "fedpub" else [])
    runs = pd.DataFrame(rows).sort_values(keys + ["seed"])
    runs.to_csv(base / "summary_runs.csv", index=False)

    cols = [c for c in runs.columns if c.startswith(("val_", "test_"))] + ["best_unit", "minutes"] + (["comm_MB"] if federated else [])
    agg = runs.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "seeds", runs.groupby(keys)["seed"].count())
    agg.to_csv(base / "summary.csv")

    unit = "best round" if federated else "best epoch"
    title = {"centralized": "Centralizado", "fedavg": "FedAvg 0-hop", "fedprox": "FedProx 0-hop",
             "ditto": "Ditto 0-hop (modelos personalizados; `global` = w de referência)",
             "fedpub": "FED-PUB 0-hop (modelos personalizados θ_i ⊙ μ_i)",
             "fedgnnhet": "FedGNNHet (FedAvg + halo de fronteira)"}[method] + f" {cfg['model'].get('name', 'leconv').upper()}"
    head = "| " + " | ".join(keys) + " | seeds | "
    lines = [f"# {title} — média ± desvio padrão entre seeds", "",
             head + " | ".join(f"test {k}" for k in metrics + extra) + f" | {unit} | min/run |" + (" comm MB |" if federated else ""),
             "|" + "---|" * (len(keys) + 1 + len(metrics) + len(extra) + 2 + (1 if federated else 0))]
    for idx, r in agg.iterrows():
        label = " | ".join(str(v) for v in (idx if isinstance(idx, tuple) else (idx,)))
        cells = [f"{r[f'test_{k}_mean']:.4f} ± {r[f'test_{k}_std']:.4f}" for k in metrics + extra]
        lines.append(f"| {label} | {int(r['seeds'])} | " + " | ".join(cells) +
                     f" | {r['best_unit_mean']:.1f} ± {r['best_unit_std']:.1f} | {r['minutes_mean']:.1f} |" +
                     (f" {r['comm_MB_mean']:.1f} |" if federated else ""))
    lines += ["", "| " + " | ".join(keys) + " | " + " | ".join(f"val {k}" for k in metrics) + " |", "|" + "---|" * (len(keys) + len(metrics))]
    for idx, r in agg.iterrows():
        label = " | ".join(str(v) for v in (idx if isinstance(idx, tuple) else (idx,)))
        lines.append(f"| {label} | " + " | ".join(f"{r[f'val_{k}_mean']:.4f} ± {r[f'val_{k}_std']:.4f}" for k in metrics) + " |")
    (base / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config" / "centralized.yaml"))
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--backend", choices=("ray", "local"), default="ray")
    p.add_argument("--force", action="store_true", help="re-treina mesmo se metrics.json existir")
    p.add_argument("--clients", nargs="*", type=int, help="federado: k pedidos (default: do YAML)")
    p.add_argument("--mu", nargs="*", type=float, help="fedprox: valores de mu (default: fedprox.proximal_mu)")
    p.add_argument("--lam", nargs="*", help="ditto: valores de lambda ou 'per_client' (default: ditto.lambda)")
    p.add_argument("--l1", nargs="*", type=float, help="fedpub: valores de lambda_l1 (default: fedpub.lambda_l1)")
    p.add_argument("--halo", nargs="*", help="fedgnnhet: variantes de halo (default: fedgnnhet.halo)")
    p.add_argument("--max-epochs", type=int, default=None, help="override de épocas (ou rodadas) para smoke test")
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.summarize_only:
        summarize(cfg)
        return
    method = _method(cfg)
    section, hp_name = FEDERATED[method][:2] if method in FEDERATED else (None, None)
    ks = (a.clients or cfg[section]["num_clients"]) if section else [None]
    ks_per_dataset = cfg[section].get("num_clients_per_dataset", {}) if (section and not a.clients) else {}
    cli_hp = {"fedprox": a.mu, "ditto": a.lam, "fedpub": a.l1, "fedgnnhet": a.halo}.get(method)
    hps = (cli_hp or cfg[section][hp_name]) if hp_name else [None]
    if method == "ditto":
        from .ditto import parse_lambda
        hps = [parse_lambda(h) for h in hps]
    specs = [(d, s, k, hp) for d in (a.datasets or cfg["datasets"]) for k in ks_per_dataset.get(d, ks) for hp in hps
             for s in (a.seeds or cfg["seeds"])]
    print(f"{len(specs)} runs: {[_label(x, cfg) for x in specs]}", flush=True)
    if a.backend == "ray":
        results = run_ray(specs, cfg, a.force, a.max_epochs)
    else:
        results = [execute(x, cfg, a.force, a.max_epochs) for x in specs]
        for r in results:
            print(f"{_label(r['spec'], cfg)}: {r['status']}", flush=True)
            if r["status"] == "failed":
                print(r["error"])
    summarize(cfg)


if __name__ == "__main__":
    main()
