"""Avaliação final de teste sobre checkpoints de uma sondagem cega, depois do protocolo congelado.

Uso previsto: o Centralizado do SUNT (alvo `n_boardings`). A sondagem `config/sunt_probe_boardings.yaml`
treinou 5 sementes × 600 épocas com `evaluate_test: false`, e o protocolo congelado (600 épocas fixas,
sem parada antecipada, checkpoint pelo menor MAE de validação) é exatamente o que ela executou. Em vez
de retreinar, este módulo avalia o teste nos `best_model.pt` dela.

Para cada semente: carrega o `best_model.pt`, confere que a época e o MAE de validação do checkpoint
batem com o metrics.json da sondagem, **recalcula** a validação e exige que reproduza o valor gravado,
avalia o teste e escreve `<saída>/<dataset>/seed_XXX/metrics.json` (com `test`, `test_evaluated: true`
e a origem em `evaluated_from`) e uma cópia do history.csv. A sondagem não é alterada.

    python -m experiments.evaluate_frozen --probe results/sunt_probe_boardings --config config/sunt_centralized_boardings.yaml

FedAvg (`method: fedavg` na config): o orçamento é por célula, então a config final traz
`federated.rounds_by_clients: {k_obtido: rodadas}` e só as células listadas são avaliadas. Além das
conferências acima, confere a partição (caminho e número de clientes, recalculados pela config) e o
número de rodadas da célula. Saída em `<saída>/<dataset>/kXXX/seed_XXX/`.

    python -m experiments.evaluate_frozen --probe results/sunt_fedavg_probe --config config/sunt_fedavg_final.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .data import WindowDataset, load_experiment
from .train import ROOT, evaluate_model, load_config, make_loader, make_model

TOLERANCE = 1e-4


def evaluate_frozen(probe: str | Path, cfg: dict, device: str | torch.device = "cpu") -> list[dict]:
    probe, out_root = ROOT / probe, ROOT / cfg["output"]
    tr = cfg["training"]
    if not tr.get("evaluate_test", True):
        raise ValueError("a config final precisa de evaluate_test: true")
    results = []
    for metrics_file in sorted(probe.glob("*/seed_*/metrics.json")):
        run_dir = metrics_file.parent
        m = json.loads(metrics_file.read_text())
        if m.get("test_evaluated", True) or "test" in m:
            raise ValueError(f"{run_dir}: a sondagem já tem teste avaliado; não é uma sondagem cega")
        if m["epochs_completed"] != int(tr["max_epochs"]):
            raise ValueError(f"{run_dir}: {m['epochs_completed']} épocas, protocolo congelado pede {tr['max_epochs']}")
        dataset, seed = m["dataset"], m["seed"]
        win = cfg["windows"][dataset]
        if m["target"] != win.get("target", m["target"]):
            raise ValueError(f"{run_dir}: alvo {m['target']} difere do protocolo ({win.get('target')})")
        best = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        if int(best["best_epoch"]) != int(m["best_epoch"]) or abs(float(best["validation_MAE"]) - m["validation"]["MAE"]) > TOLERANCE:
            raise ValueError(f"{run_dir}: best_model.pt não corresponde ao metrics.json")
        data = load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
        batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
        model = make_model(cfg, data, torch.device(device))
        model.load_state_dict(best["model_state_dict"])
        loader = lambda split: make_loader(WindowDataset(data, split), batch_size, False, seed, cfg["data"]["num_workers"])
        validation = evaluate_model(model, loader("validation"), data, device, per_node=True)
        if abs(validation["MAE"] - m["validation"]["MAE"]) > TOLERANCE:
            raise ValueError(f"{run_dir}: validação recalculada {validation['MAE']:.6f} ≠ gravada {m['validation']['MAE']:.6f}")
        final = dict(m, validation=validation, test=evaluate_model(model, loader("test"), data, device, per_node=True),
                     test_evaluated=True, evaluated_from=str(run_dir.relative_to(ROOT)))
        out = out_root / dataset / f"seed_{seed:03d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
        shutil.copyfile(run_dir / "history.csv", out / "history.csv")
        print(f"{dataset}/seed_{seed:03d} | best_epoch={final['best_epoch']} | val_MAE={validation['MAE']:.4f} "
              f"| test_MAE={final['test']['MAE']:.4f}", flush=True)
        results.append(final)
    if not results:
        raise FileNotFoundError(f"nenhum metrics.json em {probe}")
    return results


def evaluate_frozen_fedavg(probe: str | Path, cfg: dict, device: str | torch.device = "cpu") -> list[dict]:
    from .federated import evaluate_federated, load_clients, partition_dir
    probe, out_root = ROOT / probe, ROOT / cfg["output"]
    fed = cfg["federated"]
    if not fed.get("evaluate_test", True):
        raise ValueError("a config final precisa de evaluate_test: true")
    budgets = {int(k): int(v) for k, v in fed["rounds_by_clients"].items()}
    results = []
    for metrics_file in sorted(probe.glob("*/k*/seed_*/metrics.json")):
        run_dir = metrics_file.parent
        m = json.loads(metrics_file.read_text())
        k = int(m["num_clients"])
        if k not in budgets:
            continue  # célula fora do escopo desta avaliação (ex.: orçamento ainda não fixado)
        if m.get("test_evaluated", True) or "test" in m:
            raise ValueError(f"{run_dir}: a sondagem já tem teste avaliado; não é uma sondagem cega")
        if m["rounds_completed"] != budgets[k]:
            raise ValueError(f"{run_dir}: {m['rounds_completed']} rodadas, orçamento congelado da célula k={k} é {budgets[k]}")
        dataset, seed = m["dataset"], m["seed"]
        win = cfg["windows"][dataset]
        if m["target"] != win.get("target", m["target"]):
            raise ValueError(f"{run_dir}: alvo {m['target']} difere do protocolo ({win.get('target')})")
        pdir = partition_dir(dataset, int(m["requested_clients"]), int(fed["partition_seed"]))
        if str(pdir.relative_to(ROOT)) != m["partition"] or json.loads((pdir / "metadata.json").read_text())["num_clients"] != k:
            raise ValueError(f"{run_dir}: partição {m['partition']} não corresponde à config ({pdir})")
        best = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        if int(best["best_round"]) != int(m["best_round"]) or abs(float(best["validation_MAE"]) - m["validation"]["MAE"]) > TOLERANCE:
            raise ValueError(f"{run_dir}: best_model.pt não corresponde ao metrics.json")
        data = load_experiment(dataset, win["input_steps"], win["horizon"], win.get("features"), win.get("target"), win.get("graph"))
        batch_size = cfg["data"].get("batch_size_per_dataset", {}).get(dataset, cfg["data"]["batch_size"])
        clients = load_clients(data, pdir, batch_size, cfg["data"]["num_workers"], seed)
        if [int(c.nodes) for c in clients] != m["client_nodes"]:
            raise ValueError(f"{run_dir}: tamanhos dos clientes diferem do registro")
        validation = evaluate_federated(best["model_state_dict"], clients, data, cfg, "validation", device, per_node=True)
        if abs(validation["MAE"] - m["validation"]["MAE"]) > TOLERANCE:
            raise ValueError(f"{run_dir}: validação recalculada {validation['MAE']:.6f} ≠ gravada {m['validation']['MAE']:.6f}")
        final = dict(m, validation=validation,
                     test=evaluate_federated(best["model_state_dict"], clients, data, cfg, "test", device, per_node=True),
                     test_evaluated=True, evaluated_from=str(run_dir.relative_to(ROOT)))
        out = out_root / dataset / f"k{k:03d}" / f"seed_{seed:03d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
        shutil.copyfile(run_dir / "history.csv", out / "history.csv")
        print(f"{dataset}/k{k:03d}/seed_{seed:03d} | best_round={final['best_round']} | val_MAE={validation['MAE']:.4f} "
              f"| test_MAE={final['test']['MAE']:.4f}", flush=True)
        results.append(final)
    if not results:
        raise FileNotFoundError(f"nenhuma execução das células {sorted(budgets)} em {probe}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True, help="diretório da sondagem cega (ex.: results/sunt_probe_boardings)")
    p.add_argument("--config", required=True, help="config do protocolo congelado, com evaluate_test: true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = load_config(a.config)
    (evaluate_frozen_fedavg if cfg.get("method") == "fedavg" else evaluate_frozen)(a.probe, cfg, a.device)


if __name__ == "__main__":
    main()
