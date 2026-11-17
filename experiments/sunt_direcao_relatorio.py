"""Relatório do piloto de orientação do grafo do SUNT (só VALIDAÇÃO; o teste não é usado).

Lê, para cada braço (simetrico, observado, invertido) e semente, metrics.json, history.csv e validation_detail.npz,
e a auditoria das topologias (estratos fixados só pela topologia). Reporta MAE/RMSE de validação na escala
original por semente, agregados, por horizonte e por estrato; diferenças pareadas observado − simétrico,
invertido − simétrico e observado − invertido (média, dp e os cinco valores); melhor época e tempo de treino.
Confere: mesma inicialização entre braços por semente, adjacência igual à auditada, determinismo estrito, teste
não avaliado e 600 épocas. Descritivo (piloto exploratório): nenhum teste de hipótese.
Uso: python -m experiments.sunt_direcao_relatorio --base results/sunt_piloto_direcao
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

BRACOS = ("simetrico", "observado", "invertido")
CONTRASTES = (("observado", "simetrico"), ("invertido", "simetrico"), ("observado", "invertido"))
ESTRATOS = ("sem_entrada", "grande_reducao", "demais")


def carregar(base: Path, epocas: int | None = 600):
    aud = json.loads((base / "auditoria/auditoria_topologias.json").read_text())
    with open(base / "auditoria/estratos_por_no.csv") as f:
        est = list(csv.DictReader(f))
    runs, probs = {}, []
    for b in BRACOS:
        for d in sorted((base / b / "sunt").glob("seed_*")):
            m = json.loads((d / "metrics.json").read_text())
            s = int(m["seed"])
            z = np.load(d / "validation_detail.npz")
            runs[(b, s)] = {"m": m, "z": {k: z[k] for k in z.files}, "hist": list(csv.DictReader(open(d / "history.csv")))}
            if m["test_evaluated"] is not False or "test" in m:
                probs.append(f"{b}/{s}: teste avaliado")
            if not m["determinism"]["strict"] or not m["determinism"]["after_run"]["use_deterministic_algorithms"]:
                probs.append(f"{b}/{s}: sem determinismo estrito")
            if m["adj_sha256"] != aud["adj_sha256"][b]:
                probs.append(f"{b}/{s}: adjacência diferente da auditada")
            if epocas is not None and m["epochs_completed"] != epocas:
                probs.append(f"{b}/{s}: {m['epochs_completed']} épocas ≠ {epocas}")
    seeds = sorted({s for _, s in runs})
    for s in seeds:
        if any((b, s) not in runs for b in BRACOS):
            probs.append(f"semente {s}: braço ausente")
        elif len({runs[(b, s)]["m"]["init_state_sha256"] for b in BRACOS}) != 1:
            probs.append(f"semente {s}: inicialização diferente entre braços")
    return aud, est, runs, seeds, probs


def mae_rmse(z, nos=None):
    a, q, c = z["abs_error"], z["squared_error"], z["count"]
    if nos is not None:
        a, q, c = a[:, nos], q[:, nos], c[:, nos]
    n = c.sum()
    return (a.sum() / n, math.sqrt(q.sum() / n)) if n else (float("nan"), float("nan"))


def resumo_par(v):
    v = [float(x) for x in v]
    return {"valores": v, "media": statistics.mean(v), "dp": statistics.stdev(v) if len(v) > 1 else float("nan"),
            "negativos": int(sum(x < 0 for x in v)), "n": len(v)}


def relatorio(base: Path, epocas: int | None = 600) -> dict:
    aud, est, runs, seeds, probs = carregar(base, epocas)
    H = next(iter(runs.values()))["z"]["count"].shape[0]
    nos_est = {b: {e: [int(r["indice"]) for r in est if r[f"estrato_{b}"] == e] for e in ESTRATOS} for b in ("observado", "invertido")}
    out = {"problemas": probs, "sementes": seeds, "por_execucao": [], "contrastes": {}, "por_horizonte": {}, "estratos": {}}
    for (b, s), r in sorted(runs.items(), key=lambda x: (x[0][1], BRACOS.index(x[0][0]))):
        mae, rmse = mae_rmse(r["z"])
        out["por_execucao"].append({"braco": b, "semente": s, "MAE": mae, "RMSE": rmse, "MAE_metrics": r["m"]["validation"]["MAE"],
                                    "melhor_epoca": r["m"]["best_epoch"], "epocas": r["m"]["epochs_completed"],
                                    "treino_h": r["m"]["training_seconds"] / 3600, "gpu": r["m"]["gpu"],
                                    "init_state_sha256": r["m"]["init_state_sha256"]})
    val = {(x["braco"], x["semente"]): x for x in out["por_execucao"]}
    for a, b in CONTRASTES:
        k = f"{a} − {b}"
        out["contrastes"][k] = {met: resumo_par([val[(a, s)][met] - val[(b, s)][met] for s in seeds]) for met in ("MAE", "RMSE")}
        ref = statistics.mean(val[(b, s)]["MAE"] for s in seeds)
        out["contrastes"][k]["MAE_percentual_da_referencia"] = 100 * out["contrastes"][k]["MAE"]["media"] / ref
        out["por_horizonte"][k] = []
        for h in range(H):
            d = []
            for s in seeds:
                za, zb = runs[(a, s)]["z"], runs[(b, s)]["z"]
                d.append(za["abs_error"][h].sum() / za["count"][h].sum() - zb["abs_error"][h].sum() / zb["count"][h].sum())
            out["por_horizonte"][k].append({"horizonte": h + 1, **{x: y for x, y in resumo_par(d).items() if x != "valores"}})
        for eb in ([a] if a in nos_est else []) + ([b] if b in nos_est and b != a else []):
            for e in ESTRATOS:
                nos = nos_est[eb][e]
                if not nos:
                    continue
                d = [mae_rmse(runs[(a, s)]["z"], nos)[0] - mae_rmse(runs[(b, s)]["z"], nos)[0] for s in seeds]
                out["estratos"].setdefault(k, {})[f"estratos de {eb}: {e} ({len(nos)} nós)"] = resumo_par(d)
    out["por_braco"] = {b: {"MAE_media": statistics.mean(val[(b, s)]["MAE"] for s in seeds),
                            "MAE_dp": statistics.stdev(val[(b, s)]["MAE"] for s in seeds) if len(seeds) > 1 else float("nan"),
                            "RMSE_media": statistics.mean(val[(b, s)]["RMSE"] for s in seeds),
                            "por_horizonte_MAE": [statistics.mean(runs[(b, s)]["z"]["abs_error"][h].sum() / runs[(b, s)]["z"]["count"][h].sum()
                                                                  for s in seeds) for h in range(H)]} for b in BRACOS}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--epocas", type=int, default=600, help="épocas esperadas (0 = não conferir; smoke)")
    a = p.parse_args()
    out = relatorio(a.base, a.epocas or None)
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)                 # escalares numpy
    (a.base / "relatorio_validacao.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=conv) + "\n")
    print(json.dumps({"problemas": out["problemas"], "por_braco": {b: {k: v for k, v in d.items() if k != "por_horizonte_MAE"}
                                                                  for b, d in out["por_braco"].items()},
                      "contrastes": {k: {"MAE": v["MAE"], "MAE_%": v["MAE_percentual_da_referencia"]} for k, v in out["contrastes"].items()}},
                     indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    raise SystemExit(1 if out["problemas"] else 0)


if __name__ == "__main__":
    main()
