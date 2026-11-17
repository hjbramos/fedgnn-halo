"""Análise dos controles centralizados de grafo do SUNT (sem treino): global e estratificada em k=20.

Protocolo: results/sunt_graph_controls/v1/protocol.md (o script recusa executar se ele mudar). Estratos e
limites: os já fixados em results/sunt_stratified_analysis/v1/node_descriptors.csv (k=20; volume e
cobertura do treino), conferidos por hash. Braços enumerados explicitamente:

  Centralizado   results/sunt_centralized_boardings/sunt/seed_XXX            (referência existente)
  sem_arestas    results/sunt_control_noedges/sunt/seed_XXX                  (novo)
  cortes_k20     results/sunt_control_cut20/sunt/seed_XXX                    (novo)
  FedAvg_k20     results/sunt_fedavg_final/sunt/k020/seed_XXX                (existente)
  cortes_k166    results/sunt_control_cut166/sunt/seed_XXX                   (novo, desktop; protocol_k166.md)
  FedAvg_k166    results/sunt_fedavg_final/sunt/k166/seed_XXX                (existente)

Dois blocos, cada um com os estratos da sua partição: k=20 (sem_arestas − Centralizado; cortes_k20 −
Centralizado; FedAvg_k20 − cortes_k20; contexto FedAvg_k20 − Centralizado) e k=166 (cortes_k166 −
Centralizado; FedAvg_k166 − cortes_k166; contexto FedAvg_k166 − Centralizado). Nada é escrito além de
audit/checks se alguma verificação registrada falhar.

    conda activate fl_env && python -m experiments.sunt_graph_controls_analysis
    python -m experiments.sunt_graph_controls_analysis --layout-preview   # só testa o layout

`--layout-preview` substitui os braços novos por execuções existentes de OUTROS métodos (dados
substitutos), escreve só em results/sunt_graph_controls/v1/layout_preview/, marca cada figura com uma
tarja e cada tabela com um aviso, e o manifesto com "prévia_de_layout": true. Nada dessa pasta é resultado.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .train import ROOT

OUT = ROOT / "results" / "sunt_graph_controls" / "v1"
PROTOCOL = OUT / "protocol.md"
PROTOCOL_SHA256 = "89299cada66cfd5cb080ae2ef5101c61c98e4b1d239e1457645a32dc77862840"
PROTOCOL_K166 = OUT / "protocol_k166.md"
PROTOCOL_K166_SHA256 = "bc59f3a6fd9f1cd725eec92d539bc85a81a1473d4bfe14578a080c0b833482aa"
DESC = ROOT / "results" / "sunt_stratified_analysis" / "v1" / "node_descriptors.csv"
DESC_MANIFEST = ROOT / "results" / "sunt_stratified_analysis" / "v1" / "manifest.json"
CKPT_HASHES = OUT / "checkpoints_sha256.csv"          # gerado no Terumo (sha256sum dos best_model.pt), ver runs.md
SEEDS = [7, 42, 101, 123, 314]
ARMS = {
    "Centralizado": "results/sunt_centralized_boardings/sunt/seed_{seed:03d}/metrics.json",
    "sem_arestas": "results/sunt_control_noedges/sunt/seed_{seed:03d}/metrics.json",
    "cortes_k20": "results/sunt_control_cut20/sunt/seed_{seed:03d}/metrics.json",
    "FedAvg_k20": "results/sunt_fedavg_final/sunt/k020/seed_{seed:03d}/metrics.json",
    "cortes_k166": "results/sunt_control_cut166/sunt/seed_{seed:03d}/metrics.json",
    "FedAvg_k166": "results/sunt_fedavg_final/sunt/k166/seed_{seed:03d}/metrics.json",
}
FEDAVG = {"FedAvg_k20": (20, 600), "FedAvg_k166": (166, 1000)}   # braço -> (k, rodadas)
BLOCKS = {20: ["Centralizado", "sem_arestas", "cortes_k20", "FedAvg_k20"], 166: ["Centralizado", "cortes_k166", "FedAvg_k166"]}
EXPECTED_MODEL = {
    "Centralizado": {"name": "leconv", "hidden_size": 64},
    "sem_arestas": {"name": "leconv", "hidden_size": 64, "self_loops_only": True},
    "cortes_k20": {"name": "leconv", "hidden_size": 64, "cut_partition": "data/sunt/processed/partitions/leiden/k020/partition_seed_042"},
    "cortes_k166": {"name": "leconv", "hidden_size": 64, "cut_partition": "data/sunt/processed/partitions/leiden/k166/partition_seed_042"},
}
EXPECTED_EDGES = {"sem_arestas": 2903, "cortes_k20": 8994, "cortes_k166": 7998}
CONTRASTS = {20: [("sem_arestas", "Centralizado", "principal", "benefício líquido do grafo"),
                  ("cortes_k20", "Centralizado", "principal", "sensibilidade às conexões entre clientes"),
                  ("FedAvg_k20", "cortes_k20", "principal", "procedimento de treinamento com arestas correspondentes"),
                  ("FedAvg_k20", "Centralizado", "contexto", "perda federada já relatada")],
             166: [("cortes_k166", "Centralizado", "principal", "sensibilidade às conexões entre clientes"),
                   ("FedAvg_k166", "cortes_k166", "principal", "procedimento de treinamento com arestas correspondentes"),
                   ("FedAvg_k166", "Centralizado", "contexto", "perda federada já relatada")]}
LOSS = ["0", "(0;0,25]", "(0,25;0,50]", "(0,50;1]"]
TERC = ["baixo", "médio", "alto"]
DIMENSIONS = {"global": (None, ["todos"]), "fronteira": ("fronteira_k{k}", ["fronteira", "interior"]),
              "perda_vizinhança": ("perda_viz_faixa_k{k}", LOSS), "perda_peso": ("perda_peso_faixa_k{k}", LOSS),
              "volume": ("volume_tercil", TERC), "cobertura": ("cobertura_tercil", TERC)}
SPLITS = ["test", "validation"]
PREVIEW = False
PREVIEW_DIR = OUT / "layout_preview"
PREVIEW_NOTE = "PRÉVIA DE LAYOUT — DADOS SUBSTITUTOS, NÃO SÃO RESULTADOS"
PREVIEW_STANDINS = {   # braços novos -> execuções existentes de outros métodos, só para exercitar o código
    "sem_arestas": "results/sunt_fedprox_final/sunt/k020/mu0p01/seed_{seed:03d}/metrics.json",
    "cortes_k20": "results/sunt_ditto_final/sunt/k020/lam_per_client/seed_{seed:03d}/metrics.json",
    "cortes_k166": "results/sunt_fedgnnhet_final/sunt/k166/halo_1hop/seed_{seed:03d}/metrics.json",
}
RESULT_FILES = ["strata_metrics.csv", "contrasts.csv", "boundary_vs_interior.csv", "summary_tables.md", "manifest.json",
                "fig1_global.png", "fig2_fronteira_interior.png", "fig3_perda_peso.png", "fig4_volume_cobertura.png"]


def sha256(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


class Checks(list):
    def add(self, name, ok, detail=""):
        if PREVIEW and name.startswith(("auditoria results/sunt_fedprox", "auditoria results/sunt_ditto",
                                        "auditoria results/sunt_fedgnnhet", "hash do checkpoint")):
            ok, detail = True, f"[prévia: substituto, auditoria não se aplica] {detail}"
        self.append({"verificação": name, "ok": bool(ok), "detalhe": str(detail)})
        if not ok:
            print(f"FALHOU: {name} — {detail}", file=sys.stderr)


def load(checks: Checks) -> tuple[dict, pd.DataFrame]:
    ckpt = pd.read_csv(CKPT_HASHES).set_index("arquivo")["sha256"].to_dict() if CKPT_HASHES.exists() else {}
    prov = {f["arquivo"]: f for f in json.loads((ROOT / "generated-traffic" / "provenance.json").read_text())["fontes"]}
    runs, audit = {}, []
    for arm, pattern in ARMS.items():
        for seed in SEEDS:
            path = ROOT / pattern.format(seed=seed)
            if not path.exists():
                checks.add(f"existe {rel(path)}", False, "execução ausente ou incompleta")
                audit.append({"braço": arm, "semente": seed, "arquivo": rel(path), "status": "ausente"})
                continue
            m = json.loads(path.read_text())
            budget, done = (FEDAVG[arm][1], m.get("rounds_completed")) if arm in FEDAVG else (600, m.get("epochs_completed"))
            best_ckpt = (ROOT / m["evaluated_from"] / "best_model.pt") if m.get("evaluated_from") else None
            row = {"braço": arm, "semente": seed, "arquivo": rel(path), "status": "ok", "sha256": sha256(path),
                   "method": m.get("method"), "alvo": m.get("target"), "entradas": "+".join(m.get("features", [])),
                   "W": m.get("input_steps"), "H": m.get("horizon"), "orçamento": budget, "concluído": done,
                   "melhor": m.get("best_epoch", m.get("best_round")), "model_options": json.dumps(m.get("model_options"), ensure_ascii=False),
                   "model_edges": m.get("model_edges"), "gpu": m.get("gpu"), "avaliado_de": m.get("evaluated_from"),
                   "checkpoint": rel(best_ckpt) if best_ckpt else None,
                   "checkpoint_sha256": (ckpt.get(rel(best_ckpt)) if best_ckpt else None)
                   or (prov.get(rel(path), {}).get("checkpoint_sha256")),
                   "val_MAE": m["validation"]["MAE"], "test_MAE": m.get("test", {}).get("MAE"),
                   "per_node_val": "per_node" in m.get("validation", {}), "per_node_test": "per_node" in m.get("test", {})}
            audit.append(row)
            ok = (row["alvo"] == "n_boardings" and m.get("features") == ["n_boardings", "n_alightings"] and row["W"] == 168
                  and row["H"] == 24 and done == budget and m["seed"] == seed and row["per_node_val"] and row["per_node_test"]
                  and m.get("test_evaluated", False))
            if arm in EXPECTED_MODEL and arm != "Centralizado":
                ok &= m.get("model_options") == EXPECTED_MODEL[arm] and m.get("model_edges") == EXPECTED_EDGES[arm]
                ok &= m.get("graph") == "transit"
            if arm in FEDAVG:
                k = FEDAVG[arm][0]
                ok &= m.get("num_clients") == k and m.get("partition") == f"data/sunt/processed/partitions/leiden/k{k:03d}/partition_seed_042"
            if arm == "Centralizado" or arm in FEDAVG:
                ok &= prov.get(rel(path), {}).get("sha256") == row["sha256"]
            checks.add(f"auditoria {row['arquivo']}", ok, f"concluído={done}/{budget}, model_edges={row['model_edges']}")
            if arm in EXPECTED_EDGES:
                checks.add(f"hash do checkpoint registrado {row['arquivo']}", bool(row["checkpoint_sha256"]))
            runs[(arm, seed)] = m
    return runs, pd.DataFrame(audit)


def strata_metrics(runs: dict, desc: pd.DataFrame, checks: Checks) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        ref = np.asarray(runs[("Centralizado", SEEDS[0])][split]["per_node"]["count"])
        total = ref.sum()
        for (arm, seed), m in runs.items():
            pn = m[split]["per_node"]
            err, cnt = np.asarray(pn["abs_error"], dtype=np.float64), np.asarray(pn["count"])
            tag = f"{split} {arm} s{seed}"
            checks.add(f"{tag}: 2903 nós, erros finitos ≥ 0", len(err) == 2903 and np.isfinite(err).all() and (err >= 0).all())
            checks.add(f"{tag}: mesmos alvos que o Centralizado", np.array_equal(cnt, ref))
            mae = err.sum() / cnt.sum()
            checks.add(f"{tag}: Σerro/Σcount = MAE gravado", abs(mae - m[split]["MAE"]) / m[split]["MAE"] < 1e-6, f"{mae:.6f}")
            node_mae = np.where(cnt > 0, err / np.maximum(cnt, 1), np.nan)
            for k, arms in BLOCKS.items():
                if arm not in arms:
                    continue
                for dim, (col, order) in DIMENSIONS.items():
                    groups = np.full(len(err), "todos") if col is None else desc[col.format(k=k)].to_numpy()
                    checks.add(f"{tag} k={k} {dim}: estratos cobrem todos os nós", np.isin(groups, order).all())
                    recomposed = 0.0
                    for g in order:
                        sel = groups == g
                        c, e = int(cnt[sel].sum()), float(err[sel].sum())
                        if not sel.any():
                            continue
                        rows.append({"split": split, "k": k, "braço": arm, "semente": seed, "dimensão": dim, "estrato": g,
                                     "nós": int(sel.sum()), "nós_avaliáveis": int((cnt[sel] > 0).sum()), "ocorrências_alvo": c,
                                     "erro_abs": e, "MAE": e / c if c else np.nan, "participação_denominador": c / total,
                                     "MAE_médio_entre_nós": float(np.nanmean(node_mae[sel])) if c else np.nan})
                        recomposed += e / total
                    checks.add(f"{tag} k={k} {dim}: recomposição = MAE global", abs(recomposed - mae) / mae < 1e-9)
    return pd.DataFrame(rows)


def contrasts(sm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tcrit = stats.t.ppf(0.975, len(SEEDS) - 1)
    piv = sm.pivot_table(index=["split", "k", "dimensão", "estrato", "semente"], columns="braço", values="MAE")
    size = sm[sm["braço"] == "Centralizado"].groupby(["split", "k", "dimensão", "estrato"])[
        ["nós", "nós_avaliáveis", "participação_denominador"]].first()

    def summarize(v):
        mean, sd = v.mean(), v.std(ddof=1)
        half = tcrit * sd / np.sqrt(len(v))
        return {**{f"s{s}": v[s] for s in SEEDS}, "média": mean, "dp": sd, "sementes_negativas": int((v < 0).sum()),
                "IC95_t_inf": mean - half, "IC95_t_sup": mean + half}

    rows, diffs = [], []
    for k, contrasts_k in CONTRASTS.items():
        pk = piv.xs(k, level="k")
        for a, b, kind, meaning in contrasts_k:
            d = (pk[a] - pk[b]).rename("Δ").reset_index()
            d["contraste"], d["k"] = f"{a} − {b}", k
            diffs.append(d)
            for key, g in d.groupby(["split", "dimensão", "estrato"]):
                v = g.set_index("semente")["Δ"].reindex(SEEDS)
                rows.append({"split": key[0], "k": k, "dimensão": key[1], "estrato": key[2], "contraste": f"{a} − {b}",
                             "tipo": kind, "leitura": meaning, **summarize(v), **size.loc[(key[0], k, key[1], key[2])].to_dict()})
    dd = pd.concat(diffs)
    bnd = dd[dd["dimensão"] == "fronteira"].pivot_table(index=["split", "k", "contraste", "semente"], columns="estrato", values="Δ").reset_index()
    bnd["Δ"] = bnd["fronteira"] - bnd["interior"]
    bi = [{"split": key[0], "k": key[1], "contraste": key[2], **summarize(g.set_index("semente")["Δ"].reindex(SEEDS))}
          for key, g in bnd.groupby(["split", "k", "contraste"])]
    return pd.DataFrame(rows), pd.DataFrame(bi)


INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"sem_arestas − Centralizado": "#2a78d6", "cortes_k20 − Centralizado": "#eb6834",
          "FedAvg_k20 − cortes_k20": "#1baf7a", "FedAvg_k20 − Centralizado": "#8a8984",
          "cortes_k166 − Centralizado": "#eb6834", "FedAvg_k166 − cortes_k166": "#1baf7a",
          "FedAvg_k166 − Centralizado": "#8a8984"}


def _mark(fig):
    """Tarja diagonal em toda figura de prévia de layout."""
    if PREVIEW:
        w = fig.get_size_inches()[0]
        fig.text(0.5, 0.5, PREVIEW_NOTE, ha="center", va="center", rotation=25, fontsize=min(17, 1.6 * w), color="#e34948",
                 alpha=0.35, fontweight="bold", transform=fig.transFigure, zorder=100)
        for y, va in ((0.003, "bottom"), (0.997, "top")):
            fig.text(0.99, y, PREVIEW_NOTE, ha="right", va=va, fontsize=8, color="#e34948", fontweight="bold")


def figures(sm: pd.DataFrame, ct: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb"})
    t = sm[(sm["split"] == "test") & (sm["dimensão"] == "global")].drop_duplicates(["braço", "semente"])
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    arms = list(ARMS)
    for i, arm in enumerate(arms):
        v = t[t["braço"] == arm].set_index("semente").reindex(SEEDS)["MAE"]
        ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, s=16, color="#2a78d6", alpha=0.6, linewidths=0)
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=INK, lw=2)
        ax.annotate(f"{v.mean():.3f}".replace(".", ","), (i + 0.25, v.mean()), fontsize=8, color=INK, va="center")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms)
    ax.set_ylabel("MAE de teste (embarques/h)")
    ax.set_title("MAE global por braço — pontos: sementes; traço: média", fontsize=9, loc="left", color=MUTED)
    fig.tight_layout()
    _mark(fig)
    fig.savefig(OUT / "fig1_global.png", dpi=180)
    plt.close(fig)

    def panel(ax, k, dim, order, title):
        sub = ct[(ct["split"] == "test") & (ct["k"] == k) & (ct["dimensão"] == dim)]
        names = [n for n in COLORS if n in set(sub["contraste"])]
        x = np.arange(len(order))
        for j, cname in enumerate(names):
            s = sub[sub["contraste"] == cname].set_index("estrato").reindex(order)
            off = (j - (len(names) - 1) / 2) * 0.2
            for i, g in enumerate(order):
                if pd.isna(s.loc[g, "média"]):
                    continue
                ax.scatter(np.full(5, x[i] + off), [s.loc[g, f"s{k}"] for k in SEEDS], s=8, color=COLORS[cname], alpha=0.45, linewidths=0)
                ax.plot([x[i] + off - 0.08, x[i] + off + 0.08], [s.loc[g, "média"]] * 2, color=COLORS[cname], lw=2,
                        label=cname if i == 0 else None)
        ax.axhline(0, color=MUTED, lw=0.8)
        n = sub[sub["contraste"] == names[0]].set_index("estrato").reindex(order)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n{int(n.loc[g, 'nós'])} nós · {100 * n.loc[g, 'participação_denominador']:.0f}%" for g in order], fontsize=7)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_ylabel("Δ MAE (embarques/h)")

    for fname, dims in [("fig2_fronteira_interior.png", [("fronteira", ["fronteira", "interior"], "Fronteira/interior da partição k={k}")]),
                        ("fig3_perda_peso.png", [("perda_peso", LOSS, "Faixa de peso de entrada externo (k={k})")]),
                        ("fig4_volume_cobertura.png", [("volume", TERC, "Tercil de volume (treino) · bloco k={k}"),
                                                       ("cobertura", TERC, "Tercil de cobertura (treino) · bloco k={k}")])]:
        fig, axes = plt.subplots(len(BLOCKS), len(dims), figsize=(6.5 * len(dims), 4.2 * len(BLOCKS)), squeeze=False)
        for r, k in enumerate(BLOCKS):
            for ax, (dim, order, title) in zip(axes[r], dims):
                panel(ax, k, dim, order, title.format(k=k))
            axes[r, 0].legend(frameon=False, fontsize=7, loc="best")
        fig.suptitle("Contrastes pareados por semente (teste) — pontos: sementes; traço: média", fontsize=9, color=MUTED, x=0.01, ha="left")
        fig.tight_layout()
        _mark(fig)
        fig.savefig(OUT / fname, dpi=180)
        plt.close(fig)


def fmt(x, d=3):
    return "—" if pd.isna(x) else f"{x:+.{d}f}".replace(".", ",")


def tables(sm, ct, bi) -> str:
    L = ([f"> **{PREVIEW_NOTE}.** Os braços novos foram substituídos por execuções de outros métodos só para "
          "testar o código e o layout.", ""] if PREVIEW else []) + ["# Tabelas geradas — controles de grafo do SUNT (v1)", "",
         "Δ em embarques/h (MAE ponderado por alvos). sem. < 0 = sementes (de 5) em que o primeiro braço erra menos. "
         "IC95% t pareado, exploratório, sem ajuste.", "", "## MAE global por braço (média ± dp entre sementes)", "",
         "| braço | teste | validação | " + " | ".join(f"s{s} (teste)" for s in SEEDS) + " |", "|---" * (3 + len(SEEDS)) + "|"]
    g = sm[sm["dimensão"] == "global"].drop_duplicates(["split", "braço", "semente"])
    for arm in ARMS:
        cells = []
        for split in ("test", "validation"):
            v = g[(g["braço"] == arm) & (g["split"] == split)]["MAE"]
            cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}".replace(".", ","))
        tv = g[(g["braço"] == arm) & (g["split"] == "test")].set_index("semente")["MAE"]
        L.append(f"| {arm} | " + " | ".join(cells) + " | " + " | ".join(f"{tv[s]:.4f}".replace(".", ",") for s in SEEDS) + " |")
    for split in SPLITS:
        L += ["", f"## Contrastes — {split}", ""]
        for dim in DIMENSIONS:
            L += [f"### {dim}", "", "| k | estrato | nós (aval.) | % alvos | contraste | " + " | ".join(f"s{s}" for s in SEEDS)
                  + " | média ± dp | sem. < 0 | IC95% t |", "|---" * (9 + len(SEEDS)) + "|"]
            for r in ct[(ct["split"] == split) & (ct["dimensão"] == dim)].itertuples():
                L.append(f"| {r.k} | {r.estrato} | {int(r.nós)} ({int(r.nós_avaliáveis)}) | {100 * r.participação_denominador:.1f} | {r.contraste} | "
                         + " | ".join(fmt(getattr(r, f"s{s}")) for s in SEEDS)
                         + f" | {fmt(r.média)} ± {r.dp:.3f} | {r.sementes_negativas}/5 | [{fmt(r.IC95_t_inf)}; {fmt(r.IC95_t_sup)}] |".replace("± 0.", "± 0,"))
            L.append("")
        L += [f"### (Δ fronteira − Δ interior) por semente — {split}", "",
              "| k | contraste | " + " | ".join(f"s{s}" for s in SEEDS) + " | média ± dp | sem. < 0 | IC95% t |", "|---" * (len(SEEDS) + 5) + "|"]
        for r in bi[bi["split"] == split].itertuples():
            L.append(f"| {r.k} | {r.contraste} | " + " | ".join(fmt(getattr(r, f"s{s}")) for s in SEEDS)
                     + f" | {fmt(r.média)} ± {r.dp:.3f} | {r.sementes_negativas}/5 | [{fmt(r.IC95_t_inf)}; {fmt(r.IC95_t_sup)}] |".replace("± 0.", "± 0,"))
    t = sm[sm["split"] == "test"].groupby(["k", "dimensão", "estrato", "braço"])[["MAE", "MAE_médio_entre_nós"]].mean()
    for k, arms in BLOCKS.items():
        L += ["", f"## MAE absoluto por estrato, bloco k={k} (teste, média entre sementes) e MAE médio entre nós (secundário)", "",
              "| dimensão | estrato | " + " | ".join(arms) + " | MAE médio entre nós: " + " / ".join(arms) + " |", "|---" * (3 + len(arms)) + "|"]
        for dim, (_, order) in DIMENSIONS.items():
            for st in order:
                if (k, dim, st, "Centralizado") in t.index:
                    L.append(f"| {dim} | {st} | " + " | ".join(f"{t.loc[(k, dim, st, a), 'MAE']:.3f}".replace(".", ",") for a in arms)
                             + " | " + " / ".join(f"{t.loc[(k, dim, st, a), 'MAE_médio_entre_nós']:.3f}".replace(".", ",") for a in arms) + " |")
    return "\n".join(L) + "\n"


def main():
    global OUT, PREVIEW
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-preview", action="store_true", help="dados substitutos, só em layout_preview/")
    if ap.parse_args().layout_preview:
        PREVIEW, OUT = True, PREVIEW_DIR
        OUT.mkdir(parents=True, exist_ok=True)
        ARMS.update(PREVIEW_STANDINS)
        (OUT / "LEIA-ME.md").write_text(f"# {PREVIEW_NOTE}\n\nGerado por `python -m experiments.sunt_graph_controls_analysis "
                                         "--layout-preview`. Os braços sem_arestas, cortes_k20 e cortes_k166 foram substituídos por "
                                         f"execuções existentes de outros métodos:\n\n" + "".join(f"- {a}: `{p}`\n" for a, p in PREVIEW_STANDINS.items())
                                         + "\nServe só para conferir código, tabelas e figuras antes dos resultados reais. Nada aqui "
                                         "deve ser lido, citado ou comparado como resultado.\n")
    if sha256(PROTOCOL) != PROTOCOL_SHA256 or sha256(PROTOCOL_K166) != PROTOCOL_K166_SHA256:
        sys.exit("protocol.md ou protocol_k166.md mudou depois de congelado")
    checks = Checks()
    runs, audit = load(checks)
    desc = pd.read_csv(DESC)
    desc_hash = sha256(DESC)
    checks.add("node_descriptors.csv = o da análise estratificada v1 (estratos e limites fixados)",
               len(desc) == 2903 and (desc["nó"].to_numpy() == np.arange(2903)).all())
    sm = strata_metrics(runs, desc, checks) if all((a, s) in runs for a in ARMS for s in SEEDS) else None
    failed = [c for c in checks if not c["ok"]]
    audit.to_csv(OUT / "audit_runs.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(list(checks), indent=1, ensure_ascii=False) + "\n")
    lines = ["# Verificações", "", f"{len(checks)} verificações; {len(failed)} falharam.", "", "| verificação | ok | detalhe |", "|---|---|---|"]
    lines += [f"| {c['verificação']} | {'sim' if c['ok'] else '**NÃO**'} | {c['detalhe']} |" for c in failed + [c for c in checks if c["ok"]]
              if not (c["ok"] and (" dimensão" in c["verificação"] or ": estratos" in c["verificação"] or "recomposição" in c["verificação"]))]
    (OUT / "checks.md").write_text("\n".join(lines) + "\n\nVerificações por estrato completas em `checks.json`.\n")
    if failed or sm is None:
        for name in RESULT_FILES:
            (OUT / name).unlink(missing_ok=True)
        sys.exit(f"{len(failed)} verificações falharam (ver {rel(OUT / 'checks.md')}); resultados não gerados")
    ct, bi = contrasts(sm)
    if PREVIEW:   # toda linha de CSV da prévia carrega o aviso
        for df in (sm, ct, bi, audit):
            df.insert(0, "AVISO", PREVIEW_NOTE)
        audit.to_csv(OUT / "audit_runs.csv", index=False)
    sm.to_csv(OUT / "strata_metrics.csv", index=False)
    ct.to_csv(OUT / "contrasts.csv", index=False)
    bi.to_csv(OUT / "boundary_vs_interior.csv", index=False)
    figures(sm, ct)
    (OUT / "summary_tables.md").write_text(tables(sm, ct, bi))
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {"análise": "controles de grafo SUNT v1", "prévia_de_layout": PREVIEW,
                **({"aviso": PREVIEW_NOTE, "substitutos": PREVIEW_STANDINS} if PREVIEW else {}), "script": rel(Path(__file__)), "script_sha256": sha256(Path(__file__)),
                "protocolo_sha256": PROTOCOL_SHA256, "protocolo_k166_sha256": PROTOCOL_K166_SHA256, "comando": "python -m experiments.sunt_graph_controls_analysis",
                "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "python": platform.python_version(), "pacotes": {m.__name__: m.__version__ for m in (np, pd, __import__("scipy"), __import__("matplotlib"))},
                "node_descriptors": {"arquivo": rel(DESC), "sha256": desc_hash,
                                     "manifest_v1_sha256": sha256(DESC_MANIFEST) if DESC_MANIFEST.exists() else None},
                "checkpoints_sha256": rel(CKPT_HASHES) if CKPT_HASHES.exists() else None,
                "execuções": audit[["braço", "semente", "arquivo", "sha256", "checkpoint", "checkpoint_sha256", "gpu"]].to_dict("records"),
                "verificações": {"total": len(checks), "falhas": 0}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"escrito em {rel(OUT)}; {len(checks)} verificações, 0 falhas")


if __name__ == "__main__":
    main()
