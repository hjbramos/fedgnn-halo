"""Análise estratificada do SUNT (alvo n_boardings): onde a partição prejudica e onde o halo recupera.

Exploratória, sem treino e sem reavaliação: lê só o `per_node` dos metrics.json da campanha final
(enumerados explicitamente abaixo) e a estrutura das partições. O protocolo está congelado em
results/sunt_stratified_analysis/v1/protocol.md; o script recusa executar se ele mudar.

Escreve em results/sunt_stratified_analysis/v1/ (nada fora dessa pasta):
  manifest.json            insumos com SHA-256, ambiente, limites dos tercis, comando
  audit_runs.csv           uma linha por execução: método, k, semente, orçamento, alvo, janela, partição, per_node
  node_descriptors.csv     descritores por nó (índice interno) e estrato em cada k
  strata_metrics.csv       MAE por (split, método, k, semente, dimensão, estrato), contagens e participações
  contrasts.csv            contrastes pareados por semente, média, dp, sementes < 0, IC95% t
  boundary_vs_interior.csv (Δ fronteira − Δ interior) por semente
  checks.md / checks.json  verificações numéricas
  summary_tables.md        tabelas geradas para o report.md
  fig*.png                 figuras

    conda activate fl_env && python -m experiments.sunt_stratified_analysis
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

import data_loader as dl
from data_loader import sunt as sunt_loader

from .data import load_experiment
from .partition import partition_dir
from .train import ROOT

OUT = ROOT / "results" / "sunt_stratified_analysis" / "v1"
PROTOCOL = OUT / "protocol.md"
PROTOCOL_SHA256 = "c945cd86453ecd7981b816fa0700d6666019378e3c97af09345132803bdb006a"
PROVENANCE = ROOT / "generated-traffic" / "provenance.json"

SEEDS = [7, 42, 101, 123, 314]
KS = [6, 20, 166]
SECONDARY_KS = [6, 20]
TARGET, FEATURES, W, H = "n_boardings", ["n_boardings", "n_alightings"], 168, 24
BUDGET = {6: 600, 20: 600, 166: 1000}           # rodadas da campanha final; Centralizado: 600 épocas
CENT_EPOCHS = 600

CENT = "results/sunt_centralized_boardings/sunt/seed_{seed:03d}/metrics.json"
FEDERATED = {                                   # rótulo -> (padrão, campo `method` esperado, ks)
    "FedAvg": ("results/sunt_fedavg_final/sunt/k{k:03d}/seed_{seed:03d}/metrics.json", "fedavg_0hop", KS),
    "halo": ("results/sunt_fedgnnhet_final/sunt/k{k:03d}/halo_1hop/seed_{seed:03d}/metrics.json", "fedgnnhet_1hop", KS),
    "FedProx": ("results/sunt_fedprox_final/sunt/k{k:03d}/mu0p01/seed_{seed:03d}/metrics.json", "fedprox_0hop", SECONDARY_KS),
    "Ditto": ("results/sunt_ditto_final/sunt/k{k:03d}/lam_per_client/seed_{seed:03d}/metrics.json", "ditto_0hop", SECONDARY_KS),
}
METHODS = ["Centralizado", "FedAvg", "halo", "FedProx", "Ditto"]
CONTRASTS = [("FedAvg", "Centralizado", "principal"), ("halo", "FedAvg", "principal"), ("halo", "Centralizado", "principal"),
             ("FedProx", "FedAvg", "secundário"), ("Ditto", "FedAvg", "secundário"),
             ("FedProx", "Centralizado", "secundário"), ("Ditto", "Centralizado", "secundário")]
SPLITS = ["test", "validation"]

LOSS_LABELS = ["0", "(0;0,25]", "(0,25;0,50]", "(0,50;1]"]
NO_IN = "sem_entrada"
NO_OBS = "sem_obs_treino"
TERC = ["baixo", "médio", "alto"]
DIMENSIONS = {                                  # dimensão -> (coluna em node_descriptors, ordem)
    "global": (None, ["todos"]),
    "fronteira": ("fronteira", ["fronteira", "interior", NO_IN]),
    "perda_vizinhança": ("perda_viz_faixa", LOSS_LABELS + [NO_IN]),
    "perda_peso": ("perda_peso_faixa", LOSS_LABELS + [NO_IN]),
    "volume": ("volume_tercil", TERC + [NO_OBS]),
    "cobertura": ("cobertura_tercil", TERC + [NO_OBS]),
}
REL_TOL_RECOMPOSE, REL_TOL_RECORDED = 1e-9, 1e-6
RESULT_FILES = ["node_descriptors.csv", "strata_metrics.csv", "contrasts.csv", "boundary_vs_interior.csv", "summary_tables.md",
                "manifest.json", "fig1_fronteira_interior.png", "fig2_perda_peso.png", "fig3_volume_cobertura.png",
                "fig4_composicao.png"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = ""):
        self.items.append({"verificação": name, "ok": bool(ok), "detalhe": detail})
        if not ok:
            print(f"FALHOU: {name} — {detail}", file=sys.stderr)

    @property
    def all_ok(self) -> bool:
        return all(i["ok"] for i in self.items)


# ----------------------------------------------------------------------------------------------
# insumos e auditoria
# ----------------------------------------------------------------------------------------------
def run_paths() -> list[tuple[str, int | None, int, Path]]:
    runs = [("Centralizado", None, s, ROOT / CENT.format(seed=s)) for s in SEEDS]
    for label, (pattern, _, ks) in FEDERATED.items():
        runs += [(label, k, s, ROOT / pattern.format(k=k, seed=s)) for k in ks for s in SEEDS]
    return runs


def load_runs(checks: Checks, prov_hashes: dict[str, str]) -> tuple[dict, pd.DataFrame]:
    runs, audit = {}, []
    for label, k, seed, path in run_paths():
        if not path.exists():
            checks.add(f"existe {rel(path)}", False, "arquivo ausente")
            continue
        m = json.loads(path.read_text())
        digest = sha256(path)
        expected_method = "centralized" if label == "Centralizado" else FEDERATED[label][1]
        budget = CENT_EPOCHS if label == "Centralizado" else BUDGET[k]
        done = m.get("epochs_completed") if label == "Centralizado" else m.get("rounds_completed")
        row = {
            "método": label, "k": k, "semente": seed, "arquivo": rel(path), "sha256": digest,
            "sha256_igual_provenance": prov_hashes.get(rel(path)) == digest,
            "method_registrado": m.get("method"), "clientes_pedidos": m.get("requested_clients"),
            "clientes_realizados": m.get("num_clients"), "orçamento": budget, "concluído": done,
            "melhor": m.get("best_epoch", m.get("best_round")), "alvo": m.get("target"),
            "entradas": "+".join(m.get("features", [])), "W": m.get("input_steps"), "H": m.get("horizon"),
            "grafo": m.get("graph", "transit (padrão; campo ausente)"), "partição": m.get("partition"),
            "nós": m.get("nodes"), "per_node_validação": "per_node" in m.get("validation", {}),
            "per_node_teste": "per_node" in m.get("test", {}), "avaliado_de": m.get("evaluated_from"),
            "gpu": m.get("gpu"), "ditto_campo": "test (pessoal)" if label == "Ditto" else "",
        }
        audit.append(row)
        ok = (row["method_registrado"] == expected_method and row["alvo"] == TARGET and m.get("features") == FEATURES
              and row["W"] == W and row["H"] == H and row["nós"] == 2903 and done == budget and m["seed"] == seed
              and row["grafo"] in ("transit", "transit (padrão; campo ausente)"))
        if k is not None:
            ok &= (row["clientes_realizados"] == k
                   and row["partição"] == str(partition_dir("sunt", k, 42).relative_to(ROOT)) and m["partition_seed"] == 42)
        checks.add(f"configuração {row['arquivo']}", ok,
                   f"method={row['method_registrado']}, alvo={row['alvo']}, W/H={row['W']}/{row['H']}, concluído={done}/{budget}")
        checks.add(f"hash igual ao provenance {row['arquivo']}", row["sha256_igual_provenance"])
        if not (row["per_node_validação"] and row["per_node_teste"]):
            checks.add(f"per_node presente {row['arquivo']}", False, "lacuna: célula fora da análise")
            continue
        runs[(label, k, seed)] = m
    return runs, pd.DataFrame(audit)


# ----------------------------------------------------------------------------------------------
# descritores por nó
# ----------------------------------------------------------------------------------------------
def train_descriptors(data, checks: Checks) -> tuple[pd.DataFrame, dict]:
    ts = pd.DatetimeIndex(data.timestamps)
    checks.add("timestamps horários, únicos e consecutivos", ts.is_unique and bool((ts[1:] - ts[:-1] == pd.Timedelta("1h")).all()))
    t_end = int(np.searchsorted(ts, sunt_loader.TRAIN_END_EXCLUSIVE))
    checks.add("corte do treino = 1.ª origem de validação", t_end == int(data.origins["validation"].min()),
               f"t_end={t_end} ({ts[t_end]}), 1.ª origem val={int(data.origins['validation'].min())}")
    last_train_target = int(data.origins["train"].max()) + H - 1
    checks.add("último alvo de treino antes do corte", last_train_target == t_end - 1,
               f"último alvo={last_train_target} ({ts[last_train_target]})")
    ti = data.target_feature_index
    y = data.series[:t_end, :, ti].astype(np.float64)
    m = data.mask[:t_end, :, ti].astype(bool)
    n_obs = m.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        volume = np.where(n_obs > 0, (y * m).sum(0) / np.maximum(n_obs, 1), np.nan)
    coverage = n_obs / t_end
    checks.add("alvos observados do treino finitos e ≥ 0", bool(np.isfinite(y[m]).all() and (y[m] >= 0).all()))
    df = pd.DataFrame({"nó": np.arange(data.nodes), "células_treino_observadas": n_obs, "horas_treino": t_end,
                       "volume": volume, "cobertura": coverage})
    limits = {}
    for col in ("volume", "cobertura"):
        valid = n_obs > 0
        q1, q2 = np.quantile(df.loc[valid, col].to_numpy(), [1 / 3, 2 / 3])
        v = df[col].to_numpy()
        lab = np.where(~valid, NO_OBS, np.where(v <= q1, "baixo", np.where(v <= q2, "médio", "alto")))
        df[f"{col}_tercil"] = lab
        sizes = {g: int((lab == g).sum()) for g in TERC + [NO_OBS]}
        limits[col] = {"q1/3": float(q1), "q2/3": float(q2), "tamanhos": sizes,
                       "faixas_efetivas": sum(1 for g in TERC if sizes[g] > 0),
                       "empates_no_limite": {"q1/3": int((v[valid] == q1).sum()), "q2/3": int((v[valid] == q2).sum())}}
    meta = {"período_treino": f"{ts[0]} a {ts[t_end - 1]} (exclusivo {sunt_loader.TRAIN_END_EXCLUSIVE})",
            "horas_treino": t_end, "tercis": limits}
    return df, meta


def loss_bins(frac: np.ndarray, has_in: np.ndarray) -> np.ndarray:
    lab = np.select([frac == 0, frac <= 0.25, frac <= 0.5], LOSS_LABELS[:3], LOSS_LABELS[3])
    return np.where(has_in, lab, NO_IN)


def partition_strata(data, k: int, checks: Checks) -> tuple[pd.DataFrame, dict]:
    pdir = partition_dir("sunt", k, 42)
    assign = np.load(pdir / "client_assignments.npy")
    ids = np.load(pdir / "node_ids.npy", allow_pickle=True).astype(str).tolist()
    meta = json.loads((pdir / "metadata.json").read_text())
    checks.add(f"k={k}: ordem dos nós da partição = série", ids == list(data.node_ids))
    checks.add(f"k={k}: nº de clientes realizado", int(meta["num_clients"]) == k == len(np.unique(assign)))
    checks.add(f"k={k}: node_indices por cliente = client_assignments",
               all(np.array_equal(np.load(pdir / f"client_{c:03d}" / "node_indices.npy"), np.flatnonzero(assign == c))
                   for c in range(k)))
    a = np.asarray(data.adj, dtype=np.float64).copy()
    np.fill_diagonal(a, 0.0)
    e = a > 0
    cross = assign[:, None] != assign[None, :]
    in_deg, in_ext = e.sum(0), (e & cross).sum(0)            # coluna = destino (u -> v)
    out_ext = (e & cross).sum(1)
    in_w, in_w_ext = a.sum(0), (a * cross).sum(0)
    has_in = in_deg > 0
    out_only = int(((out_ext > 0) & (in_ext == 0)).sum())
    checks.add(f"k={k}: fronteira só de saída = 0 (grafo simétrico)", out_only == 0, f"{out_only} nós")
    viz = np.where(has_in, in_ext / np.maximum(in_deg, 1), np.nan)
    peso = np.where(has_in, in_w_ext / np.where(in_w > 0, in_w, 1), np.nan)
    df = pd.DataFrame({"nó": np.arange(data.nodes), f"cliente_k{k}": assign, "grau_entrada": in_deg,
                       f"arestas_entrada_externas_k{k}": in_ext, f"perda_viz_k{k}": viz, f"perda_peso_k{k}": peso})
    df[f"fronteira_k{k}"] = np.where(~has_in, NO_IN, np.where(in_ext > 0, "fronteira", "interior"))
    df[f"perda_viz_faixa_k{k}"] = loss_bins(np.nan_to_num(viz), has_in)
    df[f"perda_peso_faixa_k{k}"] = loss_bins(np.nan_to_num(peso), has_in)
    info = {"partição": rel(pdir), "client_assignments_sha256": sha256(pdir / "client_assignments.npy"),
            "node_ids_sha256": sha256(pdir / "node_ids.npy"), "metadata_sha256": sha256(pdir / "metadata.json"),
            "nós_fronteira": int((df[f"fronteira_k{k}"] == "fronteira").sum()),
            "nós_sem_entrada": int((~has_in).sum()), "fronteira_só_saída": out_only}
    return df, info


# ----------------------------------------------------------------------------------------------
# métricas por estrato
# ----------------------------------------------------------------------------------------------
def strata_metrics(runs: dict, desc: pd.DataFrame, checks: Checks) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        ref_count = np.asarray(runs[("Centralizado", None, SEEDS[0])][split]["per_node"]["count"])
        total = ref_count.sum()
        for (label, k_run, seed), m in runs.items():
            pn = m[split]["per_node"]
            err, cnt = np.asarray(pn["abs_error"], dtype=np.float64), np.asarray(pn["count"])
            tag = f"{split} {label} k={k_run} s={seed}"
            checks.add(f"{tag}: per_node com 2903 nós reais", len(err) == len(cnt) == 2903)
            checks.add(f"{tag}: erros finitos e ≥ 0", bool(np.isfinite(err).all() and (err >= 0).all()))
            checks.add(f"{tag}: contagens = Centralizado s=7", bool(np.array_equal(cnt, ref_count)))
            checks.add(f"{tag}: nós sem alvo têm erro 0", bool((err[cnt == 0] == 0).all()))
            mae = err.sum() / cnt.sum()
            recorded = m[split]["MAE"]
            checks.add(f"{tag}: Σerro/Σcount reproduz MAE gravado", abs(mae - recorded) / recorded < REL_TOL_RECORDED,
                       f"{mae:.9f} vs {recorded:.9f}")
            checks.add(f"{tag}: Σcount = observations", int(cnt.sum()) == int(m[split]["observations"]))
            node_mae = np.where(cnt > 0, err / np.maximum(cnt, 1), np.nan)
            for k in (KS if k_run is None else [k_run]):
                recompose = {}
                for dim, (col, order) in DIMENSIONS.items():
                    groups = np.full(len(err), "todos") if col is None else desc[f"{col}_k{k}" if f"{col}_k{k}" in desc else col].to_numpy()
                    for g in order:
                        sel = groups == g
                        c, e_ = int(cnt[sel].sum()), float(err[sel].sum())
                        if sel.sum() == 0:
                            continue
                        rows.append({"split": split, "método": label, "k": k, "semente": seed, "dimensão": dim, "estrato": g,
                                     "nós": int(sel.sum()), "nós_avaliáveis": int((cnt[sel] > 0).sum()),
                                     "ocorrências_alvo": c, "erro_abs": e_, "MAE": e_ / c if c else np.nan,
                                     "participação_denominador": c / total, "participação_erro": e_ / err.sum(),
                                     "MAE_médio_entre_nós": float(np.nanmean(node_mae[sel])) if c else np.nan})
                        if c:
                            recompose[dim] = recompose.get(dim, 0.0) + (c / total) * (e_ / c)
                    checks.add(f"{tag} k={k} {dim}: estratos cobrem todos os nós",
                               sum(int((groups == g).sum()) for g in order) == len(err))
                bad = {d: v for d, v in recompose.items() if abs(v - mae) / mae > REL_TOL_RECOMPOSE}
                checks.add(f"{tag} k={k}: recomposição dos estratos = MAE global", not bad, str(bad) if bad else "")
    return pd.DataFrame(rows)


def contrasts(sm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tcrit = stats.t.ppf(0.975, len(SEEDS) - 1)
    piv = sm.pivot_table(index=["split", "k", "dimensão", "estrato", "semente"], columns="método", values="MAE")
    size = sm[sm["método"] == "Centralizado"].groupby(["split", "k", "dimensão", "estrato"])[
        ["nós", "nós_avaliáveis", "ocorrências_alvo", "participação_denominador"]].first()
    rows, diffs = [], []
    for a, b, kind in CONTRASTS:
        if a not in piv or b not in piv:
            continue
        d = (piv[a] - piv[b]).dropna().rename("Δ").reset_index()
        d["contraste"], d["tipo"] = f"{a} − {b}", kind
        diffs.append(d)
        for key, g in d.groupby(["split", "k", "dimensão", "estrato"]):
            v = g.set_index("semente")["Δ"].reindex(SEEDS)
            if v.isna().any():
                continue
            mean, sd = v.mean(), v.std(ddof=1)
            rows.append({"split": key[0], "k": key[1], "dimensão": key[2], "estrato": key[3], "contraste": f"{a} − {b}",
                         "tipo": kind, **{f"s{s}": v[s] for s in SEEDS}, "média": mean, "dp": sd,
                         "sementes_negativas": int((v < 0).sum()),
                         "IC95_t_inf": mean - tcrit * sd / np.sqrt(len(v)), "IC95_t_sup": mean + tcrit * sd / np.sqrt(len(v)),
                         "contribuição_ao_global": mean * size.loc[key, "participação_denominador"],
                         **size.loc[key].to_dict()})
    ct = pd.DataFrame(rows)
    dd = pd.concat(diffs)
    b = dd[(dd["dimensão"] == "fronteira") & dd["estrato"].isin(["fronteira", "interior"])]
    b = b.pivot_table(index=["split", "k", "contraste", "tipo", "semente"], columns="estrato", values="Δ").reset_index()
    b["fronteira − interior"] = b["fronteira"] - b["interior"]
    bi = []
    for key, g in b.groupby(["split", "k", "contraste", "tipo"]):
        v = g.set_index("semente")["fronteira − interior"].reindex(SEEDS)
        mean, sd = v.mean(), v.std(ddof=1)
        bi.append({"split": key[0], "k": key[1], "contraste": key[2], "tipo": key[3], **{f"s{s}": v[s] for s in SEEDS},
                   "média": mean, "dp": sd, "sementes_negativas": int((v < 0).sum()),
                   "IC95_t_inf": mean - tcrit * sd / np.sqrt(5), "IC95_t_sup": mean + tcrit * sd / np.sqrt(5)})
    return ct, pd.DataFrame(bi)


# ----------------------------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------------------------
K_COLORS = {6: "#2a78d6", 20: "#eb6834", 166: "#1baf7a"}      # paleta categórica de referência, slots 1–3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def _dodged(ax, ct, contrast, dim, order, title):
    sub = ct[(ct["split"] == "test") & (ct["contraste"] == contrast) & (ct["dimensão"] == dim)]
    order = [g for g in order if g in set(sub["estrato"])]
    x = np.arange(len(order))
    for j, k in enumerate(KS):
        s = sub[sub["k"] == k].set_index("estrato").reindex(order)
        if s["média"].isna().all():
            continue
        off = (j - 1) * 0.25
        for i, g in enumerate(order):
            if pd.isna(s.loc[g, "média"]):
                continue
            vals = [s.loc[g, f"s{seed}"] for seed in SEEDS]
            ax.scatter(np.full(5, x[i] + off), vals, s=9, color=K_COLORS[k], alpha=0.45, linewidths=0)
            ax.plot([x[i] + off - 0.09, x[i] + off + 0.09], [s.loc[g, "média"]] * 2, color=K_COLORS[k], lw=2,
                    label=f"k={k}" if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.8)
    labels = []
    for g in order:
        n = sub[sub["estrato"] == g]
        labels.append(f"{g}\n" + "/".join(f"{int(r.nós)}" for r in n.sort_values("k").itertuples()))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def figures(ct: pd.DataFrame, sm: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    # 1. fronteira vs interior
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey="row")
    for r, contrast in enumerate(["FedAvg − Centralizado", "halo − FedAvg"]):
        for c, k in enumerate(KS):
            ax = axes[r, c]
            sub = ct[(ct["split"] == "test") & (ct["contraste"] == contrast) & (ct["dimensão"] == "fronteira") & (ct["k"] == k)]
            sub = sub.set_index("estrato").reindex([g for g in ["fronteira", "interior", NO_IN] if g in set(sub["estrato"])])
            for i, (g, row) in enumerate(sub.iterrows()):
                ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), [row[f"s{s}"] for s in SEEDS], s=14, color=K_COLORS[k], alpha=0.6, linewidths=0)
                ax.plot([i - 0.2, i + 0.2], [row["média"]] * 2, color=INK, lw=2)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels([f"{g}\n{int(r.nós)} nós · {100 * r.participação_denominador:.0f}% alvos" for g, r in sub.iterrows()], fontsize=7)
            ax.set_title(f"{contrast} · k={k}", fontsize=9, loc="left", color=INK)
            if c == 0:
                ax.set_ylabel("Δ MAE de teste (embarques/h)")
    fig.suptitle("Fronteira (≥1 vizinho de entrada externo) vs interior — pontos: sementes; traço: média", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_fronteira_interior.png", dpi=180)
    plt.close(fig)

    # 2. faixa de peso externo perdido; 3. volume e cobertura
    for fname, dims, sup in [
        ("fig2_perda_peso.png", [("perda_peso", LOSS_LABELS + [NO_IN], "faixa de peso de entrada externo")],
         "Por fração do peso de entrada externo\n(rótulos: nós em k=6/20/166)"),
        ("fig3_volume_cobertura.png", [("volume", TERC, "tercil de volume (treino)"), ("cobertura", TERC, "tercil de cobertura (treino)")],
         "Por volume e cobertura do treino\n(rótulos: nós em k=6/20/166)"),
    ]:
        contrasts_ = ["FedAvg − Centralizado", "halo − FedAvg", "halo − Centralizado"]
        fig, axes = plt.subplots(len(contrasts_), len(dims), figsize=(5.2 * len(dims) + 1, 7.5), squeeze=False, sharey="row")
        for r, contrast in enumerate(contrasts_):
            for c, (dim, order, lab) in enumerate(dims):
                _dodged(axes[r, c], ct, contrast, dim, order, f"{contrast} · {lab}")
                if c == 0:
                    axes[r, c].set_ylabel("Δ MAE (embarques/h)")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(sup + " — pontos: sementes; traço: média", fontsize=9, color=MUTED, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=180)
        plt.close(fig)

    # 4. composição do MAE global
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    present = set(sm.loc[(sm["dimensão"] == "perda_peso") & (sm["nós"] > 0), "estrato"])
    order = [g for g in LOSS_LABELS + [NO_IN] if g in present]
    greys = ["#d9d8d3", "#a9c8ee", "#5f9de3", "#1d5fae", "#8a8984"]
    base = sm[(sm["split"] == "test") & (sm["método"] == "Centralizado") & (sm["semente"] == 7) & (sm["dimensão"] == "perda_peso")]
    for j, k in enumerate(KS):
        s = base[base["k"] == k].set_index("estrato").reindex(order)
        left = 0.0
        for g, col in zip(order, greys):
            v = s.loc[g, "participação_denominador"]
            if pd.notna(v) and v > 0:
                axes[0].barh(j, v, left=left, color=col, edgecolor="#fcfcfb", linewidth=2, label=g if j == 0 else None)
                left += v
    axes[0].set_yticks(range(len(KS)))
    axes[0].set_yticklabels([f"k={k}" for k in KS])
    axes[0].set_xlabel("participação nos alvos observados de teste")
    axes[0].set_title("Faixa de peso de entrada externo:\nparticipação no denominador", fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    for ax, contrast in zip(axes[1:], ["FedAvg − Centralizado", "halo − FedAvg"]):
        sub = ct[(ct["split"] == "test") & (ct["contraste"] == contrast) & (ct["dimensão"] == "perda_peso")]
        x = np.arange(len(order))
        for j, k in enumerate(KS):
            s = sub[sub["k"] == k].set_index("estrato").reindex(order)
            ax.bar(x + (j - 1) * 0.26, s["contribuição_ao_global"].fillna(0), width=0.24, color=K_COLORS[k], label=f"k={k}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=7)
        ax.set_title(f"Contribuição ao Δ global: {contrast}\n(participação × Δ médio do estrato)", fontsize=9, loc="left")
        ax.set_ylabel("embarques/h")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_composicao.png", dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# tabelas para o relatório
# ----------------------------------------------------------------------------------------------
def fmt(x, d=3):
    return "—" if pd.isna(x) else f"{x:+.{d}f}".replace(".", ",")


def summary_tables(ct: pd.DataFrame, bi: pd.DataFrame, sm: pd.DataFrame, descmeta: dict, parts: dict) -> str:
    L = ["# Tabelas geradas — análise estratificada do SUNT (v1)", "",
         "Gerado por `python -m experiments.sunt_stratified_analysis`. Δ em embarques/h (MAE de teste, ponderado por alvos). "
         "Sementes < 0 = nº de sementes (de 5) em que o primeiro método erra menos. IC95% t exploratório, não ajustado.", ""]
    L += ["## Partições", "", "| k | nós de fronteira | nós sem entrada | só saída |", "|---|---|---|---|"]
    L += [f"| {k} | {p['nós_fronteira']} | {p['nós_sem_entrada']} | {p['fronteira_só_saída']} |" for k, p in parts.items()]
    L += ["", "## Tercis (treino)", ""]
    for col, v in descmeta["tercis"].items():
        L.append(f"- {col}: q1/3={v['q1/3']:.4f}, q2/3={v['q2/3']:.4f}; tamanhos {v['tamanhos']}; faixas efetivas {v['faixas_efetivas']}; "
                 f"empates exatamente no limite {v['empates_no_limite']}")
    for split in SPLITS:
        L += ["", f"## Contrastes por estrato — {split}", ""]
        for dim in DIMENSIONS:
            L += [f"### {dim}", "", "| k | estrato | nós (aval.) | % alvos | contraste | " + " | ".join(f"s{s}" for s in SEEDS)
                  + " | média ± dp | sem. < 0 | IC95% t | contrib. global |", "|---" * (9 + len(SEEDS)) + "|"]
            sub = ct[(ct["split"] == split) & (ct["dimensão"] == dim)]
            for k in KS:
                for g in DIMENSIONS[dim][1]:
                    for r in sub[(sub["k"] == k) & (sub["estrato"] == g)].itertuples():
                        if split == "validation" and r.tipo != "principal":
                            continue
                        vals = " | ".join(fmt(getattr(r, f"s{s}")) for s in SEEDS)
                        L.append(f"| {k} | {g} | {int(r.nós)} ({int(r.nós_avaliáveis)}) | {100 * r.participação_denominador:.1f} | {r.contraste} | {vals} | "
                                 f"{fmt(r.média)} ± {r.dp:.3f} | {r.sementes_negativas}/5 | [{fmt(r.IC95_t_inf)}; {fmt(r.IC95_t_sup)}] | {fmt(r.contribuição_ao_global, 4)} |".replace("± 0.", "± 0,"))
            L.append("")
        L += [f"### (Δ fronteira − Δ interior) por semente — {split}", "",
              "| k | contraste | " + " | ".join(f"s{s}" for s in SEEDS) + " | média ± dp | sem. < 0 | IC95% t |", "|---" * (len(SEEDS) + 5) + "|"]
        for r in bi[bi["split"] == split].sort_values(["tipo", "contraste", "k"]).itertuples():
            L.append(f"| {r.k} | {r.contraste} | " + " | ".join(fmt(getattr(r, f"s{s}")) for s in SEEDS)
                     + f" | {fmt(r.média)} ± {r.dp:.3f} | {r.sementes_negativas}/5 | [{fmt(r.IC95_t_inf)}; {fmt(r.IC95_t_sup)}] |".replace("± 0.", "± 0,"))
    L += ["", "## MAE absoluto por estrato (teste, média ± dp entre sementes; métodos principais)", "",
          "| k | dimensão | estrato | Centralizado | FedAvg | halo | MAE médio entre nós: Centr. / FedAvg / halo |", "|---|---|---|---|---|---|---|"]
    t = sm[(sm["split"] == "test") & sm["método"].isin(["Centralizado", "FedAvg", "halo"])]
    g = t.groupby(["k", "dimensão", "estrato", "método"]).agg(m=("MAE", "mean"), s=("MAE", "std"), nm=("MAE_médio_entre_nós", "mean"))
    for k in KS:
        for dim, (_, order) in DIMENSIONS.items():
            for st in order:
                if (k, dim, st, "Centralizado") not in g.index:
                    continue
                cells = [f"{g.loc[(k, dim, st, mth), 'm']:.3f} ± {g.loc[(k, dim, st, mth), 's']:.3f}".replace(".", ",") for mth in ["Centralizado", "FedAvg", "halo"]]
                nm = " / ".join(f"{g.loc[(k, dim, st, mth), 'nm']:.3f}".replace(".", ",") for mth in ["Centralizado", "FedAvg", "halo"])
                L.append(f"| {k} | {dim} | {st} | " + " | ".join(cells) + f" | {nm} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------
def main():
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        sys.exit(f"{rel(PROTOCOL)} mudou depois de congelado; atualize o protocolo como nova versão (v2)")
    checks = Checks()
    prov = json.loads(PROVENANCE.read_text())
    prov_hashes = {f["arquivo"]: f["sha256"] for f in prov["fontes"]}
    runs, audit = load_runs(checks, prov_hashes)

    # tudo é calculado em memória; nada de resultado é escrito antes de todas as verificações passarem
    data = load_experiment("sunt", W, H, FEATURES, TARGET)
    checks.add("grafo carregado = transit simétrico", data.graph == "transit" and np.allclose(data.adj, data.adj.T))
    checks.add("diagonal do grafo nula", bool((np.diag(data.adj) == 0).all()))
    desc, descmeta = train_descriptors(data, checks)
    parts = {}
    for k in KS:
        d, parts[k] = partition_strata(data, k, checks)
        desc = desc.merge(d.drop(columns=["grau_entrada"] if "grau_entrada" in desc else []), on="nó")
        for (label, k_run, seed), m in runs.items():
            if k_run == k:
                checks.add(f"{label} k={k} s={seed}: client_nodes = partição",
                           list(np.bincount(np.load(partition_dir('sunt', k, 42) / 'client_assignments.npy'))) == list(m["client_nodes"]))
    sm = strata_metrics(runs, desc, checks)

    failed = [c for c in checks.items if not c["ok"]]
    audit.to_csv(OUT / "audit_runs.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(checks.items, indent=1, ensure_ascii=False) + "\n")
    lines = ["# Verificações numéricas e de procedência", "",
             f"{len(checks.items)} verificações; {len(failed)} falharam.", "",
             "| verificação | ok | detalhe |", "|---|---|---|"]
    lines += [f"| {c['verificação']} | {'sim' if c['ok'] else '**NÃO**'} | {c['detalhe']} |"
              for c in (failed + [c for c in checks.items if c["ok"] and not any(s in c["verificação"] for s in (": estratos cobrem", " k=6 ", " k=20 ", " k=166 "))])]
    lines += ["", "Verificações por (split, método, semente, k, dimensão) de cobertura dos estratos e de recomposição "
              "estão completas em `checks.json`."]
    (OUT / "checks.md").write_text("\n".join(lines) + "\n")
    if failed:
        for name in RESULT_FILES:                  # não deixa resultados de uma execução anterior parecerem válidos
            (OUT / name).unlink(missing_ok=True)
        sys.exit(f"{len(failed)} verificações falharam (ver {rel(OUT / 'checks.md')}); resultados não gerados")

    ct, bi = contrasts(sm)
    desc.to_csv(OUT / "node_descriptors.csv", index=False)
    sm.to_csv(OUT / "strata_metrics.csv", index=False)
    ct.to_csv(OUT / "contrasts.csv", index=False)
    bi.to_csv(OUT / "boundary_vs_interior.csv", index=False)
    figures(ct, sm)
    (OUT / "summary_tables.md").write_text(summary_tables(ct, bi, sm, descmeta, parts))

    processed = dl.DATA_DIR / "sunt" / "processed"
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "análise": "SUNT estratificada v1 (exploratória)", "script": rel(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "protocolo": rel(PROTOCOL), "protocolo_sha256": PROTOCOL_SHA256, "comando": "python -m experiments.sunt_stratified_analysis",
        "git_commit": git("rev-parse", "HEAD"), "git_worktree_dirty": bool(git("status", "--porcelain", "--", "experiments", "data_loader", "results/sunt_centralized_boardings", "results/sunt_fedavg_final", "results/sunt_fedgnnhet_final", "results/sunt_fedprox_final", "results/sunt_ditto_final")),
        "python": platform.python_version(), "plataforma": platform.platform(),
        "pacotes": {m.__name__: m.__version__ for m in (np, pd, __import__("scipy"), __import__("matplotlib"))},
        "provenance": {"arquivo": rel(PROVENANCE), "sha256": sha256(PROVENANCE), "versao_exportador": prov["versao_exportador"]},
        "dados_processados": {p.name: sha256(p) for p in sorted(processed.glob("sunt_*")) if p.suffix in (".parquet", ".npz")},
        "série_alvo_e_máscara_sha256": hashlib.sha256(np.ascontiguousarray(data.series[:, :, data.target_feature_index]).tobytes()
                                                       + np.ascontiguousarray(data.mask).tobytes()).hexdigest(),
        "adjacência_sha256": hashlib.sha256(np.ascontiguousarray(data.adj).tobytes()).hexdigest(),
        "partições": parts, "descritores": descmeta,
        "execuções": audit[["método", "k", "semente", "arquivo", "sha256", "sha256_igual_provenance"]].to_dict("records"),
        "lacunas_per_node": audit.loc[~(audit["per_node_teste"] & audit["per_node_validação"]), "arquivo"].tolist(),
        "verificações": {"total": len(checks.items), "falhas": len(failed)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"escrito em {rel(OUT)}; {len(checks.items)} verificações, 0 falhas")


if __name__ == "__main__":
    main()
