"""Diagnóstico de resíduos na VALIDAÇÃO do SUNT (protocolo results/sunt_validation_diagnostics/v1/protocol.md).

Só inferência, em CPU, com os checkpoints já selecionados; nenhum treino, nenhum acesso ao teste, nada é
escrito nos metrics.json existentes. Cada máquina roda os braços cujos checkpoints guarda:

    python -m experiments.sunt_validation_diagnostics run --arms cortes_k166 halo_k20 --seeds 7          # piloto
    python -m experiments.sunt_validation_diagnostics run --arms cortes_k166 halo_k20                    # desktop
    python -m experiments.sunt_validation_diagnostics run --arms Centralizado cortes_k20 FedAvg_k20 ...  # Terumo

Por braço e semente: confere o hash do checkpoint, reproduz o MAE e o per_node de validação gravados e só
então grava as somas por nó e horizonte (e = y − ŷ) e os histogramas de resíduos por estrato. Os braços do
halo recebem também a auditoria do mecanismo. A análise (estratos, contrastes, relatório) junta os npz das
duas máquinas e é feita depois, localmente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as data_module
from .train import ROOT, load_config, make_loader, make_model

OUT = ROOT / "results" / "sunt_validation_diagnostics" / "v1"
PROTOCOL_V1 = OUT / "protocol.md"                                  # preservado como histórico
PROTOCOL_V1_SHA256 = "256d3ca9682c7ee2465d4ccd562dc90e48de0c622f9f7462512d9f986138d604"
PROTOCOL = OUT / "protocol_v2.md"                                  # emenda v2: regra de reprodução por nó
PROTOCOL_SHA256 = "b93b510da1ff9be27445d5c379163ec34f4cc45f139989afe965fe667c5de56e"
RULE = "v2"
TOL_NODE_MAE = 1e-4                                                # v2: diferença máxima no MAE do nó (embarques/h)
DESCRIPTORS = ROOT / "results" / "sunt_stratified_analysis" / "v1" / "node_descriptors.csv"
CKPT_MANIFESTS = [ROOT / "results" / "checkpoints_sha256.csv", ROOT / "results" / "sunt_graph_controls" / "v1" / "checkpoints_sha256.csv"]
SEEDS = [7, 42, 101, 123, 314]
W, H = 168, 24
BATCH = 32
TOL_MAE = 1e-4
TOL_NODE_REL, TOL_NODE_ABS = 1e-6, 1e-3                           # regra v1 (histórico; não decide mais)
TOL_HALO = 1e-4
N_HALO_WINDOWS = 24
HIST_LO, HIST_HI, HIST_STEP = -200.0, 200.0, 0.05
NB = int(round((HIST_HI - HIST_LO) / HIST_STEP)) + 2          # + caixa < −200 (0) e caixa ≥ 200 (NB−1)
HGROUPS = [(0, 6), (6, 12), (12, 18), (18, 24)]                # h = 1–6, 7–12, 13–18, 19–24 (índices 0..23)
MASE_LAGS, MASE_MIN_PAIRS = (168, 24), 100
P = "data/sunt/processed/partitions/leiden/k{k:03d}/partition_seed_042"

ARMS = {   # tipo, config que constrói o modelo, checkpoint, metrics.json com a validação gravada, k da partição
    "Centralizado": ("central", "config/sunt_probe_boardings.yaml", "results/sunt_probe_boardings/sunt/seed_{s:03d}",
                     "results/sunt_centralized_boardings/sunt/seed_{s:03d}/metrics.json", None),
    "cortes_k20": ("central", "config/sunt_control_cut20_probe.yaml", "results/sunt_control_cut20_probe/sunt/seed_{s:03d}",
                   "results/sunt_control_cut20/sunt/seed_{s:03d}/metrics.json", 20),
    "cortes_k166": ("central", "config/sunt_control_cut166_probe.yaml", "results/sunt_control_cut166_probe/sunt/seed_{s:03d}",
                    "results/sunt_control_cut166/sunt/seed_{s:03d}/metrics.json", 166),
    "sem_arestas": ("central", "config/sunt_control_noedges_probe.yaml", "results/sunt_control_noedges_probe/sunt/seed_{s:03d}",
                    "results/sunt_control_noedges/sunt/seed_{s:03d}/metrics.json", None),
    "FedAvg_k20": ("fedavg", "config/sunt_fedavg_final.yaml", "results/sunt_fedavg_probe/sunt/k020/seed_{s:03d}",
                   "results/sunt_fedavg_final/sunt/k020/seed_{s:03d}/metrics.json", 20),
    "FedAvg_k166": ("fedavg", "config/sunt_fedavg_final.yaml", "results/sunt_fedavg_probe/sunt/k166/seed_{s:03d}",
                    "results/sunt_fedavg_final/sunt/k166/seed_{s:03d}/metrics.json", 166),
    "halo_k20": ("halo", "config/sunt_fedgnnhet_final_desktop.yaml", "results/sunt_fedgnnhet_final/sunt/k020/halo_1hop/seed_{s:03d}",
                 "results/sunt_fedgnnhet_final/sunt/k020/halo_1hop/seed_{s:03d}/metrics.json", 20),
    "halo_k166": ("halo", "config/sunt_fedgnnhet_final.yaml", "results/sunt_fedgnnhet_final/sunt/k166/halo_1hop/seed_{s:03d}",
                  "results/sunt_fedgnnhet_final/sunt/k166/halo_1hop/seed_{s:03d}/metrics.json", 166),
    # sonda: parâmetros do Centralizado completo num modelo com o operador de grafo cortado (sem retreino)
    "sonda_k20": ("probe", "config/sunt_control_cut20_probe.yaml", "results/sunt_probe_boardings/sunt/seed_{s:03d}",
                  "results/sunt_centralized_boardings/sunt/seed_{s:03d}/metrics.json", 20),
    "sonda_k166": ("probe", "config/sunt_control_cut166_probe.yaml", "results/sunt_probe_boardings/sunt/seed_{s:03d}",
                   "results/sunt_centralized_boardings/sunt/seed_{s:03d}/metrics.json", 166),
}


CHECK_COLUMNS = ["braço", "semente", "host", "status", "regra", "checkpoint", "checkpoint_sha256_ok", "reproducao_ok",
                 "mae_recalculado", "mae_gravado", "contagens_iguais", "per_node_dif_MAE_max", "per_node_max_dif",
                 "halo_auditoria_ok", "sonda_reproduz_original", "sonda_parametros_identicos", "segundos", "pico_rss_gb",
                 "nota", "execucao"]


def write_checks(rows: list[dict], path: Path):
    """CSV com esquema fixo (mesmas colunas e ordem em toda execução); campos ausentes ficam vazios."""
    pd.DataFrame([{c: r.get(c) for c in CHECK_COLUMNS} for r in rows], columns=CHECK_COLUMNS).to_csv(path, index=False)


def node_rule_v2(a: np.ndarray, b: np.ndarray, cnt_a: np.ndarray, cnt_b: np.ndarray) -> tuple[bool, float]:
    """Regra v2: contagens idênticas; n_v > 0 → |a_v − b_v| / n_v ≤ 1e-4; n_v = 0 → somas nulas nos dois lados."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not np.array_equal(cnt_a, cnt_b):
        return False, float("inf")
    obs = cnt_a > 0
    dif = np.abs(a[obs] - b[obs]) / cnt_a[obs]
    zero_ok = bool(np.all(a[~obs] == 0) and np.all(b[~obs] == 0))
    worst = float(dif.max()) if dif.size else 0.0
    return bool(zero_ok and worst <= TOL_NODE_MAE), worst


def check_protocols():
    if sha256(PROTOCOL_V1) != PROTOCOL_V1_SHA256:
        sys.exit("protocol.md (v1) foi alterado; a v1 tem de ficar preservada")
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        sys.exit("protocol_v2.md mudou depois de congelado")


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


# ----------------------------------------------------------------------------------------------
# isolamento do teste: nenhum dataset de janelas pode ser construído para o split "test"
# ----------------------------------------------------------------------------------------------
def guard_test_split():
    original = data_module.WindowDataset.__init__

    def guarded(self, data, split, *args, **kwargs):
        if split != "validation":
            raise RuntimeError(f"diagnóstico só usa a validação; split '{split}' recusado")
        original(self, data, split, *args, **kwargs)

    data_module.WindowDataset.__init__ = guarded


# ----------------------------------------------------------------------------------------------
# estratos e escalas (fixados: node_descriptors da análise estratificada v1 e dados de treino)
# ----------------------------------------------------------------------------------------------
LOSS = ["0", "(0;0,25]", "(0,25;0,50]", "(0,50;1]"]
TERC = ["baixo", "médio", "alto"]


def strata_membership(desc: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    names, cols = [], []
    for k in (20, 166):
        def add(name, sel):
            names.append(f"k{k}|{name}")
            cols.append(np.asarray(sel, dtype=bool))
        add("global=todos", np.ones(len(desc)))
        for g in ("fronteira", "interior"):
            add(f"fronteira={g}", desc[f"fronteira_k{k}"] == g)
        for g in LOSS:
            add(f"perda_peso={g}", desc[f"perda_peso_faixa_k{k}"] == g)
        for g in TERC:
            add(f"volume={g}", desc["volume_tercil"] == g)
            add(f"cobertura={g}", desc["cobertura_tercil"] == g)
        for v in TERC:
            for g in ("fronteira", "interior"):
                add(f"volume×fronteira={v}×{g}", (desc["volume_tercil"] == v) & (desc[f"fronteira_k{k}"] == g))
    return names, np.stack(cols, axis=1)                          # (N, S)


SCALES_FILE = OUT / "mase_scales.npz"


def data_fingerprint(data) -> str:
    ti = data.target_feature_index
    d = hashlib.sha256()
    for a in (np.array(data.node_ids, dtype="U"), data.series[:, :, ti], data.mask[:, :, ti], data.origins["validation"]):
        d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()


def write_or_verify_scales(data, scales: dict) -> str:
    """Grava (ou confere) escalas, pares válidos e elegibilidade do MASE por nó; devolve o sha256 do artefato.

    Um erro escalado zero no acumulador pode ser erro zero ou escala inválida: a análise usa esta máscara
    de elegibilidade para aplicar o denominador correto e reportar as exclusões.
    """
    arrays = {}
    for lag, (s, pairs) in scales.items():
        arrays[f"s_{lag}"], arrays[f"pares_{lag}"], arrays[f"elegivel_{lag}"] = s, pairs, np.isfinite(s)
    fp = data_fingerprint(data)
    if SCALES_FILE.exists():
        old = np.load(SCALES_FILE, allow_pickle=False)
        same = str(old["fingerprint"]) == fp and all(
            np.array_equal(old[k], v, equal_nan=(old[k].dtype.kind == "f")) for k, v in arrays.items())
        if not same:
            raise ValueError("mase_scales.npz existente difere das escalas recalculadas; não sobrescrito")
    else:
        np.savez(SCALES_FILE, fingerprint=np.array(fp), node_ids=np.array(data.node_ids, dtype="U"),
                 min_pares=np.array(MASE_MIN_PAIRS), **arrays)
    return sha256(SCALES_FILE)


def mase_scales(data) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """s_v = média de |y_t − y_{t−m}| nos pares do treino com as duas pontas observadas; nan se < 100 pares ou s = 0."""
    t_end = int(data.origins["validation"].min())
    ti = data.target_feature_index
    y = data.series[:t_end, :, ti].astype(np.float64)
    m = data.mask[:t_end, :, ti].astype(bool)
    out = {}
    for lag in MASE_LAGS:
        both = m[lag:] & m[:-lag]
        diff = np.abs(y[lag:] - y[:-lag]) * both
        pairs = both.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = diff.sum(0) / pairs
        s = np.where((pairs >= MASE_MIN_PAIRS) & (s > 0), s, np.nan)
        out[lag] = (s, pairs)
    return out


# ----------------------------------------------------------------------------------------------
# acumulador: somas por nó e horizonte + histogramas de resíduo por estrato e grupo de horizonte
# ----------------------------------------------------------------------------------------------
class DiagAccumulator:
    def __init__(self, n_nodes: int, membership: np.ndarray, scales: dict):
        self.N, self.S = n_nodes, membership.shape[1]
        self.member = membership
        self.inv = {lag: np.where(np.isfinite(s), 1.0 / np.where(np.isfinite(s), s, 1.0), 0.0) for lag, (s, _) in scales.items()}
        z = lambda dtype=np.float64: np.zeros((n_nodes, H), dtype=dtype)
        self.n, self.npos, self.nneg, self.nzero = z(np.int64), z(np.int64), z(np.int64), z(np.int64)
        self.se, self.sabs, self.se2, self.sy = z(), z(), z(), z()
        self.sabs_scaled = {lag: z() for lag in self.inv}
        self.hist = np.zeros((self.S, len(HGROUPS) + 1, NB), dtype=np.int64)

    @staticmethod
    def bins(e: np.ndarray) -> np.ndarray:
        b = np.floor((e - HIST_LO) / HIST_STEP).astype(np.int64) + 1
        b[e < HIST_LO] = 0
        b[e >= HIST_HI] = NB - 1
        return np.clip(b, 0, NB - 1)

    def update(self, pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, nodes: np.ndarray):
        """pred, y: (B, H, n, 1) na escala física (ŷ já com max(0, ·)); mask booleana; nodes: índices globais (n,)."""
        p = pred[..., 0].double().cpu().numpy()
        t = y[..., 0].double().cpu().numpy()
        m = mask[..., 0].cpu().numpy().astype(bool)
        if not (np.isfinite(p[m]).all() and np.isfinite(t[m]).all()):
            raise ValueError("previsão ou alvo não finitos")
        e = np.where(m, t - p, 0.0)
        idx = np.asarray(nodes, dtype=np.int64)
        sumB = lambda a: a.sum(0).T                                   # (H, n) -> (n, H)
        self.n[idx] += sumB(m.astype(np.int64))
        self.npos[idx] += sumB((m & (e > 0)).astype(np.int64))
        self.nneg[idx] += sumB((m & (e < 0)).astype(np.int64))
        self.nzero[idx] += sumB((m & (e == 0)).astype(np.int64))
        self.se[idx] += sumB(e)
        self.sabs[idx] += sumB(np.abs(e))
        self.se2[idx] += sumB(e * e)
        self.sy[idx] += sumB(np.where(m, t, 0.0))
        for lag, inv in self.inv.items():
            self.sabs_scaled[lag][idx] += sumB(np.abs(e)) * inv[idx][:, None]
        # histogramas: só resíduos observados
        b_idx, h_idx, n_idx = np.nonzero(m)
        vals = e[b_idx, h_idx, n_idx]
        bins = self.bins(vals)
        g = np.searchsorted([hi for _, hi in HGROUPS], h_idx, side="right")
        member = self.member[idx[n_idx]]                               # (obs, S)
        for s in range(self.S):
            sel = member[:, s]
            if not sel.any():
                continue
            bs, gs = bins[sel], g[sel]
            self.hist[s, :len(HGROUPS)] += np.bincount(gs * NB + bs, minlength=len(HGROUPS) * NB).reshape(len(HGROUPS), NB)
            self.hist[s, len(HGROUPS)] += np.bincount(bs, minlength=NB)

    def per_node(self) -> tuple[np.ndarray, np.ndarray]:
        return self.sabs.sum(1), self.n.sum(1)

    def save(self, path: Path, meta: dict):
        np.savez_compressed(path, n=self.n, npos=self.npos, nneg=self.nneg, nzero=self.nzero, se=self.se, sabs=self.sabs,
                            se2=self.se2, sy=self.sy, **{f"sabs_scaled_{lag}": a for lag, a in self.sabs_scaled.items()},
                            hist=self.hist, meta=json.dumps(meta, ensure_ascii=False))


def hist_median(counts: np.ndarray) -> tuple[float | str, float]:
    """Mediana convencional a partir do histograma e fração em transbordamento.

    Com n observações ordenadas, a mediana é a observação de posição (n+1)/2 se n é ímpar e a média das
    posições n/2 e n/2+1 se n é par. Cada posição é representada pelo centro da sua caixa (resolução de meia
    caixa, 0,025). Se as duas posições centrais caem na mesma caixa de transbordamento, a mediana é um
    limite ("< -200" ou "> 200"); se só uma cai em transbordamento, o valor é indeterminado.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return "indefinida", float("nan")
    overflow = float(counts[0] + counts[-1]) / total
    cum = np.cumsum(counts)
    ranks = [(total + 1) // 2] if total % 2 else [total // 2, total // 2 + 1]
    bins = [int(np.searchsorted(cum, r)) for r in ranks]             # caixa da r-ésima observação (1-indexada)
    if all(b == 0 for b in bins):
        return "< -200", overflow
    if all(b == NB - 1 for b in bins):
        return "> 200", overflow
    if any(b in (0, NB - 1) for b in bins):
        return "indeterminada (posição central em transbordamento)", overflow
    return float(np.mean([HIST_LO + (b - 1 + 0.5) * HIST_STEP for b in bins])), overflow


# ----------------------------------------------------------------------------------------------
# avaliação por tipo de braço (mesmo caminho de previsão das métricas gravadas)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_central(model, data, acc: DiagAccumulator, seed: int):
    from .task import forward
    model.eval()
    loader = make_loader(data_module.WindowDataset(data, "validation"), BATCH, False, seed, 0)
    nodes = np.arange(data.nodes)
    for batch in loader:
        pred = (forward(model, batch, "cpu") * data.target_std + data.target_mean).clamp_min(0.0)
        acc.update(pred, batch["y_raw"], batch["y_mask"], nodes)


@torch.no_grad()
def predict_clients(state, clients, cfg, data, acc: DiagAccumulator, halo: bool):
    from .federated import make_client_model
    from .task import forward
    for client in clients:
        model = make_client_model(cfg, client, "cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        n = client.n_local if halo else len(client.node_indices)
        for batch in client.loaders["validation"]:
            pred = (forward(model, batch, "cpu") * data.target_std + data.target_mean).clamp_min(0.0)[:, :, :n]
            acc.update(pred, batch["y_raw"][:, :, :n], batch["y_mask"][:, :, :n], client.node_indices)


@torch.no_grad()
def halo_audit(state, clients, cfg, data) -> dict:
    """Mecanismo efetivamente usado pelo halo `1hop`, com o checkpoint treinado do próprio halo."""
    from .federated import make_client_model
    from .task import forward
    a = np.asarray(data.adj, dtype=np.float64)
    std = np.where(data.mask, (data.series - data.scaler_mean) / data.scaler_std, 0.0)
    msk = data.mask.astype(np.float64)
    ref = make_model(cfg, data, "cpu")                                 # grafo completo, MESMOS pesos do halo
    ref.load_state_dict(state, strict=True)
    ref.eval()
    full_ds = data_module.WindowDataset(data, "validation")
    windows = np.unique(np.round(np.linspace(0, len(full_ds) - 1, N_HALO_WINDOWS)).astype(int))
    full_w = ref.edge_weight.numpy()
    full_ei = ref.edge_index.numpy()
    full_norm = {(int(s), int(d)): float(w) for s, d, w in zip(full_ei[0], full_ei[1], full_w)}
    res = {"janelas": windows.tolist(), "valores_max_dif": 0.0, "mascara_max_dif": 0.0, "normalizacao_max_dif": 0.0,
           "alinhamento_max_dif": 0.0, "alvo_virtual_observado": 0, "exatidao_max_dif_padronizada": 0.0, "nos_virtuais": 0}
    ref_out = {int(i): forward(ref, {k: v.unsqueeze(0) for k, v in full_ds[int(i)].items() if k in ("x", "x_mask")}, "cpu")[0, :, :, 0]
               for i in windows}                                       # (H, N), escala padronizada
    for client in clients:
        h, nodes, n = client.halo, client.node_indices, client.n_local
        if h is None or h.size == 0:
            continue
        res["nos_virtuais"] += int(h.size)
        local = np.zeros(data.nodes, dtype=bool)
        local[nodes] = True
        ext = a[:, nodes[h.boundary_local]] * (~local)[:, None]      # (N, n_b) pesos externos u -> v
        Wv = ext.sum(0)
        res["normalizacao_max_dif"] = max(res["normalizacao_max_dif"], float(np.abs(Wv - h.weight).max()))
        ds = client.loaders["validation"].dataset
        model = make_client_model(cfg, client, "cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        # normalização: pesos de entrada dos nós de fronteira iguais aos do grafo completo (virtual = Σ externos)
        ei, ew = model.edge_index.numpy(), model.edge_weight.numpy()
        for j, (s_, d_) in enumerate(zip(ei[0], ei[1])):
            if d_ >= n:
                continue
            if s_ < n:
                target = full_norm.get((int(nodes[s_]), int(nodes[d_])))
            else:
                b = int(s_) - n
                v = int(nodes[h.boundary_local[b]])
                target = float(ext[:, b].sum() / a[:, v].sum())
            res["normalizacao_max_dif"] = max(res["normalizacao_max_dif"], abs(float(ew[j]) - (target if target is not None else np.inf)))
        for i in windows:
            o = int(data.origins["validation"][i])
            item = ds[int(i)]
            expect_v = np.einsum("tnf,nb->tbf", std[o - W:o], ext / Wv)
            expect_m = np.einsum("tnf,nb->tbf", msk[o - W:o], ext / Wv)
            res["valores_max_dif"] = max(res["valores_max_dif"], float(np.abs(item["x"][:, n:].numpy() - expect_v).max()))
            res["mascara_max_dif"] = max(res["mascara_max_dif"], float(np.abs(item["x_mask"][:, n:].numpy() - expect_m).max()))
            full_item = full_ds[int(i)]
            res["alinhamento_max_dif"] = max(res["alinhamento_max_dif"], float(
                np.abs(item["x"][:, :n].numpy() - full_item["x"][:, nodes].numpy()).max()))
            res["alvo_virtual_observado"] += int(item["y_mask"][:, n:].sum())
            out = forward(model, {"x": item["x"].unsqueeze(0), "x_mask": item["x_mask"].unsqueeze(0)}, "cpu")[0, :, :n, 0]
            res["exatidao_max_dif_padronizada"] = max(res["exatidao_max_dif_padronizada"],
                                                      float((out - ref_out[int(i)][:, nodes]).abs().max()))
    res["ok"] = bool(res["valores_max_dif"] <= 1e-5 and res["mascara_max_dif"] <= 1e-5 and res["normalizacao_max_dif"] <= 1e-5
                     and res["alinhamento_max_dif"] == 0.0 and res["alvo_virtual_observado"] == 0
                     and res["exatidao_max_dif_padronizada"] <= TOL_HALO)
    return res


def objective_audit(data, k: int, recorded: dict) -> dict:
    from .federated import load_clients
    from .partition import partition_dir
    clients = load_clients(data, partition_dir("sunt", k, 42), BATCH, 0, 7, splits=("validation",))
    ti = data.target_feature_index
    t_end = int(data.origins["validation"].min())
    y = np.where(data.mask[:t_end, :, ti], data.series[:t_end, :, ti], 0.0)
    weights = [c.train_observations for c in clients]
    steps = int(np.ceil(len(data.origins["train"]) / BATCH))
    return {"k": k, "clientes": len(clients), "pesos_iguais_ao_registrado": weights == recorded.get("client_train_observations"),
            "pesos_agregacao_ocorrencias": weights, "nos_por_cliente": [int(len(c.node_indices)) for c in clients],
            "volume_treino_por_cliente": [float(y[:, c.node_indices].sum()) for c in clients],
            "passos_por_epoca_cliente": steps, "passos_por_epoca_centralizado": steps}


# ----------------------------------------------------------------------------------------------
def run(arms: list[str], seeds: list[int], threads: int, force: bool, note: str = ""):
    check_protocols()
    torch.set_num_threads(threads)
    guard_test_split()
    host = socket.gethostname().split(".")[0]
    sums_dir = OUT / "node_horizon_sums"
    sums_dir.mkdir(parents=True, exist_ok=True)
    data = data_module.load_experiment("sunt", W, H, ["n_boardings", "n_alightings"], "n_boardings")
    ov = data.origins["validation"]
    origins_ok = bool(len(np.unique(ov)) == len(ov) and np.all(np.diff(ov) == 1))
    desc = pd.read_csv(DESCRIPTORS)
    assert (desc["nó"].to_numpy() == np.arange(data.nodes)).all()
    strata, member = strata_membership(desc)
    scales = mase_scales(data)
    scales_sha = write_or_verify_scales(data, scales)
    manifest_ckpt = {}
    for f in CKPT_MANIFESTS:
        t = pd.read_csv(f)
        manifest_ckpt.update(dict(zip(t.iloc[:, 0], t["sha256"])))
    rows, audits, objectives = [], {}, {}
    for arm in arms:
        kind, cfg_path, ckpt_dir, metrics_pat, k = ARMS[arm]
        cfg = load_config(cfg_path)
        for seed in seeds:
            t0 = time.time()
            ckpt = ROOT / ckpt_dir.format(s=seed) / "best_model.pt"
            out_npz = sums_dir / f"{arm}_s{seed:03d}.npz"
            row = {"braço": arm, "semente": seed, "host": host, "regra": RULE, "checkpoint": rel(ckpt), "nota": note}
            if out_npz.exists() and not force:
                row.update(status="já existe; não sobrescrito")
                rows.append(row)
                continue
            digest = sha256(ckpt)
            row["checkpoint_sha256_ok"] = manifest_ckpt.get(rel(ckpt)) == digest
            metrics = json.loads((ROOT / metrics_pat.format(s=seed)).read_text())
            best = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = best["model_state_dict"]
            acc = DiagAccumulator(data.nodes, member, scales)
            if kind in ("central", "probe"):
                base_cfg = load_config("config/sunt_probe_boardings.yaml") if kind == "probe" else cfg
                model = make_model(base_cfg, data, "cpu")
                model.load_state_dict(state, strict=True)
                if kind == "probe":
                    # 1–2: reproduz o Centralizado com o grafo original e confere os parâmetros
                    chk = DiagAccumulator(data.nodes, member[:, :0], {})
                    predict_central(model, data, chk, seed)
                    sabs, cnt = chk.per_node()
                    row["sonda_reproduz_original"] = bool(abs(sabs.sum() / cnt.sum() - metrics["validation"]["MAE"]) <= TOL_MAE)
                    row["sonda_parametros_identicos"] = all(torch.equal(model.state_dict()[n], state[n]) for n in state)
                    # 3–4: mesmos parâmetros no operador cortado
                    model = make_model(cfg, data, "cpu")
                    model.load_state_dict(state, strict=True)
                predict_central(model, data, acc, seed)
            else:
                from .partition import partition_dir
                pdir = partition_dir("sunt", k, 42)
                if kind == "fedavg":
                    from .federated import load_clients
                    clients = load_clients(data, pdir, BATCH, 0, seed, splits=("validation",))
                else:
                    from .fedgnnhet import load_halo_clients
                    clients = load_halo_clients(data, pdir, BATCH, 0, seed, "1hop", splits=("validation",))
                predict_clients(state, clients, cfg, data, acc, halo=(kind == "halo"))
                if kind == "halo":
                    audits[f"{arm}_s{seed:03d}"] = halo_audit(state, clients, cfg, data)
                    row["halo_auditoria_ok"] = audits[f"{arm}_s{seed:03d}"]["ok"]
                if kind == "fedavg" and k not in objectives:
                    objectives[k] = objective_audit(data, k, metrics)
            sabs, cnt = acc.per_node()
            mae = sabs.sum() / cnt.sum()
            rec = metrics["validation"]
            row["mae_recalculado"], row["mae_gravado"] = float(mae), float(rec["MAE"])
            if kind == "probe":
                row["reproducao_ok"] = bool(row["sonda_reproduz_original"] and row["sonda_parametros_identicos"])
            else:
                rec_err, rec_cnt = np.asarray(rec["per_node"]["abs_error"]), np.asarray(rec["per_node"]["count"])
                row["contagens_iguais"] = bool(np.array_equal(cnt, rec_cnt))
                row["per_node_max_dif"] = float(np.max(np.abs(sabs - rec_err) - TOL_NODE_REL * np.abs(rec_err)))   # medida v1 (histórico)
                node_ok, row["per_node_dif_MAE_max"] = node_rule_v2(sabs, rec_err, cnt, rec_cnt)
                row["reproducao_ok"] = bool(abs(mae - rec["MAE"]) <= TOL_MAE and row["contagens_iguais"] and node_ok)
            ok = row["reproducao_ok"] and row["checkpoint_sha256_ok"] and row.get("halo_auditoria_ok", True) and origins_ok
            row["segundos"] = round(time.time() - t0, 1)
            row["pico_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 2)
            if ok:
                acc.save(out_npz, {"braço": arm, "semente": seed, "host": host, "checkpoint": rel(ckpt), "checkpoint_sha256": digest,
                                   "estratos": strata, "hgroups": HGROUPS, "hist": [HIST_LO, HIST_HI, HIST_STEP],
                                   "mase_scales_sha256": scales_sha, "dados_fingerprint": data_fingerprint(data)})
                row["status"] = "gravado"
            else:
                row["status"] = "FALHA: diagnóstico não gravado"
            print(json.dumps(row, ensure_ascii=False), flush=True)
            rows.append(row)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for r in rows:
        r["execucao"] = f"run_{host}_{stamp}"
    write_checks(rows, OUT / f"reproduction_checks_{host}_{stamp}.csv")
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    run_log = {"host": host, "plataforma": platform.platform(), "torch": torch.__version__, "threads": threads,
               "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain", "--untracked-files=no"),
               "script_sha256": sha256(Path(__file__)), "protocolo_sha256": PROTOCOL_SHA256, "regra": RULE,
               "protocolo_v1_sha256": PROTOCOL_V1_SHA256, "nota": note,
               "estratos_sha256": sha256(DESCRIPTORS), "origens_validacao_unicas_consecutivas": origins_ok,
               "mase_nos_validos": {lag: int(np.isfinite(s).sum()) for lag, (s, _) in scales.items()},
               "mase_scales_sha256": scales_sha,
               "braços": arms, "sementes": seeds, "linhas": rows, "auditoria_halo": audits,
               "auditoria_objetivos": {str(k): v for k, v in objectives.items()}}
    (OUT / f"run_{host}_{stamp}.json").write_text(json.dumps(run_log, indent=1, ensure_ascii=False, default=str) + "\n")


V1_BLOCKED = {  # histórico: reprovações sob a regra v1 que motivaram a emenda v2
    ("halo_k20", 42): "v1: reprodução por nó reprovada (per_node_max_dif 3,0e-3 > 1e-3); compatível com diferenças numéricas "
                      "entre CPU e GPU; investiga_halo_k20_s042.log",
    ("FedAvg_k166", 123): "v1: reprodução por nó reprovada (per_node_max_dif 1,4e-3 > 1e-3); compatível com diferenças numéricas; "
                          "GPU original (A16) não verificada; investiga_FedAvg_k166_s123.log"}
BLOCKED: dict = {}                                                 # v2: nenhum bloqueio previsto
ARRAYS_NH = ("n", "npos", "nneg", "nzero", "se", "sabs", "se2", "sy")
OK_MEANING = ("ok=True significa que os arquivos esperados passaram, considerando os bloqueios previstos; não implica "
              "que todos os contrastes sejam publicáveis (um contraste exige cinco sementes aprovadas nos dois braços)")


def verify_files(out_dir: Path, arms: list[str], seeds: list[int] | None, *, n_nodes: int, strata_names: list[str],
                 member: np.ndarray, scales: dict, scales_sha: str, data_fp: str, recorded: dict, ckpt_sha: dict,
                 run_rows: dict, strata_sha: str, pairs_override: list | None = None) -> dict:
    """Verificação sem inferência dos npz de `out_dir/node_horizon_sums` (função pura, testável).

    recorded[(braço, semente)] = per_node de validação gravado ({"abs_error", "count"}) ou None (sonda);
    ckpt_sha[(braço, semente)] = sha256 atual do checkpoint; run_rows[(braço, semente)] = linha aprovada do run_*.json,
    com o hash dos estratos da execução de origem em "estratos_sha256_origem"; strata_sha = hash atual dos estratos.
    """
    sums = Path(out_dir) / "node_horizon_sums"
    rep = {"esperados": [], "presentes": [], "ausentes_inesperados": [], "bloqueados": [], "invalidos": [], "arquivos": {}}
    pairs = [(a, s_) for a in arms for s_ in seeds] if pairs_override is None else pairs_override
    for arm, seed in pairs:
        if True:
            key, f = (arm, seed), sums / f"{arm}_s{seed:03d}.npz"
            name = f.name
            rep["esperados"].append(name)
            if key in BLOCKED:
                rep["bloqueados"].append({"arquivo": name, "motivo": BLOCKED[key], "presente": f.exists()})
                if f.exists():
                    rep["invalidos"].append(name)                    # bloqueado não pode ter npz
                continue
            if not f.exists():
                rep["ausentes_inesperados"].append(name)
                continue
            rep["presentes"].append(name)
            try:
                info = _verify_one(f, arm, seed, key, n_nodes=n_nodes, strata_names=strata_names, member=member,
                                   scales=scales, scales_sha=scales_sha, data_fp=data_fp, recorded=recorded,
                                   ckpt_sha=ckpt_sha, run_rows=run_rows, strata_sha=strata_sha)
            except Exception as exc:                                  # arquivo ilegível ou incompleto: registra e segue
                info = {"ok": False, "erro": repr(exc)}
            rep["arquivos"][name] = info
            if not info["ok"]:
                rep["invalidos"].append(name)
    rep["sementes_aprovadas_por_braco"] = {arm: sum(1 for a2, s in pairs if a2 == arm and rep["arquivos"].get(f"{arm}_s{s:03d}.npz", {}).get("ok"))
                                           for arm in dict.fromkeys(a2 for a2, _ in pairs)}
    rep["regra"] = RULE
    rep["ok"] = bool(rep["presentes"]) and not rep["invalidos"] and not rep["ausentes_inesperados"]
    rep["significado_ok"] = OK_MEANING
    return rep


def _verify_one(f: Path, arm: str, seed: int, key, *, n_nodes, strata_names, member, scales, scales_sha, data_fp,
                recorded, ckpt_sha, run_rows, strata_sha) -> dict:
    c = {}
    z = np.load(f, allow_pickle=False)
    required = set(ARRAYS_NH) | {"hist", "meta"} | {f"sabs_scaled_{lag}" for lag in scales}
    missing = sorted(required - set(z.files))
    if missing:
        return {"ok": False, "erro": f"chaves ausentes: {missing}"}
    meta = json.loads(str(z["meta"]))
    if True:
            S = len(strata_names)
            c["dimensoes"] = all(z[k].shape == (n_nodes, H) for k in ARRAYS_NH) and z["hist"].shape == (S, len(HGROUPS) + 1, NB) \
                and all(z[f"sabs_scaled_{lag}"].shape == (n_nodes, H) for lag in scales)
            if not c["dimensoes"]:
                return {"ok": False, **c}
            floats = [z[k] for k in ("se", "sabs", "se2", "sy")] + [z[f"sabs_scaled_{lag}"] for lag in scales]
            c["finitos"] = all(np.isfinite(a).all() for a in floats)
            c["nao_negativos"] = all((z[k] >= 0).all() for k in ("n", "npos", "nneg", "nzero", "sabs", "se2", "sy", "hist")) \
                and all((z[f"sabs_scaled_{lag}"] >= 0).all() for lag in scales)
            c["contagens_somam"] = bool(np.array_equal(z["npos"] + z["nneg"] + z["nzero"], z["n"]))
            # vínculo: identidade, checkpoint, execução aprovada, estratos, escalas e dados
            c["meta_identidade"] = meta.get("braço") == arm and int(meta.get("semente", -1)) == seed
            c["meta_estratos_iguais"] = meta.get("estratos") == strata_names
            c["checkpoint_sha_igual"] = meta.get("checkpoint_sha256") is not None and meta.get("checkpoint_sha256") == ckpt_sha.get(key)
            row = run_rows.get(key)
            c["execucao_aprovada"] = bool(row and row.get("status") == "gravado" and row.get("reproducao_ok") is True)
            c["estratos_sha_origem_igual"] = bool(row) and row.get("estratos_sha256_origem") == strata_sha
            if "mase_scales_sha256" in meta:
                c["escalas_sha_igual"] = meta["mase_scales_sha256"] == scales_sha
                c["dados_fingerprint_igual"] = meta.get("dados_fingerprint") == data_fp
                vinculo = "interno (meta do npz)"
            else:
                vinculo = "externo (manifesto de verificação): npz anterior aos campos de escala/dados"
            # coerência numérica
            for lag, (s_, _) in scales.items():
                ok_nodes = np.isfinite(s_)
                expect = np.where(ok_nodes[:, None], z["sabs"] / np.where(ok_nodes, s_, 1.0)[:, None], 0.0)
                c[f"escalado_{lag}_consistente"] = bool(np.allclose(z[f"sabs_scaled_{lag}"], expect, rtol=1e-9, atol=1e-9))
            # histogramas por estrato e grupo de horizonte = contagens dos nós membros nesses horizontes
            n = z["n"]
            per_group = np.stack([n[:, lo:hi].sum(1) for lo, hi in HGROUPS] + [n.sum(1)], axis=1)   # (N, G+1)
            expect_hist = member.T.astype(np.int64) @ per_group                                      # (S, G+1)
            c["hist_por_estrato_e_horizonte"] = bool(np.array_equal(z["hist"].sum(2), expect_hist))
            # reconfirma o per_node gravado (exceto sondas)
            rec = recorded.get(key)
            if rec is not None:
                b, cb = np.asarray(rec["abs_error"], dtype=np.float64), np.asarray(rec["count"])
                a_, ca = z["sabs"].sum(1), z["n"].sum(1)
                c["per_node_contagens"] = bool(np.array_equal(ca, cb))
                c["per_node_somas"], dif = node_rule_v2(a_, b, ca, cb)
                c["per_node_dif_MAE_max"] = dif
            ok = all(v for v in c.values() if isinstance(v, bool))
            return {"ok": ok, "regra": RULE, "vinculo": vinculo, "npz_sha256": sha256(f), "checkpoint": meta.get("checkpoint"),
                    "checkpoint_sha256": meta.get("checkpoint_sha256"), "execucao": (row or {}).get("execucao"),
                    "estratos_sha256_origem": (row or {}).get("estratos_sha256_origem"), "estratos_sha256_atual": strata_sha,
                    "mase_scales_sha256_usado": meta.get("mase_scales_sha256", scales_sha), **c}


def verify(arms: list[str], seeds: list[int], pairs: list | None = None):
    """Confere os npz já gravados (sem inferência), reconstrói os CSVs a partir dos run_*.json e grava um manifesto
    de verificação. Termina com código de erro se houver arquivo inválido, ausência inesperada ou nenhum arquivo."""
    check_protocols()
    if pairs:
        arms = list(dict.fromkeys(a for a, _ in pairs))
    data = data_module.load_experiment("sunt", W, H, ["n_boardings", "n_alightings"], "n_boardings")
    scales = mase_scales(data)
    scales_sha = write_or_verify_scales(data, scales)
    desc = pd.read_csv(DESCRIPTORS)
    strata, member = strata_membership(desc)
    host = socket.gethostname().split(".")[0]
    run_rows, csvs = {}, []
    for j in sorted(OUT.glob("run_*.json")):
        r = json.loads(j.read_text())
        rows = [dict(row, execucao=j.stem) for row in r["linhas"]]
        for row in rows:
            if row.get("status") == "gravado":
                run_rows[(row["braço"], int(row["semente"]))] = dict(row, estratos_sha256_origem=r.get("estratos_sha256"))
        out = OUT / f"reproduction_checks_{j.stem.removeprefix('run_')}.csv"
        write_checks(rows, out)
        csvs.append(out.name)
    recorded, ckpt_sha = {}, {}
    for arm, seed in (pairs or [(a, s_) for a in arms for s_ in seeds]):
        kind, _, ckpt_dir, metrics_pat, _ = ARMS[arm]
        if True:
            ckpt = ROOT / ckpt_dir.format(s=seed) / "best_model.pt"
            ckpt_sha[(arm, seed)] = sha256(ckpt) if ckpt.exists() else None
            recorded[(arm, seed)] = None if kind == "probe" else json.loads((ROOT / metrics_pat.format(s=seed)).read_text())["validation"]["per_node"]
    rep = verify_files(OUT, arms, seeds, n_nodes=data.nodes, strata_names=strata, member=member, scales=scales,
                       scales_sha=scales_sha, data_fp=data_fingerprint(data), recorded=recorded, ckpt_sha=ckpt_sha, run_rows=run_rows,
                       strata_sha=sha256(DESCRIPTORS), pairs_override=pairs)
    rep.update(host=host, protocolo_sha256=PROTOCOL_SHA256, mase_scales_sha256=scales_sha, estratos_sha256=sha256(DESCRIPTORS), script_sha256=sha256(Path(__file__)),
               csv_reconstruidos=csvs)
    legacy = OUT / f"reproduction_checks_{host}.csv"
    if legacy.exists():
        kept = legacy.with_suffix(".legado.csv")
        legacy.rename(kept)
        rep["csv_legado_preservado_como"] = kept.name
    stamp = time.strftime("%Y%m%dT%H%M%S")
    (OUT / f"verify_{host}_{stamp}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: rep[k] for k in ("ok", "presentes", "ausentes_inesperados", "bloqueados", "invalidos",
                                          "sementes_aprovadas_por_braco")}, ensure_ascii=False, indent=1))
    if not rep["ok"]:
        sys.exit(1)


# ----------------------------------------------------------------------------------------------
# análise (local, sem inferência): junta os npz aprovados das duas máquinas
# ----------------------------------------------------------------------------------------------
ARM_BLOCK = {"Centralizado": (20, 166), "sem_arestas": (20, 166), "cortes_k20": (20,), "FedAvg_k20": (20,), "halo_k20": (20,),
             "sonda_k20": (20,), "cortes_k166": (166,), "FedAvg_k166": (166,), "halo_k166": (166,), "sonda_k166": (166,)}
CONTRASTS = [  # (A, B, papel, k) — exatamente os do protocolo; sem_arestas só tem métricas por braço
    ("FedAvg_k20", "cortes_k20", "principal", 20), ("FedAvg_k166", "cortes_k166", "principal", 166),
    ("cortes_k20", "Centralizado", "secundário", 20), ("cortes_k166", "Centralizado", "secundário", 166),
    ("halo_k20", "FedAvg_k20", "secundário", 20), ("halo_k166", "FedAvg_k166", "secundário", 166),
    ("FedAvg_k20", "Centralizado", "secundário", 20), ("FedAvg_k166", "Centralizado", "secundário", 166),
    ("sonda_k20", "Centralizado", "secundário", 20), ("sonda_k166", "Centralizado", "secundário", 166)]
HLABELS = ["1–6", "7–12", "13–18", "19–24", "todos"]
SMALL_CELL = 30


def cell_metrics(z: dict, sel: np.ndarray, hr: tuple[int, int], elig: dict, scales: dict, volume: np.ndarray) -> dict:
    """Métricas de um estrato (máscara de nós `sel`) num intervalo de horizontes [lo, hi). Denominador nulo → None."""
    lo, hi = hr
    take = lambda k: z[k][sel, lo:hi]
    n_all = z["n"][:, lo:hi].sum(1)                                   # por nó, todos os nós (para máscaras globais)
    n = int(n_all[sel].sum())
    sabs, se, sy = float(take("sabs").sum()), float(take("se").sum()), float(take("sy").sum())
    out = {"nos": int(sel.sum()), "nos_avaliaveis": int((sel & (n_all > 0)).sum()), "n": n,
           "MAE": sabs / n if n else None, "vies_medio": se / n if n else None,
           "prop_positiva": float(take("npos").sum()) / n if n else None, "prop_zero": float(take("nzero").sum()) / n if n else None,
           "WAPE": sabs / sy if sy > 0 else None}
    for lag in scales:
        e = sel & elig[lag]
        n_e = int(z["n"][e, lo:hi].sum())
        s_e = np.where(elig[lag], scales[lag], 0.0)
        out[f"MASE{lag}"] = float(z[f"sabs_scaled_{lag}"][e, lo:hi].sum()) / n_e if n_e else None
        den = float((z["n"][e, lo:hi].sum(1) * s_e[e]).sum())
        out[f"razao_somas_escalada{lag}"] = float(z["sabs"][e, lo:hi].sum()) / den if den > 0 else None
        out[f"MASE{lag}_nos_excluidos"] = int((sel & ~elig[lag] & (n_all > 0)).sum())
        out[f"MASE{lag}_frac_alvos_excluidos"] = (n - n_e) / n if n else None
    v = sel & (volume > 0)
    n_v = int(z["n"][v, lo:hi].sum())
    out["MAE_norm_volume_treino"] = float((z["sabs"][v, lo:hi].sum(1) / volume[v]).sum()) / n_v if n_v else None
    out["MAE_norm_volume_nos_excluidos"] = int((sel & ~(volume > 0) & (n_all > 0)).sum())
    return out


def compute_tables(files: dict, strata_names: list[str], member: np.ndarray, scales: dict, volume: np.ndarray):
    """files[(braço, semente)] = dict de arrays do npz aprovado. Devolve (métricas por braço, contrastes, bloqueados)."""
    elig = {lag: np.isfinite(s) for lag, s in scales.items()}
    sc = {lag: np.where(np.isfinite(s), s, 0.0) for lag, s in scales.items()}
    ref_n = next(iter(files.values()))["n"]
    for key, z in files.items():                                   # mesmos alvos em todos os braços e sementes
        if not np.array_equal(z["n"], ref_n):
            raise ValueError(f"contagens diferentes em {key}: braços não comparáveis")
    rows = []
    for (arm, seed), z in sorted(files.items()):
        for k in ARM_BLOCK[arm]:
            for si, name in enumerate(strata_names):
                if not name.startswith(f"k{k}|"):
                    continue
                sel = member[:, si]
                ranges = [(HLABELS[g], HGROUPS[g] if g < len(HGROUPS) else (0, H), g) for g in range(len(HLABELS))]
                ranges += [(f"h={h + 1:02d}", (h, h + 1), None) for h in range(H)]
                for lab, hr, g in ranges:
                    m = cell_metrics(z, sel, hr, elig, sc, volume)
                    med, over = hist_median(z["hist"][si, g]) if g is not None else (None, None)
                    rows.append({"braço": arm, "semente": seed, "k": k, "estrato": name.split("|", 1)[1], "horizonte": lab,
                                 **m, "mediana": med, "frac_transbordamento": over,
                                 "celula_pequena": m["nos_avaliaveis"] < SMALL_CELL})
    sm = pd.DataFrame(rows)
    from scipy import stats as st
    con, blocked = [], []
    affected = {}                                                   # braço -> sementes reprovadas na v1 (histórico)
    for (arm, seed) in V1_BLOCKED:
        affected.setdefault(arm, set()).add(seed)
    for a, b, papel, k in CONTRASTS:
        seeds_a = {s for (arm, s) in files if arm == a}
        seeds_b = {s for (arm, s) in files if arm == b}
        if not (set(SEEDS) <= seeds_a and set(SEEDS) <= seeds_b):
            blocked.append({"contraste": f"{a} − {b}", "papel": papel, "k": k, "sementes_A": sorted(seeds_a),
                            "sementes_B": sorted(seeds_b), "motivo": "exige cinco sementes aprovadas nos dois braços"})
            continue
        A = sm[(sm["braço"] == a) & (sm.k == k)].set_index(["estrato", "horizonte", "semente"])
        B = sm[(sm["braço"] == b) & (sm.k == k)].set_index(["estrato", "horizonte", "semente"])
        variants = [("principal (5 sementes)", list(SEEDS))]
        excl = affected.get(a, set()) | affected.get(b, set())
        if excl:                                                   # suplementar: os MESMOS quatro pares nos dois braços
            variants.append((f"suplementar (4 pares, sem s{'/s'.join(str(x) for x in sorted(excl))})",
                             [s_ for s_ in SEEDS if s_ not in excl]))
        for analise, seeds in variants:
            n_p = len(seeds)
            tcrit = float(st.t.ppf(0.975, n_p - 1))                 # 4 g.l. com 5 pares; 3 g.l. com 4 pares
            for (estr, hor), _ in A.groupby(level=[0, 1]):
                ia, ib = A.loc[(estr, hor)], B.loc[(estr, hor)]

                def summarize(vals, metric):
                    v = np.array([x if x is not None else np.nan for x in vals], dtype=float)
                    vals_by_seed = {f"s{s_}": None for s_ in SEEDS}
                    if np.isnan(v).any():
                        return {"metrica": metric, **vals_by_seed, "n_pares": n_p, "gl": n_p - 1, "media": None, "dp": None,
                                "IC95_inf": None, "IC95_sup": None, "indefinido": True}
                    vals_by_seed.update({f"s{s_}": float(x) for s_, x in zip(seeds, v)})
                    mean, sd = v.mean(), v.std(ddof=1)
                    half = tcrit * sd / np.sqrt(n_p)
                    return {"metrica": metric, **vals_by_seed, "n_pares": n_p, "gl": n_p - 1, "media": float(mean),
                            "dp": float(sd), "IC95_inf": float(mean - half), "IC95_sup": float(mean + half), "indefinido": False}

                base = {"contraste": f"{a} − {b}", "papel": papel, "analise": analise, "k": k, "estrato": estr, "horizonte": hor,
                        "nos_avaliaveis": int(ia.loc[seeds[0], "nos_avaliaveis"]), "celula_pequena": bool(ia.loc[seeds[0], "celula_pequena"])}
                pick = lambda frame, col: [frame.loc[s_, col] for s_ in seeds]
                diff = lambda xa, xb: [x - y if None not in (x, y) else None for x, y in zip(xa, xb)]
                mae_a, mae_b = pick(ia, "MAE"), pick(ib, "MAE")
                con.append({**base, **summarize(diff(mae_a, mae_b), "dif_MAE")})
                con.append({**base, **summarize([(x - y) / y if None not in (x, y) and y != 0 else None for x, y in zip(mae_a, mae_b)],
                                                "deterioracao_relativa_MAE")})
                for lag in scales:
                    con.append({**base, **summarize(diff(pick(ia, f"MASE{lag}"), pick(ib, f"MASE{lag}")), f"dif_MASE{lag}")})
                con.append({**base, **summarize(diff(pick(ia, "vies_medio"), pick(ib, "vies_medio")), "dif_vies_medio")})
    return sm, pd.DataFrame(con), blocked


def analyze(sources: list[Path]):
    """Junta os npz aprovados (cada fonte com node_horizon_sums/ e verify_*.json) e grava as tabelas da análise."""
    check_protocols()
    files, origin = {}, {}
    scales_shas = set()
    for src in sources:
        src = Path(src)
        approved = {}
        for v in sorted(src.glob("verify_*.json")):                   # todos os manifestos v2 da fonte; o mais recente prevalece
            rep = json.loads(v.read_text())
            if rep.get("regra") != RULE or rep.get("protocolo_sha256") != PROTOCOL_SHA256:
                continue                                              # manifestos da v1 ficam como histórico
            if not rep.get("ok"):
                sys.exit(f"{v}: verificação v2 reprovada")
            scales_shas.add(rep["mase_scales_sha256"])
            if rep.get("estratos_sha256") != sha256(DESCRIPTORS):
                sys.exit(f"{v}: estratos verificados diferem dos atuais")
            for name, info in rep["arquivos"].items():
                approved[name] = (info, v)
        if not approved:
            sys.exit(f"{src}: nenhum manifesto de verificação v2 aprovado")
        for name, (info, ver_file) in approved.items():
            ver = [ver_file]
            if not info.get("ok"):
                continue
            f = src / "node_horizon_sums" / name
            if sha256(f) != info["npz_sha256"]:
                sys.exit(f"{f}: hash difere do manifesto de verificação")
            arm, seed = name[:-4].rsplit("_s", 1)
            key = (arm, int(seed))
            if key in files:
                sys.exit(f"{name} aparece em mais de uma fonte")
            files[key] = dict(np.load(f, allow_pickle=False))
            origin[key] = {"fonte": rel(f), "npz_sha256": info["npz_sha256"], "verificacao": rel(ver[-1])}
    if len(scales_shas) != 1:
        sys.exit(f"manifestos com escalas do MASE diferentes: {scales_shas}")
    for src in sources:
        if sha256(src / "mase_scales.npz") not in scales_shas:
            sys.exit(f"{src}: mase_scales.npz difere do registrado nos manifestos de verificação")
    gc_manifest = json.loads((ROOT / "results/sunt_graph_controls/v1/manifest.json").read_text())
    if gc_manifest["node_descriptors"]["sha256"] != sha256(DESCRIPTORS):
        sys.exit("node_descriptors.csv difere do registrado no manifesto dos controles de grafo")
    sc = np.load(sources[0] / "mase_scales.npz", allow_pickle=False)
    for src in sources[1:]:
        other = np.load(src / "mase_scales.npz", allow_pickle=False)
        if str(other["fingerprint"]) != str(sc["fingerprint"]) or not all(
                np.array_equal(sc[f"s_{lag}"], other[f"s_{lag}"], equal_nan=True) for lag in MASE_LAGS):
            sys.exit("escalas do MASE diferem entre as máquinas")
    scales = {lag: sc[f"s_{lag}"] for lag in MASE_LAGS}
    desc = pd.read_csv(DESCRIPTORS)
    strata, member = strata_membership(desc)
    volume = desc["volume"].to_numpy(dtype=float)
    sm, con, blocked = compute_tables(files, strata, member, scales, volume)
    dest = OUT / "analysis"
    dest.mkdir(exist_ok=True)
    sm.to_csv(dest / "stratum_metrics.csv", index=False, float_format="%.10g")
    con.to_csv(dest / "contrasts.csv", index=False, float_format="%.10g")
    manifest = {"protocolo_sha256": PROTOCOL_SHA256, "protocolo_v1_sha256": PROTOCOL_V1_SHA256, "regra": RULE,
                "historico_v1_reprovacoes": {f"{a}_s{s:03d}": m for (a, s), m in V1_BLOCKED.items()},
                "escalas_sha256": next(iter(scales_shas)), "descritores_sha256": sha256(DESCRIPTORS), "script_sha256": sha256(Path(__file__)), "estratos_sha256": sha256(DESCRIPTORS),
                "mase_fingerprint": str(sc["fingerprint"]), "fontes": [rel(s) for s in sources],
                "arquivos": {f"{a}_s{s:03d}": v for (a, s), v in sorted(origin.items())},
                "sementes_por_braco": {a: sorted(s for (x, s) in files if x == a) for a in sorted({a for a, _ in files})},
                "contrastes_bloqueados": blocked, "bloqueios_previstos": {f"{a}_s{s:03d}": m for (a, s), m in BLOCKED.items()}}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(files)} npz; {len(sm)} linhas de métricas; {len(con)} linhas de contrastes; bloqueados: {[b['contraste'] for b in blocked]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", required=True, choices=list(ARMS))
    r.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--force", action="store_true", help="regrava npz existentes (não usar nos resultados finais)")
    r.add_argument("--note", default="", help="nota registrada em cada linha (ex.: mudança de máquina)")
    v = sub.add_parser("verify", help="confere os npz gravados, reconstrói os CSVs e grava o manifesto de verificação")
    v.add_argument("--arms", nargs="+", default=[], choices=list(ARMS))
    v.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    v.add_argument("--pairs", nargs="+", default=None, help="pares braço:semente explícitos (substituem --arms × --seeds)")
    an = sub.add_parser("analyze", help="junta os npz aprovados das máquinas e grava as tabelas (sem inferência)")
    an.add_argument("--sources", nargs="+", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "analyze":
        analyze([ROOT / p if not p.is_absolute() else p for p in a.sources])
        return
    if a.cmd == "run":
        run(a.arms, a.seeds, a.threads, a.force, a.note)
    elif a.cmd == "verify":
        if not a.arms and not a.pairs:
            ap.error("verify exige --arms ou --pairs")
        verify(a.arms, a.seeds, [(x.split(":")[0], int(x.split(":")[1])) for x in a.pairs] if a.pairs else None)


if __name__ == "__main__":
    main()
