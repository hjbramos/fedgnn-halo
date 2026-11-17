"""Exportador do pacote de resultados da dissertação, defesa e artigo: `generated-traffic/`.

Fonte exclusiva: os metrics.json de `results/` deste repositório. Não treina, não reavalia checkpoints e não
toca no `generated/` dos documentos. Passos:

1. Seleciona explicitamente as células (CELLS): dataset, pasta, k, variante, papel e config de origem.
2. Valida cada registro: sementes {7, 42, 101, 123, 314}, teste com MAE finito, metadados coerentes com o papel
   (`inventory.metadata_issues`), nós/janela/horizonte únicos por dataset e alvo, e orçamento do SUNT.
3. Escreve CSVs por semente e por célula, diferenças pareadas, tabelas LaTeX (fragmentos `tabular`, sem
   legenda: cada documento põe a sua), macros, figuras (PNG e SVG) e `provenance.json` (caminho e SHA-256 de
   cada fonte e config, versão do exportador, commit, hardware por registro).
4. Confere os números exportados contra os arquivos de origem e gera o pacote duas vezes, exigindo o mesmo
   conteúdo (exceto o carimbo de data do provenance). Qualquer falha deixa o pacote em
   `generated-traffic-INVALIDO/` e termina com código 1; só um pacote validado vai para `generated-traffic/`.

Registros não são treinamentos independentes: nas células do SUNT avaliadas a partir de sondagens cegas, o
registro final é a avaliação do checkpoint da sondagem (`evaluated_from`), não um novo treino; as sondagens
aparecem só em `dados/treinos_de_sondagem.csv`. Dados ausentes ficam ausentes (vazio no CSV, "---" na tabela);
hardware não gravado fica como "não registrado". Hashes atestam integridade, não comparabilidade.

    python -m experiments.export_traffic
"""
from __future__ import annotations

import hashlib
import json
import re
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .inventory import SEEDS, metadata_issues, role
from .train import ROOT
from .export_sunt_analysis import export as export_sunt_analysis
from .export_sunt_direcao import export as export_sunt_direcao
from .export_controles_internos import export as export_controles_internos
from .convergencia_figuras import (
    CHECKPOINT_ATOL,
    CHECKPOINT_RTOL,
    STEM_FEDERADOS,
    STEM_LOCAL,
    export as export_sunt_evolucao,
)

EXPORTER_VERSION = "1.29.0"
MANIFEST = ROOT / "results" / "checkpoints_sha256.csv"   # hashes dos best_model.pt (gerado nas máquinas de treino)
TOL = 1e-9          # tolerância relativa ao comparar números exportados (CSV com 17 algarismos significativos)
OUT = ROOT / "generated-traffic"
INVALID = ROOT / "generated-traffic-INVALIDO"
SUNT_BUDGET = {None: 600, 6: 600, 20: 600, 166: 1000}
# Grafo distribuído com o SUNT, derivado dos itinerários planejados do GTFS (1.26.0). Insumo fixo, fora do git (data/):
# baixar de SUNT_GTFS_URL e conferir o SHA-256. Só descreve a cobertura em relação ao grafo do estudo; não é usado no treino.
SUNT_GTFS_GRAPH = ROOT / "data" / "sunt" / "raw" / "graph_gtfs" / "graph_gtfs_fev_2024.parquet"
SUNT_GTFS_COMMIT = "f8d90fb38f3b1a6f74a18f0298d72fe0c6eaa54c"
SUNT_GTFS_URL = (f"https://raw.githubusercontent.com/LabIA-UFBA/SUNT/{SUNT_GTFS_COMMIT}/"
                 "data/graph_designer/graph_gtfs_fev_2024.parquet")
SUNT_GTFS_SHA256 = "106d445b118f94b9cb81a0e4b21cd7c0e9a5d33ed1ba16ba8382ac9dadf42524"


@dataclass
class Cell:
    id: str
    dataset: str
    folder: str
    k: int | None
    variant: str
    method: str          # rótulo do método nas tabelas
    group: str           # principal | controle | referência | complementar
    config: str          # arquivo de config (versão atual)
    optional: bool = False   # em execução no momento da exportação: ausência não invalida o pacote
    extra_configs: list[str] = field(default_factory=list)


def _cells() -> list[Cell]:
    c: list[Cell] = []
    for ds, tag in (("metr-la", "metrla"), ("pems-bay", "pemsbay")):
        c += [Cell(f"{tag}_cent", ds, "centralized", None, "", "Centralizado", "principal", "config/centralized.yaml"),
              Cell(f"{tag}_noedges", ds, "centralized_noedges", None, "", "Centralizado sem conexões", "controle",
                   "config/centralized_noedges.yaml"),
              # Controle da regra de parada (1.22.0): os dois centralizados das rodovias com limite 400 e paciência 30,
              # a regra das condições federadas. Protocolo: results/centralized_parada_federada/PROTOCOLO.md.
              Cell(f"{tag}_centpf", ds, "centralized_parada_federada", None, "", "Centralizado (parada federada)",
                   "controle", "config/centralized_parada_federada.yaml"),
              Cell(f"{tag}_noedgespf", ds, "centralized_noedges_parada_federada", None, "",
                   "Centralizado sem conexões (parada federada)", "controle",
                   "config/centralized_noedges_parada_federada.yaml")]
        c += [Cell(f"{tag}_fedavg_k{k}", ds, "fedavg_leconv", k, "", "FedAvg", "principal", "config/fedavg_leconv.yaml")
              for k in (5, 10, 20)]
        c += [Cell(f"{tag}_fedprox_k{k}", ds, "fedprox_leconv", k, "proximal_mu=0.01", "FedProx", "principal",
                   "config/fedprox_leconv.yaml") for k in (5, 10, 20)]          # k=5/10: lote 1 de lacunas (1.9.0)
    c += [Cell(f"metrla_halo_k{k}", "metr-la", "fedgnnhet_leconv", k, "halo=1hop", "FedGNN-Halo", "principal",
               "config/fedgnnhet_leconv.yaml") for k in (5, 10, 20)]
    c += [Cell(f"metrla_halonone_k{k}", "metr-la", "fedgnnhet_leconv", k, "halo=none", "FedAvg (ablação)", "controle",
               "config/fedgnnhet_leconv.yaml") for k in (5, 10, 20)]
    c += [Cell("metrla_halomin2_k20", "metr-la", "fedgnnhet_leconv", 20, "halo=1hop_min2", "FedGNN-Halo (dois ou mais vizinhos)",
               "controle", "config/fedgnnhet_leconv.yaml")]
    c += [Cell(f"metrla_ditto_k{k}", "metr-la", "ditto_leconv", k, "ditto_lambda=per_client", "Ditto", "principal",
               "config/ditto_leconv.yaml") for k in (5, 10, 20)]
    c += [Cell(f"metrla_local_k{k}", "metr-la", "ditto_leconv", k, "ditto_lambda=0", "Local", "referência",
               "config/ditto_leconv.yaml") for k in (5, 10, 20)]
    c += [Cell(f"metrla_dittofix{l.replace('.', 'p')}_k{k}", "metr-la", "ditto_leconv", k, f"ditto_lambda={l}",
               f"Ditto $\\lambda={l}$ (controle)", "controle", "config/ditto_leconv.yaml")
          for k in (5, 10, 20) for l in ("0.01", "0.1", "1")]
    c += [Cell(f"pemsbay_halo_k{k}", "pems-bay", "fedgnnhet_leconv", k, "halo=1hop", "FedGNN-Halo", "principal",
               "config/fedgnnhet_leconv.yaml") for k in (5, 10, 20)]            # k=5/10: lote 1 de lacunas (1.9.0)
    # Nova execução sem halo: descrição das cinco sementes, sem alterar as famílias existentes.
    c += [Cell("pemsbay_halonone_k5", "pems-bay", "controle_pems_none_k5_20261115", 5,
               "halo=none", "FedAvg (sem halo)", "controle",
               "config/completar_controles_20261114/pems-bay_none_k5.yaml")]
    c += [Cell(f"pemsbay_ditto_k{k}", "pems-bay", "ditto_leconv", k, "ditto_lambda=per_client", "Ditto", "principal",
               "config/ditto_leconv.yaml") for k in (5, 10, 20)]  # k=20: lote 3 (1.10.0); k=10: lote 4 (1.15.0)
    c += [Cell(f"pemsbay_local_k{k}", "pems-bay", "ditto_leconv", k, "ditto_lambda=0", "Local", "referência",
               "config/ditto_leconv.yaml") for k in (5, 10, 20)]                 # lote 2 de lacunas (1.10.0)
    c += [Cell("sunt_cent", "sunt", "sunt_centralized_boardings", None, "", "Centralizado", "principal",
               "config/sunt_centralized_boardings.yaml", extra_configs=["config/sunt_probe_boardings.yaml"])]
    c += [Cell(f"sunt_fedavg_k{k}", "sunt", "sunt_fedavg_final", k, "", "FedAvg", "principal",
               "config/sunt_fedavg_final.yaml", extra_configs=["config/sunt_fedavg_probe.yaml"]) for k in (6, 20, 166)]
    c += [Cell(f"sunt_halo_k{k}", "sunt", "sunt_fedgnnhet_final", k, "halo=1hop", "FedGNN-Halo", "principal",
               "config/sunt_fedgnnhet_final_desktop.yaml" if k in (6, 20) else "config/sunt_fedgnnhet_final.yaml")
          for k in (6, 20, 166)]
    c += [Cell(f"sunt_fedprox_k{k}", "sunt", "sunt_fedprox_final", k, "proximal_mu=0.01", "FedProx", "principal",
               "config/sunt_fedprox_final.yaml") for k in (6, 20, 166)]                # k=166: lote 5 (1.16.0)
    c += [Cell(f"sunt_ditto_k{k}", "sunt", "sunt_ditto_final", k, "ditto_lambda=per_client", "Ditto", "principal",
               "config/sunt_ditto_final_desktop.yaml" if k == 6 else "config/sunt_ditto_final.yaml")
          for k in (6, 20, 166)]                                                        # k=166: lote 5 (1.16.0)
    c += [Cell(f"sunt_local_k{k}", "sunt", "sunt_local_final", k, "ditto_lambda=0", "Local", "referência",
               "config/sunt_local_final.yaml") for k in (6, 20, 166)]                   # lote 6 (1.20.0)
    c += [Cell("carga_cent", "sunt", "sunt_loading_directed_centralized", None, "", "Centralizado", "complementar",
               "config/sunt_loading_directed_centralized.yaml"),
          Cell("carga_fedavg_k20", "sunt", "sunt_loading_directed_fedavg", 20, "", "FedAvg", "complementar",
               "config/sunt_loading_directed_fedavg.yaml"),
          Cell("carga_halo_k20", "sunt", "sunt_loading_directed_fedgnnhet", 20, "halo=1hop", "FedGNN-Halo",
               "complementar", "config/sunt_loading_directed_fedgnnhet_desktop.yaml", optional=True),
          Cell("sunt_fedpub_k20", "sunt", "sunt_fedpub_final", 20, "lambda_l1=1e-05", "FED-PUB (adaptação)",
               "complementar", "config/sunt_fedpub_final.yaml", optional=True)]
    return c


# Comparações pareadas por semente: (família, célula A, célula B). Diferença = A − B (negativo favorece A).
def _comparisons() -> list[tuple[str, str, str]]:
    p = []
    for k in (6, 20, 166):
        f = f"SUNT k={k}"
        p += [(f, f"sunt_fedavg_k{k}", "sunt_cent"), (f, f"sunt_halo_k{k}", f"sunt_fedavg_k{k}")]
        p += [(f, f"sunt_fedprox_k{k}", f"sunt_fedavg_k{k}"), (f, f"sunt_ditto_k{k}", f"sunt_fedavg_k{k}")]  # k=166: 1.16.0
        p += [(f, f"sunt_local_k{k}", f"sunt_fedavg_k{k}"), (f, f"sunt_local_k{k}", f"sunt_halo_k{k}")]      # lote 6 (1.20.0)
    for k in (5, 10, 20):
        f = f"METR-LA k={k}"
        p += [(f, f"metrla_fedavg_k{k}", "metrla_cent"), (f, f"metrla_halo_k{k}", f"metrla_fedavg_k{k}"),
              (f, f"metrla_halo_k{k}", f"metrla_halonone_k{k}"), (f, f"metrla_ditto_k{k}", f"metrla_fedavg_k{k}"),
              (f, f"metrla_local_k{k}", f"metrla_fedavg_k{k}"), (f, f"metrla_local_k{k}", f"metrla_halo_k{k}")]
    p += [(f"METR-LA k={k}", f"metrla_fedprox_k{k}", f"metrla_fedavg_k{k}") for k in (5, 10, 20)]   # k=5/10: 1.9.0
    p += [("METR-LA k=20", "metrla_halomin2_k20", "metrla_fedavg_k20")]
    for k in (5, 10, 20):
        p += [(f"PEMS-BAY k={k}", f"pemsbay_fedavg_k{k}", "pemsbay_cent")]
    for k in (5, 10, 20):                                                    # k=5/10: lote 1 de lacunas (1.9.0)
        p += [(f"PEMS-BAY k={k}", f"pemsbay_halo_k{k}", f"pemsbay_fedavg_k{k}"),
              (f"PEMS-BAY k={k}", f"pemsbay_fedprox_k{k}", f"pemsbay_fedavg_k{k}")]
    for k in (5, 10, 20):                                                    # lote 2 de lacunas (1.10.0)
        p += [(f"PEMS-BAY k={k}", f"pemsbay_local_k{k}", f"pemsbay_fedavg_k{k}"),
              (f"PEMS-BAY k={k}", f"pemsbay_local_k{k}", f"pemsbay_halo_k{k}")]
    for name, tag in (("METR-LA", "metrla"), ("PEMS-BAY", "pemsbay")):        # controle da regra de parada (1.22.0)
        p += [(f"{name} (regra de parada)", f"{tag}_centpf", f"{tag}_cent"),
              (f"{name} (regra de parada)", f"{tag}_noedgespf", f"{tag}_centpf")]
    p += [("PEMS-BAY k=20", "pemsbay_ditto_k20", "pemsbay_fedavg_k20")]       # lote 3 de lacunas (1.10.0)
    p += [("PEMS-BAY k=10", "pemsbay_ditto_k10", "pemsbay_fedavg_k10")]       # lote 4 de lacunas (1.15.0)
    p += [("PEMS-BAY k=5", "pemsbay_ditto_k5", "pemsbay_fedavg_k5"),
          ("Carga + grafo direcionado k=20", "carga_fedavg_k20", "carga_cent"),
          ("Carga + grafo direcionado k=20", "carga_halo_k20", "carga_fedavg_k20"),
          ("SUNT k=20 (complementar)", "sunt_fedpub_k20", "sunt_fedavg_k20")]
    return p


def inferential_family(a: str, b: str) -> str:
    """Família de ajuste de multiplicidade (Holm), definida RETROSPECTIVAMENTE, depois de conhecer os resultados
    (revisão externa de 04/10). Os p nominais continuam no CSV; os IC95% não são ajustados."""
    ds = {"sunt": "SUNT", "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}.get(a.split("_")[0], "")
    if a.endswith("pf") or b.endswith("pf"):      # controle da regra de parada: família própria por conjunto
        return f"controle: regra de parada federada ({ds})"
    if a.startswith(("carga_", "sunt_fedpub")):
        return "complementar: FED-PUB (SUNT k=20)" if a.startswith("sunt_fedpub") else "complementar: carga + grafo direcionado"
    if "_halo_k" in a and "_fedavg_k" in b:
        return "principal: halo × FedAvg (9 contrastes)"
    if "_halonone_k" in b:
        return "secundária: halo × none (METR-LA)"
    if "_fedavg_k" in a and b.endswith("_cent"):
        return f"caracterização: FedAvg × Centralizado ({ds})"
    return f"exploratória: comparadores ({ds})"


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, run, out = len(p), 0.0, {}
    for i, idx in enumerate(order):
        run = max(run, (m - i) * float(p[idx]))
        out[idx] = min(1.0, run)
    return pd.Series(out).loc[p.index]


def evidence(diff: float, p_nom: float, p_adj: float) -> str:
    """Do ponto de vista de A (negativo = A com menor MAE): melhora ou piora, observada ou com evidência."""
    if diff == 0:
        return "sem diferença observada"
    sense = "melhora" if diff < 0 else "piora"
    if p_adj < 0.05:
        return f"{sense} com evidência após ajuste"
    if p_nom < 0.05:
        return f"{sense} com evidência nominal"
    return f"{sense} observada"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variant(m: dict) -> str:
    for key in ("halo", "proximal_mu", "ditto_lambda", "lambda_l1"):
        if m.get(key) is not None:
            return f"{key}={m[key]}"
    return ""


def _environment(gpu) -> str:
    if not gpu:
        return "não registrado"
    if "RTX" in gpu:
        return "estação de trabalho"
    return "servidor do laboratório"


@lru_cache(maxsize=1)
def _manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    m = pd.read_csv(MANIFEST)
    return dict(zip(m.caminho, m.sha256))


@lru_cache(maxsize=None)
def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def results_commit(path: str) -> str:
    """Commit que versionou o metrics.json (o mais antigo que o adicionou)."""
    out = _git("log", "--diff-filter=A", "--format=%H", "--", path).splitlines()
    return out[-1] if out else ""


def config_at(commit: str, config: str) -> str:
    """SHA-256 da config na revisão do commit que versionou o resultado (aproximação da versão usada)."""
    if not commit:
        return "resultado não versionado"
    r = subprocess.run(["git", "show", f"{commit}:{config}"], cwd=ROOT, capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else "ausente no commit"


def load_records() -> pd.DataFrame:
    manifest = _manifest()
    rows = []
    for p in sorted((ROOT / "results").glob("**/metrics.json")):
        rel = p.relative_to(ROOT)
        folder = rel.parts[1]
        if folder.startswith("_"):
            continue
        m = json.loads(p.read_text())
        t, v = m.get("test") or {}, m.get("validation") or {}
        ds = m.get("dataset")
        rows.append({
            "pasta": folder, "dataset": ds, "k": m.get("num_clients"), "k_pedido": m.get("requested_clients"),
            "variante": _variant(m), "semente": m.get("seed"),
            "alvo": m.get("target") or ("speed" if ds in ("metr-la", "pems-bay") else None),
            "entradas": ",".join(m.get("features") or []) or None, "grafo": m.get("graph"),
            "nos": m.get("nodes"), "janela": m.get("input_steps"), "horizonte": m.get("horizon"),
            "iteracoes": m.get("rounds_completed", m.get("epochs_completed")),
            "melhor_iteracao": m.get("best_round", m.get("best_epoch")),
            "test_MAE": t.get("MAE"), "test_RMSE": t.get("RMSE"), "test_MSE": t.get("MSE"), "test_R2": t.get("R2"),
            "test_WAPE_percent": t.get("WAPE_percent"), "test_macro_client_MAE": t.get("macro_client_MAE"),
            "test_global_MAE": (m.get("test_global") or {}).get("MAE"),
            "val_MAE": v.get("MAE"), "teste_presente": "test" in m,
            "comm_modelo_bytes": m.get("model_communication_bytes"), "halo_bytes": m.get("halo_bytes"),
            "tempo_treino_s": m.get("training_seconds"),
            "halo_nos": m.get("halo_nodes_total"), "halo_vizinho_unico_pct": m.get("halo_single_neighbour_pct"),
            "halo_exatidao": (m.get("halo_exactness_max_abs_error") or {}).get("todos")
            if isinstance(m.get("halo_exactness_max_abs_error"), dict) else m.get("halo_exactness_max_abs_error"),
            **_fedpub_fields(m),
            "particao": m.get("partition"), "clientes_nos": json.dumps(m.get("client_nodes")) if m.get("client_nodes") else None,
            "janelas_treino": (m.get("samples") or {}).get("train"), "janelas_validacao": (m.get("samples") or {}).get("validation"),
            "janelas_teste": (m.get("samples") or {}).get("test"),
            "tipo_registro": "avaliação de checkpoint de sondagem" if m.get("evaluated_from") else "treino com avaliação ao fim",
            "checkpoint_de": m.get("evaluated_from"),
            "checkpoint": f"{m.get('evaluated_from') or rel.parent.as_posix()}/best_model.pt",
            "checkpoint_sha256": manifest.get(f"{m.get('evaluated_from') or rel.parent.as_posix()}/best_model.pt", ""),
            **_probe(m, p),
            "gpu": m.get("gpu") or "não registrado", "ambiente": _environment(m.get("gpu")),
            "papel_inventario": role(folder, m), "problemas_metadados": "; ".join(metadata_issues(role(folder, m), m)),
            "fonte": str(rel), "sha256": sha256(p),
        })
    return pd.DataFrame(rows)


def _fedpub_fields(m: dict) -> dict:
    """FED-PUB: esparsidade das máscaras e resumo da matriz α, ambas no melhor checkpoint (o que o metrics.json grava).

    Esparsidade = fração das entradas de máscara com |μ| ≤ limiar (não entradas exatamente zero), nos tensores mascaráveis.
    α_ij = peso do modelo do cliente j na agregação personalizada do cliente i (linhas somam 1)."""
    if m.get("alpha") is None:
        return {}
    a = np.asarray(m["alpha"], dtype=float)
    n = len(a)
    ent = [float(-(r[r > 0] * np.log(r[r > 0])).sum() / np.log(n)) for r in a]
    cs = m.get("client_mask_sparsity") or []
    return {"fedpub_limiar_mascara": m.get("mask_threshold"),
            "fedpub_esparsidade": m.get("mask_sparsity"),
            "fedpub_esparsidade_cliente_min": min(cs) if cs else None,
            "fedpub_esparsidade_cliente_max": max(cs) if cs else None,
            "fedpub_alpha_min": float(a.min()), "fedpub_alpha_max": float(a.max()),
            "fedpub_alpha_diag_min": float(np.diag(a).min()), "fedpub_alpha_diag_max": float(np.diag(a).max()),
            "fedpub_alpha_diag_media": float(np.diag(a).mean()),
            "fedpub_alpha_razao_linha_max": float((a.max(1) / a.min(1)).max()),
            "fedpub_alpha_entropia_norm_min": min(ent),
            "fedpub_comm_densa_bytes": m.get("dense_communication_bytes")}


def _probe(m: dict, p: Path) -> dict:
    """Para avaliações de checkpoint de sondagem: confere a sondagem de origem (melhor rodada e rodadas)."""
    src = m.get("evaluated_from")
    if not src:
        return {"sonda_metrics_sha256": "", "sonda_confere": None}
    pm_ = ROOT / src / "metrics.json"
    if not pm_.exists():
        return {"sonda_metrics_sha256": "", "sonda_confere": False}
    pm = json.loads(pm_.read_text())
    best = lambda d: d.get("best_round", d.get("best_epoch"))
    its = lambda d: d.get("rounds_completed", d.get("epochs_completed"))
    ok = best(pm) == best(m) and its(pm) == its(m) and pm.get("seed") == m.get("seed") and "test" not in pm
    return {"sonda_metrics_sha256": sha256(pm_), "sonda_confere": bool(ok)}


def select(records: pd.DataFrame, cell: Cell) -> pd.DataFrame:
    r = records[(records.pasta == cell.folder) & (records.dataset == cell.dataset) & (records.variante == cell.variant)]
    r = r[r.k.isna()] if cell.k is None else r[r.k == cell.k]
    return r.sort_values("semente")


def validate(records: pd.DataFrame, cells: list[Cell]) -> tuple[list[str], list[str]]:
    errors, notes = [], []
    seen = []
    for cell in cells:
        r = select(records, cell)
        if r.empty:
            (notes if cell.optional else errors).append(
                f"{cell.id}: {'em execução, fora do pacote' if cell.optional else 'célula ausente'}")
            continue
        seeds = set(int(s) for s in r.semente)
        if seeds != SEEDS:
            (notes if cell.optional else errors).append(f"{cell.id}: sementes {sorted(seeds)} (esperado {sorted(SEEDS)})")
            if cell.optional:
                continue
        if r.semente.duplicated().any():
            errors.append(f"{cell.id}: semente duplicada")
        bad = r[[not (isinstance(x, (int, float)) and math.isfinite(x)) for x in r.test_MAE]]
        if len(bad):
            errors.append(f"{cell.id}: teste ausente ou não finito em {len(bad)} registro(s)")
        meta = sorted({x for s in r.problemas_metadados for x in s.split("; ") if x})
        if meta:
            errors.append(f"{cell.id}: metadados {meta}")
        if cell.dataset == "sunt" and cell.group in ("principal", "referência") and not (r.iteracoes == SUNT_BUDGET[cell.k]).all():
            errors.append(f"{cell.id}: orçamento diferente de {SUNT_BUDGET[cell.k]}")
        if cell.dataset == "sunt" and cell.group == "principal" and cell.folder in ("sunt_fedavg_final", "sunt_centralized_boardings"):
            if not r.checkpoint_de.notna().all():
                errors.append(f"{cell.id}: avaliação sem referência ao checkpoint da sondagem")
            elif not r.sonda_confere.all():
                errors.append(f"{cell.id}: sondagem de origem não confere (melhor rodada, rodadas, semente ou teste)")
            if (r.checkpoint_sha256 == "").any():
                errors.append(f"{cell.id}: checkpoint da sondagem sem hash no manifesto")
        missing_ck = int((r.checkpoint_sha256 == "").sum())
        if missing_ck and not (cell.dataset == "sunt" and cell.folder in ("sunt_fedavg_final", "sunt_centralized_boardings")):
            notes.append(f"{cell.id}: {missing_ck} checkpoint(s) sem hash no manifesto")
        if not (ROOT / cell.config).exists():
            errors.append(f"{cell.id}: config {cell.config} inexistente")
        seen.append(r.assign(celula=cell.id))
    sel = pd.concat(seen) if seen else pd.DataFrame()
    for (ds, alvo), g in sel.groupby(["dataset", "alvo"]):
        for col in ("nos", "janela", "horizonte"):
            vals = set(g[col].dropna())
            if len(vals) > 1:
                errors.append(f"{ds}/{alvo}: {col} inconsistente {sorted(vals)}")
    return errors, notes


def paired(runs: pd.DataFrame, a: str, b: str) -> dict | None:
    ra, rb = runs[runs.celula == a].sort_values("semente"), runs[runs.celula == b].sort_values("semente")
    if ra.empty or rb.empty or set(ra.semente) != SEEDS or set(rb.semente) != SEEDS:
        return None
    x = ra.set_index("semente").test_MAE.astype(float)
    y = rb.set_index("semente").test_MAE.astype(float).loc[x.index]
    d = (x - y).to_numpy()
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    t_p = float(stats.ttest_rel(x, y).pvalue)
    w_p = float(stats.wilcoxon(d, method="exact").pvalue) if np.any(d != 0) else float("nan")
    return {"dif_MAE_media": mean, "dif_MAE_dp": sd, "IC95_inf": mean - half, "IC95_sup": mean + half,
            "p_t_pareado": t_p, "p_wilcoxon_exato": w_p, "sementes_favoraveis_a_A": int((d < 0).sum()), "n": n,
            **{f"dif_semente_{int(s)}": float(v) for s, v in zip(x.index, d)}}


# ---------------------------------------------------------------- escrita
NUMWORD = {5: "Five", 6: "Six", 10: "Ten", 20: "Twenty", 166: "OneSixtySix", None: ""}
DSWORD = {"sunt": "SUNT", "metr-la": "MetrLA", "pems-bay": "PemsBay"}


def macro_prefix(cell: Cell) -> str:
    ds = "Carga" if cell.id.startswith("carga") else DSWORD[cell.dataset]
    digit = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
             "8": "Eight", "9": "Nine", "p": "Point"}
    token = cell.id.split("_")[1]
    meth = "".join(digit[ch] if ch.isdigit() or (ch == "p" and i and token[i - 1].isdigit()) else ch
                   for i, ch in enumerate(token)).title().replace(" ", "")
    meth = "".join(ch for ch in meth if ch.isalpha())
    return f"Traffic{ds}{meth}{'K' + NUMWORD[cell.k] if cell.k else ''}"


def fmt(x, nd):
    return "---" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def pm(s: pd.Series, nd: int) -> str:
    s = s.dropna().astype(float)
    return "---" if s.empty else f"{s.mean():.{nd}f} $\\pm$ {s.std(ddof=1):.{nd}f}"


PARTITION_KS = {"metr-la": (5, 10, 20), "pems-bay": (5, 10, 20), "sunt": (6, 20, 166)}


def _array_sha256(a: np.ndarray) -> str:
    """Hash do conteúdo da matriz (float64, ordem C, com a forma), independente de como foi carregada."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


@lru_cache(maxsize=None)
def _adjacency(dataset: str) -> np.ndarray:
    from .data import load_experiment
    win = {"sunt": (168, 24, ["n_boardings", "n_alightings"], "n_boardings")}.get(dataset, (12, 12, None, None))
    return np.asarray(load_experiment(dataset, *win).adj, dtype=np.float64)


@lru_cache(maxsize=None)
def _target_stats(dataset: str) -> tuple[float, float]:
    """Descritivos do alvo na série completa (1.23.0): percentual de posições nó-instante sem observação do alvo e
    média do alvo nas posições observadas, na unidade original. Mesma carga de dados de `_adjacency`."""
    from .data import load_experiment
    win = {"sunt": (168, 24, ["n_boardings", "n_alightings"], "n_boardings")}.get(dataset, (12, 12, None, None))
    d = load_experiment(dataset, *win)
    j = int(d.target_feature_index)
    obs = np.asarray(d.mask)[:, :, j].astype(bool)
    val = np.asarray(d.series)[:, :, j]
    return 100.0 * float(1.0 - obs.mean()), float(val[obs].mean())


def partition_stats() -> pd.DataFrame:
    """Frações de corte das partições Leiden usadas (semente 42), calculadas da adjacência e das atribuições.

    Quatro medidas, com definições explícitas (no SUNT, simétrico, as duas primeiras coincidem):
      frac_arestas_dirigidas_cortadas_pct  arestas u->v (sem laços) com u e v em clientes diferentes (definição usada
                                           nos textos anteriores: METR-LA 6,3 / 16,2 / 39,1 %)
      frac_pares_cortados_pct              pares não direcionados {u, v} cortados
      nos_fronteira_pct                    nós com ao menos uma aresta cortada, em qualquer sentido
      nos_com_entrada_externa_pct          nós que perderam ao menos um vizinho de ENTRADA (o que o halo repõe)"""
    rows = []
    for ds, ks in PARTITION_KS.items():
        a = _adjacency(ds).copy()
        np.fill_diagonal(a, 0.0)
        directed = a > 0
        und = np.triu(np.maximum(a, a.T) > 0, 1)
        und_full = np.maximum(a, a.T) > 0
        pairs = int(und.sum())
        index_path = ROOT / "data" / ds / "processed" / "partitions" / "leiden" / "partitions.csv"
        index = pd.read_csv(index_path)
        for k in ks:
            row = index[(index.num_clients == k) & (index.partition_seed == 42)].iloc[0]
            pdir = ROOT / row.path
            assign = np.load(pdir / "client_assignments.npy")
            cross = assign[:, None] != assign[None, :]
            sizes = np.bincount(assign)
            entrada_externa = ((a > 0) & cross).any(0)          # nó que perdeu ao menos um vizinho de entrada
            rows.append({"dataset": ds, "k_pedido": int(row.requested_clients), "k": k, "resolucao": float(row.resolution),
                         "nos": len(assign), "arestas_dirigidas": int(directed.sum()),
                         "nos_sem_conexoes": int((~und_full.any(axis=1)).sum()),
                         "arestas_dirigidas_cortadas": int((directed & cross).sum()),
                         "frac_arestas_dirigidas_cortadas_pct": 100 * float((directed & cross).sum()) / float(directed.sum()),
                         "pares": pairs, "pares_cortados": int((und & cross).sum()),
                         "frac_pares_cortados_pct": 100 * float((und & cross).sum()) / pairs,
                         "nos_fronteira_pct": 100 * float((und_full & cross).any(1).mean()),
                         "nos_com_entrada_externa_pct": 100 * float(entrada_externa.mean()),
                         "nos_por_cliente_min": int(sizes.min()), "nos_por_cliente_mediana": float(np.median(sizes)),
                         "nos_por_cliente_max": int(sizes.max()),
                         "particao": row.path, "clientes_nos": json.dumps(sizes.tolist()),
                         "grafo_estatisticas": "metr-la/pems-bay: dirigido de distâncias; sunt: transit simétrico (embarques)",
                         "adjacencia_sha256": _array_sha256(_adjacency(ds)),
                         "partitions_csv_sha256": sha256(index_path),
                         "client_assignments_sha256": sha256(pdir / "client_assignments.npy"),
                         "node_ids_sha256": sha256(pdir / "node_ids.npy") if (pdir / "node_ids.npy").exists() else "",
                         "metadata_sha256": sha256(pdir / "metadata.json") if (pdir / "metadata.json").exists() else ""})
    return pd.DataFrame(rows)


def partition_link_errors(runs: pd.DataFrame, part: pd.DataFrame) -> list[str]:
    """Cada execução federada deve apontar para a partição das estatísticas e ter os mesmos tamanhos de cliente.

    O metrics.json registra o caminho da partição e `client_nodes` (nós por cliente, na ordem dos clientes), mas não um
    hash; a conferência dos tamanhos contra `bincount(client_assignments.npy)` é o vínculo disponível."""
    errs = []
    ref = {(r.dataset, int(r.k)): r for r in part.itertuples()}
    for r in runs[runs.k.notna()].itertuples():
        p = ref.get((r.dataset, int(r.k)))
        if p is None:
            errs.append(f"{r.fonte}: sem partição de referência para {r.dataset} k={int(r.k)}")
        elif r.particao != p.particao:
            errs.append(f"{r.fonte}: partição {r.particao} difere de {p.particao}")
        elif not isinstance(r.clientes_nos, str) or json.loads(r.clientes_nos) != json.loads(p.clientes_nos):
            errs.append(f"{r.fonte}: nós por cliente não conferem com client_assignments.npy")
    return errs


def sunt_gtfs_comparison() -> pd.DataFrame:
    """Cobertura do grafo do estudo no SUNT em relação ao grafo do GTFS distribuído com o conjunto (1.26.0).

    Nós: paradas do grafo do estudo (as que têm séries) e extremidades das arestas do grafo do GTFS, pelo stop_id.
    Pares: conexões não direcionadas {u, v} entre paradas distintas; os autolaços do GTFS são contados à parte."""
    if not SUNT_GTFS_GRAPH.exists():
        raise ValueError(f"grafo do GTFS ausente: baixar {SUNT_GTFS_URL} para {SUNT_GTFS_GRAPH.relative_to(ROOT)}")
    if sha256(SUNT_GTFS_GRAPH) != SUNT_GTFS_SHA256:
        raise ValueError(f"{SUNT_GTFS_GRAPH.relative_to(ROOT)}: SHA-256 difere do fixado ({SUNT_GTFS_SHA256})")
    from .data import load_experiment
    d = load_experiment("sunt", 168, 24, ["n_boardings", "n_alightings"], "n_boardings")
    ids = np.array([int(s) for s in d.node_ids], dtype=np.int64)
    a = np.asarray(d.adj, dtype=np.float64).copy()
    np.fill_diagonal(a, 0.0)
    iu, iv = np.nonzero(np.triu(np.maximum(a, a.T) > 0, 1))
    ours = set(ids.tolist())
    ours_pairs = {frozenset((int(ids[u]), int(ids[v]))) for u, v in zip(iu, iv)}
    g = pd.read_parquet(SUNT_GTFS_GRAPH, columns=["src", "dst"])
    src, dst = g.src.astype("int64").to_numpy(), g.dst.astype("int64").to_numpy()
    gtfs = set(src.tolist()) | set(dst.tolist())
    loops = src == dst
    gtfs_pairs = {frozenset((int(u), int(v))) for u, v in zip(src[~loops], dst[~loops])}
    return pd.DataFrame([{
        "nos_estudo": len(ours), "nos_gtfs": len(gtfs), "nos_comuns": len(ours & gtfs),
        "nos_so_estudo": len(ours - gtfs), "nos_so_gtfs": len(gtfs - ours),
        "pares_estudo": len(ours_pairs), "pares_gtfs": len(gtfs_pairs), "pares_comuns": len(ours_pairs & gtfs_pairs),
        "pares_so_estudo": len(ours_pairs - gtfs_pairs), "pares_so_gtfs": len(gtfs_pairs - ours_pairs),
        "arestas_gtfs": len(g), "autolacos_gtfs": int(loops.sum()),
        "grafo_gtfs": str(SUNT_GTFS_GRAPH.relative_to(ROOT)), "grafo_gtfs_sha256": SUNT_GTFS_SHA256,
        "grafo_gtfs_url": SUNT_GTFS_URL, "adjacencia_estudo_sha256": _array_sha256(np.asarray(d.adj, dtype=np.float64))}])


def write_structure(d: Path, runs: pd.DataFrame, by_id: dict) -> None:
    sunt_gtfs_comparison().to_csv(d / "sunt_grafo_gtfs.csv", index=False)
    part = partition_stats()
    errs = partition_link_errors(runs, part)
    if errs:
        raise ValueError("vínculo execução-partição: " + " | ".join(errs[:10]))
    part.to_csv(d / "particoes.csv", index=False, float_format="%.17g")
    halo = (runs[runs.halo_nos.notna()].groupby(["celula", "dataset", "alvo", "k"], sort=False)
            .agg(nos_virtuais=("halo_nos", "first"), nos_virtuais_unicos=("halo_nos", "nunique"),
                 vizinho_unico_pct=("halo_vizinho_unico_pct", "first"), exatidao_max=("halo_exatidao", "max"),
                 halo_MB=("halo_bytes", lambda x: x.max() / 1e6)).reset_index())
    halo.to_csv(d / "halo.csv", index=False, float_format="%.17g")
    comm = (runs[runs.comm_modelo_bytes.notna()].groupby(["celula", "metodo", "dataset", "alvo", "k"], sort=False)
            .agg(comm_modelo_GB_media=("comm_modelo_bytes", lambda x: x.mean() / 1e9),
                 comm_modelo_GB_dp=("comm_modelo_bytes", lambda x: x.std(ddof=1) / 1e9),
                 iteracoes_media=("iteracoes", "mean"), halo_MB=("halo_bytes", lambda x: x.max() / 1e6)).reset_index())
    comm.to_csv(d / "comunicacao_por_execucao.csv", index=False, float_format="%.17g")
    # Custo computacional (1.16.0): tempo de treino gravado pelo próprio treino (`training_seconds`), por registro e por
    # célula. DESCRITIVO: GPU, concorrência e período variam entre registros, mesmo no mesmo servidor; não ordena métodos.
    t = runs[runs.tempo_treino_s.notna()].copy()
    t["tempo_treino_h"] = t.tempo_treino_s / 3600
    t["s_por_iteracao"] = t.tempo_treino_s / t.iteracoes
    t[["celula", "metodo", "dataset", "alvo", "k", "semente", "iteracoes", "tempo_treino_h", "s_por_iteracao", "gpu",
       "ambiente"]].to_csv(d / "custo_computacional_por_execucao.csv", index=False, float_format="%.17g")
    (t.groupby(["celula", "metodo", "dataset", "alvo", "k"], sort=False)
     .agg(execucoes=("semente", "count"), tempo_treino_h_media=("tempo_treino_h", "mean"),
          tempo_treino_h_dp=("tempo_treino_h", lambda x: x.std(ddof=1)), s_por_iteracao_media=("s_por_iteracao", "mean"),
          gpus=("gpu", lambda x: "; ".join(sorted(set(x)))), ambientes=("ambiente", lambda x: "; ".join(sorted(set(x)))))
     .reset_index().to_csv(d / "custo_computacional_por_celula.csv", index=False, float_format="%.17g"))
    cent = runs[runs.celula.str.endswith("_cent")]
    data = (cent.groupby(["dataset", "alvo"], sort=False)
            .agg(nos=("nos", "first"), janela=("janela", "first"), horizonte=("horizonte", "first"),
                 janelas_treino=("janelas_treino", "first"), janelas_validacao=("janelas_validacao", "first"),
                 janelas_teste=("janelas_teste", "first")).reset_index())
    principal = data.alvo != "loading"      # descritivos do alvo só para os três cenários principais (1.23.0)
    data["alvo_ausente_pct"] = [_target_stats(ds)[0] if ok else float("nan") for ds, ok in zip(data.dataset, principal)]
    data["alvo_media_observada"] = [_target_stats(ds)[1] if ok else float("nan") for ds, ok in zip(data.dataset, principal)]
    data.to_csv(d / "datasets.csv", index=False, float_format="%.17g")


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    return (runs.groupby(["celula", "grupo", "metodo", "dataset", "alvo", "k"], dropna=False, sort=False)
            .agg(n=("semente", "size"), MAE_media=("test_MAE", "mean"), MAE_dp=("test_MAE", "std"),
                 RMSE_media=("test_RMSE", "mean"), RMSE_dp=("test_RMSE", "std"), R2_media=("test_R2", "mean"),
                 R2_dp=("test_R2", "std"), melhor_iteracao_media=("melhor_iteracao", "mean"),
                 iteracoes=("iteracoes", "max")).reset_index())


def comparisons(runs: pd.DataFrame, by_id: dict) -> pd.DataFrame:
    comp = []
    for fam, a, b in _comparisons():
        res = paired(runs, a, b)
        if res is not None:
            comp.append({"familia": fam, "familia_inferencial": inferential_family(a, b), "A": a, "B": b,
                         "rotulo": f"{by_id[a].method} − {by_id[b].method}", **res})
    comp = pd.DataFrame(comp)
    if comp.empty:
        return comp
    comp["p_t_holm"] = pd.concat([holm(g.p_t_pareado) for _, g in comp.groupby("familia_inferencial")]).loc[comp.index]
    comp["p_wilcoxon_holm"] = pd.concat([holm(g.p_wilcoxon_exato.fillna(1.0)) for _, g in comp.groupby("familia_inferencial")]).loc[comp.index]
    comp["tamanho_familia"] = comp.groupby("familia_inferencial").A.transform("size")
    comp["classificacao"] = [evidence(d, pn, pa) for d, pn, pa in zip(comp.dif_MAE_media, comp.p_t_pareado, comp.p_t_holm)]
    order = ["principal", "secundária", "caracterização", "exploratória", "complementar", "controle"]
    comp["_o"] = comp.familia_inferencial.map(lambda f: next(i for i, o in enumerate(order) if f.startswith(o)))
    return comp.sort_values(["_o", "familia_inferencial"], kind="stable").drop(columns="_o").reset_index(drop=True)


def write_package(dest: Path, records: pd.DataFrame, cells: list[Cell], notes: list[str]) -> dict:
    (dest / "dados").mkdir(parents=True)
    (dest / "tabelas").mkdir()
    (dest / "figuras").mkdir()
    by_id = {c.id: c for c in cells}
    chosen = []
    for c in cells:
        r = select(records, c)
        if r.empty or set(r.semente) != SEEDS:
            continue
        chosen.append(r.assign(celula=c.id, grupo=c.group, metodo=c.method.replace("\\_", "_").replace("$\\lambda=", "λ=").replace("$", "")))
    runs = pd.concat(chosen, ignore_index=True)
    cfg = {c.id: c.config for c in cells}
    runs["commit_resultado"] = [results_commit(f) for f in runs.fonte]
    runs["config"] = [cfg[c] for c in runs.celula]
    runs["config_sha256_no_commit"] = [config_at(cm, cf) for cm, cf in zip(runs.commit_resultado, runs.config)]
    cols = ["celula", "grupo", "metodo", "dataset", "alvo", "entradas", "grafo", "k", "k_pedido", "variante", "semente",
            "nos", "janela", "horizonte", "iteracoes", "melhor_iteracao", "test_MAE", "test_RMSE", "test_MSE", "test_R2",
            "test_WAPE_percent", "test_macro_client_MAE", "test_global_MAE", "val_MAE", "tipo_registro", "checkpoint_de",
            "gpu", "ambiente", "fonte", "sha256", "checkpoint", "checkpoint_sha256", "sonda_metrics_sha256",
            "commit_resultado", "config", "config_sha256_no_commit", "comm_modelo_bytes", "halo_bytes", "halo_nos",
            "halo_vizinho_unico_pct", "halo_exatidao", "particao", "clientes_nos"]
    cols += [c for c in runs.columns if c.startswith("fedpub_") and runs[c].notna().any()]
    runs[cols].to_csv(dest / "dados" / "resultados_por_semente.csv", index=False, float_format="%.17g")

    summ = summarize(runs)
    summ.to_csv(dest / "dados" / "resumo_por_celula.csv", index=False, float_format="%.17g")

    probes = records[records.papel_inventario.str.startswith("sondagem")]
    probes[["pasta", "dataset", "k", "semente", "iteracoes", "melhor_iteracao", "val_MAE", "teste_presente", "gpu",
            "ambiente", "fonte", "sha256"]].sort_values(["pasta", "k", "semente"]).to_csv(
        dest / "dados" / "treinos_de_sondagem.csv", index=False, float_format="%.17g")
    hist = records[records.papel_inventario.str.startswith(("histórico", "exploratório"))]
    hist[["papel_inventario", "pasta", "dataset", "alvo", "k", "variante", "semente", "test_MAE", "fonte", "sha256"]].sort_values(
        ["pasta", "fonte"]).to_csv(dest / "dados" / "historico_fora_da_evidencia.csv", index=False, float_format="%.17g")

    write_structure(dest / "dados", runs, by_id)
    comp = comparisons(runs, by_id)
    comp.to_csv(dest / "dados" / "diferencas_pareadas.csv", index=False, float_format="%.17g")

    write_tables(dest / "tabelas", summ, comp, by_id)
    macros = write_macros(dest / "tabelas" / "macros_resultados.tex", summ, comp, by_id)
    write_figures(dest / "figuras", summ, comp)
    write_figures(dest / "figuras", summ, comp, lang="en")   # variante para o artigo (inglês)
    from .export_panorama import write_panorama
    write_panorama(dest, summ, COLORS, MARKERS)
    analises = export_sunt_analysis(dest)
    direcao = export_sunt_direcao(dest)     # 1.18.0: piloto de orientação do grafo (só validação); acrescenta macros
    evolucao_sunt = export_sunt_evolucao(dest)  # 1.21.0: evolução do MAE de validação no SUNT (históricos auditados)
    controles = export_controles_internos(dest)
    from .export_contrast_figures import write_contrast_figures
    write_contrast_figures(dest)
    return {"controles_arestas_internas": controles, "registros_selecionados": len(runs), "celulas_exportadas": sorted(runs.celula.unique()),
            "macros": macros, "comparacoes": len(comp), "analises_sunt": analises,
            "piloto_direcao_sunt": direcao, "evolucao_validacao_sunt": evolucao_sunt}


def _row_cells(summ, by_id, ids, nd=4, key="MAE"):
    out = []
    for i in ids:
        s = summ[summ.celula == i]
        out.append("---" if s.empty else f"{s[f'{key}_media'].iloc[0]:.{nd}f} $\\pm$ {s[f'{key}_dp'].iloc[0]:.{nd}f}")
    return out


# Células da tabela de custo do SUNT com k=166 (1.17.0), na ordem das tabelas principais.
COST_CELLS = (("Centralizado", "sunt_cent"), ("FedAvg", "sunt_fedavg_k166"), ("FedProx", "sunt_fedprox_k166"),
              ("Ditto", "sunt_ditto_k166"), ("FedGNN-Halo", "sunt_halo_k166"))
# Células que só geram macros de tempo (1.25.0): FedAvg nas demais partições do SUNT, para comparação com o centralizado.
COST_MACRO_EXTRA = ("sunt_fedavg_k6", "sunt_fedavg_k20")


def write_tables(d: Path, summ, comp, by_id):
    head = "% Gerado por experiments/export_traffic.py; não editar. Fragmento tabular, sem legenda.\n"

    def table_label(value):
        # Nomes de apresentação: não altera identificadores, famílias estatísticas ou números.
        if value == "Ditto":
            return r"\shortstack[l]{Ditto\\(seleção adaptada)}"
        value = re.sub(r"\bDitto\b(?! \()", "Ditto (seleção adaptada)", value)
        return (value.replace("Centralizado sem conexões", "Centralizado sem arestas entre nós distintos")
                     .replace("(parada federada)", "(limite e paciência ampliados)")
                     .replace("regra de parada federada", "limite e paciência ampliados")
                     .replace("halo × none", "FedGNN-Halo × FedAvg (ablação)")
                     .replace("exploratória: comparadores", "exploratória: métodos de comparação"))

    def table(name, header, rows, align, label=None):
        label = label or table_label
        rows = [None if row is None else [label(cell) for cell in row] for row in rows]
        body = [head, f"\\begin{{tabular}}{{{align}}}", "\\toprule", header + " \\\\", "\\midrule"]
        body += [("\\midrule" if r is None else " & ".join(r) + " \\\\") for r in rows] + ["\\bottomrule", "\\end{tabular}", ""]
        (d / name).write_text("\n".join(body))

    table("pemsbay_sem_halo_k5.tex", "Condição & MAE (mph) & RMSE (mph)",
          [[label, pm_cell(summ, cell, "MAE", 3), pm_cell(summ, cell, "RMSE", 3)]
           for label, cell in (("FedAvg (comparações gerais)", "pemsbay_fedavg_k5"),
                               ("FedAvg (sem halo)", "pemsbay_halonone_k5"),
                               ("FedGNN-Halo", "pemsbay_halo_k5"))], "lcc")

    # RMSE das rodovias: mesmas médias/desvios, incluindo o Local, sem inferência.
    for dataset in ("metrla", "pemsbay"):
        rows = [["Centralizado", pm_cell(summ, f"{dataset}_cent", "RMSE", 3), "---", "---", "---"], None]
        for method, key in (("Local", "local"), ("FedAvg", "fedavg"), ("FedGNN-Halo", "halo"), ("FedProx", "fedprox"), ("Ditto", "ditto")):
            rows.append([method, "---"] + [pm_cell(summ, f"{dataset}_{key}_k{k}", "RMSE", 3) for k in (5, 10, 20)])
        table(f"{dataset}_rmse.tex", r"Condição & Sem partição & $k=5$ & $k=10$ & $k=20$", rows, "lcccc", label=lambda x: x)

    # SUNT principal: MAE e RMSE por k
    ks = (6, 20, 166)
    rows = [["Centralizado (referência)"] + [pm_cell(summ, "sunt_cent", "MAE", 4)] * 0 +
            [f"\\multicolumn{{3}}{{c}}{{{pm_cell(summ, 'sunt_cent', 'MAE', 4)}}}",
             f"\\multicolumn{{3}}{{c}}{{{pm_cell(summ, 'sunt_cent', 'RMSE', 2)}}}"]]
    for meth, pre in (("FedAvg", "sunt_fedavg"), ("FedProx", "sunt_fedprox"), ("Ditto", "sunt_ditto"), ("FedGNN-Halo", "sunt_halo")):
        rows.append([meth] + [pm_cell(summ, f"{pre}_k{k}", "MAE", 4) for k in ks] + [pm_cell(summ, f"{pre}_k{k}", "RMSE", 2) for k in ks])
    table("sunt_principal.tex", "Método & \\multicolumn{3}{c}{MAE ($k$ = 6 / 20 / 166)} & \\multicolumn{3}{c}{RMSE ($k$ = 6 / 20 / 166)}",
          rows, "lcccccc")
    # Mesma informação em duas tabelas legíveis em largura de página (uma métrica por tabela).
    # Desde o 1.14.0, o Centralizado fica numa coluna própria "Sem partição" (não pertence a nenhum k), separado dos
    # métodos federados por um traço; nas colunas por k só há métodos federados.
    def by_k(name, cent, fed, kk, key, nd):
        rows = [["Centralizado", pm_cell(summ, cent, key, nd)] + ["---"] * len(kk), None]
        for meth, pre in fed:
            vals = [pm_cell(summ, f"{pre}_k{k}", key, nd) for k in kk]
            if any(v != "---" for v in vals):
                rows.append([meth, "---"] + [bold_if_best(summ, [f"{p}_k{k}" for _, p in fed], f"{pre}_k{k}", key, nd, v)
                                             for k, v in zip(kk, vals)])
        table(name, "Método & Sem partição & " + " & ".join(f"$k={k}$" for k in kk), rows, "l" + "c" * (len(kk) + 1))

    for key, nd in (("MAE", 4), ("RMSE", 2)):
        by_k(f"sunt_{key.lower()}.tex", "sunt_cent", (("FedAvg", "sunt_fedavg"), ("FedProx", "sunt_fedprox"),
             ("Ditto", "sunt_ditto"), ("FedGNN-Halo", "sunt_halo")), ks, key, nd)

    # Tabelas principais das rodovias: Centralizado na coluna "Sem partição" e os métodos federados por k (1.14.0).
    # Local (referência) e os controles (centralizado sem arestas, halo none, 1hop_min2) continuam nos CSVs, nas macros
    # e nas diferenças pareadas, fora destas tabelas.
    for ds, tag, kk in (("metr-la", "metrla", (5, 10, 20)), ("pems-bay", "pemsbay", (5, 10, 20))):
        by_k(f"{tag}_mae.tex", f"{tag}_cent", (("FedAvg", f"{tag}_fedavg"), ("FedProx", f"{tag}_fedprox"),
             ("Ditto", f"{tag}_ditto"), ("FedGNN-Halo", f"{tag}_halo")), kk, "MAE", 4)

    n = lambda x, f: f"\\num{{{x:{f}}}}"
    rows = [[r.familia_inferencial, r.familia, re.sub(r"(?<!\\)_", r"\\_", r.rotulo), n(r.dif_MAE_media, "+.4f"),
             f"[{n(r.IC95_inf, '+.4f')}; {n(r.IC95_sup, '+.4f')}]", n(r.p_t_pareado, ".4f"), n(r.p_t_holm, ".4f"),
             n(r.p_wilcoxon_exato, ".4f"), f"{r.sementes_favoraveis_a_A}/{r.n}", r.classificacao] for r in comp.itertuples()]
    rows = [[c.replace("\\num{0.0000}", "\\num{<0.0001}") for c in row] for row in rows]
    table("diferencas_pareadas.tex", "Família de ajuste & Célula & Comparação (A − B) & $\\Delta$MAE & IC95\\% (t, não ajustado) & "
          "$p_t$ & $p_{\\mathrm{Holm}}$ & $p_W$ exato & A melhor & Classificação", rows, "lllccccccl")
    # A mesma tabela dividida por grupo de famílias (principal, secundária, caracterização, exploratória,
    # complementar), sem a coluna de família, para leitura em fonte normal.
    slug = {"principal": "principal", "secundária": "secundaria", "caracterização": "caracterizacao",
            "exploratória": "exploratoria", "complementar": "complementar", "controle": "controle_parada"}
    ds_slug = {"(METR-LA)": "_metrla", "(PEMS-BAY)": "_pemsbay", "(SUNT)": "_sunt"}
    parts = []
    for grupo, nome in slug.items():
        fams = sorted({row[0] for row in rows if row[0].split(":")[0] == grupo}, key=lambda f: [r[0] for r in rows].index(f))
        if grupo == "exploratória":      # 17 linhas não cabem numa página: uma tabela por conjunto (família de Holm)
            parts += [(f"{nome}{next(v for k, v in ds_slug.items() if f.endswith(k))}", [f]) for f in fams]
        else:
            parts.append((nome, fams))
    for nome, fams in parts:
        # Só nas tabelas por família (1.23.2): k em modo matemático na primeira coluna; os CSVs não mudam.
        sub = [[re.sub(r"k=(\d+)", r"$k=\1$", row[1])] + row[2:] for row in rows if row[0] in fams]
        if sub:
            table(f"diferencas_{nome}.tex", "Configuração & Comparação (A $-$ B) & $\\Delta$MAE & IC95\\% & $p_t$ & "
                  "$p_{\\mathrm{Holm}}$ & $p_W$ & A melhor & Classificação", sub,
                  (">{\\raggedright\\arraybackslash}p{2.0cm}>{\\raggedright\\arraybackslash}p{5.0cm}cccccc"
                   if nome == "controle_parada" else
                   ">{\\raggedright\\arraybackslash}l>{\\raggedright\\arraybackslash}p{3.3cm}cccccc")
                  + ">{\\raggedright\\arraybackslash}p{3.3cm}")  # requer o pacote array

    carga = (("Centralizado", "carga_cent"), ("FedAvg ($k=20$)", "carga_fedavg_k20"), ("FedGNN-Halo ($k=20$)", "carga_halo_k20"))
    rows = [[meth] + [bold_if_best(summ, [c for _, c in carga], cid, key, nd, pm_cell(summ, cid, key, nd), maior=key == "R2")
                      for key, nd in (("MAE", 4), ("RMSE", 3), ("R2", 4))] for meth, cid in carga]
    table("carga_direcionado.tex", "Método & MAE & RMSE & $R^2$", rows, "lccc")

    # Custo no SUNT com k=166 (1.17.0): comunicação estimada de modelos, payload do halo e tempo de treino gravado pelo
    # próprio treino (média ± dp entre sementes; segundos por rodada = tempo / rodadas). Tempos indicativos (servidor
    # compartilhado, GPUs A16 e L40S); a legenda fica a cargo de cada documento.
    dd = d.parent / "dados"
    if (dd / "custo_computacional_por_execucao.csv").exists() and (dd / "comunicacao_por_execucao.csv").exists():
        cost = pd.read_csv(dd / "custo_computacional_por_execucao.csv")
        comm = pd.read_csv(dd / "comunicacao_por_execucao.csv").set_index("celula")
        rows = []
        for meth, cid in COST_CELLS:
            g = cost[cost.celula == cid]
            if g.empty:
                continue
            if cid not in comm.index:        # centralizado (1.25.0): sem troca de parâmetros; tempo em épocas
                rows.append([meth, "---", "---",
                             f"\\num{{{g.tempo_treino_h.mean():.1f}}} $\\pm$ \\num{{{g.tempo_treino_h.std(ddof=1):.1f}}}",
                             f"\\num{{{g.s_por_iteracao.mean():.0f}}}"])
                continue
            halo = comm.loc[cid, "halo_MB"]
            rows.append([meth, f"\\num{{{comm.loc[cid, 'comm_modelo_GB_media']:.2f}}}",
                         "---" if not (isinstance(halo, float) and math.isfinite(halo)) else f"\\num{{{halo:.1f}}}",
                         f"\\num{{{g.tempo_treino_h.mean():.1f}}} $\\pm$ \\num{{{g.tempo_treino_h.std(ddof=1):.1f}}}",
                         f"\\num{{{g.s_por_iteracao.mean():.0f}}}"])
        if rows:
            # 1.26.2: tabela do corpo do texto, que chama o método apenas de Ditto; as tabelas do apêndice mantêm
            # "Ditto (seleção adaptada)" para distingui-lo das execuções com λ fixo.
            table("sunt_custo_k166.tex", "Método & Comunicação de modelos (GB) & Halo (MB) & Treino (h) & s por rodada ou época",
                  rows, "lcccc", label=lambda cell: cell if cell == "Ditto" else table_label(cell))


def fmt_p(p: float) -> str:
    """p com quatro casas; abaixo de 1e-4, comparador (siunitx aceita \\num{<0.0001})."""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def pm_cell(summ, cid, key, nd):
    """Números dentro de \\num{} (siunitx): o separador decimal segue a configuração de cada documento."""
    s = summ[summ.celula == cid]
    if s.empty:
        return "---"
    return f"\\num{{{s[f'{key}_media'].iloc[0]:.{nd}f}}} $\\pm$ \\num{{{s[f'{key}_dp'].iloc[0]:.{nd}f}}}"


def bold_if_best(summ, cells, cid, key, nd, cell, maior=False):
    """Negrito na melhor média entre `cells`, comparada com as casas impressas (empate no que se lê fica em negrito nos
    dois). É descritivo: não indica diferença detectável, e a legenda de cada documento deve dizê-lo."""
    media = lambda c: round(float(summ.loc[summ.celula == c, f"{key}_media"].iloc[0]), nd)
    vals = [media(c) for c in cells if not summ[summ.celula == c].empty]
    if cell == "---" or media(cid) != (max(vals) if maior else min(vals)):
        return cell
    # mode=text: no modo matemático padrão, o \\num do siunitx 3 não herda o negrito
    return f"\\textbf{{\\boldmath\\sisetup{{mode=text,detect-weight=true}}{cell}}}"


def write_macros(path: Path, summ, comp, by_id) -> int:
    lines = ["% Gerado por experiments/export_traffic.py; não editar.",
             f"\\newcommand{{\\TrafficSeeds}}{{{len(SEEDS)}}}",
             f"\\newcommand{{\\TrafficSeedList}}{{{', '.join(str(s) for s in sorted(SEEDS))}}}"]
    for r in summ.itertuples():
        pre = macro_prefix(by_id[r.celula])
        lines += [f"\\newcommand{{\\{pre}MAE}}{{{r.MAE_media:.4f}}}", f"\\newcommand{{\\{pre}MAEStd}}{{{r.MAE_dp:.4f}}}",
                  f"\\newcommand{{\\{pre}RMSE}}{{{r.RMSE_media:.3f}}}",
                  f"\\newcommand{{\\{pre}RMSEStd}}{{{r.RMSE_dp:.3f}}}", f"\\newcommand{{\\{pre}RTwo}}{{{r.R2_media:.4f}}}"]
    for r in comp.itertuples():
        pa, pb = macro_prefix(by_id[r.A]), macro_prefix(by_id[r.B])
        name = f"{pa}Minus{pb.removeprefix('Traffic')}"
        lines += [f"\\newcommand{{\\{name}DiffMAE}}{{{r.dif_MAE_media:+.4f}}}",
                  f"\\newcommand{{\\{name}CILow}}{{{r.IC95_inf:+.4f}}}", f"\\newcommand{{\\{name}CIHigh}}{{{r.IC95_sup:+.4f}}}",
                  f"\\newcommand{{\\{name}PNominal}}{{{fmt_p(r.p_t_pareado)}}}", f"\\newcommand{{\\{name}PHolm}}{{{fmt_p(r.p_t_holm)}}}"]
    # Recuperação do halo, descritiva e só no METR-LA: (FedAvg − halo) / (FedAvg − Centralizado), sobre as médias.
    # No SUNT nenhuma diferença halo − FedAvg é detectável, e uma "porcentagem recuperada" induziria leitura indevida.
    for ds, tag, kk in (("METR-LA", "metrla", (5, 10, 20)),):
        for k in kk:
            g = {c: summ[summ.celula == c].MAE_media for c in (f"{tag}_fedavg_k{k}", f"{tag}_halo_k{k}", f"{tag}_cent")}
            if all(len(v) for v in g.values()):
                fa, ha, ce = (float(g[c].iloc[0]) for c in (f"{tag}_fedavg_k{k}", f"{tag}_halo_k{k}", f"{tag}_cent"))
                lines.append(f"\\newcommand{{\\Traffic{DSWORD['metr-la' if tag == 'metrla' else 'sunt']}HaloK{NUMWORD[k]}"
                             f"RecoveryPct}}{{{100 * (fa - ha) / (fa - ce):.1f}}}")
    d = path.parent.parent / "dados"
    dsname = {"metr-la": "MetrLA", "pems-bay": "PemsBay", "sunt": "SUNT"}
    if (d / "particoes.csv").exists():
        for r in pd.read_csv(d / "particoes.csv").itertuples():
            pre = f"Traffic{dsname[r.dataset]}PartK{NUMWORD[int(r.k)]}"
            lines += [f"\\newcommand{{\\{pre}CutPct}}{{{r.frac_arestas_dirigidas_cortadas_pct:.1f}}}",
                      f"\\newcommand{{\\{pre}PairCutPct}}{{{r.frac_pares_cortados_pct:.1f}}}",
                      f"\\newcommand{{\\{pre}CutPairs}}{{{r.pares_cortados}}}",
                      f"\\newcommand{{\\{pre}BoundaryPct}}{{{r.nos_fronteira_pct:.1f}}}",
                      f"\\newcommand{{\\{pre}InBoundaryPct}}{{{r.nos_com_entrada_externa_pct:.1f}}}",
                      f"\\newcommand{{\\{pre}Requested}}{{{r.k_pedido}}}",
                      f"\\newcommand{{\\{pre}MinNodes}}{{{int(r.nos_por_cliente_min)}}}",
                      f"\\newcommand{{\\{pre}MedianNodes}}{{{r.nos_por_cliente_mediana:g}}}",
                      f"\\newcommand{{\\{pre}MaxNodes}}{{{int(r.nos_por_cliente_max)}}}"]
        for ds, g in pd.read_csv(d / "particoes.csv").groupby("dataset"):
            lines.append(f"\\newcommand{{\\Traffic{dsname[ds]}IsolatedNodes}}{{{int(g.nos_sem_conexoes.iloc[0])}}}")
            lines += [f"\\newcommand{{\\Traffic{dsname[ds]}Pairs}}{{{int(g.pares.iloc[0])}}}",
                      f"\\newcommand{{\\Traffic{dsname[ds]}DirectedEdges}}{{{int(g.arestas_dirigidas.iloc[0])}}}",
                      f"\\newcommand{{\\Traffic{dsname[ds]}Nodes}}{{{int(g.nos.iloc[0])}}}"]
    if (d / "sunt_grafo_gtfs.csv").exists():      # 1.26.0: cobertura em relação ao grafo do GTFS do SUNT
        r = pd.read_csv(d / "sunt_grafo_gtfs.csv").iloc[0]
        for col, nm in (("nos_gtfs", "Nodes"), ("nos_comuns", "CommonNodes"), ("nos_so_estudo", "MissingSeriesStops"),
                        ("nos_so_gtfs", "ExtraStops"), ("pares_gtfs", "Pairs"), ("pares_comuns", "CommonPairs"),
                        ("pares_so_estudo", "ObservedOnlyPairs"), ("pares_so_gtfs", "GTFSOnlyPairs"),
                        ("autolacos_gtfs", "SelfLoops")):
            lines.append(f"\\newcommand{{\\TrafficSUNTGTFS{nm}}}{{{int(r[col])}}}")
    if (d / "halo.csv").exists():
        for r in pd.read_csv(d / "halo.csv").itertuples():
            pre = macro_prefix(by_id[r.celula])
            lines += [f"\\newcommand{{\\{pre}VirtualNodes}}{{{int(r.nos_virtuais)}}}",
                      f"\\newcommand{{\\{pre}SingleNeighbourPct}}{{{r.vizinho_unico_pct:.1f}}}",
                      f"\\newcommand{{\\{pre}ExactnessMax}}{{{r.exatidao_max:.1e}}}",
                      f"\\newcommand{{\\{pre}HaloMB}}{{{r.halo_MB:.1f}}}"]
    if (d / "comunicacao_por_execucao.csv").exists():
        for r in pd.read_csv(d / "comunicacao_por_execucao.csv").itertuples():
            pre = macro_prefix(by_id[r.celula])
            lines += [f"\\newcommand{{\\{pre}CommGB}}{{{r.comm_modelo_GB_media:.2f}}}",
                      f"\\newcommand{{\\{pre}CommMB}}{{{1000 * r.comm_modelo_GB_media:.1f}}}"]
    if (d / "custo_computacional_por_execucao.csv").exists():      # 1.17.0: só as células da tabela de custo
        cost = pd.read_csv(d / "custo_computacional_por_execucao.csv")
        for cid in [c for _, c in COST_CELLS] + list(COST_MACRO_EXTRA):
            g = cost[cost.celula == cid]
            if len(g):
                pre = macro_prefix(by_id[cid])
                lines += [f"\\newcommand{{\\{pre}TrainHours}}{{{g.tempo_treino_h.mean():.1f}}}",
                          f"\\newcommand{{\\{pre}SecPerRound}}{{{g.s_por_iteracao.mean():.0f}}}"]
    if (d / "resultados_por_semente.csv").exists():
        for cid, g in pd.read_csv(d / "resultados_por_semente.csv").groupby("celula", sort=False):
            pre = macro_prefix(by_id[cid])
            lines += [f"\\newcommand{{\\{pre}BestIterMin}}{{{int(g.melhor_iteracao.min())}}}",
                      f"\\newcommand{{\\{pre}BestIterMax}}{{{int(g.melhor_iteracao.max())}}}",
                      f"\\newcommand{{\\{pre}Iterations}}{{{int(g.iteracoes.max())}}}"]
    if (d / "resultados_por_semente.csv").exists():
        rs = pd.read_csv(d / "resultados_por_semente.csv")
        if "fedpub_esparsidade" in rs:
            for cid, g in rs[rs.fedpub_esparsidade.notna()].groupby("celula", sort=False):
                pre = macro_prefix(by_id[cid])
                pct = lambda x: f"{100 * x:.1f}"
                lines += [  # três níveis: média global (média das médias por semente), médias por semente, faixa por cliente
                    f"\\newcommand{{\\{pre}MaskSparsityPct}}{{{pct(g.fedpub_esparsidade.mean())}}}",
                    f"\\newcommand{{\\{pre}MaskSparsitySeedMinPct}}{{{pct(g.fedpub_esparsidade.min())}}}",
                    f"\\newcommand{{\\{pre}MaskSparsitySeedMaxPct}}{{{pct(g.fedpub_esparsidade.max())}}}",
                    f"\\newcommand{{\\{pre}MaskSparsityClientMinPct}}{{{pct(g.fedpub_esparsidade_cliente_min.min())}}}",
                    f"\\newcommand{{\\{pre}MaskSparsityClientMaxPct}}{{{pct(g.fedpub_esparsidade_cliente_max.max())}}}",
                    f"\\newcommand{{\\{pre}MaskThreshold}}{{{g.fedpub_limiar_mascara.iloc[0]:g}}}",
                    f"\\newcommand{{\\{pre}AlphaMin}}{{{g.fedpub_alpha_min.min():.3f}}}",
                    f"\\newcommand{{\\{pre}AlphaMax}}{{{g.fedpub_alpha_max.max():.3f}}}",
                    f"\\newcommand{{\\{pre}AlphaDiagMin}}{{{g.fedpub_alpha_diag_min.min():.3f}}}",
                    f"\\newcommand{{\\{pre}AlphaDiagMax}}{{{g.fedpub_alpha_diag_max.max():.3f}}}",
                    f"\\newcommand{{\\{pre}AlphaSelfMean}}{{{g.fedpub_alpha_diag_media.mean():.3f}}}",
                    f"\\newcommand{{\\{pre}AlphaRowRatioMin}}{{{g.fedpub_alpha_razao_linha_max.min():.2f}}}",
                    f"\\newcommand{{\\{pre}AlphaRowRatioMax}}{{{g.fedpub_alpha_razao_linha_max.max():.2f}}}",
                    f"\\newcommand{{\\{pre}AlphaEntropyMin}}{{{g.fedpub_alpha_entropia_norm_min.min():.3f}}}",
                    f"\\newcommand{{\\{pre}DenseCommGB}}{{{g.fedpub_comm_densa_bytes.mean() / 1e9:.2f}}}",
                    f"\\newcommand{{\\{pre}Uniform}}{{{1 / int(g.k.iloc[0]):.3f}}}"]
    if (d / "datasets.csv").exists():
        for r in pd.read_csv(d / "datasets.csv").itertuples():
            pre = "TrafficCarga" if r.alvo == "loading" else f"Traffic{dsname[r.dataset]}"
            for col, nm in (("janelas_treino", "TrainWindows"), ("janelas_validacao", "ValWindows"), ("janelas_teste", "TestWindows")):
                v = getattr(r, col)
                if v == v:
                    lines.append(f"\\newcommand{{\\{pre}{nm}}}{{{int(v)}}}")
            if "alvo_ausente_pct" in r._fields and r.alvo_ausente_pct == r.alvo_ausente_pct:
                lines += [f"\\newcommand{{\\{pre}MissingPct}}{{{r.alvo_ausente_pct:.1f}}}",
                          f"\\newcommand{{\\{pre}TargetMean}}{{{r.alvo_media_observada:.2f}}}"]
    names = [l.split("{")[1].split("}")[0] for l in lines if l.startswith("\\newcommand")]
    assert len(names) == len(set(names)), "macros duplicadas"
    path.write_text("\n".join(lines) + "\n")
    return len(names)


COLORS = {"FedAvg": "#2a78d6", "FedGNN-Halo": "#eb6834", "FedProx": "#1baf7a", "Ditto": "#eda100", "Local": "#e87ba4"}
MARKERS = {"FedAvg": "o", "FedGNN-Halo": "s", "FedProx": "^", "Ditto": "D", "Local": "v"}
# Rótulos das figuras por idioma (o artigo é em inglês). Os dados e as cores não mudam com o idioma.
FIG_TEXT = {
    "pt": {"suffix": "", "cent": "Centralizado", "ylabel": "MAE de teste (média ± desvio-padrão)",
           "sunt": "SUNT, embarques por hora", "metrla": "METR-LA, velocidade (mph)",
           "xpaired": "Diferença pareada de MAE contra o FedAvg\n(IC95%, t pareado; negativo favorece o método)",
           "names": {"Ditto": "Ditto (seleção adaptada)"}, "terms": {}},
    "en": {"suffix": "_en", "cent": "Centralized (reference)", "ylabel": "Test MAE (mean ± sd, 5 seeds)",
           "sunt": "SUNT, boardings (W168/H24)", "metrla": "METR-LA, speed (W12/H12)",
           "xpaired": "Paired MAE difference vs. FedAvg\n(95% CI, paired t; negative favors the method)",
           "names": {"Ditto": "Ditto (adapted selection)", "FedGNN-Halo": "FedGNN-Halo", "FED-PUB (adaptação)": "FED-PUB (adapted)"},
           "terms": {"(complementar)": "(complementary)"}},
}


def write_figures(d: Path, summ, comp, lang: str = "pt"):
    T = FIG_TEXT[lang]
    disp = lambda name: T["names"].get(name, name)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "fed-stgnn-traffic", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#5e5e59", "ytick.color": "#5e5e59", "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "axes.axisbelow": True})

    def mae_panel(ax, tag, ks, methods):
        cent = summ[summ.celula == f"{tag}_cent"]
        if len(cent):
            ax.axhline(cent.MAE_media.iloc[0], color="#8a8a85", lw=1.2, ls="--", label=T["cent"])
        width = 0.13
        for j, (label, pre) in enumerate(methods):
            xs, ys, es = [], [], []
            for i, k in enumerate(ks):
                s = summ[summ.celula == f"{tag}_{pre}_k{k}"]
                if len(s):
                    xs.append(i + (j - (len(methods) - 1) / 2) * width)
                    ys.append(s.MAE_media.iloc[0])
                    es.append(s.MAE_dp.iloc[0])
            if xs:
                ax.errorbar(xs, ys, yerr=es, fmt=MARKERS[label], color=COLORS[label], ms=6, lw=1.4, capsize=2.5,
                            mec="#fcfcfb", mew=0.8, label=disp(label))
        ax.set_xticks(range(len(ks)), [f"$k = {k}$" for k in ks])
        ax.set_ylabel(T["ylabel"])
        if lang == "pt":                 # vírgula decimal nas figuras em português (1.24.0)
            from .export_sunt_analysis import _comma_axis
            _comma_axis(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(d / f"{name}{T['suffix']}.svg", metadata={"Date": None, "Creator": None})
        fig.savefig(d / f"{name}{T['suffix']}.png", dpi=200, metadata={"Software": None})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    mae_panel(ax, "sunt", (6, 20, 166), [("FedAvg", "fedavg"), ("FedProx", "fedprox"), ("Ditto", "ditto"), ("FedGNN-Halo", "halo")])
    ax.set_title(T["sunt"], fontsize=10, loc="left")
    save(fig, "sunt_mae_por_k")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    mae_panel(ax, "metrla", (5, 10, 20), [("FedAvg", "fedavg"), ("FedProx", "fedprox"), ("Ditto", "ditto"), ("Local", "local"),
                                           ("FedGNN-Halo", "halo")])
    ax.set_title(T["metrla"], fontsize=10, loc="left")
    save(fig, "metrla_mae_por_k")

    # Local (referência, lote 6) fica fora desta figura: a perda de 0,1–0,6 contra o FedAvg esticaria o eixo e espremeria
    # os contrastes entre federados, que são o objeto dela. Ele segue na tabela exploratória, nas macros e nos CSVs.
    sub = comp[comp.familia.str.startswith("SUNT k=") & (comp.B.str.contains("fedavg")) & ~comp.A.str.startswith("sunt_local")].copy()
    if len(sub):
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.axvline(0, color="#8a8a85", lw=1)
        for i, r in enumerate(sub.itertuples()):
            meth = r.rotulo.split(" − ")[0]
            ax.errorbar(r.dif_MAE_media, i, xerr=[[r.dif_MAE_media - r.IC95_inf], [r.IC95_sup - r.dif_MAE_media]],
                        fmt=MARKERS.get(meth, "o"), color=COLORS.get(meth, "#5e5e59"), ms=6, lw=1.4, capsize=2.5)
        def ylab(r):
            fam = r.familia.replace("SUNT ", "")
            for a, b in T["terms"].items():
                fam = fam.replace(a, b)
            return f"{fam}: {disp(r.rotulo.split(' − ')[0])}"
        ax.set_yticks(range(len(sub)), [ylab(r) for r in sub.itertuples()])
        ax.invert_yaxis()
        ax.set_xlabel(T["xpaired"])
        ax.grid(axis="y", visible=False)
        save(fig, "sunt_diferencas_pareadas")


# ---------------------------------------------------------------- validação dos números exportados
def _close(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= TOL * max(1.0, abs(b))


def check_numbers(dest: Path, records: pd.DataFrame, cells: list[Cell]) -> list[str]:
    """Confere o pacote a partir dos próprios CSVs exportados, com tolerância relativa TOL:
    registros contra as fontes (MAE, RMSE, R², hashes); resumos (MAE, RMSE e R², média e dp) recalculados;
    diferenças pareadas (média, IC95%, p do t e p do Wilcoxon) recalculadas; tabelas e macros re-renderizadas
    a partir dos valores recalculados e comparadas texto a texto com as do pacote."""
    errors = []
    runs = pd.read_csv(dest / "dados" / "resultados_por_semente.csv")
    src = records.set_index("fonte")
    for r in runs.itertuples():
        s = src.loc[r.fonte]
        for col in ("test_MAE", "test_RMSE", "test_R2"):
            a, b = getattr(r, col), s[col]
            if not (isinstance(b, float) and math.isfinite(b) and abs(a - b) <= 1e-9 * max(1.0, abs(b))):
                errors.append(f"{r.fonte}: {col} exportado {a} ≠ fonte {b}")
        if sha256(ROOT / r.fonte) != r.sha256:
            errors.append(f"{r.fonte}: hash diverge")
    part = pd.read_csv(dest / "dados" / "particoes.csv")
    errors += partition_link_errors(runs, part)
    for r in part.itertuples():
        pdir = ROOT / r.particao
        if sha256(pdir / "client_assignments.npy") != r.client_assignments_sha256:
            errors.append(f"{r.particao}: client_assignments.npy diverge do hash exportado")
        if _array_sha256(_adjacency(r.dataset)) != r.adjacencia_sha256:
            errors.append(f"{r.dataset}: adjacência diverge do hash exportado")
    by_id = {c.id: c for c in cells}
    summ = pd.read_csv(dest / "dados" / "resumo_por_celula.csv")
    re_summ = summarize(runs).set_index("celula")
    for r in summ.itertuples():
        e = re_summ.loc[r.celula]
        for col in ("MAE_media", "MAE_dp", "RMSE_media", "RMSE_dp", "R2_media", "R2_dp"):
            if not _close(getattr(r, col), float(e[col])):
                errors.append(f"{r.celula}: {col} do resumo não confere com os registros")
    comp = pd.read_csv(dest / "dados" / "diferencas_pareadas.csv")
    re_comp = comparisons(runs, by_id).set_index(["A", "B"])
    for r in comp.itertuples():
        e = re_comp.loc[(r.A, r.B)]
        for col in ("dif_MAE_media", "IC95_inf", "IC95_sup", "p_t_pareado", "p_wilcoxon_exato", "p_t_holm", "p_wilcoxon_holm"):
            if not _close(getattr(r, col), float(e[col])):
                errors.append(f"{r.A} − {r.B}: {col} não confere")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "tabelas"
        t.mkdir()
        shutil.copytree(dest / "dados", Path(tmp) / "dados")      # macros de estrutura leem os CSVs exportados
        write_tables(t, summarize(runs), comparisons(runs, by_id), by_id)
        write_macros(t / "macros_resultados.tex", summarize(runs), comparisons(runs, by_id), by_id)
        for f in sorted(t.iterdir()):
            if f.read_text() != (dest / "tabelas" / f.name).read_text():
                errors.append(f"tabelas/{f.name}: re-renderização a partir do CSV difere do pacote")
    ev_seed_path = dest / "dados" / "sunt_evolucao_validacao_por_semente.csv"
    ev_sum_path = dest / "dados" / "sunt_evolucao_validacao_resumo.csv"
    ev_cnt_path = dest / "dados" / "sunt_evolucao_validacao_contagens.json"
    if not ev_seed_path.exists() or not ev_sum_path.exists() or not ev_cnt_path.exists():
        errors.append("dados/sunt_evolucao_validacao_*.{csv,json} ausente")
    else:
        ev_seed = pd.read_csv(ev_seed_path, keep_default_na=False)
        ev_cnt = json.loads(ev_cnt_path.read_text())
        runs_idx = runs.set_index(["celula", "semente"])
        if len(ev_seed) != 80:
            errors.append(f"sunt_evolucao_validacao_por_semente.csv: esperadas 80 linhas, encontradas {len(ev_seed)}")
        fed166 = ev_seed[ev_seed.celula.isin(["sunt_fedavg_k166", "sunt_fedprox_k166", "sunt_ditto_k166", "sunt_halo_k166"])]
        c166 = ev_cnt.get("federados_k166") or {}
        lim_tail = int(c166.get("limiar_ultimos_10pct", 901))
        lim_blk = int(c166.get("limiar_bloco_final_observado", 983))
        if int((fed166.melhor_iteracao.astype(int) >= lim_tail).sum()) != int(c166.get("execucoes_ultimos_10pct", -1)):
            errors.append("sunt_evolucao_validacao_contagens.json: contagem nos últimos 10% diverge do CSV")
        if int((fed166.melhor_iteracao.astype(int) >= lim_blk).sum()) != int(c166.get("execucoes_desde_limiar_bloco_final", -1)):
            errors.append(f"sunt_evolucao_validacao_contagens.json: contagem >= {lim_blk} diverge do CSV")
        for r in ev_seed.itertuples():
            key = (r.celula, int(r.semente))
            if key not in runs_idx.index:
                errors.append(f"evolucao_sunt {key}: execução não pertence a resultados_por_semente.csv")
                continue
            ref = runs_idx.loc[key]
            if int(r.iteracoes) != int(ref.iteracoes) or int(r.melhor_iteracao) != int(ref.melhor_iteracao):
                errors.append(f"evolucao_sunt {key}: iterações/checkpoint divergem de resultados_por_semente.csv")
            v_h, v_m = float(r.val_MAE_checkpoint_historico), float(ref.val_MAE)
            if abs(v_h - v_m) > CHECKPOINT_ATOL + CHECKPOINT_RTOL * max(1.0, abs(v_m)):
                errors.append(f"evolucao_sunt {key}: MAE no checkpoint do histórico ({v_h}) ≠ val_MAE ({v_m})")
            if sha256(ROOT / r.historico) != r.historico_sha256:
                errors.append(f"evolucao_sunt {key}: hash de {r.historico} diverge")
            if r.historico_sonda and (
                sha256(ROOT / r.historico_sonda) != r.historico_sonda_sha256
                or r.historico_sonda_sha256 != r.historico_sha256
            ):
                errors.append(f"evolucao_sunt {key}: hash de {r.historico_sonda} diverge do histórico final")
            if sha256(ROOT / r.config) != r.config_sha256:
                errors.append(f"evolucao_sunt {key}: hash de {r.config} diverge")
    for stem in (STEM_FEDERADOS, STEM_LOCAL):
        for suffix in (".pdf", ".png", "_pb.png", ".svg"):
            fp = dest / "figuras" / f"{stem}{suffix}"
            if not fp.is_file() or fp.stat().st_size == 0:
                errors.append(f"figuras/{stem}{suffix}: figura ausente ou vazia")
    return errors


def content_hashes(dest: Path, skip_meta: bool = True) -> dict[str, str]:
    out = {}
    for p in sorted(dest.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and not (skip_meta and p.name in ("provenance.json", "VALIDACAO.md")):
            out[str(p.relative_to(dest))] = sha256(p)
    return out


def comparable(dest: Path) -> dict[str, str]:
    """Hashes de todos os arquivos, com o carimbo de data retirado do provenance.json e do VALIDACAO.md."""
    out = content_hashes(dest, skip_meta=False)
    prov = json.loads((dest / "provenance.json").read_text())
    prov.pop("gerado_em", None)
    out["provenance.json"] = hashlib.sha256(json.dumps(prov, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    val = "\n".join(l for l in (dest / "VALIDACAO.md").read_text().splitlines() if not l.startswith("- Gerado em"))
    out["VALIDACAO.md"] = hashlib.sha256(val.encode()).hexdigest()
    return out


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "worktree_dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def committed_source_errors(records: pd.DataFrame, cells: list[Cell]) -> list[str]:
    """Não confundir árvore limpa com fontes não rastreadas: exigir presença no HEAD."""
    committed = set(_git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
    required = {"experiments/export_traffic.py", "experiments/export_controles_internos.py",
                "experiments/export_sunt_analysis.py", "experiments/export_sunt_direcao.py",
                "experiments/convergencia_figuras.py", "experiments/export_panorama.py",
                "experiments/export_contrast_figures.py"}
    for cell in cells:
        required.update([cell.config, *cell.extra_configs])
        for source in select(records, cell).fonte:
            required.add(source)
    # Insumos novos: históricos, configurações, manifestos e instruções de recuperação.
    for folder in ["results/completar_controles_20261114", "results/controle_pems_none_k5_20261115",
                   "config/completar_controles_20261114"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix not in {".pt", ".log", ".pyc"}:
                required.add(path.relative_to(ROOT).as_posix())
    return ["fonte ausente do commit HEAD: " + path for path in sorted(required - committed)]


def build(dest: Path, generated_at: str) -> tuple[bool, list[str], list[str], dict]:
    """Uma geração completa e independente: carrega as fontes, valida, escreve e confere o pacote em `dest`."""
    _manifest.cache_clear()
    cells = _cells()
    records = load_records()
    errors, notes = validate(records, cells)
    errors += committed_source_errors(records, cells)
    dest.mkdir(parents=True)
    info = {}
    if not errors:
        try:
            info = write_package(dest, records, cells, notes)
        except ValueError as e:
            errors.append(str(e))
        if not errors:
            errors += check_numbers(dest, records, cells)
    valid = not errors
    configs = sorted({c.config for c in cells} | {x for c in cells for x in c.extra_configs})
    sel = (pd.read_csv(dest / "dados" / "resultados_por_semente.csv") if (dest / "dados" / "resultados_por_semente.csv").exists()
           else pd.DataFrame(columns=["fonte"]))
    sources = records[records.fonte.isin(sel.fonte)]
    prov = {
        "pacote": "generated-traffic", "fonte_experimental": "fed-stgnn-traffic", "exportador": "experiments/export_traffic.py",
        "versao_exportador": EXPORTER_VERSION, "sha256_exportador": sha256(Path(__file__)),
        "sha256_exportador_contrastes": sha256(ROOT / "experiments/export_contrast_figures.py"),
        "sha256_exportador_panorama": sha256(ROOT / "experiments/export_panorama.py"),
        "sha256_exportador_analises": sha256(ROOT / "experiments/export_sunt_analysis.py"),
        "analises_sunt": info.get("analises_sunt", {}),
        "controles_arestas_internas": info.get("controles_arestas_internas", {}),
        "sha256_exportador_piloto_direcao": sha256(ROOT / "experiments/export_sunt_direcao.py"),
        "piloto_direcao_sunt": info.get("piloto_direcao_sunt", {}),
        "sha256_exportador_evolucao_sunt": sha256(ROOT / "experiments/convergencia_figuras.py"),
        "evolucao_validacao_sunt": info.get("evolucao_validacao_sunt", {}),
        "git_commit": _git("rev-parse", "HEAD"), "git_worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!generated-traffic")),
        "gerado_em": generated_at, "python": sys.version.split()[0], "plataforma": platform.platform(),
        "pacotes": {m: __import__(m).__version__ for m in ("numpy", "pandas", "scipy", "matplotlib")},
        "sementes": sorted(SEEDS), "valido": valid, "erros": errors, "observacoes": notes,
        "celulas": [{"id": c.id, "dataset": c.dataset, "pasta": c.folder, "k": c.k, "variante": c.variant, "metodo": c.method,
                     "grupo": c.group, "opcional": c.optional, "config": c.config, "configs_extra": c.extra_configs} for c in cells],
        "configs_versao_atual": {p: sha256(ROOT / p) for p in configs if (ROOT / p).exists()},
        "nota_configs": "por registro, `config_sha256_no_commit` em dados/resultados_por_semente.csv é o hash da config na revisão do "
                        "commit que versionou aquele resultado (aproximação da versão usada; a de lançamento pode ser anterior). "
                        "`configs_versao_atual` traz a versão atual.",
        "insumos_estruturais": (pd.read_csv(dest / "dados" / "particoes.csv")[
            ["dataset", "k", "particao", "grafo_estatisticas", "adjacencia_sha256", "partitions_csv_sha256",
             "client_assignments_sha256", "node_ids_sha256", "metadata_sha256"]].to_dict("records")
            if (dest / "dados" / "particoes.csv").exists() else []),
        "nota_insumos_estruturais": "particoes.csv, halo.csv e as macros de corte vêm da adjacência e de client_assignments.npy "
                                    "atuais (hashes acima; adjacência com hash do conteúdo da matriz). Cada execução federada "
                                    "foi conferida contra essas partições pelo caminho registrado e pelos nós por cliente "
                                    "(client_nodes); o metrics.json não guarda hash da partição, então o vínculo é por caminho "
                                    "e tamanhos, não por hash.",
        "grafo_gtfs_sunt": {"arquivo": str(SUNT_GTFS_GRAPH.relative_to(ROOT)), "sha256": SUNT_GTFS_SHA256,
                            "url": SUNT_GTFS_URL, "repositorio_commit": SUNT_GTFS_COMMIT,
                            "uso": "descritivo: cobertura de nós e pares do grafo do estudo em relação ao grafo do GTFS "
                                   "(dados/sunt_grafo_gtfs.csv e macros \\TrafficSUNTGTFS...); não entra no treino"},
        "manifesto_checkpoints": {"arquivo": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST) if MANIFEST.exists() else ""},
        "fontes": [{"arquivo": r.fonte, "sha256": r.sha256, "checkpoint": r.checkpoint, "checkpoint_sha256": r.checkpoint_sha256,
                    "gpu": r.gpu, "ambiente": r.ambiente} for r in sources.itertuples()],
        "estatistica": "diferença pareada de MAE de teste por semente (A − B; negativo favorece A), testes bilaterais, α = 0,05. "
                       "IC95% pelo t pareado (supõe diferenças aproximadamente normais), NÃO ajustado. Wilcoxon exato como "
                       "sensibilidade (com 5 pares, p bilateral mínimo de 0,0625). Ajuste de Holm dentro de cada família de "
                       "'familia_inferencial': principal = os 9 contrastes halo × FedAvg (SUNT k=6/20/166, METR-LA k=5/10/20, "
                       "PEMS-BAY k=5/10/20; ampliada de 7 para 9 no 1.9.0, com o lote 1, por decisão registrada antes de ver "
                       "os resultados: results/lote1_lacunas/decisao_familias_holm.md); exploratória do PEMS-BAY ampliada de 4 para 11 no "
                       "1.10.0 (Local − FedAvg e Local − halo k=5/10/20, lote 2; Ditto − FedAvg k=20, lote 3), também por decisão "
                       "registrada antes: results/lote2_local/ e results/lote3_ditto_pemsbay/decisao_familias_holm.md), e de 11 para 12 no 1.15.0 (Ditto − FedAvg k=10, lote 4: "
                       "results/lote4_ditto_pemsbay_k10/decisao_familias_holm.md); exploratória do SUNT ampliada de 4 para 6 no 1.16.0 (FedProx − FedAvg e "
                       "Ditto − FedAvg k=166, lote 5: results/lote5_sunt_k166/decisao_familias_holm.md), e de 6 para 12 no 1.20.0 (Local − FedAvg e "
                       "Local − halo k=6/20/166, lote 6, registrada antes dos resultados: results/lote6_sunt_local/REGISTRO.md §7); secundária = halo × none no METR-LA (3); caracterização = FedAvg × Centralizado por "
                       "dataset; exploratória = demais comparadores por dataset; complementar = carga + grafo direcionado e "
                       "FED-PUB; controle = regra de parada federada nos centralizados das rodovias (1.22.0; duas comparações "
                       "por dataset, registradas antes das execuções em results/centralized_parada_federada/PROTOCOLO.md). Essa organização foi DEFINIDA RETROSPECTIVAMENTE, depois de conhecer os resultados (revisão "
                       "externa de 04/10), e deve ser apresentada assim. Classificação, do ponto de vista de A: 'melhora'/'piora' "
                       "'com evidência após ajuste' (p_t Holm < 0,05), 'com evidência nominal' (p_t < 0,05 só sem ajuste) ou "
                       "'observada' (sem evidência). Os p nominais são preservados.",
        "ambientes": "servidor do laboratório (GPUs A16/L40S) e estação de trabalho (RTX 2060); hardware por registro em 'fontes'; "
                     "registros antigos sem GPU gravada constam como 'não registrado'. Não comparar tempos entre ambientes.",
        **({"resumo": {k: v for k, v in info.items() if k not in ("celulas_exportadas", "analises_sunt", "piloto_direcao_sunt", "evolucao_validacao_sunt")}} if valid else {}),
    }
    if valid:
        (dest / "README.md").write_text(README)
    prov["artefatos"] = content_hashes(dest)
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
    (dest / "VALIDACAO.md").write_text(validation_report(prov, info))
    return valid, errors, notes, info


def main():
    generated_at = datetime.now(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        valid, errors, notes, info = build(a, generated_at)
        if valid:
            valid_b, errors_b, _, _ = build(b, generated_at)
            ha, hb = comparable(a), comparable(b)
            if not valid_b or ha != hb:
                diff = sorted(k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k))
                errors.append(f"segunda geração independente difere em: {diff}" if diff else f"segunda geração inválida: {errors_b}")
                valid = False
                prov = json.loads((a / "provenance.json").read_text())
                prov.update(valido=False, erros=errors)
                (a / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
                (a / "VALIDACAO.md").write_text(validation_report(prov, info))
        target = OUT if valid else INVALID
        # Pacote válido substitui o anterior e apaga um inválido antigo; falha grava só o INVALIDO e preserva o
        # último pacote válido (que continua conferindo com as fontes que ele lista).
        for old in ((OUT, INVALID) if valid else (INVALID,)):
            if old.exists():
                shutil.rmtree(old)
        shutil.copytree(a, target, ignore=shutil.ignore_patterns(".*"))
    print(("PACOTE VÁLIDO em " if valid else "PACOTE INVÁLIDO em ") + str(target.relative_to(ROOT)))
    for e in errors:
        print("  ERRO:", e)
    for n in notes:
        print("  obs:", n)
    sys.exit(0 if valid else 1)


def validation_report(prov, info) -> str:
    ok = prov["valido"]
    lines = ["# Relatório de validação do pacote `generated-traffic`", "",
             f"- Resultado: **{'VÁLIDO' if ok else 'INVÁLIDO'}**",
             f"- Exportador {prov['versao_exportador']} ({prov['exportador']}), commit `{prov['git_commit'][:7]}`"
             f"{' (árvore com alterações não versionadas)' if prov['git_worktree_dirty'] else ''}",
             f"- Gerado em {prov['gerado_em']}", ""]
    if ok:
        lines += [f"- {info['registros_selecionados']} registros selecionados em {len(info['celulas_exportadas'])} células; "
                  f"{info['comparacoes']} comparações pareadas; {info['macros']} macros.", "",
                  "- Ampliação de arestas internas: 35 registros em sete células adicionais e 14 contrastes em uma família complementar de Holm, exportados separadamente em dados/controles_arestas_internas/.", "",
                  "## O que o exportador verifica (todas as conferências passaram)", "",
                  "1. Cada célula não opcional tem exatamente as sementes 7, 42, 101, 123 e 314, sem duplicatas.",
                  "2. Todo registro selecionado tem MAE de teste finito.",
                  "3. Metadados gravados coerentes com o papel da célula: alvo, entradas, grafo, janela e horizonte.",
                  "4. Nós, janela e horizonte únicos por dataset e alvo entre as células selecionadas.",
                  "5. SUNT principal: 600 épocas no Centralizado e 600 / 600 / 1 000 rodadas em k = 6 / 20 / 166. Nas "
                  "avaliações de checkpoint de sondagem (Centralizado e FedAvg), a sondagem de origem existe, tem a mesma "
                  "semente, melhor rodada e total de rodadas, não tem teste, e o `best_model.pt` associado a ela tem hash no manifesto.",
                  f"6. Consistência numérica, com tolerância relativa {TOL:g}: MAE, RMSE e R² de cada registro conferem com o "
                  "metrics.json de origem, cujo hash confere; a partir dos CSVs exportados, os resumos (média e dp de MAE, "
                  "RMSE e R²) e as diferenças pareadas (média, IC95%, p do t e p do Wilcoxon) foram recalculados e conferem; "
                  "tabelas e macros re-renderizadas a partir desses valores são idênticas às do pacote.",
                  "7. Estatísticas estruturais (particoes.csv e macros de corte): cada execução federada aponta para a "
                  "partição usada nas estatísticas e tem os mesmos nós por cliente (`client_nodes`) que "
                  "`bincount(client_assignments.npy)`; os hashes das atribuições, de `node_ids.npy`, `metadata.json`, "
                  "`partitions.csv` e do conteúdo da adjacência estão em `insumos_estruturais` no provenance e foram "
                  "reconferidos contra o disco. O metrics.json não guarda hash da partição: o vínculo é por caminho e tamanhos.",
                  "8. Duas gerações independentes, cada uma recarregando as fontes do disco, produziram arquivos idênticos "
                  "(todos, inclusive provenance.json e este relatório, exceto o carimbo de data).", "",
                  "## O que não é verificado", "",
                  "- Arquitetura, normalização e configuração efetiva de cada treino (a partição é conferida como no item 7). Por registro, há o hash da "
                  "config na revisão do commit que versionou o resultado, que aproxima a versão usada, sem garanti-la.",
                  "- Que o checkpoint produziu a métrica: o manifesto identifica o `best_model.pt` ASSOCIADO a cada resultado "
                  "(rastreabilidade), mas sem vínculo gravado na avaliação original nem reprodução da avaliação ele não "
                  "demonstra, sozinho, que aquele arquivo produziu o número (reprodução). O exportador não carrega checkpoints.",
                  "- Comparabilidade entre células: hashes atestam integridade, não comparabilidade experimental.",
                  "- Multiplicidade: os p ajustados (Holm) seguem famílias definidas retrospectivamente, depois de conhecer os "
                  "resultados; os IC95% não são ajustados (ver `estatistica` no provenance).", ""]
    else:
        lines += ["## Erros", ""] + [f"- {e}" for e in prov["erros"]] + [""]
    if prov["observacoes"]:
        lines += ["## Observações (não invalidam)", ""] + [f"- {n}" for n in prov["observacoes"]] + [""]
    if ok and prov.get("analises_sunt"):
        lines += ["## Análises adicionais do SUNT", "",
                  "Estratificação e controles centralizados de grafo: hashes das métricas, scripts e protocolos conferidos; "
                  "MAE e denominadores de cada estrato reconstruídos a partir de per_node; diferenças das cinco sementes, "
                  "médias, desvios e IC95% recalculados. Auditorias de origem sem falhas. "
                  "São análises exploratórias com IC95% não ajustados, fora das famílias Holm originais. "
                  "Procedência em analises_sunt no provenance.json; tabelas, macros e figuras próprias no pacote.", ""]
    if ok and prov.get("piloto_direcao_sunt"):
        lines += ["## Piloto de orientação do grafo do SUNT (só validação)", "",
                  "Insumos versionados e com hash; auditoria das topologias sem falhas; 15 execuções com rc=0, 600 épocas, "
                  "determinismo estrito, teste não avaliado, adjacência igual à auditada e mesma inicialização entre os "
                  "braços de cada semente. MAE e RMSE reconstituídos de validation_detail.npz e conferidos com o metrics.json; "
                  "diferenças pareadas e por estrato recalculadas e conferidas com o relatório do piloto. Descritivo: sem IC, "
                  "sem p e fora das famílias Holm. Procedência em piloto_direcao_sunt no provenance.json.", ""]
    if ok and prov.get("evolucao_validacao_sunt"):
        ev = prov["evolucao_validacao_sunt"]
        c166 = ev["contagens_automaticas"]["federados_k166"]
        cloc = ev["contagens_automaticas"]["local"]
        lines += ["## Evolução do MAE de validação no SUNT (orçamento limitado)", "",
                  f"{ev['execucoes_verificadas']} execuções selecionadas nas {ev['celulas_verificadas']} células do SUNT "
                  "(Centralizado, FedAvg, FedProx, Ditto, FedGNN-Halo e Local em k = 6, 20 e 166; sementes 7, 42, 101, 123 e 314): "
                  "todos os `history.csv`, `metrics.json` e `config/*.yaml` existem, estão versionados no git e conferem em "
                  "orçamento (600 épocas no Centralizado; 600, 600 e 1 000 rodadas em k = 6, 20 e 166), número de observações "
                  f"de validação por etapa ({ev['observacoes_validacao_por_etapa']:,}".replace(",", " ") + ") e seleção de checkpoint "
                  "(`argmin(validation_MAE) == melhor_iteracao`, coincidente com `metrics.json`). Nas "
                  f"{ev['historicos_sonda_verificados']} execuções avaliadas a partir de sondagens cegas (`sunt_cent` e `sunt_fedavg_k*`), "
                  "o `history.csv` final tem o mesmo SHA-256 do `history.csv` da sonda de origem. No Ditto, utiliza-se "
                  "explicitamente a validação dos modelos personalizados (`validation_MAE`, distinta de `validation_global_MAE`). "
                  f"Contagens recalculadas automaticamente dos históricos: em k = 166, {c166['execucoes_ultimos_10pct']}/{c166['total_execucoes']} "
                  f"execuções federadas selecionaram o checkpoint nos últimos 10% do orçamento (rodadas >= {c166['limiar_ultimos_10pct']}) e "
                  f"{c166['execucoes_desde_limiar_bloco_final']}/{c166['total_execucoes']} a partir da rodada {c166['limiar_bloco_final_observado']} "
                  f"(inclusive); no Local, {cloc['sementes_com_piora_pos_minimo']}/{cloc['total_execucoes']} execuções terminaram "
                  "com MAE de validação acima do checkpoint selecionado. O exportador garante a **reprodução determinística das figuras e "
                  "estatísticas** a partir dos históricos e métricas versionados, sem executar nem reproduzir treinamentos. "
                  f"Procedência e hashes dos {len(ev['historicos'])} arquivos `history.csv` auditados em `evolucao_validacao_sunt` no `provenance.json`.", ""]
    return "\n".join(lines)


README = """# generated-traffic

Pacote de resultados da dissertação, defesa e artigo, gerado por `python -m experiments.export_traffic` no
repositório `fed-stgnn-traffic`. **Não editar à mão**: regenere. Validação em `VALIDACAO.md`; procedência
(fontes, hashes, configs, commit, hardware por registro) em `provenance.json`.

## Conteúdo

- `dados/sunt_stratified_analysis/` e `dados/sunt_graph_controls/`: análises adicionais auditadas do SUNT,
  com protocolos, manifestos, auditorias, estratos e contrastes. As tabelas `sunt_estratos_fronteira.tex`,
  `sunt_controles_grafo.tex` e `macros_analises_sunt.tex` são próprias dessa etapa. IC95% exploratórios
  não ajustados, sem correção de multiplicidade; não alteram as famílias Holm originais. Figuras de
  prévia de layout não são exportadas. Não há novo treinamento no exportador.
- `dados/sunt_piloto_direcao/`, `tabelas/piloto_direcao.tex` e as macros `\\TrafficSUNTDir...` (desde o 1.18.0, em
  `macros_analises_sunt.tex`): piloto de orientação do grafo do SUNT no centralizado com alvo de embarques: versões
  simétrica (vigente), direcionada (sentidos observados) e invertida, com a mesma afinidade por par. **Só validação**
  (o teste não foi avaliado; a validação também escolheu os checkpoints); não se compara com as métricas de teste.
  Descritivo: diferenças pareadas por semente sem IC, sem p e fora das famílias Holm.
- `dados/resultados_por_semente.csv`: um registro por célula e semente (métricas de teste, validação,
  iterações, tipo de registro, checkpoint de origem, hardware e hash da fonte). Ausente fica vazio.
- `dados/resumo_por_celula.csv`: média e desvio padrão amostral por célula.
- `figuras/panorama_relativo_centralizado.{pdf,png,svg}` e `dados/panorama_relativo_centralizado.csv` (1.28.0):
  três painéis com todos os métodos, incluindo Local; 100 × (MAE médio do método − MAE médio centralizado) /
  MAE médio centralizado. Sem barras de dispersão ou classificação estatística; escalas verticais distintas
  entre rodovias e SUNT, posições de k igualmente espaçadas e linhas apenas como guia visual.
- `dados/diferencas_pareadas.csv`: A − B por semente, IC95% (t pareado), p do t e Wilcoxon exato (sensibilidade).
- `dados/treinos_de_sondagem.csv`: treinos das sondagens cegas do SUNT (sem teste), cujos checkpoints foram
  avaliados nas células finais. Não são repetições adicionais.
- `dados/historico_fora_da_evidencia.csv`: registros históricos e exploratórios, fora da evidência.
- `dados/particoes.csv`: estrutura das partições Leiden (arestas dirigidas e pares cortados, nós de fronteira em
  qualquer sentido e nós com vizinho de entrada externo; definições no docstring de `partition_stats`).
  Desde o 1.26.1, `nos_sem_conexoes` conta os nós sem arestas com outros nós em nenhum dos sentidos,
  desconsiderando autolaços, e fornece as macros `...IsolatedNodes` dos três conjuntos.
- `dados/sunt_grafo_gtfs.csv` e as macros `\\TrafficSUNTGTFS...` (desde o 1.26.0): cobertura do grafo do estudo no SUNT em
  relação ao grafo derivado do GTFS distribuído com o conjunto (`graph_gtfs_fev_2024.parquet`, SHA-256 e commit de origem no
  provenance): paradas em comum, paradas com séries ausentes do GTFS e o inverso, e pares não direcionados em comum ou
  exclusivos de cada grafo (autolaços do GTFS à parte). Descritivo; o grafo do GTFS não entra no treino.
- `dados/halo.csv`, `dados/comunicacao_por_execucao.csv`, `dados/datasets.csv`: nós virtuais, vizinho único,
  exatidão do halo e payload; comunicação de modelos por execução (contabilização do código, não tráfego medido);
  nós, janela, horizonte e número de janelas por split.
- `dados/custo_computacional_por_execucao.csv` e `dados/custo_computacional_por_celula.csv` (desde o 1.16.0): tempo de
  treino gravado pelo próprio treino (`training_seconds`), em horas e por iteração, com GPU e ambiente. **Descritivos**:
  GPU, concorrência com outros treinos e período variam entre registros, mesmo no mesmo servidor; não servem para
  ordenar métodos. A diferença estrutural de computação local (por exemplo, o Ditto com λ por cliente treina o modelo
  global e um modelo pessoal por candidato a cada rodada) está descrita nos métodos, não medida aqui.
- `tabelas/sunt_custo_k166.tex` e as macros `...TrainHours` e `...SecPerRound` (desde o 1.17.0): SUNT com $k=166$,
  comunicação estimada de modelos, payload do halo e tempo de treino (média ± dp entre sementes) e segundos por rodada,
  para FedAvg, FedProx, Ditto e FedGNN-Halo. Tempos indicativos, medidos no servidor compartilhado (GPUs A16 e L40S).
- `tabelas/*.tex`: fragmentos `tabular` (booktabs), sem legenda, números em `\\num{}` (siunitx: o separador
  decimal segue cada documento), com rótulos em português (a versão em inglês
  para o artigo ainda não existe); `tabelas/macros_resultados.tex`: macros. As macros `...HaloK...RecoveryPct`
  existem só no METR-LA e são **descritivas** em todos os k: (FedAvg − halo) / (FedAvg − Centralizado) sobre as
  médias, não redução percentual de MAE e não evidência inferencial (em k=5 e k=10 a diferença halo − FedAvg não é
  detectável). No SUNT não há macro de recuperação.
- Diferenças pareadas: p nominais preservados e p com Holm dentro de famílias definidas **retrospectivamente**
  (principal = 9 contrastes halo × FedAvg, ampliada de 7 no 1.9.0 com o lote 1; secundária = halo × none no METR-LA; caracterização e comparadores à
  parte, exploratórios); os IC95% não são ajustados. A coluna `classificacao` separa, do ponto de vista de A,
  melhora ou piora observada, com evidência nominal ou com evidência após ajuste.
- Checkpoints: `checkpoint_sha256` identifica o arquivo associado a cada resultado (rastreabilidade), não prova que
  ele produziu a métrica (reprodução).
- `figuras/*.{png,svg}` e `figuras/sunt_evolucao_validacao_{federados,local}.{pdf,png,svg}` (mais `*_pb.png`, desde o 1.21.0):
  MAE por k (SUNT e METR-LA), diferenças pareadas no SUNT e duas figuras separadas da evolução do MAE de validação no SUNT
  sob orçamento limitado, dimensionadas em 16,0 cm (Figura A: Centralizado e métodos federados com linha de trajetórias
  completas e linha de ampliações do trecho final; Figura B: Local sob escala comum), acompanhadas de
  `dados/sunt_evolucao_validacao_por_semente.csv`, `dados/sunt_evolucao_validacao_resumo.csv` e
  `dados/sunt_evolucao_validacao_contagens.json`.

## Figuras e RMSE adicionais (1.29.0)

As figuras `halo_fracao_cortes` e `contrastes_controles_centralizados` usam os contrastes pareados já auditados, com IC95% sem ajuste. A escala relativa divide a diferença e os limites pela média do MAE do FedAvg; não estima contribuições causais. Os CSVs correspondentes preservam a precisão numérica. As tabelas `metrla_rmse` e `pemsbay_rmse` apresentam médias e desvios existentes com três casas, incluindo Local. O eixo do Local usa épocas por modelo.

## Grupos

`principal` (comparação principal), `controle` (halo none, 1hop_min2, Ditto com λ fixo, Centralizado sem
arestas), `referência` (Local = Ditto λ=0) e `complementar` (carga com grafo direcionado; FED-PUB adaptado).
As tabelas principais `metrla_mae.tex`, `pemsbay_mae.tex`, `sunt_mae.tex` e `sunt_rmse.tex` têm o Centralizado numa
coluna própria, "Sem partição", e FedAvg, FedProx, Ditto e FedGNN-Halo nas colunas por k (desde o 1.14.0;
controles retirados no 1.11.0, Local no 1.12.0). Desde o 1.19.0, a menor média de cada coluna de k fica em negrito,
com as casas impressas (empates em negrito nos dois); o Centralizado, sozinho na sua coluna, não. Em `carga_direcionado.tex`,
o negrito marca a melhor média de cada métrica. É descritivo, e não indica diferença detectável. Local e controles ficam em `dados/resumo_por_celula.csv`, nas macros
e nas diferenças pareadas; `sunt_principal.tex` não muda. A macro `...RMSEStd` (desvio do RMSE) existe desde o 1.13.0.
O Ditto fica visível nas tabelas do METR-LA e do PEMS-BAY, com k=5, 10 e 20 nos dois (no PEMS-BAY, k=10 desde o
1.15.0; o Ditto k=20 foi treinado na estação de trabalho, e a referência FedAvg, no servidor). No SUNT, FedProx e
Ditto têm k=166 desde o 1.16.0 (lote 5), e a exploratória do SUNT passa de 4 para 6 comparações. Desde o 1.20.0
(lote 6), o SUNT tem o Local (`sunt_local_k6/20/166`, caminho `ditto.local_only`: só os modelos pessoais
com λ=0, sem treinar o modelo global, com a mesma trajetória pessoal do Ditto λ=0; `results/lote6_sunt_local/`), fora
das tabelas por k, como nas rodovias; a exploratória do SUNT passa de 6 para 12 comparações (Local − FedAvg e
Local − halo). O tempo do Local do SUNT não é comparável ao do Local das rodovias, que ainda treinava o modelo global.

## Ambientes

Servidor do laboratório e estação de trabalho; hardware por registro na procedência. Não comparar tempos
entre ambientes; dentro do mesmo servidor, os tempos também dependem da GPU e da concorrência de cada período
(`dados/custo_computacional_*.csv`). Registros antigos sem GPU gravada constam como "não registrado".
"""


if __name__ == "__main__":
    main()
