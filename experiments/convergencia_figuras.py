"""Evolução do MAE de validação no SUNT sob orçamento limitado de épocas/rodadas.

Lê exclusivamente os `history.csv`, `metrics.json` e arquivos `config/*.yaml` das 16 células do SUNT (embarques,
W168/H24) selecionadas em `generated-traffic/dados/resultados_por_semente.csv`: Centralizado, FedAvg, FedProx,
Ditto (validação dos modelos personalizados), FedGNNHet e Local nas partições k = 6, 20 e 166, nas cinco sementes
{7, 42, 101, 123, 314} (80 execuções). Não executa treinamentos, não reavalia checkpoints e não substitui arquivos
ausentes por execuções de outra campanha.

Gera duas figuras científicas independentes dimensionadas para a largura real da mancha de texto A4 (16,0 cm),
com fontes >= 8 pt no tamanho final de impressão, em PDF vetorial, PNG 300 DPI (cor e escala de cinza) e SVG:
  - Figura A (`sunt_evolucao_validacao_federados`): Centralizado, FedAvg, FedProx, Ditto e FedGNNHet em 2 linhas ×
    3 colunas (trajetórias completas na linha superior e ampliações explícitas do trecho final na linha inferior).
  - Figura B (`sunt_evolucao_validacao_local`): Local em 1 linha × 3 colunas sob escala vertical comum.

Também calcula e exporta automaticamente todas as contagens e estatísticas descritivas dos históricos em
`dados/sunt_evolucao_validacao_contagens.json`, `dados/sunt_evolucao_validacao_por_semente.csv` e
`dados/sunt_evolucao_validacao_resumo.csv`, evitando números derivados manualmente em relatórios.

Uso autônomo (lendo `generated-traffic` já exportado e gravando em `--saida`):
    python -m experiments.convergencia_figuras [--pacote generated-traffic] [--saida results/figuras_convergencia]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .train import ROOT

SEEDS = (7, 42, 101, 123, 314)
KS = (6, 20, 166)
SUNT_BUDGET: dict[int | None, int] = {None: 600, 6: 600, 20: 600, 166: 1000}
ZOOM_START: dict[int, int] = {6: 120, 20: 120, 166: 400}
SUNT_VAL_OBSERVATIONS = 10_867_874
SUNT_NODES = 2903
CHECKPOINT_RTOL = 1e-7   # acomoda redução float32 em GPU ao reavaliar best_model.pt (máx. observado: 6,17e-9)
CHECKPOINT_ATOL = 1e-8

# Largura física exata da mancha de texto A4 da dissertação (16,0 cm -> escala 1:1 na inclusão LaTeX).
FIG_WIDTH_CM = 16.0
CM_TO_IN = 1.0 / 2.54
MIN_FONT_PT = 8.0

# Nomes acadêmicos consistentes entre figuras (sem nomes de variáveis, sufixos internos ou caminhos).
FED_PANEL_METHODS: tuple[tuple[str, str], ...] = (
    ("Centralizado", "cent"),
    ("FedAvg", "fedavg"),
    ("FedProx", "fedprox"),
    ("Ditto", "ditto"),
    ("FedGNNHet", "halo"),
)
LOCAL_METHODS: tuple[tuple[str, str], ...] = (
    ("Local", "local"),
)
ALL_METHODS: tuple[tuple[str, str], ...] = FED_PANEL_METHODS + LOCAL_METHODS

# Paleta sóbria de alto contraste em fundo branco + luminâncias escalonadas, tracejados e marcadores distintos em P&B.
METHOD_COLOR = {
    "Centralizado": "#141413",
    "FedAvg": "#133b78",
    "FedProx": "#1a8f5c",
    "Ditto": "#c98200",
    "FedGNNHet": "#d94814",
    "Local": "#781f46",
}

METHOD_LINESTYLE = {
    "Centralizado": "-",
    "FedAvg": (0, (7.0, 2.0)),
    "FedProx": (0, (2.6, 1.8)),
    "Ditto": (0, (1.1, 1.6)),
    "FedGNNHet": (0, (4.5, 1.7, 1.0, 1.7)),
    "Local": (0, (6.0, 1.6, 1.0, 1.6, 1.0, 1.6)),
}

METHOD_MARKER = {
    "Centralizado": "P",
    "FedAvg": "o",
    "FedProx": "^",
    "Ditto": "D",
    "FedGNNHet": "s",
    "Local": "v",
}

STEM_FEDERADOS = "sunt_evolucao_validacao_federados"
STEM_LOCAL = "sunt_evolucao_validacao_local"


@dataclass(frozen=True)
class AuditedCurve:
    celula: str
    metodo: str
    k: int | None
    semente: int
    unidade: str
    orcamento: int
    melhor_iteracao: int
    curve: pd.Series
    val_mae_checkpoint_hist: float
    val_mae_checkpoint_metrics: float
    historico_rel: str
    historico_sha256: str
    historico_sonda_rel: str
    historico_sonda_sha256: str
    config_rel: str
    config_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _tracked_files() -> frozenset[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return frozenset(line.strip() for line in out.splitlines() if line.strip())


def _tracked(path: Path) -> bool:
    rel = str(path.resolve().relative_to(ROOT))
    return rel in _tracked_files()


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(f"Evolução da validação SUNT: {msg}")


def _close_val(a: float, b: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= CHECKPOINT_ATOL + CHECKPOINT_RTOL * max(1.0, abs(b)))


def expected_sunt_cells() -> list[tuple[str, str, int | None]]:
    out: list[tuple[str, str, int | None]] = [("sunt_cent", "Centralizado", None)]
    for label, token in ALL_METHODS[1:]:
        for k in KS:
            out.append((f"sunt_{token}_k{k}", label, k))
    return out


def _verify_config_correspondence(cid: str, metodo: str, k: int | None, fonte_rel: str, cfg_rel: str, metrics: dict) -> None:
    cfg_path = ROOT / cfg_rel
    _require(cfg_path.is_file(), f"{cid}: config ausente: {cfg_rel}")
    _require(_tracked(cfg_path), f"{cid}: config não versionada no git: {cfg_rel}")
    cfg = yaml.safe_load(cfg_path.read_text())
    _require(cfg.get("datasets") == ["sunt"], f"{cid}: datasets inválido em {cfg_rel}: {cfg.get('datasets')}")
    _require(tuple(cfg.get("seeds") or ()) == SEEDS, f"{cid}: seeds em {cfg_rel} diferem de {SEEDS}")

    win = (cfg.get("windows") or {}).get("sunt") or {}
    _require(
        win.get("input_steps") == 168
        and win.get("horizon") == 24
        and win.get("target") == "n_boardings"
        and win.get("features") == ["n_boardings", "n_alightings"],
        f"{cid}: janela/alvo/features incorretos em {cfg_rel}",
    )
    mdl = cfg.get("model") or {}
    _require(mdl.get("name") == "leconv" and mdl.get("hidden_size") == 64, f"{cid}: modelo incorreto em {cfg_rel}")
    _require(metrics.get("model") == "leconv", f"{cid}: modelo em metrics.json difere de leconv")

    folder = Path(fonte_rel).parts[1]
    _require(cfg.get("output") == f"results/{folder}", f"{cid}: output de {cfg_rel} ({cfg.get('output')}) ≠ results/{folder}")

    if k is None:
        tr = cfg.get("training") or {}
        _require(
            tr.get("learning_rate") == 0.001
            and tr.get("max_epochs") == SUNT_BUDGET[None]
            and tr.get("early_stopping_min_delta") == 0.0
            and tr.get("gradient_clip_norm") == 5.0,
            f"{cid}: bloco training inconsistente em {cfg_rel}",
        )
        return

    block_key = {
        "FedAvg": "federated",
        "FedProx": "fedprox",
        "Ditto": "ditto",
        "Local": "ditto",
        "FedGNNHet": "fedgnnhet",
    }[metodo]
    blk = cfg.get(block_key) or {}
    _require(
        blk.get("partition_seed") == 42
        and blk.get("learning_rate") == 0.001
        and blk.get("optimizer") == "adam_reset_each_round"
        and blk.get("local_epochs") == 1
        and blk.get("early_stopping_min_delta") == 0.0
        and blk.get("gradient_clip_norm") == 5.0
        and blk.get("participation_fraction") == 1.0
        and blk.get("aggregation_weight") == "train_observations",
        f"{cid}: hiperparâmetros federados inconsistentes em {cfg_rel}",
    )
    _require(metrics.get("local_epochs") == 1 and metrics.get("partition_seed") == 42,
             f"{cid}: local_epochs/partition_seed divergentes no metrics.json")

    if metodo == "FedProx":
        _require(blk.get("proximal_mu") == [0.01] and float(metrics.get("proximal_mu", -1)) == 0.01,
                 f"{cid}: proximal_mu inconsistente")
    elif metodo == "Ditto":
        _require(
            blk.get("lambda") == ["per_client"]
            and not blk.get("local_only", False)
            and metrics.get("ditto_lambda") == "per_client"
            and "validation_global" in metrics,
            f"{cid}: configuração Ditto per_client inconsistente",
        )
    elif metodo == "Local":
        _require(
            blk.get("lambda") == [0]
            and blk.get("local_only") is True
            and str(metrics.get("ditto_lambda")) in ("0", "0.0")
            and "validation_global" not in metrics,
            f"{cid}: configuração Local (ditto.local_only, lambda=0) inconsistente",
        )
    elif metodo == "FedGNNHet":
        _require(blk.get("halo") == ["1hop"] and metrics.get("halo") == "1hop", f"{cid}: halo=1hop inconsistente")


def compute_automatic_counts(curves_by_cell: dict[str, list[AuditedCurve]], seed_df: pd.DataFrame) -> dict:
    """Calcula todas as contagens e estatísticas descritivas diretamente dos históricos auditados."""
    fed_tokens = [tok for _, tok in FED_PANEL_METHODS if tok != "cent"]
    fed_k166_ids = [f"sunt_{tok}_k166" for tok in fed_tokens]
    fed_k166 = seed_df[seed_df.celula.isin(fed_k166_ids)].copy()
    b166 = SUNT_BUDGET[166]
    tail10_k166 = int(b166 - 0.1 * b166 + 1)  # 901..1000 = últimos 10% de 1000 rodadas

    # A menor rodada selecionada entre FedAvg, FedProx e FedGNNHet em k=166 é calculada dos dados (983).
    non_ditto_k166_min = int(
        seed_df[seed_df.celula.isin([f"sunt_{t}_k166" for t in ("fedavg", "fedprox", "halo")])].melhor_iteracao.min()
    )

    up_step_counts_k166: list[int] = []
    for cid in fed_k166_ids:
        for c in curves_by_cell[cid]:
            diffs = np.diff(c.curve.to_numpy(dtype=float))
            up_step_counts_k166.append(int((diffs > 0.0).sum()))

    local_df = seed_df[seed_df.metodo == "Local"].copy()
    cent_df = seed_df[seed_df.celula == "sunt_cent"].copy()
    tail10_cent = int(SUNT_BUDGET[None] - 0.1 * SUNT_BUDGET[None] + 1)  # 541..600

    by_method_k166 = {}
    for label, tok in FED_PANEL_METHODS[1:]:
        g = seed_df[seed_df.celula == f"sunt_{tok}_k166"]
        by_method_k166[label] = {
            "melhores_rodadas_ordenadas": [int(x) for x in sorted(g.melhor_iteracao.tolist())],
            "ultimos_10pct": int((g.melhor_iteracao >= tail10_k166).sum()),
            f"desde_rodada_{non_ditto_k166_min}": int((g.melhor_iteracao >= non_ditto_k166_min).sum()),
            "na_ultima_rodada": int((g.melhor_iteracao == b166).sum()),
            "final_acima_checkpoint": int((g.delta_final_menos_checkpoint > 0.0).sum()),
        }

    local_by_k = {}
    for k in KS:
        g = local_df[local_df.k == k]
        local_by_k[f"k={k}"] = {
            "orcamento": SUNT_BUDGET[k],
            "melhores_rodadas_ordenadas": [int(x) for x in sorted(g.melhor_iteracao.tolist())],
            "sementes_com_piora_pos_minimo": int((g.delta_final_menos_checkpoint > 0.0).sum()),
            "delta_final_menos_checkpoint_min": float(g.delta_final_menos_checkpoint.min()),
            "delta_final_menos_checkpoint_media": float(g.delta_final_menos_checkpoint.mean()),
            "delta_final_menos_checkpoint_max": float(g.delta_final_menos_checkpoint.max()),
        }

    return {
        "total_celulas": int(seed_df.celula.nunique()),
        "total_execucoes": int(len(seed_df)),
        "execucoes_ultimo_passo_igual_checkpoint": int((seed_df.melhor_iteracao == seed_df.iteracoes).sum()),
        "execucoes_final_acima_checkpoint": int((seed_df.delta_final_menos_checkpoint > 0.0).sum()),
        "centralizado": {
            "orcamento_epocas": SUNT_BUDGET[None],
            "limiar_ultimos_10pct": tail10_cent,
            "melhores_epocas_ordenadas": [int(x) for x in sorted(cent_df.melhor_iteracao.tolist())],
            "sementes_ultimos_10pct": int((cent_df.melhor_iteracao >= tail10_cent).sum()),
            "sementes_final_acima_checkpoint": int((cent_df.delta_final_menos_checkpoint > 0.0).sum()),
        },
        "federados_k166": {
            "total_execucoes": int(len(fed_k166)),
            "orcamento_rodadas": b166,
            "limiar_ultimos_10pct": tail10_k166,
            "execucoes_ultimos_10pct": int((fed_k166.melhor_iteracao >= tail10_k166).sum()),
            "limiar_bloco_final_observado": non_ditto_k166_min,
            "execucoes_desde_limiar_bloco_final": int((fed_k166.melhor_iteracao >= non_ditto_k166_min).sum()),
            "execucoes_minimo_na_rodada_1000": int((fed_k166.melhor_iteracao == b166).sum()),
            "execucoes_final_acima_checkpoint": int((fed_k166.delta_final_menos_checkpoint > 0.0).sum()),
            "melhores_rodadas_ordenadas": [int(x) for x in sorted(fed_k166.melhor_iteracao.tolist())],
            "transicoes_subida_por_execucao_em_999": {
                "min": int(min(up_step_counts_k166)),
                "max": int(max(up_step_counts_k166)),
            },
            "por_metodo": by_method_k166,
        },
        "local": {
            "total_execucoes": int(len(local_df)),
            "sementes_com_piora_pos_minimo": int((local_df.delta_final_menos_checkpoint > 0.0).sum()),
            "por_k": local_by_k,
        },
    }


def audit_and_load_sunt(runs_df: pd.DataFrame) -> tuple[dict[str, list[AuditedCurve]], pd.DataFrame, pd.DataFrame, dict]:
    """Audita as 80 execuções SUNT selecionadas e devolve curvas, tabela por semente, resumo por célula e metadados."""
    _tracked_files.cache_clear()
    expected = expected_sunt_cells()
    expected_ids = {cid for cid, _, _ in expected}
    present_sunt_main = set(runs_df[(runs_df.dataset == "sunt") & (runs_df.alvo == "n_boardings")
                                    & (runs_df.grupo.isin(["principal", "referência"]))].celula.unique())
    _require(present_sunt_main == expected_ids,
             f"conjunto de células SUNT principal/referência ({sorted(present_sunt_main)}) ≠ esperado ({sorted(expected_ids)})")

    curves_by_cell: dict[str, list[AuditedCurve]] = {}
    seed_rows: list[dict] = []
    tracked_histories: list[dict[str, str]] = []
    seen_files: set[str] = set()
    max_rel_diff = 0.0

    for cid, metodo, k in expected:
        sub = runs_df[runs_df.celula == cid].sort_values("semente")
        _require(len(sub) == len(SEEDS), f"{cid}: esperadas {len(SEEDS)} execuções, encontradas {len(sub)}")
        _require(tuple(sub.semente.astype(int)) == SEEDS, f"{cid}: sementes {tuple(sub.semente)} ≠ {SEEDS}")

        budget = SUNT_BUDGET[k]
        step_col = "epoch" if k is None else "round"
        forbidden_col = "round" if k is None else "epoch"
        unidade = "época" if k is None else ("rodada local" if metodo == "Local" else "rodada federada")
        cell_curves: list[AuditedCurve] = []

        for r in sub.itertuples():
            seed = int(r.semente)
            metrics_path = ROOT / str(r.fonte)
            hist_path = metrics_path.parent / "history.csv"
            _require(metrics_path.is_file(), f"{cid}/seed_{seed:03d}: metrics.json ausente ({r.fonte})")
            _require(hist_path.is_file(), f"{cid}/seed_{seed:03d}: history.csv ausente ({hist_path.relative_to(ROOT)})")
            _require(_tracked(metrics_path), f"{cid}/seed_{seed:03d}: metrics.json não versionado ({r.fonte})")
            _require(_tracked(hist_path), f"{cid}/seed_{seed:03d}: history.csv não versionado ({hist_path.relative_to(ROOT)})")

            m = json.loads(metrics_path.read_text())
            _verify_config_correspondence(cid, metodo, k, str(r.fonte), str(r.config), m)

            _require(
                int(m.get("seed", -1)) == seed
                and m.get("dataset") == "sunt"
                and m.get("target") == "n_boardings"
                and m.get("features") == ["n_boardings", "n_alightings"]
                and int(m.get("nodes", -1)) == SUNT_NODES
                and int(m.get("input_steps", -1)) == 168
                and int(m.get("horizon", -1)) == 24,
                f"{cid}/seed_{seed:03d}: metadados básicos divergentes em metrics.json",
            )
            if k is None:
                _require(m.get("num_clients") is None, f"{cid}/seed_{seed:03d}: num_clients deve ser None")
                completed = int(m["epochs_completed"])
                best_m = int(m["best_epoch"])
            else:
                _require(int(m.get("num_clients", -1)) == k, f"{cid}/seed_{seed:03d}: num_clients ≠ {k}")
                completed = int(m["rounds_completed"])
                best_m = int(m["best_round"])

            _require(completed == budget and int(r.iteracoes) == budget,
                     f"{cid}/seed_{seed:03d}: iterações ({completed}/{r.iteracoes}) ≠ orçamento {budget}")
            _require(best_m == int(r.melhor_iteracao) and 1 <= best_m <= budget,
                     f"{cid}/seed_{seed:03d}: melhor_iteracao ({best_m}/{r.melhor_iteracao}) fora de 1..{budget}")

            hist_sha = _sha256(hist_path)
            hist_rel = str(hist_path.relative_to(ROOT))
            if hist_rel not in seen_files:
                tracked_histories.append({"arquivo": hist_rel, "sha256": hist_sha, "papel": "execucao_selecionada"})
                seen_files.add(hist_rel)

            probe_hist_rel, probe_hist_sha = "", ""
            chk_from = getattr(r, "checkpoint_de", "")
            if isinstance(chk_from, str) and chk_from:
                probe_dir = ROOT / chk_from
                probe_metrics = probe_dir / "metrics.json"
                probe_hist = probe_dir / "history.csv"
                _require(probe_metrics.is_file() and probe_hist.is_file(),
                         f"{cid}/seed_{seed:03d}: arquivos da sonda ausentes em {chk_from}")
                _require(_tracked(probe_metrics) and _tracked(probe_hist),
                         f"{cid}/seed_{seed:03d}: arquivos da sonda não versionados em {chk_from}")
                probe_hist_sha = _sha256(probe_hist)
                probe_hist_rel = str(probe_hist.relative_to(ROOT))
                _require(probe_hist_sha == hist_sha,
                         f"{cid}/seed_{seed:03d}: history.csv diverge do history.csv da sonda ({probe_hist_rel})")
                if probe_hist_rel not in seen_files:
                    tracked_histories.append({"arquivo": probe_hist_rel, "sha256": probe_hist_sha, "papel": "sondagem_origem"})
                    seen_files.add(probe_hist_rel)

            h = pd.read_csv(hist_path)
            _require(step_col in h.columns and forbidden_col not in h.columns,
                     f"{cid}/seed_{seed:03d}: coluna de etapa esperada '{step_col}' (sem '{forbidden_col}') em {hist_rel}")
            _require(len(h) == budget, f"{cid}/seed_{seed:03d}: history.csv tem {len(h)} linhas, esperado {budget}")

            steps = h[step_col].to_numpy(dtype=int)
            _require(np.array_equal(steps, np.arange(1, budget + 1)),
                     f"{cid}/seed_{seed:03d}: etapas não formam sequência contígua 1..{budget}")

            _require("validation_observations" in h.columns,
                     f"{cid}/seed_{seed:03d}: validation_observations ausente em {hist_rel}")
            obs = h["validation_observations"].to_numpy(dtype=int)
            _require(
                bool((obs == SUNT_VAL_OBSERVATIONS).all())
                and int((m.get("validation") or {}).get("observations", -1)) == SUNT_VAL_OBSERVATIONS,
                f"{cid}/seed_{seed:03d}: validation_observations ≠ {SUNT_VAL_OBSERVATIONS}",
            )

            vals = h["validation_MAE"].to_numpy(dtype=float)
            _require(bool(np.isfinite(vals).all()), f"{cid}/seed_{seed:03d}: validation_MAE contém NaN/Inf")

            if metodo == "Ditto":
                _require("validation_global_MAE" in h.columns,
                         f"{cid}/seed_{seed:03d}: Ditto deve registrar validation_global_MAE separado de validation_MAE")
                gvals = h["validation_global_MAE"].to_numpy(dtype=float)
                _require(not np.allclose(vals, gvals),
                         f"{cid}/seed_{seed:03d}: validation_MAE pessoal idêntico ao global no Ditto")
                g_at_best = float(gvals[best_m - 1])
                g_metrics = float((m.get("validation_global") or {}).get("MAE", float("nan")))
                _require(_close_val(g_at_best, g_metrics),
                         f"{cid}/seed_{seed:03d}: validation_global_MAE na melhor rodada não confere com metrics.json")

            argmin_step = int(steps[int(np.argmin(vals))])
            _require(argmin_step == best_m,
                     f"{cid}/seed_{seed:03d}: argmin(validation_MAE)={argmin_step} ≠ melhor_iteracao={best_m}")

            v_hist = float(vals[best_m - 1])
            v_metrics = float((m.get("validation") or {}).get("MAE", float("nan")))
            v_csv = float(r.val_MAE)
            _require(_close_val(v_hist, v_metrics) and _close_val(v_hist, v_csv),
                     f"{cid}/seed_{seed:03d}: MAE na etapa {best_m} ({v_hist}) ≠ metrics.json ({v_metrics})")

            rel_diff = abs(v_hist - v_metrics) / max(1.0, abs(v_metrics))
            max_rel_diff = max(max_rel_diff, rel_diff)

            cfg_sha = _sha256(ROOT / str(r.config))
            diffs = np.diff(vals)
            curve = pd.Series(vals, index=pd.Index(steps, name=step_col), name=f"seed_{seed:03d}")
            audited = AuditedCurve(
                celula=cid,
                metodo=metodo,
                k=k,
                semente=seed,
                unidade=unidade,
                orcamento=budget,
                melhor_iteracao=best_m,
                curve=curve,
                val_mae_checkpoint_hist=v_hist,
                val_mae_checkpoint_metrics=v_metrics,
                historico_rel=hist_rel,
                historico_sha256=hist_sha,
                historico_sonda_rel=probe_hist_rel,
                historico_sonda_sha256=probe_hist_sha,
                config_rel=str(r.config),
                config_sha256=cfg_sha,
            )
            cell_curves.append(audited)
            seed_rows.append({
                "celula": cid,
                "metodo": metodo,
                "k": k,
                "semente": seed,
                "unidade_iteracao": unidade,
                "iteracoes": budget,
                "melhor_iteracao": best_m,
                "val_MAE_inicial": float(vals[0]),
                "val_MAE_checkpoint_historico": v_hist,
                "val_MAE_checkpoint_metrics": v_metrics,
                "dif_abs_checkpoint": abs(v_hist - v_metrics),
                "val_MAE_final": float(vals[-1]),
                "delta_final_menos_checkpoint": float(vals[-1] - v_hist),
                "transicoes_subida": int((diffs > 0.0).sum()),
                "observacoes_validacao": SUNT_VAL_OBSERVATIONS,
                "historico": hist_rel,
                "historico_sha256": hist_sha,
                "historico_sonda": probe_hist_rel,
                "historico_sonda_sha256": probe_hist_sha,
                "config": str(r.config),
                "config_sha256": cfg_sha,
            })

        curves_by_cell[cid] = cell_curves

    seed_df = pd.DataFrame(seed_rows)
    summary_rows: list[dict] = []
    for cid, metodo, k in expected:
        runs_cell = curves_by_cell[cid]
        mean_curve = mean_where_all_active(runs_cell)
        budget = SUNT_BUDGET[k]
        _require(len(mean_curve) == budget, f"{cid}: média ativa tem {len(mean_curve)} etapas, esperado {budget}")
        g = seed_df[seed_df.celula == cid]
        tail_threshold = int(budget - 0.1 * budget + 1)  # últimos 10%: 541..600 ou 901..1000
        summary_rows.append({
            "celula": cid,
            "metodo": metodo,
            "k": k,
            "unidade_iteracao": g.unidade_iteracao.iloc[0],
            "orcamento_iteracoes": budget,
            "sementes_ativas_orcamento": len(runs_cell),
            "val_MAE_inicial_min": float(g.val_MAE_inicial.min()),
            "val_MAE_inicial_media": float(g.val_MAE_inicial.mean()),
            "val_MAE_inicial_max": float(g.val_MAE_inicial.max()),
            "melhor_iteracao_min": int(g.melhor_iteracao.min()),
            "melhor_iteracao_mediana": float(g.melhor_iteracao.median()),
            "melhor_iteracao_max": int(g.melhor_iteracao.max()),
            "sementes_ultimos_10pct": int((g.melhor_iteracao >= tail_threshold).sum()),
            "sementes_desde_983": int((g.melhor_iteracao >= 983).sum()) if budget == 1000 else 0,
            "sementes_final_acima_checkpoint": int((g.delta_final_menos_checkpoint > 0.0).sum()),
            "val_MAE_checkpoint_media": float(g.val_MAE_checkpoint_historico.mean()),
            "val_MAE_checkpoint_dp": float(g.val_MAE_checkpoint_historico.std(ddof=1)),
            "val_MAE_final_media": float(mean_curve.iloc[-1]),
            "val_MAE_final_dp": float(g.val_MAE_final.std(ddof=1)),
            "delta_final_menos_checkpoint_media": float(g.delta_final_menos_checkpoint.mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    contagens = compute_automatic_counts(curves_by_cell, seed_df)

    meta = {
        "celulas_verificadas": len(expected),
        "execucoes_verificadas": len(seed_df),
        "historicos_sonda_verificados": int((seed_df.historico_sonda != "").sum()),
        "observacoes_validacao_por_etapa": SUNT_VAL_OBSERVATIONS,
        "orcamento_por_k": {"Centralizado": 600, "k=6": 600, "k=20": 600, "k=166": 1000},
        "ditto_coluna_validacao": "validation_MAE (modelos personalizados {v_k}; distinta de validation_global_MAE)",
        "diferenca_relativa_max_checkpoint_historico_vs_metrics": max_rel_diff,
        "suavizacao": False,
        "prolongamento_series_encerradas": False,
        "largura_figura_cm": FIG_WIDTH_CM,
        "fonte_minima_pt": MIN_FONT_PT,
        "contagens_automaticas": contagens,
        "historicos": tracked_histories,
    }
    return curves_by_cell, seed_df, summary_df, meta


def mean_where_all_active(curves: list[AuditedCurve]) -> pd.Series:
    """Média aritmética ponto a ponto restrita às etapas em que TODAS as 5 sementes possuem observação, sem suavizar."""
    _require(len(curves) == len(SEEDS), f"esperadas {len(SEEDS)} sementes para cálculo da média, recebidas {len(curves)}")
    common_index = curves[0].curve.index
    for c in curves[1:]:
        common_index = common_index.intersection(c.curve.index)
    _require(len(common_index) > 0, "interseção vazia de etapas entre sementes")
    mat = pd.concat([c.curve.loc[common_index] for c in curves], axis=1)
    _require(not bool(mat.isna().any().any()), "valores ausentes na interseção das sementes")
    return mat.mean(axis=1)


def _fmt_pt(val: float, decimals: int = 1) -> str:
    if decimals == 0 or abs(val - round(val)) < 1e-9 and decimals == 0:
        return f"{int(round(val))}"
    return f"{val:.{decimals}f}".replace(".", ",")


def _base_rc() -> dict:
    return {
        "svg.hashsalt": "fed-stgnn-traffic",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Latin Modern Roman", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#6e6e68",
        "axes.linewidth": 0.75,
        "axes.labelcolor": "#1f1f1d",
        "xtick.color": "#333330",
        "ytick.color": "#333330",
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.55,
        "axes.axisbelow": True,
    }


def _save_deterministic_formats(fig, out_dir: Path, stem: str) -> list[Path]:
    """Salva PDF vetorial, PNG 300 DPI em cores, PNG 300 DPI em escala de cinza e SVG determinísticos."""
    from PIL import Image, PngImagePlugin

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    pb_path = out_dir / f"{stem}_pb.png"
    svg_path = out_dir / f"{stem}.svg"

    fig.savefig(
        pdf_path, format="pdf",
        metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None},
    )
    fig.savefig(
        png_path, format="png", dpi=300,
        metadata={"Software": None},
    )
    fig.savefig(
        svg_path, format="svg",
        metadata={"Date": None, "Creator": None},
    )

    # Conversão determinística para luminância sRGB (ITU-R BT.709) a 300 DPI para conferência/impressão P&B.
    with Image.open(png_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    lum = np.clip(np.rint(0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]), 0, 255).astype(np.uint8)
    gray_img = Image.fromarray(lum, mode="L")
    gray_img.save(pb_path, format="PNG", dpi=(300, 300), pnginfo=PngImagePlugin.PngInfo())

    return [pdf_path, png_path, pb_path, svg_path]


def render_sunt_federados_figure(curves_by_cell: dict[str, list[AuditedCurve]], out_dir: Path,
                                 stem: str = STEM_FEDERADOS) -> list[Path]:
    """Figura A: Centralizado e métodos federados (linha 1: trajetória completa; linha 2: ampliação final)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    out_dir.mkdir(parents=True, exist_ok=True)
    int_fmt = FuncFormatter(lambda v, _: _fmt_pt(v, 0))
    dec_fmt = FuncFormatter(lambda v, _: _fmt_pt(v, 1))

    with plt.rc_context(_base_rc()):
        fig = plt.figure(figsize=(FIG_WIDTH_CM * CM_TO_IN, 11.8 * CM_TO_IN))
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[1.0, 1.15],
            left=0.105, right=0.968, top=0.935, bottom=0.155,
            hspace=0.38, wspace=0.17,
        )
        top_axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
        bot_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]

        cent_curves = curves_by_cell["sunt_cent"]
        cent_mean = mean_where_all_active(cent_curves)
        zoom_y0, zoom_y1 = 4.83, 5.76

        for j, k in enumerate(KS):
            ax_full = top_axes[j]
            ax_zoom = bot_axes[j]
            budget = SUNT_BUDGET[k]
            zx0 = ZOOM_START[k]

            for metodo, token in FED_PANEL_METHODS:
                runs = cent_curves if metodo == "Centralizado" else curves_by_cell[f"sunt_{token}_k{k}"]
                mean = cent_mean if metodo == "Centralizado" else mean_where_all_active(runs)
                color = METHOD_COLOR[metodo]
                ls = METHOD_LINESTYLE[metodo]
                mk = METHOD_MARKER[metodo]
                lw_mean = 1.75 if metodo == "Ditto" else 1.50
                halo_fx = [pe.Stroke(linewidth=lw_mean + 0.9, foreground="#ffffff", alpha=0.82), pe.Normal()]

                for target_ax, ms, seed_alpha in ((ax_full, 3.6, 0.16), (ax_zoom, 4.3, 0.18)):
                    for c in runs:
                        target_ax.plot(
                            c.curve.index.to_numpy(), c.curve.to_numpy(),
                            color=color, lw=0.45, alpha=seed_alpha, ls="-", zorder=2,
                        )
                    target_ax.plot(
                        mean.index.to_numpy(), mean.to_numpy(),
                        color=color, lw=lw_mean, ls=ls, path_effects=halo_fx, zorder=3.6,
                    )
                    for c in runs:
                        target_ax.plot(
                            c.melhor_iteracao, c.val_mae_checkpoint_hist,
                            linestyle="None", marker=mk, color=color,
                            ms=ms, mec="#ffffff", mew=0.48, zorder=4.6,
                        )

            # Demarcação discreta do trecho ampliado no painel completo.
            ax_full.add_patch(Rectangle(
                (zx0, zoom_y0), budget - zx0, zoom_y1 - zoom_y0,
                fill=False, edgecolor="#767670", linestyle=(0, (2.6, 1.5)), linewidth=0.70, zorder=5,
            ))

            if k == 166:
                ax_zoom.axvline(600, color="#666660", lw=0.65, ls=(0, (1.6, 1.6)), ymax=0.36, zorder=2.5)
                ax_zoom.text(
                    612, 4.95, "600 ép.", fontsize=8.0, color="#333330",
                    ha="left", va="center",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "#ffffff", "ec": "none", "alpha": 0.88},
                )

            ax_full.set_xlim(0, budget * 1.025)
            ax_full.set_ylim(4.55, 19.6)
            ax_full.set_yticks([5, 10, 15, 19])
            ax_full.yaxis.set_major_formatter(int_fmt)

            ax_zoom.set_xlim(zx0, budget * 1.02)
            ax_zoom.set_ylim(zoom_y0, zoom_y1)
            ax_zoom.set_yticks([4.9, 5.1, 5.3, 5.5, 5.7])
            ax_zoom.yaxis.set_major_formatter(dec_fmt)

            if k in (6, 20):
                ax_full.set_xticks([0, 200, 400, 600])
                ax_zoom.set_xticks([120, 300, 450, 600])
            else:
                ax_full.set_xticks([0, 300, 600, 1000])
                ax_zoom.set_xticks([400, 600, 800, 1000])

            if j > 0:
                ax_full.tick_params(axis="y", labelleft=False)
                ax_zoom.tick_params(axis="y", labelleft=False)

            ax_full.set_title(f"({chr(ord('a') + j)}) $k = {k}$", fontsize=8.8, loc="left", pad=4, fontweight="bold")
            ax_zoom.set_title(
                f"({chr(ord('d') + j)}) Ampliação: {zx0}–{budget}",
                fontsize=8.5, loc="left", pad=4, fontweight="bold",
            )

        top_axes[0].set_ylabel("MAE de validação\n(embarques)", fontsize=8.5)
        bot_axes[0].set_ylabel("MAE de validação\n(embarques)", fontsize=8.5)
        fig.text(
            0.536, 0.076,
            "Rodadas federadas (Centralizado: épocas)",
            fontsize=8.6, color="#1f1f1d", ha="center", va="center",
        )

        handles = [
            Line2D(
                [0], [0],
                color=METHOD_COLOR[m], lw=1.75 if m == "Ditto" else 1.50,
                ls=METHOD_LINESTYLE[m], marker=METHOD_MARKER[m],
                ms=4.5, mec="#ffffff", mew=0.45, label="FedGNN-Halo" if m == "FedGNNHet" else m,
            )
            for m, _ in FED_PANEL_METHODS
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=5, frameon=False,
            fontsize=8.4, bbox_to_anchor=(0.50, 0.004),
            handlelength=2.7, columnspacing=1.25, handletextpad=0.45,
        )

        saved = _save_deterministic_formats(fig, out_dir, stem)
        plt.close(fig)
    return saved


def render_sunt_local_figure(curves_by_cell: dict[str, list[AuditedCurve]], out_dir: Path,
                             stem: str = STEM_LOCAL) -> list[Path]:
    """Figura B: Local em k = 6, 20 e 166 sob escala vertical comum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    out_dir.mkdir(parents=True, exist_ok=True)
    int_fmt = FuncFormatter(lambda v, _: _fmt_pt(v, 0))

    with plt.rc_context(_base_rc()):
        fig = plt.figure(figsize=(FIG_WIDTH_CM * CM_TO_IN, 6.4 * CM_TO_IN))
        gs = fig.add_gridspec(
            1, 3,
            left=0.105, right=0.968, top=0.880, bottom=0.275,
            wspace=0.17,
        )
        axes = [fig.add_subplot(gs[0, j]) for j in range(3)]

        color = METHOD_COLOR["Local"]
        ls = METHOD_LINESTYLE["Local"]
        mk = METHOD_MARKER["Local"]
        halo_fx = [pe.Stroke(linewidth=2.5, foreground="#ffffff", alpha=0.82), pe.Normal()]

        for j, (ax, k) in enumerate(zip(axes, KS)):
            budget = SUNT_BUDGET[k]
            runs = curves_by_cell[f"sunt_local_k{k}"]
            mean = mean_where_all_active(runs)

            for c in runs:
                ax.plot(
                    c.curve.index.to_numpy(), c.curve.to_numpy(),
                    color=color, lw=0.52, alpha=0.24, ls="-", zorder=2,
                )
            ax.plot(
                mean.index.to_numpy(), mean.to_numpy(),
                color=color, lw=1.60, ls=ls, path_effects=halo_fx, zorder=3.6,
            )
            for c in runs:
                ax.plot(
                    c.melhor_iteracao, c.val_mae_checkpoint_hist,
                    linestyle="None", marker=mk, color=color,
                    ms=4.5, mec="#ffffff", mew=0.52, zorder=4.6,
                )

            ax.set_xlim(0, budget * 1.025)
            ax.set_ylim(5.0, 10.8)
            ax.set_yticks([5, 6, 7, 8, 9, 10])
            ax.yaxis.set_major_formatter(int_fmt)
            if k in (6, 20):
                ax.set_xticks([0, 200, 400, 600])
            else:
                ax.set_xticks([0, 300, 600, 1000])

            if j > 0:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(f"({chr(ord('a') + j)}) $k = {k}$", fontsize=8.8, loc="left", pad=4, fontweight="bold")

        axes[0].set_ylabel("MAE de validação\n(embarques)", fontsize=8.5)
        fig.text(
            0.536, 0.142,
            "Épocas por modelo (sem agregação de parâmetros)",
            fontsize=8.5, color="#1f1f1d", ha="center", va="center",
        )

        handles = [
            Line2D([0], [0], color=color, lw=0.65, alpha=0.45, ls="-", label="Sementes (5)"),
            Line2D([0], [0], color=color, lw=1.60, ls=ls, label="Local (média)"),
            Line2D([0], [0], color=color, linestyle="None", marker=mk, ms=4.5,
                   mec="#ffffff", mew=0.52, label="Checkpoint selecionado"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=3, frameon=False,
            fontsize=8.4, bbox_to_anchor=(0.50, 0.008),
            handlelength=2.7, columnspacing=1.50, handletextpad=0.45,
        )

        saved = _save_deterministic_formats(fig, out_dir, stem)
        plt.close(fig)
    return saved


def render_sunt_validation_figures(curves_by_cell: dict[str, list[AuditedCurve]], out_dir: Path) -> list[Path]:
    """Renderiza Figura A (Centralizado + federados) e Figura B (Local) nos formatos vetoriais, cor e P&B."""
    return (
        render_sunt_federados_figure(curves_by_cell, out_dir)
        + render_sunt_local_figure(curves_by_cell, out_dir)
    )


def export(dest: Path) -> dict:
    """Integra auditoria dos históricos, contagens automáticas e figuras A e B ao pacote `generated-traffic`."""
    runs_csv = dest / "dados" / "resultados_por_semente.csv"
    _require(runs_csv.is_file(), f"resultados_por_semente.csv ausente em {dest}")
    runs_df = pd.read_csv(runs_csv)
    curves_by_cell, seed_df, summary_df, meta = audit_and_load_sunt(runs_df)

    seed_csv = dest / "dados" / "sunt_evolucao_validacao_por_semente.csv"
    summary_csv = dest / "dados" / "sunt_evolucao_validacao_resumo.csv"
    counts_json = dest / "dados" / "sunt_evolucao_validacao_contagens.json"
    seed_df.to_csv(seed_csv, index=False, float_format="%.17g")
    summary_df.to_csv(summary_csv, index=False, float_format="%.17g")
    counts_json.write_text(json.dumps(meta["contagens_automaticas"], indent=2, ensure_ascii=False) + "\n")

    fig_files = render_sunt_validation_figures(curves_by_cell, dest / "figuras")
    meta["artefatos_gerados"] = [str(p.relative_to(dest)) for p in [seed_csv, summary_csv, counts_json, *fig_files]]
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita históricos SUNT e gera figuras A e B de evolução da validação.")
    ap.add_argument("--pacote", type=Path, default=ROOT / "generated-traffic",
                    help="Diretório do pacote validado com dados/resultados_por_semente.csv.")
    ap.add_argument("--saida", type=Path, default=ROOT / "results" / "figuras_convergencia",
                    help="Diretório de saída para execução autônoma (fora de generated-traffic).")
    args = ap.parse_args()

    runs_csv = args.pacote / "dados" / "resultados_por_semente.csv"
    _require(runs_csv.is_file(), f"arquivo não encontrado: {runs_csv}")
    runs_df = pd.read_csv(runs_csv)
    curves_by_cell, seed_df, summary_df, meta = audit_and_load_sunt(runs_df)

    args.saida.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(args.saida / "sunt_evolucao_validacao_por_semente.csv", index=False, float_format="%.17g")
    summary_df.to_csv(args.saida / "sunt_evolucao_validacao_resumo.csv", index=False, float_format="%.17g")
    (args.saida / "sunt_evolucao_validacao_contagens.json").write_text(
        json.dumps(meta["contagens_automaticas"], indent=2, ensure_ascii=False) + "\n"
    )
    files = render_sunt_validation_figures(curves_by_cell, args.saida)
    c166 = meta["contagens_automaticas"]["federados_k166"]
    print(
        f"Auditoria concluída: {meta['execucoes_verificadas']} execuções em {meta['celulas_verificadas']} células "
        f"(k=166 federados: {c166['execucoes_ultimos_10pct']}/{c166['total_execucoes']} nos últimos 10%, "
        f"{c166['execucoes_desde_limiar_bloco_final']}/{c166['total_execucoes']} >= {c166['limiar_bloco_final_observado']})."
    )
    for p in files:
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)


if __name__ == "__main__":
    main()
