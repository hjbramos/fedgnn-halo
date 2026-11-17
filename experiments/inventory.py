"""Inventário das execuções de `results/` para o pacote de resultados da dissertação (passo 1 da migração).

Lê todos os metrics.json (exceto `results/_smoke`), classifica cada REGISTRO pelo papel que ele tem na
dissertação e confere, por célula (pasta, configuração, dataset, k, variante), a cobertura de sementes, o
teste (MAE finito; ausente nas sondagens), os metadados da classificação (alvo, entradas, janela, horizonte e
grafo) e o orçamento, além das células esperadas que estejam ausentes. Não treina, não reavalia e não altera
resultados.

É um levantamento estrutural, não uma auditoria científica: não confere arquitetura, partição, normalização
nem a configuração efetiva de cada execução, e hashes (no exportador) atestam integridade, não
comparabilidade. Registros não são treinamentos independentes: as sondagens e as campanhas finais do SUNT que
avaliaram os checkpoints delas aparecem em pastas separadas e não são repetições adicionais.

Papéis:
  principal     entra na comparação principal (SUNT embarques; METR-LA e PEMS-BAY nos protocolos próprios)
  controle      controle interno no mesmo módulo (halo none, 1hop_min2, Ditto λ fixo)
  referência    referência que não é método concorrente (Local = Ditto λ=0)
  complementar  campanha complementar (carga com grafo direcionado)
  sondagem      sondagem cega sem teste; os checkpoints foram avaliados nas campanhas finais
  exploratório  execução isolada (GRU com uma semente)
  histórico     fora da evidência final (SUNT com alvo loading anterior a 23/09; halo static pré-correção)

    python -m experiments.inventory          # escreve results/inventario/{inventario.csv,cobertura.csv,cobertura.md}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .train import ROOT

OUT = ROOT / "results" / "inventario"
SEEDS = {7, 42, 101, 123, 314}
SUNT_ROUNDS = {6: 600, 20: 600, 166: 1000}
ROAD = {"metr-la", "pems-bay"}
DATASETS = ROAD | {"sunt"}
# Células esperadas (pasta, dataset, k, variante) da evidência principal, controles, referências e complementar.
# Uma célula daqui sem registro aparece como "célula ausente".
EXPECTED = (
    [("centralized", d, None, "") for d in ROAD] + [("centralized_noedges", d, None, "") for d in ROAD]
    + [(f, d, None, "") for f in ("centralized_parada_federada", "centralized_noedges_parada_federada") for d in ROAD]
    + [("fedavg_leconv", d, k, "") for d in ROAD for k in (5, 10, 20)]
    + [("fedgnnhet_leconv", "metr-la", k, f"halo={h}") for k in (5, 10, 20) for h in ("1hop", "none")]
    + [("fedgnnhet_leconv", "metr-la", 20, "halo=1hop_min2")]
    + [("fedgnnhet_leconv", "pems-bay", k, "halo=1hop") for k in (5, 10, 20)]          # k=5/10: lote 1 de lacunas
    + [("fedprox_leconv", d, k, "proximal_mu=0.01") for d in ROAD for k in (5, 10, 20)]  # k=5/10: lote 1 de lacunas
    + [("ditto_leconv", "metr-la", k, f"ditto_lambda={l}") for k in (5, 10, 20) for l in ("per_client", "0", "0.01", "0.1", "1")]
    + [("ditto_leconv", "pems-bay", k, "ditto_lambda=per_client") for k in (5, 10, 20)]  # k=20: lote 3; k=10: lote 4
    + [("ditto_leconv", "pems-bay", k, "ditto_lambda=0") for k in (5, 10, 20)]            # lote 2 de lacunas
    + [("sunt_centralized_boardings", "sunt", None, "")]
    + [("sunt_fedavg_final", "sunt", k, "") for k in (6, 20, 166)]
    + [("sunt_fedgnnhet_final", "sunt", k, "halo=1hop") for k in (6, 20, 166)]
    + [("sunt_fedprox_final", "sunt", k, "proximal_mu=0.01") for k in (6, 20, 166)]       # k=166: lote 5 de lacunas
    + [("sunt_ditto_final", "sunt", k, "ditto_lambda=per_client") for k in (6, 20, 166)]  # k=166: lote 5 de lacunas
    + [("sunt_loading_directed_centralized", "sunt", None, ""), ("sunt_loading_directed_fedavg", "sunt", 20, "")]
)
HISTORICO_SUNT_LOADING = "histórico: SUNT com alvo loading, protocolo anterior a 23/09"


def variant(m: dict) -> str:
    for key in ("halo", "proximal_mu", "ditto_lambda", "lambda_l1"):
        if m.get(key) is not None:
            return f"{key}={m[key]}"
    return ""


def role(folder: str, m: dict) -> str:
    ds, var = m.get("dataset"), variant(m)
    if folder in ("centralized", "centralized_noedges"):
        return "principal" if ds in ROAD else HISTORICO_SUNT_LOADING
    if folder in ("centralized_parada_federada", "centralized_noedges_parada_federada"):
        return "controle: regra de parada federada"
    if folder in ("centralized_uniform", "centralized_edge_constant", "centralized_edge_dynamic", "sunt_fixed", "sunt_probe"):
        return HISTORICO_SUNT_LOADING
    if folder == "centralized_gru":
        return "exploratório: GRU, uma semente"
    if folder in ("fedavg_leconv", "fedprox_leconv"):
        return "principal"
    if folder in ("ditto_leconv", "sunt_ditto_final"):
        lam = str(m.get("ditto_lambda"))
        return {"per_client": "principal", "0": "referência: Local (Ditto λ=0)"}.get(lam, "controle: Ditto λ fixo")
    if folder == "fedgnnhet_leconv":
        return {"halo=1hop": "principal", "halo=none": "controle: halo none",
                "halo=1hop_min2": "controle: halo 1hop_min2",
                "halo=static": "histórico: halo static anterior à correção de vazamento"}.get(var, "?")
    if folder in ("sunt_centralized_boardings", "sunt_fedavg_final", "sunt_fedgnnhet_final", "sunt_fedprox_final"):
        return "principal"
    if folder in ("sunt_fedavg_probe", "sunt_probe_boardings"):
        return "sondagem: sem teste (checkpoints avaliados na campanha final)"
    if folder.startswith("sunt_loading_directed"):
        return "complementar: carga + grafo direcionado"
    if folder == "sunt_fedpub_final":
        return "complementar: FED-PUB (limitação)"
    return "?"


def metadata_issues(papel: str, m: dict) -> list[str]:
    """Confere se os metadados gravados correspondem ao papel atribuído pela pasta."""
    ds, issues = m.get("dataset"), []
    feats, graph = m.get("features"), m.get("graph")
    w, h = m.get("input_steps"), m.get("horizon")
    if ds == "sunt" and (papel == "principal" or papel.startswith("sondagem")):
        if m.get("target") != "n_boardings" or feats not in (None, ["n_boardings", "n_alightings"]):
            issues.append("alvo/entradas")
        if graph not in (None, "transit"):
            issues.append(f"grafo {graph}")
        if (w, h) not in ((168, 24), (None, None)):
            issues.append(f"W/H {w}/{h}")
    if papel.startswith("complementar: carga"):
        if m.get("target") != "loading" or graph != "transit_directed":
            issues.append(f"alvo {m.get('target')}/grafo {graph}")
        if feats != ["n_boardings", "n_alightings", "loading"]:
            issues.append("entradas")
    if ds in ROAD and papel != "?" and (w, h) not in ((12, 12), (None, None)):
        issues.append(f"W/H {w}/{h}")
    return issues


def test_ok(m: dict) -> bool:
    mae = (m.get("test") or {}).get("MAE")
    return isinstance(mae, (int, float)) and math.isfinite(mae)


def collect() -> pd.DataFrame:
    rows = []
    for p in sorted((ROOT / "results").glob("**/metrics.json")):
        rel = p.relative_to(ROOT / "results")
        if rel.parts[0].startswith("_"):
            continue
        m = json.loads(p.read_text())
        folder = rel.parts[0]
        config = rel.parts[1] if len(rel.parts) > 2 and rel.parts[1] not in DATASETS else ""
        test = m.get("test") or {}
        papel = role(folder, m)
        rows.append({
            "pasta": folder, "configuracao": config, "dataset": m.get("dataset"), "metodo": m.get("method"), "variante": variant(m),
            "k": m.get("num_clients"), "k_pedido": m.get("requested_clients"), "semente": m.get("seed"),
            "alvo": m.get("target") or ("speed" if m.get("dataset") in ROAD else None),
            "grafo": m.get("graph"), "iteracoes": m.get("rounds_completed", m.get("epochs_completed")),
            "melhor": m.get("best_round", m.get("best_epoch")), "teste_presente": "test" in m,
            "teste_avaliado": test_ok(m), "metadados": "; ".join(metadata_issues(papel, m)),
            "test_MAE": test.get("MAE"), "test_RMSE": test.get("RMSE"), "test_R2": test.get("R2"),
            "val_MAE": (m.get("validation") or {}).get("MAE"), "gpu": m.get("gpu"),
            "papel": papel, "arquivo": str(rel),
        })
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(["papel", "pasta", "configuracao", "dataset", "alvo", "k", "variante"], dropna=False):
        papel, pasta, config, ds, alvo, k, var = key
        seeds = set(int(s) for s in g["semente"].dropna())
        checks = []
        if not papel.startswith(("histórico", "exploratório")):
            if seeds != SEEDS:
                checks.append(f"sementes {sorted(seeds)}")
        if not papel.startswith(("sondagem", "histórico")) and not g["teste_avaliado"].all():
            checks.append("teste ausente ou não finito")
        if papel.startswith("sondagem") and g["teste_presente"].any():
            checks.append("sondagem com teste")
        meta = sorted({x for v in g["metadados"] for x in v.split("; ") if x})
        if meta:
            checks.append("metadados: " + ", ".join(meta))
        if ds == "sunt" and papel == "principal" and k == k and k in SUNT_ROUNDS:
            if not (g["iteracoes"] == SUNT_ROUNDS[int(k)]).all():
                checks.append(f"rodadas ≠ {SUNT_ROUNDS[int(k)]}")
        if ds == "sunt" and papel == "principal" and k != k and not (g["iteracoes"] == 600).all():
            checks.append("épocas ≠ 600")
        if ds == "sunt" and papel == "principal" and set(g["alvo"]) != {"n_boardings"}:
            checks.append(f"alvo {set(g['alvo'])}")
        if g.duplicated(["semente"]).any():
            checks.append("semente duplicada")
        out.append({"papel": papel, "pasta": pasta + (f"/{config}" if config else ""), "dataset": ds, "alvo": alvo,
                    "k": "" if k != k else int(k), "variante": var, "n": len(g),
                    "sementes": ",".join(str(s) for s in sorted(seeds)),
                    "test_MAE_media": round(g["test_MAE"].mean(), 4) if g["teste_avaliado"].any() else None,
                    "gpus": ",".join(sorted({str(x).replace("NVIDIA ", "") for x in g["gpu"].dropna()})),
                    "conferencia": "ok" if not checks else "; ".join(checks)})
    present = {(r.pasta.split("/")[0], r.dataset, None if r.k == "" else int(r.k), r.variante) for r in pd.DataFrame(out).itertuples()}
    for pasta, ds, k, var in EXPECTED:
        if (pasta, ds, k, var) not in present:
            out.append({"papel": "esperada", "pasta": pasta, "dataset": ds, "alvo": None, "k": "" if k is None else k,
                        "variante": var, "n": 0, "sementes": "", "test_MAE_media": None, "gpus": "",
                        "conferencia": "célula ausente"})
    order = ["esperada", "principal", "controle", "referência", "complementar", "sondagem", "exploratório", "histórico"]
    cov = pd.DataFrame(out)
    cov["_o"] = cov["papel"].map(lambda r: next((i for i, o in enumerate(order) if r.startswith(o)), 99))
    return cov.sort_values(["_o", "dataset", "pasta", "k", "variante"]).drop(columns="_o")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "inventario.csv", index=False)
    cov = coverage(df)
    cov.to_csv(OUT / "cobertura.csv", index=False)
    lines = ["# Inventário das execuções (gerado por `python -m experiments.inventory`)", "",
             f"{len(df)} registros (metrics.json) fora de `results/_smoke`, em {len(cov)} células. Registros não são",
             "treinamentos independentes (sondagens e avaliações finais do SUNT ficam em pastas separadas). É um",
             "levantamento estrutural, não auditoria científica; papéis e conferências no docstring de",
             "`experiments/inventory.py`. Execuções em andamento no momento da geração não aparecem.", "",
             "| papel | pasta | dataset | alvo | k | variante | n | sementes | teste MAE (média) | GPUs | conferência |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cov.itertuples():
        lines.append(f"| {r.papel} | {r.pasta} | {r.dataset} | {r.alvo} | {r.k} | {r.variante} | {r.n} | {r.sementes} | "
                     f"{'' if r.test_MAE_media is None or r.test_MAE_media != r.test_MAE_media else r.test_MAE_media} | {r.gpus} | {r.conferencia} |")
    (OUT / "cobertura.md").write_text("\n".join(lines) + "\n")
    bad = cov[cov["conferencia"] != "ok"]
    print(f"{len(df)} registros, {len(cov)} células; {len(bad)} com observação. Escrito em {OUT.relative_to(ROOT)}")
    if len(bad):
        print(bad[["papel", "pasta", "dataset", "k", "variante", "conferencia"]].to_string(index=False))


if __name__ == "__main__":
    main()
