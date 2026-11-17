"""Auditoria das topologias do piloto de orientação do grafo do SUNT (só topologia; nenhum erro de previsão).

Braços: simétrico (`transit`), observado (`transit_pair_observed`) e invertido (`transit_pair_reversed`). Confere
universo de nós, suporte de pares, simetrização, a MESMA afinidade por par nos três braços e a transposição; conta
arestas, reciprocidade, graus, componentes e nós sem entrada; grava os ESTRATOS (definidos só pela topologia, antes
de examinar erros) por braço direcionado, relativos ao grau de entrada do simétrico:
    sem_entrada     grau de entrada 0 no braço
    grande_reducao  0 < grau de entrada no braço < 0,5 · grau de entrada no simétrico
    demais          grau de entrada no braço ≥ 0,5 · grau de entrada no simétrico
Uso: python -m experiments.sunt_direcao_auditoria --saida results/sunt_piloto_direcao/auditoria
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from data_loader import sunt

BRACOS = {"simetrico": "transit", "observado": "transit_pair_observed", "invertido": "transit_pair_reversed"}
LIMIAR_GRANDE_REDUCAO = 0.5


def estratos(in_braco: np.ndarray, in_sim: np.ndarray) -> np.ndarray:
    """Estrato topológico de cada nó (regra fixa, ver docstring do módulo)."""
    e = np.full(in_braco.shape, "demais", dtype=object)
    e[(in_braco > 0) & (in_braco < LIMIAR_GRANDE_REDUCAO * in_sim)] = "grande_reducao"
    e[in_braco == 0] = "sem_entrada"
    return e


def auditar() -> tuple[dict, dict]:
    g = {b: sunt.GRAPHS[nome]() for b, nome in BRACOS.items()}
    ids = g["simetrico"]["stop_ids"]
    a = {b: np.asarray(x["adj"], dtype=np.float32) for b, x in g.items()}
    n = len(ids)
    out = {"nos": int(n), "universo_igual": all(np.array_equal(x["stop_ids"], ids) for x in g.values()),
           "adj_sha256": {b: hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest() for b, m in a.items()}}
    sim, obs, inv = a["simetrico"], a["observado"], a["invertido"]
    pares = lambda m: (np.triu((m > 0) | (m.T > 0), 1))
    out["pares"] = {b: int(pares(m).sum()) for b, m in a.items()}
    out["mesmo_suporte_de_pares"] = all(np.array_equal(pares(m), pares(sim)) for m in a.values())
    out["simetrico_e_simetrizacao_do_observado"] = bool(np.array_equal(sim > 0, (obs > 0) | (obs.T > 0)))
    out["invertido_e_transposta_do_observado"] = bool(np.array_equal(inv, obs.T))
    out["mesma_afinidade_por_par"] = bool(np.array_equal(obs[obs > 0], sim[obs > 0]) and np.array_equal(inv[inv > 0], sim[inv > 0]))
    out["diagonal_zero"] = all(not np.any(np.diag(m)) for m in a.values())
    cont = g["observado"]["transition_counts"]
    out["contagens_somadas_reproduzem_simetrico"] = bool(np.array_equal(
        (cont + cont.T)[np.triu_indices(n, 1)] > 0, (sim > 0)[np.triu_indices(n, 1)]))
    por = {}
    for b, m in a.items():
        ind, outd = (m > 0).sum(0), (m > 0).sum(1)
        wcc = connected_components(csr_matrix(m > 0), directed=True, connection="weak")[0]
        scc = connected_components(csr_matrix(m > 0), directed=True, connection="strong")[0]
        rec = int(((m > 0) & (m.T > 0)).sum() // 2)
        por[b] = {"arestas_orientadas": int((m > 0).sum()), "pares_reciprocos": rec,
                  "grau_entrada": {"min": int(ind.min()), "mediana": float(np.median(ind)), "max": int(ind.max()),
                                   "zero": int((ind == 0).sum())},
                  "grau_saida": {"min": int(outd.min()), "mediana": float(np.median(outd)), "max": int(outd.max()),
                                 "zero": int((outd == 0).sum())},
                  "soma_pesos_entrada": {"min": float(m.sum(0).min()), "mediana": float(np.median(m.sum(0))),
                                         "max": float(m.sum(0).max())},
                  "componentes_fracas": int(wcc), "componentes_fortes": int(scc),
                  "nos_sem_entrada": [str(ids[i]) for i in np.where(ind == 0)[0]]}
    out["por_braco"] = por
    in_sim = (sim > 0).sum(0)
    tab = {"indice": np.arange(n), "stop_id": ids, "grau_entrada_simetrico": in_sim}
    for b in ("observado", "invertido"):
        ind = (a[b] > 0).sum(0)
        tab[f"grau_entrada_{b}"] = ind
        tab[f"estrato_{b}"] = estratos(ind, in_sim)
        out.setdefault("estratos", {})[b] = {e: int((tab[f"estrato_{b}"] == e).sum()) for e in ("sem_entrada", "grande_reducao", "demais")}
    out["regra_estratos"] = (f"sem_entrada: grau 0; grande_reducao: 0 < grau < {LIMIAR_GRANDE_REDUCAO} × grau no simétrico; "
                             "demais: o resto (grau = número de arestas de entrada; definidos só pela topologia)")
    out["afinidade"] = "log1p(contagem do par u↔v no treino)/máx (regra do simétrico), igual nos três braços"
    return out, tab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saida", type=Path, required=True)
    a = p.parse_args()
    out, tab = auditar()
    a.saida.mkdir(parents=True, exist_ok=True)
    (a.saida / "auditoria_topologias.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    cols = list(tab)
    with open(a.saida / "estratos_por_no.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(tab["indice"])):
            f.write(",".join(str(tab[c][i]) for c in cols) + "\n")
    falhas = [k for k in ("universo_igual", "mesmo_suporte_de_pares", "simetrico_e_simetrizacao_do_observado",
                          "invertido_e_transposta_do_observado", "mesma_afinidade_por_par", "diagonal_zero",
                          "contagens_somadas_reproduzem_simetrico") if not out[k]]
    print(json.dumps({k: v for k, v in out.items() if k not in ("por_braco", "adj_sha256")}, indent=1, ensure_ascii=False))
    print(json.dumps({b: {k: v for k, v in d.items() if k != "nos_sem_entrada"} for b, d in out["por_braco"].items()}, indent=1))
    raise SystemExit(1 if falhas else 0)


if __name__ == "__main__":
    main()
