"""Exporta o piloto de orientação do grafo do SUNT (centralizado, embarques, só VALIDAÇÃO) para `generated-traffic`.

Fonte: `results/sunt_piloto_direcao/` (protocolo, auditoria das topologias, estratos, 15 metrics.json e
validation_detail.npz, códigos de saída e relatório). Não treina nem avalia checkpoints. Recusa o piloto se algum
insumo não estiver versionado, se a auditoria tiver falhado, se algum treino não terminou com rc=0, se o teste tiver
sido avaliado, sem determinismo estrito, com épocas ≠ 600, com adjacência diferente da auditada ou com inicialização
diferente entre braços da mesma semente. MAE e RMSE são recalculados das somas por horizonte e nó (validation_detail)
e conferidos contra o metrics.json; diferenças pareadas e por estrato são recalculadas e conferidas contra o
relatório do piloto. Descritivo: sem IC, sem p e fora das famílias Holm.
"""
from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from .export_sunt_analysis import close, digest, require_tracked
from .train import ROOT

BASE = ROOT / "results" / "sunt_piloto_direcao"
SEEDS = [7, 42, 101, 123, 314]
EPOCAS = 600
BRACOS = ("simetrico", "observado", "invertido")
CONTRASTES = (("observado", "simetrico"), ("invertido", "simetrico"), ("observado", "invertido"))
ROTULO = {"simetrico": "Simétrica (vigente)", "observado": "Direcionada (observada)", "invertido": "Invertida"}
CHAVE = {"simetrico": "Sym", "observado": "Obs", "invertido": "Rev"}
AUDITORIA_OK = ("universo_igual", "mesmo_suporte_de_pares", "simetrico_e_simetrizacao_do_observado",
                "invertido_e_transposta_do_observado", "mesma_afinidade_por_par", "diagonal_zero",
                "contagens_somadas_reproduzem_simetrico")


def require(ok, message):
    if not ok:
        raise ValueError("Piloto de orientação do SUNT: " + message)


def _mae_rmse(z, nos=None):
    a, q, c = z["abs_error"], z["squared_error"], z["count"]
    if nos is not None:
        a, q, c = a[:, nos], q[:, nos], c[:, nos]
    return a.sum() / c.sum(), float(np.sqrt(q.sum() / c.sum()))


def export(dest: Path) -> dict:
    fixos = [BASE / f for f in ("PROTOCOLO.md", "RELATORIO.md", "relatorio_validacao.json", "auditoria/auditoria_topologias.json",
                                "auditoria/estratos_por_no.csv", "logs/lancamento.txt")]
    fixos += [BASE / f"logs/seed{s}_rc.txt" for s in SEEDS]
    fixos += [ROOT / f"config/sunt_piloto_direcao_{b}.yaml" for b in BRACOS]
    fixos += [ROOT / f for f in ("experiments/sunt_direcao_auditoria.py", "experiments/sunt_direcao_relatorio.py",
                                 "scripts/sunt_piloto_direcao.sh")]
    runs_dir = {(b, s): BASE / b / "sunt" / f"seed_{s:03d}" for b in BRACOS for s in SEEDS}
    require_tracked(*fixos, *[d / f for d in runs_dir.values() for f in ("metrics.json", "validation_detail.npz", "history.csv")])

    aud = json.loads((BASE / "auditoria/auditoria_topologias.json").read_text())
    require(all(aud[k] is True for k in AUDITORIA_OK), "auditoria das topologias reprovada")
    est = pd.read_csv(BASE / "auditoria/estratos_por_no.csv")
    require(len(est) == aud["nos"] and (est.indice.to_numpy() == np.arange(aud["nos"])).all(), "estratos incompatíveis com a auditoria")
    for b in ("observado", "invertido"):
        require(est[f"estrato_{b}"].value_counts().reindex(aud["estratos"][b].keys(), fill_value=0).to_dict() == aud["estratos"][b],
                f"contagem dos estratos de {b} diverge da auditoria")
    for s in SEEDS:
        rc = (BASE / f"logs/seed{s}_rc.txt").read_text().split("\n")
        require([l.split()[-1] for l in rc if " rc=" in l] == ["rc=0"] * 3 and "FIM" in rc, f"semente {s}: execução incompleta ou rc≠0")

    linhas, z = [], {}
    for (b, s), d in runs_dir.items():
        m = json.loads((d / "metrics.json").read_text())
        require(m["seed"] == s and m["dataset"] == "sunt", f"{b}/{s}: metrics.json de outra execução")
        require(m["test_evaluated"] is False and "test" not in m, f"{b}/{s}: teste avaliado")
        require(m["determinism"]["strict"] and m["determinism"]["after_run"]["use_deterministic_algorithms"], f"{b}/{s}: sem determinismo estrito")
        require(m["epochs_completed"] == EPOCAS, f"{b}/{s}: {m['epochs_completed']} épocas")
        require(m["adj_sha256"] == aud["adj_sha256"][b], f"{b}/{s}: adjacência diferente da auditada")
        require(digest(d / "validation_detail.npz") == m["validation_detail_sha256"], f"{b}/{s}: validation_detail diverge do hash")
        zz = np.load(d / "validation_detail.npz")
        z[b, s] = {k: zz[k] for k in ("abs_error", "squared_error", "count")}
        require(z[b, s]["count"].shape[1] == aud["nos"], f"{b}/{s}: número de nós")
        mae, rmse = _mae_rmse(z[b, s])
        require(close(mae, m["validation"]["MAE"]) and close(rmse, m["validation"]["RMSE"]), f"{b}/{s}: MAE/RMSE não reconstituídos")
        linhas.append({"braco": b, "semente": s, "val_MAE": mae, "val_RMSE": rmse, "melhor_epoca": m["best_epoch"],
                       "epocas": m["epochs_completed"], "treino_h": m["training_seconds"] / 3600, "gpu": m["gpu"],
                       "init_state_sha256": m["init_state_sha256"], "adj_sha256": m["adj_sha256"],
                       "fonte": str((d / "metrics.json").relative_to(ROOT)), "sha256": digest(d / "metrics.json")})
    runs = pd.DataFrame(linhas)
    for s in SEEDS:
        require(runs[runs.semente == s].init_state_sha256.nunique() == 1, f"semente {s}: inicialização diferente entre braços")

    rel = json.loads((BASE / "relatorio_validacao.json").read_text())
    require(rel["problemas"] == [] and rel["sementes"] == SEEDS, "relatório do piloto com problemas")
    v = runs.set_index(["braco", "semente"])
    nos_est = {b: {e: est.index[est[f"estrato_{b}"] == e].to_numpy() for e in aud["estratos"][b]} for b in ("observado", "invertido")}
    cont = []
    for a, b in CONTRASTES:
        k = f"{a} − {b}"
        d = (v.loc[a].val_MAE - v.loc[b].val_MAE).loc[SEEDS].to_numpy()
        dr = (v.loc[a].val_RMSE - v.loc[b].val_RMSE).loc[SEEDS].to_numpy()
        require(close(d, rel["contrastes"][k]["MAE"]["valores"]) and close(dr, rel["contrastes"][k]["RMSE"]["valores"]),
                f"{k}: diferenças não conferem com o relatório")
        cont.append({"contraste": k, "estrato": "todos", "nos": aud["nos"], "metrica": "MAE", **{f"s{s}": x for s, x in zip(SEEDS, d)}})
        cont.append({"contraste": k, "estrato": "todos", "nos": aud["nos"], "metrica": "RMSE", **{f"s{s}": x for s, x in zip(SEEDS, dr)}})
        for eb in [x for x in (a, b) if x in nos_est]:
            for e, nos in nos_est[eb].items():
                if not len(nos):
                    continue
                de = np.array([_mae_rmse(z[a, s], nos)[0] - _mae_rmse(z[b, s], nos)[0] for s in SEEDS])
                chave = f"estratos de {eb}: {e} ({len(nos)} nós)"
                require(close(de, rel["estratos"][k][chave]["valores"]), f"{k}, {chave}: não confere com o relatório")
                cont.append({"contraste": k, "estrato": f"{e} ({eb})", "nos": len(nos), "metrica": "MAE",
                             **{f"s{s}": x for s, x in zip(SEEDS, de)}})
    ct = pd.DataFrame(cont)
    vals = ct[[f"s{s}" for s in SEEDS]].to_numpy()
    ct["media"], ct["dp"], ct["positivas"] = vals.mean(1), vals.std(1, ddof=1), (vals > 0).sum(1)

    alvo = dest / "dados" / "sunt_piloto_direcao"
    alvo.mkdir()
    copiados = ["PROTOCOLO.md", "RELATORIO.md", "relatorio_validacao.json", "auditoria/auditoria_topologias.json", "auditoria/estratos_por_no.csv"]
    for f in copiados:
        shutil.copy2(BASE / f, alvo / Path(f).name)
    runs.to_csv(alvo / "por_execucao.csv", index=False, float_format="%.17g")
    ct.to_csv(alvo / "diferencas_pareadas.csv", index=False, float_format="%.17g")

    def num(x, nd):
        return "\\num{" + f"{x:.{nd}f}" + "}"

    def pick(contraste, estrato="todos", metrica="MAE"):
        r = ct[(ct.contraste == contraste) & (ct.estrato == estrato) & (ct.metrica == metrica)]
        require(len(r) == 1, f"contraste ausente: {contraste} / {estrato}")
        return r.iloc[0]

    lin = ["% Gerado por experiments/export_sunt_direcao.py; não editar. Métricas de VALIDAÇÃO (o teste não foi avaliado).",
           r"\begin{tabular}{lcccc}\toprule",
           r"Versão do grafo & MAE & RMSE & $\Delta$MAE (sementes $>0$) & Melhores épocas\\\midrule"]
    mac = ["% Piloto de orientação do grafo do SUNT (centralizado, só validação); descritivo, fora das famílias Holm."]
    for b in BRACOS:
        g = runs[runs.braco == b]
        delta = "---" if b == "simetrico" else (lambda r: "\\num{" + f"{r.media:+.4f}" + "} (" + f"{r.positivas}/5)")(pick(f"{b} − simetrico"))
        lin.append(f"{ROTULO[b]} & {num(g.val_MAE.mean(), 4)} $\\pm$ {num(g.val_MAE.std(ddof=1), 4)} & "
                   f"{num(g.val_RMSE.mean(), 2)} $\\pm$ {num(g.val_RMSE.std(ddof=1), 2)} & {delta} & "
                   f"\\num{{{g.melhor_epoca.min()}}}--\\num{{{g.melhor_epoca.max()}}}" + r"\\")
        mac += [f"\\newcommand{{\\TrafficSUNTDir{CHAVE[b]}ValMAE}}{{{g.val_MAE.mean():.4f}}}",
                f"\\newcommand{{\\TrafficSUNTDir{CHAVE[b]}ValMAEStd}}{{{g.val_MAE.std(ddof=1):.4f}}}",
                f"\\newcommand{{\\TrafficSUNTDir{CHAVE[b]}ValRMSE}}{{{g.val_RMSE.mean():.2f}}}"]
    lin.append(r"\bottomrule\end{tabular}")
    (dest / "tabelas" / "piloto_direcao.tex").write_text("\n".join(lin) + "\n")

    pb = aud["por_braco"]
    mac += [f"\\newcommand{{\\TrafficSUNTDirObsEdges}}{{{pb['observado']['arestas_orientadas']}}}",
            f"\\newcommand{{\\TrafficSUNTDirReciprocalPairs}}{{{pb['observado']['pares_reciprocos']}}}",
            f"\\newcommand{{\\TrafficSUNTDirSymInDegreeMedian}}{{{pb['simetrico']['grau_entrada']['mediana']:g}}}",
            f"\\newcommand{{\\TrafficSUNTDirObsInDegreeMedian}}{{{pb['observado']['grau_entrada']['mediana']:g}}}",
            f"\\newcommand{{\\TrafficSUNTDirObsNoIncoming}}{{{pb['observado']['grau_entrada']['zero']}}}",
            f"\\newcommand{{\\TrafficSUNTDirObsLargeReductionNodes}}{{{aud['estratos']['observado']['grande_reducao']}}}",
            f"\\newcommand{{\\TrafficSUNTDirRevLargeReductionNodes}}{{{aud['estratos']['invertido']['grande_reducao']}}}"]
    for (a, b), nome in zip(CONTRASTES, ("ObsMinusSym", "RevMinusSym", "ObsMinusRev")):
        r = pick(f"{a} − {b}")
        mac += [f"\\newcommand{{\\TrafficSUNTDir{nome}ValDiffMAE}}{{{r.media:+.4f}}}",
                f"\\newcommand{{\\TrafficSUNTDir{nome}ValPositive}}{{{r.positivas}}}"]
    for a, nome in (("observado", "ObsMinusSym"), ("invertido", "RevMinusSym")):
        mac += [f"\\newcommand{{\\TrafficSUNTDir{nome}ValLargeReductionDiffMAE}}{{{pick(f'{a} − simetrico', f'grande_reducao ({a})').media:+.4f}}}",
                f"\\newcommand{{\\TrafficSUNTDir{nome}ValOtherDiffMAE}}{{{pick(f'{a} − simetrico', f'demais ({a})').media:+.4f}}}"]
    with open(dest / "tabelas" / "macros_analises_sunt.tex", "a") as f:     # já incluído pela dissertação
        f.write("\n".join(mac) + "\n")

    return {"base": str(BASE.relative_to(ROOT)), "registros": len(runs),
            "fontes": [{"arquivo": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in fixos]
                      + [{"arquivo": r.fonte, "sha256": r.sha256} for r in runs.itertuples()]
                      + [{"arquivo": str((d / "validation_detail.npz").relative_to(ROOT)), "sha256": digest(d / "validation_detail.npz")}
                         for d in runs_dir.values()],
            "macros": len(mac) - 1,
            "estatistica": "Descritiva: diferenças pareadas por semente (média, dp, sementes > 0), só validação; sem IC, "
                           "sem p e fora das famílias Holm. O teste não foi avaliado."}
