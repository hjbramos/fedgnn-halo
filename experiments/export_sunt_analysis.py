"""Export audited SUNT analyses without changing the original comparison families.

Reconstruct each stratum from the original per-node errors and verify paired
contrasts before exporting. No training, checkpoint evaluation or layout previews.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .train import ROOT

SEEDS = [7, 42, 101, 123, 314]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracked(path) -> bool:
    """Insumo versionado no git. A verificação de árvore suja do exportador ignora arquivos não rastreados;
    sem esta exigência, um pacote poderia citar um commit que não contém as fontes das análises."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    return subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0


def require_tracked(*paths):
    for path in paths:
        require(tracked(path), f"insumo não versionado no git: {Path(path).relative_to(ROOT)}")


def require(ok, message):
    if not ok:
        raise ValueError("Análises SUNT: " + message)


def close(a, b):
    return bool(np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True))


def export(dest: Path) -> dict:
    descriptors_path = ROOT / "results/sunt_stratified_analysis/v1/node_descriptors.csv"
    require_tracked(descriptors_path)
    descriptors = pd.read_csv(descriptors_path)
    cols = {"fronteira": "fronteira_k{k}", "perda_vizinhança": "perda_viz_faixa_k{k}",
            "perda_peso": "perda_peso_faixa_k{k}", "volume": "volume_tercil", "cobertura": "cobertura_tercil"}
    provenance, frames = {}, {}
    for name, script in [("sunt_stratified_analysis", "sunt_stratified_analysis.py"),
                         ("sunt_graph_controls", "sunt_graph_controls_analysis.py")]:
        base = ROOT / "results" / name / "v1"
        require_tracked(ROOT / "experiments" / script, *[base / f for f in ("manifest.json", "checks.json", "protocol.md",
                        "strata_metrics.csv", "contrasts.csv", "boundary_vs_interior.csv", "audit_runs.csv")])
        if name == "sunt_graph_controls":
            require_tracked(base / "protocol_k166.md", base / "checkpoints_sha256.csv")
        manifest = json.loads((base / "manifest.json").read_text())
        require(not manifest.get("prévia_de_layout", False), "prévia de layout recusada")
        checks = json.loads((base / "checks.json").read_text())
        require(bool(checks) and all(c["ok"] for c in checks), name + ": auditoria reprovada")
        require(digest(ROOT / "experiments" / script) == manifest["script_sha256"], name + ": script mudou")
        require(digest(base / "protocol.md") == manifest["protocolo_sha256"], name + ": protocolo mudou")
        if name == "sunt_graph_controls":
            require(digest(base / "protocol_k166.md") == manifest["protocolo_k166_sha256"], "protocolo k166 mudou")
            require(digest(descriptors_path) == manifest["node_descriptors"]["sha256"], "estratos mudaram")
        methods = {}
        for run in manifest["execuções"]:
            source = ROOT / run["arquivo"]
            require_tracked(source)
            require(digest(source) == run["sha256"], "hash divergente: " + run["arquivo"])
            m = json.loads(source.read_text())
            label = run.get("braço", run.get("método"))
            k = m.get("num_clients") if name == "sunt_stratified_analysis" else None
            methods[label, k, int(run["semente"])] = m
        sm = pd.read_csv(base / "strata_metrics.csv")
        method_col = "método" if "método" in sm else "braço"
        for row in sm.to_dict("records"):
            label, k, seed = row[method_col], int(row["k"]), int(row["semente"])
            source_k = None if name == "sunt_graph_controls" or label == "Centralizado" else k
            m = methods[label, source_k, seed]
            pn = m[row["split"]]["per_node"]
            err, count = np.asarray(pn["abs_error"]), np.asarray(pn["count"])
            require(len(err) == len(count) == len(descriptors), "ordem/tamanho dos nós incompatível")
            require(np.isfinite(err).all() and (err >= 0).all() and np.isfinite(count).all()
                    and (count >= 0).all() and (count == np.floor(count)).all(), "erros/contagens inválidos")
            require(np.isclose(err.sum() / count.sum(), m[row["split"]]["MAE"], rtol=1e-6, atol=1e-10), "MAE original incompatível")
            dim = row["dimensão"]
            mask = np.ones(len(err), dtype=bool) if dim == "global" else (
                descriptors[cols[dim].format(k=k)].to_numpy() == row["estrato"])
            n = count[mask].sum()
            require(close(n, row["ocorrências_alvo"]) and close(mask.sum(), row["nós"]), "denominador incompatível")
            require(close(err[mask].sum(), row["erro_abs"]) and close((count[mask] > 0).sum(), row["nós_avaliáveis"])
                    and close(n/count.sum(), row["participação_denominador"]), "somas/participação incompatíveis")
            require(close(err[mask].sum() / n if n else np.nan, row["MAE"]), "MAE por estrato incompatível")
        ct = pd.read_csv(base / "contrasts.csv")
        pivot = sm.pivot(index=["split", "k", "dimensão", "estrato", "semente"], columns=method_col, values="MAE")
        for row in ct.to_dict("records"):
            a, b = row["contraste"].split(" − ")
            keys = [(row["split"], row["k"], row["dimensão"], row["estrato"], s) for s in SEEDS]
            values = np.array([pivot.loc[key, a] - pivot.loc[key, b] for key in keys])
            mean, sd = values.mean(), values.std(ddof=1)
            margin = stats.t.ppf(.975, 4) * sd / np.sqrt(5)
            require(close(values, [row[f"s{s}"] for s in SEEDS]) and close(
                [mean, sd, mean-margin, mean+margin],
                [row["média"], row["dp"], row["IC95_t_inf"], row["IC95_t_sup"]]), "contraste pareado incompatível")
        for row in pd.read_csv(base / "boundary_vs_interior.csv").to_dict("records"):
            a, b = row["contraste"].split(" − ")
            values = []
            for seed in SEEDS:
                f = pivot.loc[(row["split"], row["k"], "fronteira", "fronteira", seed)]
                i = pivot.loc[(row["split"], row["k"], "fronteira", "interior", seed)]
                values.append((f[a]-f[b])-(i[a]-i[b]))
            values = np.asarray(values)
            mean, sd = values.mean(), values.std(ddof=1)
            margin = stats.t.ppf(.975, 4)*sd/np.sqrt(5)
            require(close(values, [row[f"s{s}"] for s in SEEDS]) and close(
                [mean, sd, mean-margin, mean+margin],
                [row["média"], row["dp"], row["IC95_t_inf"], row["IC95_t_sup"]]), "contraste fronteira/interior incompatível")
        files = ["protocol.md", "manifest.json", "checks.json", "audit_runs.csv", "strata_metrics.csv",
                 "contrasts.csv", "boundary_vs_interior.csv"]
        if name == "sunt_graph_controls":
            files += ["protocol_k166.md", "checkpoints_sha256.csv"]
        else:
            files += ["node_descriptors.csv"]
        target = dest / "dados" / name
        target.mkdir()
        for file in files:
            shutil.copy2(base / file, target / file)
        provenance[name] = {"manifesto": str((base / "manifest.json").relative_to(ROOT)),
                            "fontes": [{"arquivo": str((base / f).relative_to(ROOT)), "sha256": digest(base / f)} for f in files],
                            "verificacoes_origem": len(checks), "registros": len(manifest["execuções"]),
                            "estatistica": "Exploratória; diferenças pareadas por semente; IC95% t não ajustado; sem ajuste de multiplicidade. Não integra as famílias Holm originais."}
        frames[name] = ct

    def number(x):
        return "\\num{" + f"{x:.3f}" + "}"

    def interval(row):
        return "[" + number(row.IC95_t_inf) + "; " + number(row.IC95_t_sup) + "]"

    def pick(frame, k, contrast, group="fronteira", dim="fronteira"):
        rows = frame[(frame.split == "test") & (frame.k == k) & (frame.contraste == contrast)
                     & (frame["dimensão"] == dim) & (frame.estrato == group)]
        require(len(rows) == 1, "contraste ausente ou duplicado")
        return rows.iloc[0]

    ct = frames["sunt_stratified_analysis"]
    lines = ["% Gerado e conferido contra per_node; não editar.",
             r"\begin{tabular}{lrrrrr}\toprule",
             r"$k$ & Nós & Alvos (\%) & FedAvg $-$ cent. & \shortstack{FedGNN-Halo\\$-$ FedAvg} & IC95\%\\\midrule"]
    for k in [6, 20, 166]:
        a = pick(ct, k, "FedAvg − Centralizado")
        h = pick(ct, k, "halo − FedAvg")
        lines.append(f"{k} & \\num{{{int(h['nós'])}}} & {number(100*h.participação_denominador)} & {number(a['média'])} & {number(h['média'])} & {interval(h)}" + r"\\")
    lines += [r"\bottomrule\end{tabular}"]
    (dest / "tabelas/sunt_estratos_fronteira.tex").write_text("\n".join(lines) + "\n")

    ct = frames["sunt_graph_controls"]
    pairs = [(20, "sem_arestas − Centralizado", r"\shortstack[l]{Centralizado sem arestas entre nós distintos\\$-$ centralizado com grafo completo}", "Noedges"),
             (20, "cortes_k20 − Centralizado", "Cortes $k=20$ $-$ completo", "CutTwenty"),
             (166, "cortes_k166 − Centralizado", "Cortes $k=166$ $-$ completo", "CutOneSixtySix"),
             (20, "FedAvg_k20 − cortes_k20", "FedAvg $k=20$ $-$ cortes", "GapTwenty"),
             (166, "FedAvg_k166 − cortes_k166", "FedAvg $k=166$ $-$ cortes", "GapOneSixtySix")]
    lines = ["% Gerado e conferido contra per_node; não editar.", r"\begin{tabular}{lrr}\toprule",
             r"Contraste & $\Delta$MAE & IC95\%\\\midrule"]
    macros = ["% Análises adicionais exploratórias; não alteram as famílias Holm originais."]
    for k, contrast, label, key in pairs:
        row = pick(ct, k, contrast, "todos", "global")
        lines.append(label + " & " + number(row['média']) + " & " + interval(row) + r"\\")
        for suffix, value in [("Diff", row['média']), ("CILow", row.IC95_t_inf), ("CIHigh", row.IC95_t_sup)]:
            macros.append(r"\newcommand{\TrafficControl" + key + suffix + "}{" + f"{value:.3f}" + "}")
    lines.append(r"\bottomrule\end{tabular}")
    (dest / "tabelas/sunt_controles_grafo.tex").write_text("\n".join(lines) + "\n")
    (dest / "tabelas/macros_analises_sunt.tex").write_text("\n".join(macros) + "\n")
    for name, filename, target in [("sunt_stratified_analysis", "fig1_fronteira_interior.png", "sunt_estratos_fronteira.png"),
                                   ("sunt_graph_controls", "fig2_fronteira_interior.png", "sunt_controles_fronteira.png")]:
        source = ROOT / "results" / name / "v1" / filename
        require_tracked(source)
        if name == "sunt_stratified_analysis":
            boundary_figure(frames[name], dest / "figuras" / target)
        else:
            controls_figure(frames[name], dest / "figuras" / target)
        provenance[name]["figura"] = {"arquivo": str(source.relative_to(ROOT)), "sha256": digest(source)}
    provenance["sunt_stratified_analysis"]["figura_apresentacao"] = {
        "gerador": "experiments/export_sunt_analysis.py:boundary_figure",
        "nota": "Regerada dos contrastes auditados; nome de apresentação FedGNN-Halo, sem alteração dos dados."
    }
    provenance["sunt_graph_controls"]["figura_apresentacao"] = {
        "gerador": "experiments/export_sunt_analysis.py:controls_figure",
        "nota": "Regerada dos contrastes auditados com os rótulos da dissertação, sem alteração dos dados."
    }
    return provenance


def _milhar(x):
    return f"{int(x):,}".replace(",", ".")


def _comma_axis(ax):
    """Vírgula decimal nos rótulos do eixo vertical (figuras em português)."""
    from matplotlib.ticker import ScalarFormatter

    class _Comma(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")

    ax.yaxis.set_major_formatter(_Comma())


def boundary_figure(ct, target):
    import matplotlib
    with matplotlib.rc_context():
        _boundary_figure(ct, target)


def controls_figure(ct, target):
    import matplotlib
    with matplotlib.rc_context():
        _controls_figure(ct, target)


def _controls_figure(ct, target):
    """Contrastes auditados dos controles com os rótulos da dissertação; sem nova análise (1.24.0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sunt_stratified_analysis import _style, SEEDS, INK, MUTED
    from .sunt_graph_controls_analysis import COLORS, BLOCKS
    _style(plt)
    order = ["fronteira", "interior"]

    def label(name, k):
        a, b = name.split(" − ")
        nomes = {"Centralizado": "centralizado com grafo completo",
                 "sem_arestas": "Centralizado sem arestas entre nós distintos",
                 f"cortes_k{k}": "centralizado com arestas internas",
                 f"FedAvg_k{k}": "FedAvg"}
        a, b = nomes[a], nomes[b]
        return f"{a[0].upper()}{a[1:]}\n− {b}"

    fig, axes = plt.subplots(len(BLOCKS), 1, figsize=(8, 8.4), squeeze=False)
    for ax, k in zip(axes[:, 0], BLOCKS):
        sub = ct[(ct["split"] == "test") & (ct["k"] == k) & (ct["dimensão"] == "fronteira")]
        names = [n for n in COLORS if n in set(sub["contraste"])]
        x = np.arange(len(order))
        for j, cname in enumerate(names):
            s = sub[sub["contraste"] == cname].set_index("estrato").reindex(order)
            off = (j - (len(names) - 1) / 2) * 0.2
            for i, g in enumerate(order):
                if pd.isna(s.loc[g, "média"]):
                    continue
                ax.scatter(np.full(5, x[i] + off), [s.loc[g, f"s{seed}"] for seed in SEEDS], s=12, color=COLORS[cname],
                           alpha=0.5, linewidths=0)
                ax.plot([x[i] + off - 0.08, x[i] + off + 0.08], [s.loc[g, "média"]] * 2, color=COLORS[cname], lw=2,
                        label=label(cname, k) if i == 0 else None)
        ax.axhline(0, color=MUTED, lw=0.8)
        n = sub[sub["contraste"] == names[0]].set_index("estrato").reindex(order)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g.capitalize()}\n{_milhar(n.loc[g, 'nós'])} nós · {100 * n.loc[g, 'participação_denominador']:.0f}% dos alvos"
                            for g in order], fontsize=9)
        ax.set_title(f"Partição $k={k}$", fontsize=10, loc="left", color=INK)
        ax.set_ylabel("Δ MAE de teste (embarques/h)")
        _comma_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.01), columnspacing=2, labelspacing=1)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(target, dpi=300)
    plt.close(fig)


def _boundary_figure(ct, target):
    """Render audited contrasts with the dissertation method name; no new analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sunt_stratified_analysis import _style, KS, SEEDS, NO_IN, K_COLORS, INK, MUTED
    _style(plt)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey="row")
    for r, contrast in enumerate(["FedAvg − Centralizado", "halo − FedAvg"]):
        for c, k in enumerate(KS):
            ax = axes[r, c]
            sub = ct[(ct["split"] == "test") & (ct["contraste"] == contrast)
                     & (ct["dimensão"] == "fronteira") & (ct["k"] == k)]
            groups = [g for g in ["fronteira", "interior", NO_IN] if g in set(sub["estrato"])]
            sub = sub.set_index("estrato").reindex(groups)
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5),
                           [row[f"s{s}"] for s in SEEDS], s=14,
                           color=K_COLORS[k], alpha=0.6, linewidths=0)
                ax.plot([i - 0.2, i + 0.2], [row["média"]] * 2, color=INK, lw=2)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels([f"{g.capitalize()}\n{_milhar(row.nós)} nós\n{100 * row.participação_denominador:.0f}% dos alvos"
                                for g, row in sub.iterrows()], fontsize=8.5)
            ax.set_title(f"{contrast.replace('halo', 'FedGNN-Halo')}, $k={k}$",
                         fontsize=9.5, loc="left", color=INK)
            _comma_axis(ax)
            if c == 0:
                ax.set_ylabel("Δ MAE de teste (embarques/h)")
    fig.tight_layout()
    fig.savefig(target, dpi=300)
    plt.close(fig)
