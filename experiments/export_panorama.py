"""Panorama descritivo: diferenças percentuais entre médias, sem inferência.

Usa as mesmas médias por célula das tabelas. Não calcula a média das razões
por semente, não altera resultados e não executa treinamentos.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def write_panorama(dest: Path, summary: pd.DataFrame, colors: dict, markers: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    methods = [("FedAvg", "fedavg"), ("FedGNN-Halo", "halo"),
               ("FedProx", "fedprox"), ("Ditto", "ditto"), ("Local", "local")]
    panels = [("METR-LA", "metrla", (5, 10, 20)),
              ("PEMS-BAY", "pemsbay", (5, 10, 20)),
              ("SUNT", "sunt", (6, 20, 166))]
    rows = []
    for dataset, prefix, ks in panels:
        central = summary.loc[summary.celula == f"{prefix}_cent", "MAE_media"]
        assert len(central) == 1 and central.iloc[0] > 0
        reference = float(central.iloc[0])
        for method, key in methods:
            for k in ks:
                cell = f"{prefix}_{key}_k{k}"
                source = summary.loc[summary.celula == cell]
                assert len(source) == 1 and int(source.n.iloc[0]) == 5, cell
                mean = float(source.MAE_media.iloc[0])
                relative = 100 * (mean - reference) / reference
                assert np.isfinite(relative), cell
                rows.append(dict(conjunto=dataset, celula=cell, metodo=method, k=k,
                                 MAE_media=mean, MAE_centralizado=reference,
                                 diferenca_percentual=relative))
    data = pd.DataFrame(rows)
    data.to_csv(dest / "dados" / "panorama_relativo_centralizado.csv",
                index=False, float_format="%.17g")
    with plt.rc_context({"font.size": 9, "svg.hashsalt": "fed-stgnn-traffic-panorama",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True}):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.25))
        for ax, (dataset, _, ks), letter in zip(axes, panels, "abc"):
            ax.axhline(0, color="#666666", lw=1, ls="--", zorder=1)
            for method, _ in methods:
                values = data[(data.conjunto == dataset) & (data.metodo == method)]
                ax.plot(range(3), values.diferenca_percentual,
                        color=colors[method], marker=markers[method],
                        ms=4.5, lw=1.1, label=method, markeredgewidth=0.5,
                        markeredgecolor="white")
            ax.set_title(f"({letter}) {dataset}", fontsize=10)
            ax.set_xticks(range(3), [str(k) for k in ks])
            ax.set_xlabel("Número de clientes ($k$)")
            ax.set_xlim(-0.18, 2.18)
            ax.grid(axis="y", color="#e2e2e2", linewidth=0.6)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
            if dataset == "SUNT":
                ax.set_ylim(-1, 27)
                ax.set_yticks([0, 5, 10, 15, 20, 25])
            else:
                ax.set_ylim(-3.2, 4.5)
                ax.set_yticks([-2, 0, 2, 4])
        axes[0].set_ylabel("Diferença de MAE frente\nao centralizado (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
                   fontsize=9, handlelength=1.8, columnspacing=1.3)
        fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.0)
        base = dest / "figuras" / "panorama_relativo_centralizado"
        fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None, "Creator": None})
        fig.savefig(base.with_suffix(".svg"), metadata={"Date": None, "Creator": None})
        fig.savefig(base.with_suffix(".png"), dpi=250, metadata={"Software": None})
        plt.close(fig)
