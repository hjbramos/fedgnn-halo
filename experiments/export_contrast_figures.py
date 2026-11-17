"""Apresentação dos contrastes auditados; não altera famílias nem resultados.

ICs pareados sem ajuste. Percentuais usam a média do FedAvg como escala comum.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

def write_contrast_figures(dest: Path):
    G = dest
    OUT = dest / "figuras"
    SEEDS = [7, 42, 101, 123, 314]
    PANELS = [("METR-LA", "metr-la", (5, 10, 20)), ("PEMS-BAY", "pems-bay", (5, 10, 20)),
              ("SUNT", "sunt", (6, 20, 166))]

    d = pd.read_csv(G / "dados/resultados_por_semente.csv")
    ctl = pd.read_csv(G / "dados/controles_arestas_internas/resultados_por_semente.csv")
    aud = pd.read_csv(G / "dados/sunt_graph_controls/audit_runs.csv")
    part = pd.read_csv(G / "dados/particoes.csv")
    macros = dict(re.findall(r"\\newcommand\{\\Traffic(\w+)\}\{([^}]*)\}",
                             (G / "tabelas/macros_resultados.tex").read_text()))


    def serie(df, col="test_MAE"):
        assert len(df) == len(SEEDS) and not df.semente.duplicated().any()
        s = df.set_index("semente")[col].reindex(SEEDS)
        assert s.notna().all(), df.head()
        return s.to_numpy(float)


    def principal(ds, metodo, k=None):
        q = d[(d.grupo == "principal") & (d.dataset == ds) & (d.metodo == metodo)]
        if k is not None:
            q = q[q.k == k]
        return serie(q)


    def interno(ds, k):
        if ds == "sunt" and k in (20, 166):
            return serie(aud[aud["braço"] == f"cortes_k{k}"])
        return serie(ctl[(ctl.dataset == ds) & (ctl.k == k)], "MAE")


    def ic(x):
        m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return m, m - h, m + h


    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    rc = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
          "svg.hashsalt": "fed-stgnn-traffic-contrastes"}

    # ---------------- Figura 1: decomposição ----------------
    rows = []
    for nome, ds, ks in PANELS:
        cent = principal(ds, "Centralizado")
        for k in ks:
            fed, cut = principal(ds, "FedAvg", k), interno(ds, k)
            for comp, x in [("total", fed - cent), ("retirada", cut - cent), ("restante", fed - cut)]:
                m, lo, hi = ic(x)
                rows.append(dict(conjunto=nome, k=k, componente=comp, media=m, ic_inf=lo, ic_sup=hi,
                                 **{f"s{s}": v for s, v in zip(SEEDS, x)}))
    dec = pd.DataFrame(rows)
    for nome, _, ks in PANELS:
        for k in ks:
            vals = dec[(dec.conjunto == nome) & (dec.k == k)].set_index("componente")
            for seed in SEEDS:
                assert np.isclose(vals.loc["total", f"s{seed}"], vals.loc["retirada", f"s{seed}"] + vals.loc["restante", f"s{seed}"], atol=1e-12)
    dec.to_csv(dest / "dados" / "contrastes_controles_centralizados.csv", index=False, float_format="%.17g")

    SER = [("retirada", "Centralizado com arestas internas − centralizado completo", "#C0612B", "s"),
           ("restante", "FedAvg − centralizado com arestas internas", "#2a78d6", "o"),
           ("total", "FedAvg − centralizado completo", "#1A1D21", "D")]
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.4))
        for ax, (nome, _, ks), letra in zip(axes, PANELS, "abc"):
            ax.axhline(0, color="#666666", lw=1, ls="--", zorder=1)
            for j, (comp, _, cor, mk) in enumerate(SER):
                v = dec[(dec.conjunto == nome) & (dec.componente == comp)]
                xs = np.arange(3) + (j - 1) * 0.22
                ax.errorbar(xs, v.media, yerr=[v.media - v.ic_inf, v.ic_sup - v.media], fmt=mk,
                            color=cor, ms=5, lw=1.2, capsize=2.2,
                            mfc="white" if comp == "total" else cor, mew=1.1, zorder=3)
            ax.set_title(f"({letra}) {nome}", fontsize=10)
            ax.set_xticks(range(3), [str(k) for k in ks]); ax.set_xlim(-0.5, 2.5)
            ax.set_xlabel("Número de clientes ($k$)")
            ax.grid(axis="y", color="#e2e2e2", lw=0.6); ax.yaxis.set_major_formatter(fmt)
        axes[0].set_ylabel("Diferença de MAE (mph)"); axes[2].set_ylabel("Diferença de MAE (embarques/h)")
        hs = [Line2D([], [], color=c, marker=m, lw=1.2, mfc="white" if k == "total" else c, ms=5)
              for k, _, c, m in SER]
        fig.legend(hs, [l for _, l, _, _ in SER], loc="lower center", ncol=2, frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"contrastes_controles_centralizados.{ext}", dpi=250, metadata={"CreationDate": None, "ModDate": None, "Creator": None} if ext == "pdf" else {"Software": None})
        plt.close(fig)

    # ---------------- Figura 2: halo × arestas cortadas ----------------
    CODE = {"METR-LA": "MetrLA", "PEMS-BAY": "PemsBay", "SUNT": "SUNT"}
    KW = {5: "Five", 6: "Six", 10: "Ten", 20: "Twenty", 166: "OneSixtySix"}
    CORES = {"METR-LA": ("#0F86A0", "o"), "PEMS-BAY": ("#7B5EA7", "^"), "SUNT": ("#C0612B", "s")}
    rows = []
    for nome, ds, ks in PANELS:
        for k in ks:
            fed, halo = principal(ds, "FedAvg", k), principal(ds, "FedGNN-Halo", k)
            x = 100 * (halo - fed) / fed.mean()
            m, lo, hi = ic(x)
            comparison = pd.read_csv(G / "dados/diferencas_pareadas.csv")
            prefix = {"METR-LA": "metrla", "PEMS-BAY": "pemsbay", "SUNT": "sunt"}[nome]
            ref = comparison[(comparison.A == f"{prefix}_halo_k{k}") & (comparison.B == f"{prefix}_fedavg_k{k}")]
            assert len(ref) == 1
            scale = 100 / fed.mean()
            assert np.allclose([m, lo, hi], ref.iloc[0][["dif_MAE_media", "IC95_inf", "IC95_sup"]].to_numpy(float) * scale, atol=1e-10)
            p = part[(part.dataset == ds) & (part.k == k)].iloc[0]
            c = CODE[nome]
            ph = macros[f"{c}HaloK{KW[k]}Minus{c}FedavgK{KW[k]}PHolm"]
            rows.append(dict(conjunto=nome, k=k, cortes_pct=p.frac_arestas_dirigidas_cortadas_pct,
                             entrada_externa_pct=p.nos_com_entrada_externa_pct, rel_media=m,
                             rel_ic_inf=lo, rel_ic_sup=hi, diff_abs=(halo - fed).mean(), p_holm=ph,
                             sig=ph.startswith("<") or float(ph) < 0.05))
    hal = pd.DataFrame(rows)
    hal.to_csv(dest / "dados" / "halo_fracao_cortes.csv", index=False, float_format="%.17g")

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        ax.axhline(0, color="#666666", lw=1, ls="--", zorder=1)
        for nome, (cor, mk) in CORES.items():
            v = hal[hal.conjunto == nome]
            for _, r in v.iterrows():
                ax.errorbar(r.cortes_pct, r.rel_media, yerr=[[r.rel_media - r.rel_ic_inf], [r.rel_ic_sup - r.rel_media]],
                            fmt=mk, color=cor, ms=6, lw=1.2, capsize=2.5, mfc=cor if r.sig else "white", mew=1.2, zorder=3)
                guia = {("PEMS-BAY", 5): (5.2, 0.95), ("PEMS-BAY", 20): (19.5, 0.65), ("METR-LA", 10): (19.5, -0.7)}
                if (nome, r.k) in guia:
                    ax.annotate(f"$k={r.k}$", (r.cortes_pct, r.rel_media), xytext=guia[(nome, r.k)], textcoords="data",
                                ha="left", va="center", fontsize=7.5, color="#4A525A",
                                arrowprops=dict(arrowstyle="-", color="#A7ADB3", lw=0.7, shrinkA=2, shrinkB=4))
                else:
                    ax.annotate(f"$k={r.k}$", (r.cortes_pct, r.rel_ic_sup), xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.5, color="#4A525A")
        ax.set_xlabel("Arestas cortadas na partição (%)")
        ax.set_ylabel("FedGNN-Halo − FedAvg\n(% do MAE do FedAvg)")
        ax.grid(axis="y", color="#e2e2e2", lw=0.6)
        ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
        hs = [Line2D([], [], color=c, marker=m, ls="", ms=6) for c, m in CORES.values()]
        hs += [Line2D([], [], color="#4A525A", marker="o", ls="", ms=6, mfc="#4A525A"),
               Line2D([], [], color="#4A525A", marker="o", ls="", ms=6, mfc="white")]
        fig.legend(hs, list(CORES) + ["$p_{\\mathrm{Holm}}<0{,}05$", "inconclusiva"], frameon=False,
                   fontsize=8, loc="lower center", ncol=5, columnspacing=1.2, handletextpad=0.3)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"halo_fracao_cortes.{ext}", dpi=250, metadata={"CreationDate": None, "ModDate": None, "Creator": None} if ext == "pdf" else {"Software": None})
        plt.close(fig)

