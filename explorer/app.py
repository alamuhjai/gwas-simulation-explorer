"""Interactive GWAS explorer: statistical power, and why population structure
breaks a naive GWAS (simulated data only)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from shiny import App, reactive, render, ui

import structure as sd

SIDEBAR_OPEN = {"desktop": "open", "mobile": "closed"}


def stat(label, outputId):
    return ui.div(ui.div(label, class_="lbl"),
                  ui.div(ui.output_text(outputId, inline=True), class_="val"),
                  class_="stat")


def plotCard(titleOutput, plotId):
    return ui.card(ui.card_header(ui.output_text(titleOutput, inline=True)),
                   ui.output_plot(plotId, height="40vh"), full_screen=True)


appUi = ui.page_sidebar(
    ui.sidebar(
        ui.h5("View"),
        ui.input_radio_buttons(
            "view_mode",
            None,
            {
                "power": "Power explorer",
                "structure": "Population structure",
            },
            selected="power",
            width="100%",
        ),
        ui.hr(style="margin: 12px 0;"),
        ui.panel_conditional(
            "input.view_mode === 'power'",
            ui.h5("Parameters"),
            ui.input_slider("n", "Sample size (N)", 1000, 500000, 50000, step=1000),
            ui.input_slider("maf", "Minor allele frequency", 0.01, 0.5, 0.2, step=0.01),
            ui.input_slider("beta", "Effect size β (residual SD per allele)",
                            0.01, 0.5, 0.05, step=0.005),
            ui.input_select("alpha", "Significance threshold",
                            {"5e-8": "5e-8 (genome-wide)", "1e-5": "1e-5 (suggestive)",
                             "0.05": "0.05 (single test)"}),
            ui.input_slider("m", "Number of causal SNPs to simulate", 1, 50, 10),
        ),
        ui.panel_conditional(
            "input.view_mode === 'structure'",
            ui.h5("Parameters"),
            ui.input_slider("shift", "Ancestry effect on the trait (non-genetic shift, SD)",
                            0.0, 1.5, 0.6, step=0.1),
            ui.input_slider("npc", "Principal components used as covariates",
                            0, 5, 2, step=1),
            ui.p("Fixed settings: 1,000 individuals in two subpopulations "
                 "(Balding-Nichols, Fst = 0.05), 5,000 SNPs, 20 causal SNPs, h² = 0.3. "
                 "The trait differs between the subpopulations for non-genetic reasons, "
                 "so any SNP whose allele frequency differs between them looks "
                 "associated. Naive = no covariates; adjusted = top principal "
                 "components of the genotype matrix as covariates.", class_="note"),
            ui.p("All data are simulated. Nothing here is real cohort data.",
                 class_="note"),
        ),
        width=320,
        open=SIDEBAR_OPEN,
    ),
    ui.tags.head(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.include_css(Path(__file__).parent / "styles.css"),
    ),
    ui.panel_conditional(
        "input.view_mode === 'power'",
        ui.div(
            stat("Statistical power", "power_val"),
            stat("Variance explained (r²)", "r2_val"),
            stat("Non-centrality (NCP)", "ncp_val"),
            stat("Threshold", "alpha_val"),
            class_="stats",
        ),
        ui.layout_columns(
            ui.card(ui.card_header("Power vs sample size"),
                    ui.output_plot("power_plot", height="46vh"), full_screen=True),
            ui.card(ui.card_header("Simulated Manhattan plot"),
                    ui.output_plot("manhattan_plot", height="46vh"), full_screen=True),
            col_widths={"sm": (12, 12), "xl": (6, 6)}),
    ),
    ui.panel_conditional(
        "input.view_mode === 'structure'",
        ui.div(
            stat("Naive inflation factor (lambda)", "lam_naive"),
            stat("Adjusted inflation factor", "lam_adj"),
            stat("Naive genome-wide hits", "hits_naive"),
            stat("Adjusted genome-wide hits", "hits_adj"),
            class_="stats",
        ),
        ui.layout_columns(
            plotCard("title_naive_man", "naive_manhattan"),
            plotCard("title_naive_qq", "naive_qq"),
            col_widths={"sm": (12, 12), "xl": (6, 6)}),
        ui.layout_columns(
            plotCard("title_adj_man", "adj_manhattan"),
            plotCard("title_adj_qq", "adj_qq"),
            col_widths={"sm": (12, 12), "xl": (6, 6)}),
    ),
    title=ui.span(
        ui.span("GWAS Explorer", class_="app-title"),
        ui.span("Power and population structure on simulated data", class_="app-sub"),
    ),
    window_title="GWAS Explorer",
    fillable=False,
)


def varExplained(maf, beta):
    """r2 of an additive SNP: y = beta*G + e, Var(e) = 1, Var(G) = 2p(1-p).

    r2 = 2p(1-p)beta^2 / (1 + 2p(1-p)beta^2), so N*r2/(1-r2) = N*2p(1-p)*beta^2.
    """
    v = 2 * maf * (1 - maf) * beta ** 2
    return v / (1 + v)


def power(n, r2, alpha):
    """Power via the non-central chi-square with 1 df; NCP = N * r2 / (1 - r2)."""
    ncp = n * r2 / (1 - r2)
    crit = stats.chi2.isf(alpha, 1)
    return stats.ncx2.sf(crit, 1, ncp)


def missProb(n, r2, alpha):
    """Type II error (1 - power) for a 1-df test, accurate when tiny.

    Test statistic |Z + d|, d = sqrt(NCP); reject if |Z + d| > c, c = sqrt(chi2 crit).
    Miss = P(-c < Z + d < c) = Phi(c - d) - Phi(-c - d).
    """
    d = np.sqrt(n * r2 / (1 - r2))
    c = np.sqrt(stats.chi2.isf(alpha, 1))
    return stats.norm.cdf(c - d) - stats.norm.cdf(-c - d)


def fmtPower(n, r2, alpha):
    miss = missProb(n, r2, alpha)
    p = 1 - miss
    if p < 0.999:
        return f"{p:.1%}"
    if miss > 1e-6:
        return f"{p:.4%}"
    if miss > 1e-15:
        return f"1 − {miss:.0e}"
    return "> 99.9999999999999%"


def server(input, output, session):
    # ----------------------------------------------------------------- power
    def r2():
        return varExplained(input.maf(), input.beta())

    @render.text
    def power_val():
        return fmtPower(input.n(), r2(), float(input.alpha()))

    @render.text
    def r2_val():
        return f"{r2():.3%}"

    @render.text
    def ncp_val():
        return f"{input.n() * r2() / (1 - r2()):.1f}"

    @render.text
    def alpha_val():
        return f"{float(input.alpha()):g}"

    @render.plot
    def power_plot():
        a = float(input.alpha())
        ns = np.logspace(3, np.log10(500000), 100)
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(ns, power(ns, r2(), a), lw=2.5, color="#3b5b92")
        ax.fill_between(ns, power(ns, r2(), a), alpha=.12, color="#3b5b92")
        ax.axvline(input.n(), color="#e4572e", ls="--")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Sample size N")
        ax.set_ylabel("Power")
        ax.grid(alpha=.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return fig

    @render.plot
    def manhattan_plot():
        rng = np.random.default_rng(0)
        nSnps = 5000
        n = input.n()
        z = rng.normal(size=nSnps)
        causalSnps = rng.choice(nSnps, input.m(), replace=False)
        # Each causal SNP has the chosen MAF and effect (random sign):
        # under H1, z ~ N(sqrt(NCP), 1).
        shift = np.sqrt(n * r2() / (1 - r2()))
        z[causalSnps] += shift * rng.choice([-1, 1], len(causalSnps))
        logp = -stats.chi2.logsf(z ** 2, 1) / np.log(10)
        fig, ax = plt.subplots(layout="constrained")
        chrom = np.arange(nSnps) * 10 // nSnps
        ax.scatter(np.arange(nSnps), logp,
                   c=np.where(chrom % 2 == 0, "#3b5b92", "#8aa4d6"), s=5)
        ax.axhline(-np.log10(float(input.alpha())), color="red", ls="--", lw=1)
        ax.set_xlabel("SNP index")
        ax.set_ylabel("-log10(p)")
        ax.grid(alpha=.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return fig

    # ------------------------------------------------------ population structure
    @reactive.calc
    def analysis():
        data = sd.simDataset(input.shift())
        k = int(input.npc())
        naive, adjusted = sd.analyseData(data, k)
        return {
            "causal": data["causal"],
            "naive": naive,
            "adjusted": adjusted,
            "sNaive": sd.summariseResults(naive, data["causal"]),
            "sAdj": sd.summariseResults(adjusted, data["causal"]),
            "limit": sd.sharedLimits(naive, adjusted),
            "k": k,
        }

    def adjLabel():
        k = analysis()["k"]
        return "Adjusted for 0 PCs (same as naive)" if k == 0 else f"Adjusted for {k} PC{'s' if k != 1 else ''}"

    def hitsText(s):
        return f"{s['hits']} ({s['true_hits']} causal)"

    @render.text
    def lam_naive():
        return f"{analysis()['sNaive']['lam']:.2f}"

    @render.text
    def lam_adj():
        return f"{analysis()['sAdj']['lam']:.2f}"

    @render.text
    def hits_naive():
        return hitsText(analysis()["sNaive"])

    @render.text
    def hits_adj():
        return hitsText(analysis()["sAdj"])

    @render.text
    def title_naive_man():
        return "Naive: Manhattan"

    @render.text
    def title_naive_qq():
        return "Naive: QQ"

    @render.text
    def title_adj_man():
        return f"{adjLabel()}: Manhattan"

    @render.text
    def title_adj_qq():
        return f"{adjLabel()}: QQ"

    @render.plot
    def naive_manhattan():
        a = analysis()
        return sd.makeFigure("manhattan", a["naive"], a["causal"], "", limit=a["limit"])

    @render.plot
    def naive_qq():
        a = analysis()
        return sd.makeFigure("qq", a["naive"], a["causal"], "", lam=a["sNaive"]["lam"],
                             limit=a["limit"])

    @render.plot
    def adj_manhattan():
        a = analysis()
        return sd.makeFigure("manhattan", a["adjusted"], a["causal"], "", limit=a["limit"])

    @render.plot
    def adj_qq():
        a = analysis()
        return sd.makeFigure("qq", a["adjusted"], a["causal"], "", lam=a["sAdj"]["lam"],
                             limit=a["limit"])


app = App(appUi, server)