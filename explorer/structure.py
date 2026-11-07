"""Glue for the "Population structure" tab: naive vs PC-adjusted GWAS.

No GWAS logic lives here. Simulation, regression, PCA, lambda and the plot
functions all come from the gwas modules; this file only wires them to the
app's sliders.
"""
import numpy as np
import matplotlib.pyplot as plt

from gwas.simulate import simGenotypes, simPhenotype
from gwas.association import compPCs, runGWAS, genoInflation
from gwas.plot import manhattan, qq, GW_THRESHOLD

_BASE = {}


def baseGenotypes(fst=0.05, n=1000, m=5000, maxPCs=5):
    """Genotypes and PCs depend only on Fst, so compute them once and reuse."""
    key = (fst, n, m, maxPCs)
    if key not in _BASE:
        G, pop = simGenotypes(n=n, m=m, fst=fst, seed=1)
        _BASE[key] = {"G": G, "pop": pop,
                      "pcs": compPCs(G, k=maxPCs, method="gram")}
    return _BASE[key]


def simDataset(popShift, fst=0.05, n=1000, m=5000, nCausal=20, h2=0.3,
               maxPCs=5):
    """Genotypes (cached) plus a phenotype with the chosen ancestry shift."""
    base = baseGenotypes(fst, n, m, maxPCs)
    y, causalSnps = simPhenotype(base["G"], base["pop"], nCausal=nCausal,
                                 h2=h2, popShift=popShift, seed=2)
    return {"G": base["G"], "y": y, "pop": base["pop"], "causal": causalSnps,
            "pcs": base["pcs"]}


def analyseData(data, k):
    """Naive GWAS and GWAS adjusted for the top k PCs (k = 0 means naive)."""
    naive = runGWAS(data["G"], data["y"])
    adjusted = naive if k == 0 else runGWAS(data["G"], data["y"],
                                            data["pcs"][:, :k])
    return naive, adjusted


def summariseResults(res, causalSnps):
    """Lambda, genome-wide hits, how many hits are truly causal, share p<0.05."""
    p = res["p"].values
    hits = np.flatnonzero(p < GW_THRESHOLD)
    trueHits = np.intersect1d(hits, causalSnps)
    return {
        "lam": genoInflation(p),
        "hits": int(len(hits)),
        "true_hits": int(len(trueHits)),
        "false_hits": int(len(hits) - len(trueHits)),
        "frac_05": float(np.nanmean(p < 0.05)),
    }


def sharedLimits(*results):
    """One shared axis limit so the naive and adjusted plots are comparable."""
    top = 0.0
    for res in results:
        p = res["p"].values
        p = p[np.isfinite(p)]
        top = max(top, float(-np.log10(p.min())))
    return top + 1.0


def makeFigure(kind, res, causalSnps, title, lam=None, limit=None):
    """Build one Manhattan or QQ figure using the gwas plotting functions."""
    fig, ax = plt.subplots(layout="constrained")
    if kind == "manhattan":
        manhattan(ax, res, causalSnps=causalSnps, title=title, ymax=limit)
    else:
        qq(ax, res["p"].values, lam=lam, title=title, lim=limit)
    ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig
