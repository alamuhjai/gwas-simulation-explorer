"""Manhattan and QQ plots."""
import numpy as np
import matplotlib.pyplot as plt

GW_THRESHOLD = 5e-8


def manhattan(ax, res, causalSnps=None, nChr=10, title="", ymax=None):
    """Manhattan plot; SNPs are split evenly into pseudo-chromosomes."""
    m = len(res)
    logp = -np.log10(res["p"].values)
    chrom = np.minimum(np.arange(m) * nChr // m, nChr - 1)
    colors = np.where(chrom % 2 == 0, "#3b5b92", "#8aa4d6")
    ax.scatter(np.arange(m), logp, c=colors, s=5)
    if causalSnps is not None:
        ax.scatter(causalSnps, logp[causalSnps], facecolors="none", edgecolors="#e4572e",
                   s=50, label="true causal SNP")
        ax.legend(loc="upper right")
    ax.axhline(-np.log10(GW_THRESHOLD), color="red", ls="--", lw=1)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_xlabel("SNP index")
    ax.set_ylabel("-log10(p)")
    ax.set_title(title)


def qq(ax, p, lam=None, title="", lim=None):
    """QQ plot of observed vs expected -log10 p-values."""
    p = np.sort(np.asarray(p)[np.isfinite(p)])
    exp = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
    obs = -np.log10(p)
    ax.scatter(exp, obs, s=5, color="#3b5b92")
    end = max(exp.max(), obs.max()) if lim is None else lim
    ax.plot([0, end], [0, end], color="red", lw=1)
    if lim is not None:
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    ax.set_xlabel("expected -log10(p)")
    ax.set_ylabel("observed -log10(p)")
    ax.set_title(title + (f"  (λ = {lam:.2f})" if lam is not None else ""))
