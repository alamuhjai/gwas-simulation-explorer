"""Per-SNP linear regression GWAS, PCA, and genomic inflation."""
import numpy as np
import pandas as pd
from scipy import stats


def compPCs(G, k=2, method="svd"):
    """Top-k principal components of the standardized genotype matrix.

    method="svd" (default) is the exact reference version. method="gram"
    eigendecomposes the n x n matrix X X^T instead, which gives the same
    components (up to sign, ~1e-12) several times faster when n << m. The
    browser demo uses it so the first load stays quick.
    """
    keep = G.std(0) > 0
    X = (G[:, keep] - G[:, keep].mean(0)) / G[:, keep].std(0)
    if method == "gram":
        w, V = np.linalg.eigh(X @ X.T)
        idx = np.argsort(w)[::-1][:k]
        return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * S[:k]


def runGWAS(G, y, covars=None):
    """Fast vectorised OLS of y on each SNP, adjusting for covariates.

    Uses the Frisch-Waugh-Lovell trick: residualise y and every SNP on the
    covariates (and intercept), then a simple regression gives the SNP effect.
    """
    n, m = G.shape
    C = np.ones((n, 1))
    if covars is not None:
        C = np.column_stack([C, covars])
    Q, _ = np.linalg.qr(C)
    yr = y - Q @ (Q.T @ y)
    Gr = G - Q @ (Q.T @ G)
    sxx = (Gr ** 2).sum(0)
    ok = sxx > 1e-12
    beta = np.full(m, np.nan)
    se = np.full(m, np.nan)
    beta[ok] = (Gr[:, ok].T @ yr) / sxx[ok]
    df = n - C.shape[1] - 1
    rss = ((yr[:, None] - Gr * np.nan_to_num(beta)) ** 2).sum(0)
    se[ok] = np.sqrt(rss[ok] / df / sxx[ok])
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return pd.DataFrame({"snp": np.arange(m), "beta": beta, "se": se, "p": p})


def genoInflation(p):
    """Lambda GC: median observed chi-square over its expected value under H0."""
    p = np.asarray(p)
    p = p[np.isfinite(p)]
    chi2 = stats.chi2.isf(p, 1)
    return float(np.median(chi2) / stats.chi2.ppf(0.5, 1))
