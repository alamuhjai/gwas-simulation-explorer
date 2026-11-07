"""Simulate genotypes with population structure and a confounded phenotype."""
import numpy as np


def simGenotypes(n=1000, m=5000, fst=0.05, seed=1):
    """Two subpopulations via the Balding-Nichols model.

    Returns genotype matrix G (n x m, values 0/1/2) and population labels (n,).
    """
    rng = np.random.default_rng(seed)
    pop = np.repeat([0, 1], [n // 2, n - n // 2])
    anc = rng.uniform(0.05, 0.95, m)
    a = anc * (1 - fst) / fst
    b = (1 - anc) * (1 - fst) / fst
    freqs = np.vstack([rng.beta(a, b), rng.beta(a, b)])
    G = rng.binomial(2, freqs[pop]).astype(float)
    return G, pop


def simPhenotype(G, pop, nCausal=20, h2=0.3, popShift=0.0, seed=2):
    """Quantitative trait: causal SNP effects + optional population shift + noise.

    popShift > 0 creates confounding by ancestry (non-genetic mean difference).
    Returns phenotype y and the indices of the causal SNPs.
    """
    rng = np.random.default_rng(seed)
    n, m = G.shape
    causalSnps = rng.choice(m, nCausal, replace=False)
    Z = (G[:, causalSnps] - G[:, causalSnps].mean(0)) / G[:, causalSnps].std(0)
    beta = rng.normal(size=nCausal)
    g = Z @ beta
    g = g / g.std() * np.sqrt(h2)
    e = rng.normal(size=n) * np.sqrt(1 - h2)
    y = g + e + popShift * pop
    return y, np.sort(causalSnps)
