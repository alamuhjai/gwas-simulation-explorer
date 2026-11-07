import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
from gwas.simulate import simGenotypes, simPhenotype
from gwas.association import runGWAS, compPCs, genoInflation


def testLinregressMatch():
    G, pop = simGenotypes(n=300, m=50, seed=3)
    y, _ = simPhenotype(G, pop, nCausal=5, seed=4)
    res = runGWAS(G, y)
    ref = stats.linregress(G[:, 0], y)
    assert np.isclose(res.loc[0, "beta"], ref.slope)
    assert np.isclose(res.loc[0, "p"], ref.pvalue)


def testPcCorrection():
    G, pop = simGenotypes(n=1000, m=3000, fst=0.05)
    y, _ = simPhenotype(G, pop, popShift=0.6)
    lamNaive = genoInflation(runGWAS(G, y)["p"])
    lamAdj = genoInflation(runGWAS(G, y, compPCs(G, 2))["p"])
    assert lamNaive > 1.3
    assert lamAdj < lamNaive
    assert lamAdj < 1.2


def testNullCalibration():
    G, pop = simGenotypes(n=800, m=3000, seed=7)
    y = np.random.default_rng(0).normal(size=800)
    assert abs(genoInflation(runGWAS(G, y)["p"]) - 1) < 0.15