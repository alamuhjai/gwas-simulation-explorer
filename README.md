# GWAS Simulation and Power Explorer

**Live app:** [alamuhjai.github.io/gwas-simulation-explorer](https://alamuhjai.github.io/gwas-simulation-explorer)

An interactive genome-wide association study (GWAS) simulator in Python, compiled to WebAssembly with Shinylive so it runs in the browser without a server. All data are simulated.

The **Power explorer** view shows how statistical power (non-central $\chi^2$) and a simulated Manhattan plot change with sample size, minor allele frequency, effect size, and significance threshold. The **Population structure** view simulates 1,000 individuals across two subpopulations (Balding–Nichols model, $F_{\text{ST}} = 0.05$), 5,000 SNPs, and a quantitative trait with 20 causal SNPs ($h^2 = 0.3$) plus a non-genetic shift between the subpopulations:

| Model (shift = 0.6 SD) | $\lambda_{\text{GC}}$ | Genome-wide hits ($p < 5\times 10^{-8}$) | Causal SNPs found |
|---|---|---|---|
| Naive (no covariates) | 6.38 | 79 | 5 / 20 |
| Adjusted for 2 PCs | 0.99 | 4 | 4 / 20 |

Without covariates, any SNP whose allele frequency differs between the two subpopulations correlates with the trait, producing 74 false positives ($\lambda_{\text{GC}} = 6.38$). Including the top 2 principal components of the genotype matrix as covariates removes the false positives and restores calibration ($\lambda_{\text{GC}} = 0.99$).

## Running locally

Core simulation, vectorised OLS (Frisch–Waugh–Lovell QR residualisation), and PCA live in `gwas/`, the Shiny UI in `explorer/`, and tests against `scipy.stats.linregress` in `test/testGWAS.py`.

```bash
pip install -r explorer/requirements.txt pytest
pytest test/testGWAS.py
shiny run explorer/app.py
```
