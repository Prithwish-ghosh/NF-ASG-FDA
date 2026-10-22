# Normalizing-Flow Priors for Functional Principal Component Scores

Simulation code for learning the **distribution of functional principal component (FPC) scores** with a normalizing flow, and using it as a prior for score estimation in sparse functional data, with posterior computation by importance sampling or by the **flip-ASG** slice sampler.

Standard FPCA keeps only the mean and variances of the scores, which describes them completely only if they are Gaussian. For sparse curves, PACE (Yao, Müller & Wang, 2005) estimates scores under a Gaussian prior. When the scores are non-Gaussian (clusters, skewness, heavy tails, curved dependence), this prior pulls estimates toward the wrong place. This repository replaces the Gaussian with a RealNVP normalizing flow learned from the data and compares the two approaches.

---

## Model

Each curve is observed with noise:

$$
Y_i(t_{ij}) = \mu(t_{ij}) + \sum_{k=1}^{K} \xi_{ik}\,\phi_k(t_{ij}) + \varepsilon_{ij},
\qquad \varepsilon_{ij} \sim N(0, \sigma^2).
$$

For each curve, the scores are estimated from the posterior

$$
p(\boldsymbol\xi_i \mid \mathbf Y_i) \;\propto\; L_i(\boldsymbol\xi_i)\, q(\boldsymbol\xi_i),
$$

where $q$ is either a Gaussian (PACE) or a normalizing flow fitted by stochastic EM.

---

## Repository contents

| File | Purpose |
|---|---|
| `fpca_score_flow.R` | **Study 1 (dense curves).** Simulates curves with known non-Gaussian scores, runs classical FPCA, fits a Gaussian and a normalizing flow to the scores, and compares both with the true score density. |
| `compare_fpca_flow.R` | **Study 2 (sparse curves).** PACE vs. a flow prior fitted by stochastic EM, with posteriors by importance sampling; the true prior is included as a benchmark ("oracle"). |
| `compare_fpca_flow_asg.R` | **Study 3 (sparse curves).** As Study 2, but each posterior is computed both by importance sampling and by the flip-ASG sampler (inside the EM loop and for the final posterior). |
| `paired_analysis.R` | Paired comparison of Study 3 results across replications, and the three-panel summary figure. |
| `asg_flip_fast.R` | flip-ASG slice sampler (required by Study 3). |
| `effective_support_cached.R` | Effective-support bounds for the sampler (sourced by `asg_flip_fast.R`). |

---

## Requirements

- R ≥ 4.1
- [`torch`](https://torch.mlverse.org/) for the normalizing flow
- Base R packages: `splines`, `stats`

First-time setup:

```r
install.packages("torch")
torch::install_torch()      # downloads the libtorch backend once
```

Set `use_flow <- FALSE` at the top of `compare_fpca_flow.R` or `compare_fpca_flow_asg.R` to run every part except the flow, with no `torch` needed.

---

## Quick start

Run each script from the repository folder (Study 3 sources the sampler files by relative path).

```r
source("fpca_score_flow.R")        # Study 1: a few minutes
source("compare_fpca_flow.R")      # Study 2: a few min per replication
source("compare_fpca_flow_asg.R")  # Study 3: ~8-10 min per replication
source("paired_analysis.R")        # after Study 3
```

### Settings (top of each script)

| Variable | Meaning | Default |
|---|---|---|
| `scenario` | True score distribution: `"gaussian"`, `"mixture"`, `"banana"`, `"skew_heavy"` | `"mixture"` |
| `n` | Curves per data set | 400 (Study 1), 500 (Studies 2–3) |
| `m_range` | Observations per curve (sparse studies) | `3:6` |
| `sigma` | Noise standard deviation | 0.2 (Study 1), 0.5 (Studies 2–3) |
| `n_rep` | Replications | 10 (Study 2), 5 (Study 3) |
| `S` | Prior draws for importance sampling | 20000 |
| `n_draw` | Posterior draws per curve | 1000 (Study 2), 500 (Study 3) |
| `n_em`, `n_per` | Stochastic-EM iterations / draws per curve per E-step (Study 3) | 6 / 5 |
| `asg_burn` | flip-ASG burn-in per curve (Study 3) | 100 |

### Outputs

| Script | Files written |
|---|---|
| `fpca_score_flow.R` | `score_flow_<scenario>.pdf`, `eigenfunctions_check.pdf` |
| `compare_fpca_flow.R` | `compare_fpca_flow_<scenario>.pdf` |
| `compare_fpca_flow_asg.R` | `compare_fpca_flow_asg_<scenario>.csv` (per replication), `compare_fpca_flow_asg_<scenario>.pdf` |
| `paired_analysis.R` | `fig_paired.pdf`, plus paired tables printed to the console |

---

## Methods at a glance

```
simulate scores ξ ~ p  ──►  simulate curves  ──►  FPCA: μ̂, φ̂_k, λ̂_k, σ̂²
                                                        │
                         ┌──────────────────────────────┴───────────────────────────┐
                         ▼                                                          ▼
                PACE: Gaussian prior                       Flow prior, stochastic EM:
                closed-form posterior                        init on PACE posterior draws
                                                             repeat: E-step (IS or flip-ASG)
                                                                     M-step (refit flow)
                         │                                                          │
                         └─────────────► compare with the truth ◄───────────────────┘
                 score MSE · curve MSE · 95% band coverage/width · KL(true ‖ prior) · ESS · time
```

- **Sparse FPCA (shared by all methods):**
  - mean by B-spline least squares
  - covariance surface fitted to off-diagonal raw products
  - eigenfunctions from that surface
  - λ_k and σ² by Gaussian marginal likelihood given φ̂_k
- **Normalizing flow:** RealNVP with affine coupling layers (6–8 layers, 64 tanh hidden units), trained by maximum likelihood with Adam.
- **Stochastic EM:** start from draws of the PACE posteriors, then alternate:
  - E-step: draws from each curve's posterior under the current flow
  - M-step: refit the flow to those draws

  At a fixed point the prior equals the average posterior, so the flow learns the distribution of the scores, not of their noisy estimates.
- **flip-ASG in Study 3:**
  - It samples each curve's negative log kernel ‖r_i − Φ_i ξ‖²/(2σ²) − log q(ξ).
  - log q is tabulated once on a 300 × 300 grid and interpolated, so the sampler never calls `torch`.
  - Bounds use `freeze_burnin` mode.
- **Oracle:** the true score density used as the prior, with the same estimated μ̂, φ̂_k, σ̂². It is the best any prior can do in this pipeline.

---

## Results (mixture scenario)

| Study | Compared | Winner | Key result | Why |
|---|---|---|---|---|
| 1. Dense curves (1 data set, n = 400) | Gaussian vs. flow fit to FPCA scores | **Flow** | KL to truth 0.106 vs. 1.068 (−90%) | A Gaussian has one mode, at 0, the empty gap between the clusters. Even the best Gaussian has KL ≈ 1.07. |
| 2. Sparse curves (3–6 points, 10 replications) | PACE vs. flow prior | **Flow** | Score MSE 0.083 vs. 0.103 (−20%); 72% of the oracle's gain | PACE pulls curves into the gap; the flow pulls each curve toward its own cluster. |
| 3. Sparse curves (5 replications) | Importance sampling vs. flip-ASG | **Tie on accuracy; ASG more reliable** | Score MSE identical (+0.4%). Worst-curve ESS: ASG 29–97, IS as low as 1.7. Hardest data set: KL 1.37 (ASG) vs. 2.60 (IS) | In 2D, importance sampling with 20,000 draws gets the means right but breaks down on some curves. ASG explores each posterior directly. |

In Study 3, both flow versions beat PACE in all 5 replications (score MSE −21% on average, paired SE 0.002).

### Known limitations

- **Coverage is 0.86–0.89, not 0.95, for every method,** including the oracle, because μ̂, φ̂_k and σ̂² are treated as known. Sampling them as well (fully Bayesian) is the natural fix.
- **flip-ASG is 1.7–21× slower than importance sampling** in this 2D setting, and its flip move (ξ → −ξ) is almost never accepted here (≈ 0.0002). Moves between clusters come from the effective-support bounds.
- **Only the mixture scenario has been run so far,** with 5–10 replications. The Gaussian scenario is the control: there the flow should tie PACE.
- **The flow fitted by EM varies between data sets** (KL from the truth 0.20–2.60 across replications).

---

## Notes on the simulation design

- **The true scores are uncorrelated with decreasing variances.** FPC scores have that property by definition, so the mixture is rotated by the eigenvectors of its population covariance. Correlated "true" scores would not be the scores FPCA estimates.
- **Eigenfunction signs are aligned with the truth for evaluation only.** Curve estimates don't depend on the sign.
- **Methods are compared on the same data sets** (seeds `1000 + r`), so paired differences are the right basis for comparison.

---

## Citation

If you use this code, please cite:

```
[Author(s)], "Learning the Distribution of Functional Principal Component Scores
with a Normalizing Flow", [year]. Code: https://github.com/[user]/[repository]
```

## References

- Yao, F., Müller, H.-G., & Wang, J.-L. (2005). Functional data analysis for sparse longitudinal data. *JASA*, 100(470), 577–590.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using Real NVP. *ICLR*.
- Dockhorn, T., Ritchie, J. A., Yu, Y., & Murray, I. (2020). Density deconvolution with normalizing flows. arXiv:2006.09396.
- James, G. M., Hastie, T. J., & Sugar, C. A. (2000). Principal component models for sparse functional data. *Biometrika*, 87(3), 587–602.

