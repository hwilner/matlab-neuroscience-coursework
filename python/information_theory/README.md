# Information theory — MI ports + bias-corrected estimation for neuron ranking

## Introduction

Python port of the MATLAB information-theory exercises (`MINF.m`, `joint.m`,
`calculate_spike_direction_MI.m`, `COMP3.m`, and the `ass6.m` synergy matrix),
plus a novel research component: **finite-sample bias-corrected mutual
information** (Miller–Madow and shuffle correction) applied to the problem of
ranking neurons by the information their responses carry about stimuli.

## Extended introduction

The plug-in (histogram) MI estimator the coursework used is *upward-biased* at
finite sample sizes: empirical entropies are biased downward by roughly
`(m − 1)/2N` each, so a neuron with many response bins can appear more
informative than a genuinely informative neuron with few bins. The neuroscience
standard is the Panzeri–Treves / Miller–Madow analytical correction, often
combined with an empirical **shuffle correction** (subtract the mean MI
recomputed under label-shuffled surrogates). This package implements both,
plus the coursework's synergy/redundancy matrix
`SR(i,j) = I(Ri,Rj;S) − I(Ri;S) − I(Rj;S)`, and measures when the corrections
actually matter.

## Methods

- `mi.py` — `entropy`, `joint_histogram` (port of `joint.m`),
  `mutual_information_plugin` (port of `MINF.m`),
  `mutual_information_miller_madow`, `mutual_information_shuffle_corrected`
  (B = 200 label-permutation surrogates), `neuron_ranking` (port of
  `calculate_spike_direction_MI.m` with a `method=` switch).
- `roc_auc.py` — neurometric ROC/AUC from two rate distributions
  (port of `COMP3.m`).
- `synergy.py` — `synergy_matrix` with plugin / Miller–Madow / shuffle
  variants (port of the `ass6.m` synergy computation).
- `experiment_mi_bias.py` — 22 synthetic neurons, 8 stimulus classes,
  dependence levels ρᵢ ∈ [0, 0.9] and heterogeneous response-bin counts
  ({4, 8, 12, 16}) so that true MI and true SR are analytic; trials per
  stimulus N ∈ {10, 25, 50, 100, 400} × 200 repetitions.

## Results

Full captioned tables in [`RESULTS.md`](RESULTS.md). Headline findings:

- **Bias:** at N = 10 trials/stimulus the plugin estimator overshoots true MI
  by +0.52 bits on average; Miller–Madow cuts that to +0.36 and the shuffle
  correction to −0.08. All estimators converge at N = 400, as theory requires
  — corrections are a *small-sample* fix, not an asymptotic improvement.
- **Ranking:** Spearman ρ between estimated and true neuron rankings at
  N = 10 rises from 0.903 (plugin) to 0.967 (shuffle-corrected).
- **Honest negative result:** for the *sign* of synergy at small N,
  Miller–Madow is **worse than plugin** (its nonzero-cell correction
  under-estimates the joint term); only the empirical shuffle correction keeps
  SR signs reliable (1–5% sign-flips vs 45–68%).

## Tests

```bash
python -m pytest python/information_theory -q   # 10 tests
```

## Part 3 status — complete at Part 2

Retrospective assessment (2026): the bias corrections implemented here are
the *established* standard (Miller-Madow 1955; Panzeri-Treves 1996; shuffle
correction as recommended by Panzeri et al. 2007). The package's
contribution is the quantified benchmark itself -- including the honest
negative result that Miller-Madow is worse than plug-in for synergy signs
at tiny N. There is no defensible path to methodological novelty, so this
topic is intentionally not developed further.
