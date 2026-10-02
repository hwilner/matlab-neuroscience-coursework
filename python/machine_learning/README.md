# Machine learning — perceptron & kNN ports + GeoKNN (Riemannian kNN)

## Introduction

Python port of the MATLAB machine-learning exercises (perceptron with the
online update rule; multiclass kNN with 7-fold cross-validation) plus a novel
research component, **GeoKNN**: a geodesic-kernel-weighted kNN that classifies
samples as SPD covariance descriptors using Riemannian geometry.

## Extended introduction

The coursework kNN compares samples by Euclidean distance between summed
features. That collapses each multivariate window to a mean-like statistic,
so any class structure carried by *covariance* — correlations between
channels, anisotropy, orientation — is invisible to it. The BCI/EEG
literature (Barachant et al.; pyRiemann) shows that covariance matrices
classified on the SPD manifold (log-Euclidean or affine-invariant metrics,
minimum-distance-to-mean classifiers) capture exactly that structure.

GeoKNN goes beyond the textbook minimum-distance-to-mean (MDM) classifier in
two ways: it is **instance-based** (kNN over individual SPD descriptors rather
than class means), and it votes with a **Riemannian Gaussian kernel**
`w = exp(−d²/2σ²)` whose bandwidth `σ` is self-tuned per query as the median
neighbour distance — a scale-free voting rule on the manifold.

## Methods

- `perceptron.py` — perceptron with online update rule and `MAX_TURNS` bound.
- `knn.py` — faithful port of the coursework multiclass kNN + 7-fold CV.
- `geoknn.py` — SPD embedding `C = XXᵀ/T + εI`, log-Euclidean distances via
  `eigh`-based matrix logarithms, Karcher mean (fixed-point), tangent-space
  mapping, `GeoKNNClassifier`, and `MDMClassifier` as a literature bridge.
- `experiment_geoknn.py` — two synthetic regimes, 7 classes, 8 channels:
  (a) *covariance-discriminative* — identical class means, classes differ by
  covariance rotation/anisotropy; (b) *control* — shared covariance, classes
  differ in mean location. 7-fold CV, k swept 1–15, train sizes {100, 400},
  3 seeds.

## Results

Full captioned tables in [`RESULTS.md`](RESULTS.md). Headline findings:

- **Covariance-discriminative regime:** GeoKNN **96.9 ± 1.6%** accuracy where
  the original Euclidean kNN achieves **18.6%** (chance ≈ 14.3%) — covariance
  geometry recovers structure that summed features erase.
- **Control regime:** Euclidean kNN wins (100.0% vs GeoKNN 98.7%) — when class
  information lives in the means, the covariance descriptor is the wrong
  representation, and we say so.
- Kernel voting gives a flat k-sensitivity curve (92.6→97.5% for k=1→15),
  i.e. the self-tuned bandwidth makes the classifier robust to the choice of k.
- Runtime cost is modest: ~0.21 ms/query vs ~0.13 ms for Euclidean kNN.

## Part 3 (2026 retrospective) — population decoding

GeoKNN was developed into a neural-decoding task styled on the coursework's
22-neuron recording (`ass6.m`): decode one of 8 stimuli from the activity of
22 neurons. Two regimes are simulated — *covariance-coded* (identity only in
correlation structure; mean rates identical) and *rate-coded* (identity only
in mean rates). GeoKNN gained a second metric: the **affine-invariant
Riemannian metric (AIRM)** via generalized eigenvalues, congruence-invariant
and complementary to log-Euclidean.

Headline results (full table in [`RESULTS.md`](RESULTS.md), 5-fold CV):

- **Covariance-coded:** GeoKNN (log-Euclidean and AIRM) and MDM decode
  **perfectly (100%)** at every sample size, while Euclidean kNN / LDA /
  SVM on rate vectors stay near chance (0.13–0.25; Euclidean kNN leaks a
  little covariance structure into its rate features — reported, not hidden).
- **Rate-coded control:** rate-based methods are perfect (100%) and the
  geometry methods drop to chance (~0.12) — the covariance descriptor is
  deliberately the wrong representation there.
- AIRM ≈ log-Euclidean here (no congruence distortion in the simulation);
  AIRM at 200 trials/stimulus is **deferred** (pairwise generalized-
  eigenvalue cost), per the repo's cheap-tests policy.

## Tests

```bash
python -m pytest python/machine_learning -q   # 10 tests
```
