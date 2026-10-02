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

## Tests

```bash
python -m pytest python/machine_learning -q   # 7 tests
```
