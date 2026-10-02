# GeoKNN Experiment Results

Synthetic 7-class classification of multivariate windows (T=64, d=8). Two regimes: **(a) covariance-discriminative** (all class means zero, classes differ only by covariance rotation/anisotropy) and **(b) control** (shared covariance, classes differ only in mean location). k swept over 1..15 with 7-fold CV; best-k test accuracy reported as mean +/- std over 3 seeds (test set: 50 windows/class). Runtime: total experiment 13.2 s.


## Table: Regime (a): covariance-discriminative — test accuracy (%) at CV-best k

| Method | 100/class | 400/class |
|---|---|---|
| Euclidean kNN (summed features, original) | 17.8 +/- 3.0 | 18.6 +/- 0.6 |
| Euclidean kNN (vectorized covariance) | 95.0 +/- 3.1 | 96.5 +/- 2.0 |
| GeoKNN (Riemannian, kernel-weighted) | 96.3 +/- 2.7 | 96.9 +/- 1.6 |
| MDM (log-Euclidean class means) | 98.3 +/- 1.1 | 97.6 +/- 0.9 |

## Table: Regime (b): control (mean-discriminative) — test accuracy (%) at CV-best k

| Method | 100/class | 400/class |
|---|---|---|
| Euclidean kNN (summed features, original) | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| Euclidean kNN (vectorized covariance) | 99.0 +/- 0.4 | 98.5 +/- 1.1 |
| GeoKNN (Riemannian, kernel-weighted) | 99.3 +/- 0.5 | 98.7 +/- 1.0 |
| MDM (log-Euclidean class means) | 99.5 +/- 0.4 | 99.0 +/- 0.8 |

## Table: k-sensitivity — 7-fold CV accuracy (%) at 400 train/class

| Regime | Method | k=1 | k=3 | k=5 | k=11 | k=15 |
|---|---|---|---|---|---|---|
| covariance | Euclidean kNN (summed features, original) | 17.4 +/- 0.7 | 17.9 +/- 0.3 | 17.9 +/- 0.6 | 18.7 +/- 0.8 | 18.6 +/- 1.7 |
| covariance | Euclidean kNN (vectorized covariance) | 91.6 +/- 3.2 | 94.4 +/- 2.6 | 95.6 +/- 2.0 | 96.8 +/- 1.5 | 97.0 +/- 1.5 |
| covariance | GeoKNN (Riemannian, kernel-weighted) | 92.6 +/- 3.0 | 95.2 +/- 2.3 | 96.4 +/- 1.7 | 97.4 +/- 1.6 | 97.5 +/- 1.5 |
| covariance | MDM (log-Euclidean class means) | n/a | n/a | n/a | n/a | n/a |
| control | Euclidean kNN (summed features, original) | 99.7 +/- 0.2 | 99.8 +/- 0.1 | 99.8 +/- 0.1 | 99.8 +/- 0.2 | 99.8 +/- 0.1 |
| control | Euclidean kNN (vectorized covariance) | 96.4 +/- 2.5 | 97.4 +/- 1.8 | 98.0 +/- 1.4 | 98.3 +/- 1.3 | 98.3 +/- 1.3 |
| control | GeoKNN (Riemannian, kernel-weighted) | 96.8 +/- 2.3 | 98.0 +/- 1.4 | 98.5 +/- 1.1 | 98.8 +/- 0.8 | 98.8 +/- 0.8 |
| control | MDM (log-Euclidean class means) | n/a | n/a | n/a | n/a | n/a |

## Table: runtime per query (ms) at 400 train/class (mean over seeds)

| Method | covariance | control |
|---|---|---|
| Euclidean kNN (summed features, original) | 0.129 | 0.128 |
| Euclidean kNN (vectorized covariance) | 0.131 | 0.130 |
| GeoKNN (Riemannian, kernel-weighted) | 0.217 | 0.207 |
| MDM (log-Euclidean class means) | 0.293 | 0.272 |

## Where Euclidean kNN wins, and why

In the covariance-discriminative regime (a), GeoKNN reaches 96.9% at 400/class while the original Euclidean kNN on time-summed features is at 18.6% — far below, and only weakly above the 14.3% chance level. Every class has zero mean, so the window sums contain no first-order class signal; the residual above-chance accuracy comes from second-order leakage (the distribution of a window sum still depends on the class covariance orientation). The SPD embedding captures the class-specific covariance rotations directly. In the control regime (b), the roles reverse: Euclidean kNN on summed features achieves 100.0% vs. 98.7% for GeoKNN. The summed features are a sufficient statistic for the class mean (signal-to-noise grows linearly with window length), while the SPD embedding only sees the means through the second moment mu mu^T, which is a much weaker and partially ambiguous signal — classes whose means are negatives of each other share nearly the same second moment. Euclidean distance in the right feature space is unbeatable when that space aligns with the class signal; Riemannian geometry pays off precisely when the discriminative information lives in covariance structure rather than in means. MDM tracks GeoKNN closely in regime (a), consistent with MDM being the large-k (class-mean) limit of Riemannian kNN.



---

## Part 3 (2026 retrospective) -- population decoding

GeoKNN developed into a neural-decoding task: 22 neurons, 8 stimuli, styled on the coursework's 22-neuron recording (ass6). In the covariance-coded regime, stimulus identity lives only in the correlation structure (mean rates identical); in the rate-coded control, identity lives in mean rates. AIRM cells at 200 trials/stimulus are deferred (pairwise generalized-eigenvalue cost), per the repo's cheap-tests policy.

**Table 3.** Population decoding accuracy (5-fold CV, mean +/- std over 3 seeds). Chance = 0.125.

| Method | cov-coded n=20 | rate-coded n=20 | cov-coded n=50 | rate-coded n=50 | cov-coded n=200 | rate-coded n=200 |
|---|---:|---:|---:|---:|---:|---:|
| Euclidean kNN (rates) | 0.177 +/- 0.031 | 1.000 +/- 0.000 | 0.197 +/- 0.014 | 1.000 +/- 0.000 | 0.246 +/- 0.001 | 1.000 +/- 0.000 |
| LDA (rates) | 0.158 +/- 0.019 | 1.000 +/- 0.000 | 0.129 +/- 0.024 | 1.000 +/- 0.000 | 0.143 +/- 0.005 | 1.000 +/- 0.000 |
| SVM-RBF (rates) | 0.154 +/- 0.025 | 1.000 +/- 0.000 | 0.129 +/- 0.007 | 1.000 +/- 0.000 | 0.158 +/- 0.008 | 1.000 +/- 0.000 |
| GeoKNN log-Euclidean | 1.000 +/- 0.000 | 0.123 +/- 0.013 | 1.000 +/- 0.000 | 0.123 +/- 0.008 | 1.000 +/- 0.000 | 0.126 +/- 0.005 |
| GeoKNN AIRM | 1.000 +/- 0.000 | 0.125 +/- 0.027 | 1.000 +/- 0.000 | 0.118 +/- 0.009 | deferred | deferred |
| MDM | 1.000 +/- 0.000 | 0.117 +/- 0.010 | 1.000 +/- 0.000 | 0.133 +/- 0.009 | 1.000 +/- 0.000 | 0.116 +/- 0.009 |

Total runtime 264 s.
