# Spike Sorting Experiment Results

Synthetic ground-truth study: three biphasic-Gaussian spike templates (differing in width, asymmetry, and amplitude) fired by Poisson trains with a 2 ms refractory period on 1/f-coloured Gaussian noise (fs = 30000 Hz, 8 s per recording, 10 seeds; runtime 78.3 s). Detection is scored by precision/recall at a +/-0.5 ms matching tolerance; clustering runs on identical (MAD) detections and is scored by Hungarian best-match accuracy and adjusted Rand index (ARI). Entries are mean +/- std over seeds.

**Table 1.** Detection performance of the robust MAD threshold (`k*median(|x|)/0.6745`, k=4) vs. the legacy coursework threshold (`std(x)/1.5`). The legacy threshold sits near 0.7x the noise standard deviation, so it fires constantly on noise: recall looks high but precision collapses to ~0.1-0.2, and its F1 is far below the MAD detector's in every regime. The MAD detector keeps precision near ceiling throughout, and its recall approaches the legacy detector's once spike peaks rise above the threshold (SNR >= 5).

| SNR | Overlap | MAD precision | MAD recall | MAD F1 | std/1.5 precision | std/1.5 recall | std/1.5 F1 |
|----:|--------:|--------------:|-----------:|-------:|------------------:|---------------:|-----------:|
| 3 | 0.00 | 0.919 +/- 0.029 | 0.222 +/- 0.019 | 0.358 +/- 0.025 | 0.125 +/- 0.005 | 0.759 +/- 0.020 | 0.215 +/- 0.008 |
| 3 | 0.15 | 0.939 +/- 0.027 | 0.282 +/- 0.018 | 0.434 +/- 0.024 | 0.129 +/- 0.006 | 0.689 +/- 0.012 | 0.217 +/- 0.009 |
| 5 | 0.00 | 0.920 +/- 0.012 | 0.806 +/- 0.014 | 0.859 +/- 0.010 | 0.146 +/- 0.007 | 0.853 +/- 0.009 | 0.249 +/- 0.010 |
| 5 | 0.15 | 0.931 +/- 0.012 | 0.726 +/- 0.016 | 0.816 +/- 0.014 | 0.143 +/- 0.005 | 0.754 +/- 0.012 | 0.240 +/- 0.008 |
| 10 | 0.00 | 0.975 +/- 0.016 | 0.916 +/- 0.022 | 0.944 +/- 0.018 | 0.172 +/- 0.007 | 0.920 +/- 0.017 | 0.290 +/- 0.010 |
| 10 | 0.15 | 0.979 +/- 0.009 | 0.801 +/- 0.020 | 0.881 +/- 0.015 | 0.188 +/- 0.010 | 0.792 +/- 0.021 | 0.304 +/- 0.013 |

**Table 2.** Clustering performance of SpikeSPD (SPD-manifold log-Euclidean k-means) vs. the classic PCA(3) + k-means baseline on identical MAD detections.

| SNR | Overlap | SpikeSPD acc | SpikeSPD ARI | PCA+kmeans acc | PCA+kmeans ARI |
|----:|--------:|-------------:|-------------:|---------------:|---------------:|
| 3 | 0.00 | 0.599 +/- 0.066 | 0.232 +/- 0.065 | 0.606 +/- 0.057 | 0.305 +/- 0.125 |
| 3 | 0.15 | 0.460 +/- 0.039 | 0.060 +/- 0.034 | 0.445 +/- 0.052 | 0.032 +/- 0.038 |
| 5 | 0.00 | 0.722 +/- 0.123 | 0.492 +/- 0.109 | 0.608 +/- 0.028 | 0.388 +/- 0.041 |
| 5 | 0.15 | 0.595 +/- 0.023 | 0.351 +/- 0.028 | 0.570 +/- 0.019 | 0.261 +/- 0.022 |
| 10 | 0.00 | 0.947 +/- 0.017 | 0.847 +/- 0.046 | 0.741 +/- 0.141 | 0.598 +/- 0.181 |
| 10 | 0.15 | 0.601 +/- 0.020 | 0.368 +/- 0.047 | 0.617 +/- 0.096 | 0.346 +/- 0.125 |

**Table 3.** Per-regime winners (detection: higher mean F1, balancing precision and recall; clustering: higher mean accuracy, with accuracy ties broken by mean ARI; ARI winner shown in parentheses). 'tie' = means within 0.01.

| SNR | Overlap | Detection winner | Clustering winner |
|----:|--------:|-----------------:|------------------:|
| 3 | 0.00 | MAD | PCA+kmeans (ARI: PCA+kmeans) |
| 3 | 0.15 | MAD | SpikeSPD (ARI: SpikeSPD) |
| 5 | 0.00 | MAD | SpikeSPD (ARI: SpikeSPD) |
| 5 | 0.15 | MAD | SpikeSPD (ARI: SpikeSPD) |
| 10 | 0.00 | MAD | SpikeSPD (ARI: SpikeSPD) |
| 10 | 0.15 | MAD | PCA+kmeans (ARI: SpikeSPD) |

## Where the baseline wins or ties

Honest accounting of regimes where the classic PCA(3) + k-means baseline matches or beats SpikeSPD in clustering (verdict = mean accuracy, ties broken by mean ARI):

- **PCA+kmeans wins:** SNR=3, overlap=0.00; SNR=10, overlap=0.15.
- **Statistical ties (means within 0.01):** none.

The pattern is consistent with theory. (i) At SNR=3 the detected waveforms are noise-dominated, so SpikeSPD's rank-1 descriptor `x x^T + eps I` is mostly a noise outer product with little stable orienting information, and PCA+kmeans matches or edges out SpikeSPD. (ii) With a 15% overlap rate at high SNR, roughly a quarter of the extracted waveforms are superpositions of two templates that neither method can attribute to a single unit; both sit near the resulting accuracy ceiling and the verdict becomes metric-dependent (here PCA+kmeans on accuracy but SpikeSPD on ARI). SpikeSPD's clearest advantage is on isolated spikes at moderate-to-high SNR (SNR 5-10, overlap 0), where the second-order structure of the covariance descriptor separates the three templates far more reliably than a fixed 3-component linear projection (e.g. SNR=10: ~0.95 vs ~0.74 mean accuracy). Finally, PCA+kmeans is much cheaper (SPD descriptors are L x L matrices), so for clean, well-separated units the baseline remains the pragmatic choice.


---

## Part 3 (2026 retrospective) -- developed SpikeSPD v2

Development over the Part-2 prototype: time-delay-embedding (TDE) covariance descriptors replacing rank-1 outer products, zero-phase low-pass denoising, template-adaptive realignment, and overlap screening. Track A stress-tests on synthetic recordings with amplitude drift, a bursting unit, and extraction jitter; Track B benchmarks on SpikeInterface ground-truth recordings against SpyKING CIRCUS 2.

**Table 4.** Track A: clustering accuracy (Hungarian best match) under amplitude drift and bursting, on identical MAD detections. `fusion` = block-diagonal TDE + outer-product descriptor; `fusion screened` excludes spikes flagged as overlaps. Mean +/- std over seeds.

| SNR | Overlap | Jitter | v1 acc | TDE acc | fusion acc | fusion screened acc | v1 ARI | TDE ARI | fusion ARI |
|----:|--------:|-------:|-------:|--------:|-----------:|--------------------:|-------:|--------:|-----------:|
| 5 | 0 | 0 | 0.688 +/- 0.033 | 0.730 +/- 0.095 | 0.697 +/- 0.038 | 0.699 +/- 0.039 | 0.473 +/- 0.033 | 0.551 +/- 0.081 | 0.499 +/- 0.037 |
| 5 | 0 | 2 | 0.713 +/- 0.044 | 0.705 +/- 0.076 | 0.707 +/- 0.031 | 0.708 +/- 0.031 | 0.506 +/- 0.035 | 0.532 +/- 0.064 | 0.518 +/- 0.045 |
| 5 | 0.15 | 0 | 0.657 +/- 0.036 | 0.706 +/- 0.014 | 0.697 +/- 0.017 | 0.698 +/- 0.017 | 0.358 +/- 0.048 | 0.464 +/- 0.019 | 0.428 +/- 0.042 |
| 5 | 0.15 | 2 | 0.678 +/- 0.041 | 0.695 +/- 0.027 | 0.702 +/- 0.030 | 0.703 +/- 0.029 | 0.373 +/- 0.062 | 0.427 +/- 0.033 | 0.431 +/- 0.026 |
| 10 | 0 | 0 | 0.792 +/- 0.124 | 0.768 +/- 0.094 | 0.859 +/- 0.118 | 0.864 +/- 0.118 | 0.594 +/- 0.193 | 0.605 +/- 0.098 | 0.734 +/- 0.178 |
| 10 | 0 | 2 | 0.816 +/- 0.147 | 0.748 +/- 0.088 | 0.867 +/- 0.128 | 0.872 +/- 0.131 | 0.642 +/- 0.209 | 0.584 +/- 0.078 | 0.754 +/- 0.192 |
| 10 | 0.15 | 0 | 0.710 +/- 0.075 | 0.734 +/- 0.030 | 0.750 +/- 0.051 | 0.756 +/- 0.057 | 0.439 +/- 0.096 | 0.519 +/- 0.029 | 0.508 +/- 0.058 |
| 10 | 0.15 | 2 | 0.686 +/- 0.019 | 0.730 +/- 0.027 | 0.732 +/- 0.037 | 0.739 +/- 0.031 | 0.397 +/- 0.038 | 0.526 +/- 0.027 | 0.488 +/- 0.048 |

**Table 5.** Track B: SpikeInterface ground-truth benchmark (single channel, 4 units, 30 s, 300-6000 Hz band-pass). Hungarian-matched per-unit agreement with ground truth (coincidence +/-0.4 ms). Caveat: SpikeSPD variants are given the true unit count; SpyKING CIRCUS 2 estimates it.

| Seed | SpikeSPD v1 | SpikeSPD v2 (TDE) | SpikeSPD v2 (fusion) | SpyKING CIRCUS 2 |
|-----:|------------:|------------------:|---------------------:|-----------------:|
| 0 | 0.874 | 0.623 | 0.867 | 0.211 |
| 1 | 0.800 | 0.887 | 0.880 | 0.472 |
| 2 | 0.862 | 0.881 | 0.864 | 0.629 |

Mean: v1 0.845 +/- 0.039, TDE 0.797 +/- 0.151, fusion 0.870 +/- 0.009, SC2 0.438 +/- 0.211. Total runtime 413 s.
