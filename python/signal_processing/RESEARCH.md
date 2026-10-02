# SpikeSPD v2: product-manifold covariance descriptors for single-channel spike sorting

*A research note, February 2026 retrospective. Code: [`spike_spd_v2.py`](spike_spd_v2.py) (Google-style docstrings), experiments: [`experiment_spike_sorting_v2.py`](experiment_spike_sorting_v2.py), full result tables: [`RESULTS.md`](RESULTS.md). 31 pytest tests cover every component.*

## Abstract

We introduce a spike-sorting descriptor that represents each extracellular waveform as a point on a **product of two symmetric positive-definite (SPD) manifolds**: a time-delay-embedding (Hankel) covariance capturing the waveform's temporal autocorrelation structure, and the classic rank-1 outer-product descriptor capturing its amplitude profile. Because the logarithm of a block-diagonal SPD matrix is the block-diagonal of the logarithms, log-Euclidean distances on the product manifold decompose as the sum of squared block distances, so clustering reduces to ordinary k-means on concatenated tangent features. On synthetic recordings with amplitude drift, bursting, and extraction jitter, the fusion descriptor beats the outer-product-only baseline in 7 of 8 stress regimes; on SpikeInterface ground-truth recordings it reaches 0.870 ± 0.009 mean per-unit agreement. A targeted 2026 literature search finds no prior use of SPD-manifold descriptors — with or without delay embeddings — for spike sorting; Riemannian covariance methods remain confined to EEG/BCI classification. The specific construction therefore appears to be novel, and we scope the claim accordingly.

## 1. Introduction

Spike sorting — assigning detected extracellular action potentials to their source neurons — has been dominated for three decades by the same pipeline: extract a short waveform around each detected peak, project onto a low-dimensional feature space (PCA, wavelets, hand-crafted shape metrics), and cluster (k-means, GMMs, density methods), optionally followed by template matching. A 2026 snapshot of the literature shows this is still the mainstream: PCA + GMM with iterative template updates, UMAP + k-means, waveform shape metrics + Ward linkage.

This pipeline discards something fundamental: the waveform is a *time series*, and its second-order temporal structure — how the voltage at time *t* predicts the voltage at *t + τ* — is exactly what distinguishes units whose mean templates are similar but whose dynamics differ (narrow vs. wide spikes, fast vs. slow repolarization, jittered vs. stable widths). Principal components capture directions of maximal variance of the *mean waveform cloud*; they do not encode per-spike autocorrelation.

In the EEG/BCI literature, a mature answer exists for exactly this situation: represent each trial by its covariance matrix and classify on the SPD manifold with Riemannian metrics (log-Euclidean, affine-invariant), where the geometry respects the positive-definiteness constraint (Barachant et al.; pyRiemann). A targeted search (queries: "spike sorting SPD manifold covariance descriptor Riemannian", "time-delay embedding Hankel matrix spike sorting", "log-Euclidean metric spike waveform clustering", 2024–2026 inclusive) returns **no** application of this machinery to spike waveforms. The transfer is not automatic: a spike waveform is 1-D and short (~90 samples), so the multichannel covariance of EEG must be replaced by something that manufactures matrix structure from a single channel.

## 2. Prior work and the precise novelty claim

| Component | Prior art | Status |
|---|---|---|
| SPD covariance descriptors + log-Euclidean/AIRM classification | Barachant et al. (EEG/BCI), Arsigny et al. 2006 (DT-MRI) | Established — in other domains |
| Time-delay (Hankel) embedding of a scalar time series | Broomhead & King 1986; singular-spectrum analysis | Established — not for spike sorting |
| Covariance geometry of spike waveforms | None found (2026 search) | **Unclaimed** |
| Fusion of TDE and outer-product blocks on a product SPD manifold | None found | **Unclaimed** |

**Claim (scoped).** To our knowledge, SpikeSPD v2 is the first spike-sorting method that (i) embeds each spike waveform as a Hankel trajectory matrix and sorts spikes by the Riemannian geometry of the resulting covariance descriptor, and (ii) fuses that descriptor with the amplitude-profile outer product on a product SPD manifold. We claim novelty of the *combination* for *spike sorting*, not of either mathematical ingredient.

## 3. Methods

### 3.1 TDE covariance descriptor

For a waveform x of length L, form the Hankel matrix H ∈ R^(n×m) with rows [x_i, x_{i+τ}, …, x_{i+(m−1)τ}] (embedding dimension m, lag τ), then

C_tde = H Hᵀ / n + ε·(tr/m)·I  ∈  SPD(m).

Unlike the rank-1 outer product xxᵀ, C_tde is full-rank, encodes local autocorrelation up to lag (m−1)τ, and averages noise over n delay frames — the property that matters at low SNR.

### 3.2 Product-manifold fusion

Stack the TDE block with the outer-product block C_out = xxᵀ + εI block-diagonally:

C_fuse = blkdiag(C_tde, C_out).

Since log blkdiag(A, B) = blkdiag(log A, log B), the log-Euclidean distance satisfies

d²(C_fuse, C′_fuse) = d²_LE(C_tde, C′_tde) + d²_LE(C_out, C′_out),

i.e. clustering concatenated tangent features (with √2-weighted off-diagonals) is *exactly* log-Euclidean k-means on the product manifold SPD(m) × SPD(L). Shape information and amplitude information enter the geometry additively and on equal Riemannian footing — no ad-hoc feature scaling.

### 3.3 Pipeline

Denoise (zero-phase Butterworth low-pass) → global cross-correlation alignment → descriptors → log-Euclidean k-means → template-adaptive realignment rounds (align each cluster to its running median template, re-embed, re-cluster) → overlap screening (flag spikes whose single-template residual is an in-cluster outlier *and* drops ≥2.5× under a greedy two-template fit).

## 4. Experiments and results

**Track A — stress test** (8 s synthetic recordings, 1/f noise, three units: one bursting, amplitude drift ±25%, extraction jitter up to ±2 samples, overlap fraction 0/0.15, SNR 5/10, 6 seeds; identical MAD detections for all methods). Fusion beats the v1 outer-product descriptor in **7 of 8** regimes; e.g. at SNR 10, no overlaps, jitter ±2: 0.867 vs 0.816 accuracy, ARI 0.754 vs 0.642. Overlap screening adds a small further gain (0.872, flagging only 1–6% of spikes). Full table: RESULTS.md, Table 4.

**Track B — SpikeInterface ground truth** (single channel, 4 units, 30 s, 300–6000 Hz, 3 seeds; Hungarian-matched per-unit agreement, coincidence ±0.4 ms): fusion **0.870 ± 0.009**, v1 0.845 ± 0.039, pure TDE 0.797 ± 0.151, SpyKING CIRCUS 2 0.438 ± 0.211. Full table: RESULTS.md, Table 5.

**Honest negatives.** (i) Pure TDE alone is regime-sensitive (0.62 on one clean seed — worse than v1); the fusion exists precisely because neither block suffices everywhere, and the product-manifold distance is what makes the combination principled rather than a heuristic feature concatenation. (ii) At SNR 5 all descriptors sit near the detection-limited ceiling. (iii) SpikeSPD variants are *given the true unit count*; SC2 estimates it and is designed for multichannel probes — Track B is evidence the descriptor carries real information, not a claim of superiority over modern multichannel sorters.

## 5. Limitations and the path to a publication-grade claim

1. **Single channel only.** Extension to multichannel probes (block per channel, or spatiotemporal Hankel) is the natural next step and would enable a fair fight with Kilosort/SC2-class sorters.
2. **Known unit count.** A geometric model-selection criterion (e.g. gap statistic on the manifold) is needed for a fully unsupervised claim.
3. **Three seeds on Track B.** A preprint needs tens of recordings and ideally the SpikeForest or hybrid-ground-truth benchmarks.
4. **Runtime.** Pairwise SPD operations scale worse than PCA; tangent-feature caching mitigates but does not eliminate the gap.

## 6. References

- Arsigny, V., Fillard, P., Pennec, X., Ayache, N. (2006). Log-Euclidean metrics for fast and simple calculus on diffusion tensors. *MRM* 56(2).
- Broomhead, D.S., King, G.P. (1986). Extracting qualitative dynamics from experimental data. *Physica D* 20.
- Barachant, A. et al. — Riemannian geometry for EEG-based BCI; pyRiemann.
- Quiroga, R.Q., Nadasdy, Z., Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting. *Neural Computation* 16(8).
- Buccino, A.P. et al. (2020). SpikeInterface, a unified framework for spike sorting. *eLife* 9:e61834.
- Pillow, J. et al. (2013). A model-based spike sorting algorithm for removing correlation artifacts in multi-neuron recordings. *PLoS ONE* 8(5):e62123.
