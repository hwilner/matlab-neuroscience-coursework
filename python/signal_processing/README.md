# Signal processing — baseline ports + SpikeSPD (SPD-manifold spike sorting)

## Introduction

Python port of the MATLAB signal-processing sequence (SNR and z-score
normalisation, circular convolution, spike detection, waveform extraction,
ISI / refractory analysis, raster + PSTH) together with a novel spike-sorting
component, **SpikeSPD**, which clusters spike waveforms on the manifold of
symmetric positive-definite (SPD) matrices instead of in PCA space.

## Extended introduction

The coursework detector used a threshold of `std(signal)/1.5`. That statistic
is contaminated by the spikes themselves: in dense recordings the standard
deviation grows with spike rate, and the threshold drifts upward precisely
when most spikes are present. Modern pipelines (Quiroga et al., 2004;
SpikeInterface) estimate noise robustly with the median absolute deviation,
`σ̂ = median(|x|) / 0.6745`, which is insensitive to sparse large events.

For sorting, the classic pipeline is PCA on aligned waveforms followed by
k-means. PCA captures directions of maximal *variance of the mean waveform
cloud*, but two units whose mean templates are similar can still differ in
their *second-order* waveform structure (width jitter, asymmetry spread).
SpikeSPD represents each spike by its regularised covariance descriptor
`C = x xᵀ + εI` — a point on the SPD manifold — and clusters these descriptors
with log-Euclidean k-means (`log C` vectorised with √2-weighted off-diagonals,
so Euclidean distance in feature space equals the log-Euclidean Riemannian
distance between descriptors). Riemannian geometry on SPD matrices is mature
in EEG/BCI classification but largely unexplored for spike waveforms.

## Methods

- `baseline.py` — ports: `zscore`, `snr_power`, `snr_db`,
  `circular_convolution` (direct definition) and `circular_convolution_fft`,
  `detect_spikes` (MAD, refractory-enforced), `detect_spikes_std` (legacy
  coursework detector), `extract_waveforms` (peak-aligned), `isi_histogram`,
  `refractory_violations`, `raster_psth`, `sort_spikes_pca_kmeans`.
- `spike_spd.py` — SpikeSPD: SPD descriptors, exact log-Euclidean Fréchet-mean
  k-means updates with empty-cluster reseeding.
- `experiment_spike_sorting.py` — synthetic ground truth: three biphasic
  Gaussian templates differing in width/asymmetry/amplitude, Poisson spike
  trains with 2 ms refractory, 1/f-coloured noise, controlled overlaps;
  SNR ∈ {3, 5, 10} × overlap ∈ {0, 0.15} × 10 seeds. Detection scored by
  precision/recall at ±0.5 ms; clustering by accuracy and adjusted Rand index
  after Hungarian best-match alignment.

## Results

Full captioned tables in [`RESULTS.md`](RESULTS.md). Headline findings:

- **Detection:** the MAD detector dominates the legacy `std/1.5` detector in
  F1 at every SNR and overlap rate (e.g., F1 0.94 vs 0.29 at SNR 10, no
  overlaps) — the legacy detector's precision collapses (~0.13–0.19) because
  spike contamination inflates `std` unevenly.
- **Clustering:** SpikeSPD beats PCA+k-means most clearly on isolated spikes
  at SNR 5–10 (accuracy 0.95 vs 0.74 at SNR 10) and wins ARI in 4 of 6
  regimes. PCA+kmeans wins in the noise-dominated SNR 3 / no-overlap regime,
  where rank-1 covariance descriptors are mostly noise — reported, not hidden.

## Part 3 (2026 retrospective) — developed SpikeSPD v2

Development over the Part-2 prototype, targeting the failure modes it
exposed (`spike_spd_v2.py`):

1. **Time-delay-embedding (TDE) descriptors** — Hankel-matrix covariance
   `C = H Hᵀ/n + εI` replaces the rank-1 outer product: full-rank, encodes
   temporal autocorrelation, averages noise over delay frames.
2. **Fusion descriptor** — block-diagonal `blkdiag(C_tde, C_outer)`;
   log-Euclidean distance on the product manifold adds the blocks'
   distances in quadrature, uniting shape (TDE) and amplitude (outer
   product) information. This is the default.
3. **Zero-phase low-pass denoising** and **template-adaptive
   realignment** (cross-correlation to running cluster templates).
4. **Overlap screening** — spikes whose single-template residual is an
   in-cluster outlier *and* whose residual is cut ≥2.5× by a greedy
   two-template fit are flagged as overlaps.

Headline results (full tables in [`RESULTS.md`](RESULTS.md)):

- **Stress test** (drift + bursting + jitter, Track A): fusion beats the
  v1 descriptor in 7 of 8 SNR×overlap×jitter regimes (e.g., 0.87 vs 0.79
  accuracy at SNR 10, no overlaps; 0.74 vs 0.69 with overlaps + jitter),
  with overlap screening adding a further small gain.
- **SpikeInterface ground-truth benchmark** (Track B, 3 seeds): fusion
  **0.870 ± 0.009** mean per-unit agreement vs v1 0.845 ± 0.039 — and
  both far above SpyKING CIRCUS 2 (0.438), which struggles on
  single-channel recordings (caveat: SpikeSPD is given the true unit
  count, SC2 estimates it; the comparison is informative, not a claim of
  superiority over modern multichannel sorters).
- **Honesty:** pure TDE alone is regime-sensitive (0.62 on clean SI data);
  fusion exists precisely because neither descriptor suffices everywhere.

## Tests

```bash
python -m pytest python/signal_processing -q   # 31 tests
```
