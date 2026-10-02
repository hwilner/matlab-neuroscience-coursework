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

## Tests

```bash
python -m pytest python/signal_processing -q   # 20 tests
```
