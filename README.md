# Computational neuroscience coursework — MATLAB originals + Python research ports

Undergraduate coursework from a **neuroscience bachelor's at Bar-Ilan
University**, originally written in MATLAB, now reorganised into two layers:

1. **`src/`** — the original MATLAB exercises, anonymised and consolidated.
2. **`python/`** — faithful Python re-implementations of each topic, each
   extended with a **small but genuine research improvement** that is
   implemented, tested (`pytest`), and benchmarked quantitatively against the
   baseline on synthetic ground-truth data.

This was originally a single `Matlab/` folder inside
[`small--projects`](https://github.com/hwilner/small--projects), where it sat
among Python machine-learning notebooks and did not belong.

---

## Repository layout

| Folder | Contents |
|---|---|
| [`src/information-theory/`](src/information-theory/) | Mutual information between spike direction and stimulus class; joint-distribution and firing-rate utilities |
| [`src/signal-processing/`](src/signal-processing/) | Fourier analysis, DFT/FFT, sampling, convolution, SNR — the computational neuroscience core sequence |
| [`src/machine-learning/`](src/machine-learning/) | Perceptron, k-nearest-neighbours with cross-validation |
| [`src/language/`](src/language/) | Language identification from raw text; letter extraction |
| [`python/signal_processing/`](python/signal_processing/) | Python port + **SpikeSPD**: SPD-manifold (Riemannian) spike sorting vs PCA+k-means |
| [`python/machine_learning/`](python/machine_learning/) | Python port + **GeoKNN**: geodesic-kernel-weighted Riemannian kNN on covariance descriptors |
| [`python/information_theory/`](python/information_theory/) | Python port + **bias-corrected MI** (Miller–Madow, shuffle correction) for small-sample neuron ranking |
| [`python/language_id/`](python/language_id/) | Python port + **bigram SPRT** with Laplace smoothing vs unigram Wald SPRT |
| [`data/`](data/) | Text corpora used by the language-identification exercise |
| [`assignments/`](assignments/) | Submitted write-ups and course PDFs (anonymised) |
| [`contributors/`](contributors/) | One contributed file that is **not** the repo owner's work — see below |

---

## The original MATLAB code

### Information theory — `src/information-theory/`

**`calculate_spike_direction_MI.m`** is the centrepiece. It computes, for each
neuron, the mutual information between the direction a spike travels and the
stimulus class that elicited it:

```
MI = Σₓ Σ_y  p(x, y) · log₂( p(x, y) / (p(x) · p(y)) )
```

It builds a joint histogram over spike-direction bins × stimulus class,
marginalises it, and accumulates the information term over bins with non-zero
probability, then returns the index of the neuron with the highest MI.

**`MINF.m`** and **`joint.m`** are the extracted utilities — MI from a joint
distribution table, and the table itself from paired stimulus/response vectors.

**`COMP3.m`** computes per-cell mean firing, normalised firing-rate
distributions, and ROC/AUC neurometric curves across two recorded cells.

### Signal processing — `src/signal-processing/`

| File | Topic |
|---|---|
| `ex2.m` | Signal sampling and plotting |
| `ex3_snr.m` | Signal-to-noise ratio, z-score normalisation |
| `Assignment4.m` | Fourier representation of sinusoids, phase relationships, spike-train cross-correlation |
| `Assignment5_convolution.m` | Circular convolution from first principles |
| `Assignment6_convolution_matrix.m`, `ass6.m` | Convolution as a matrix operator; joint distributions over 22 recorded neurons conditioned on stimulus class |
| `ex7.m`, `ex8_harel_wilner.m` | DFT of periodic functions, sampling rate and aliasing |
| `assign9_harel_wilner.m` | FFT, circular shift, and its effect in the frequency domain |
| `ex11_wilner_harel.m` | PCA on tabular data |

### Machine learning — `src/machine-learning/`

- **`ex1_perceptron.m`** — the perceptron, with the online update rule, a
  turn counter and an explicit `MAX_TURNS` bound so a non-separable dataset
  cannot loop forever.
- **`get_kNN_classification.m`**, **`predict_kNN_multiclass_classification.m`** —
  k-NN with k-fold cross-validation, reporting held-out performance.
- **`ex1.m`** — neural-recording analysis: spike detection, waveform
  extraction, ISI histogram, raster plot and PSTH.

### Language — `src/language/`

`decide_text_language.m` and `extract_letters.m` classify text by language with
a **Wald sequential probability ratio test (SPRT)** over single-letter
statistics, using the corpora in [`data/`](data/).

---

## The Python research layer — `python/`

Each topic ships a baseline port, a novel improvement, an experiment script
with synthetic ground truth, and a `RESULTS.md` of captioned result tables.
All code uses Google-style docstrings and is covered by `pytest`.

**Headline results** (all on synthetic ground truth; full tables in each
topic's `RESULTS.md`):

| Topic | Baseline | Novel method | Headline result |
|---|---|---|---|
| Spike detection & sorting | `std/1.5` threshold + PCA+k-means | MAD robust detection + SpikeSPD (Riemannian clustering) | Detection F1 0.94 vs 0.29 (SNR 10); sorting accuracy 0.95 vs 0.74 on isolated spikes |
| kNN classification | Euclidean kNN on summed features | GeoKNN (log-Euclidean SPD kNN, kernel-weighted) | 96.9% vs 18.6% on covariance-discriminative data; Euclidean wins the mean-discriminative control (100%) |
| Mutual information | Plug-in histogram MI | Miller–Madow + shuffle-corrected MI | Small-sample bias +0.52 → −0.08 bits; ranking Spearman ρ 0.90 → 0.97 at N=10 |
| Language ID | Unigram Wald SPRT | Laplace-smoothed bigram SPRT | ~2–3× fewer letters to decision at equal-or-lower error (e.g., 10.3 vs 20.6 letters at α=0.01) |

**Honesty policy.** Every experiment reports the regimes where the novel method
loses or ties (e.g., mean-discriminative data for GeoKNN; large-sample regimes
for bias-corrected MI). Gains are claimed only where measured.

---

## Running

MATLAB sources require MATLAB/Octave. Python layer:

```bash
pip install numpy scipy scikit-learn pytest
python -m pytest python/ -q
python python/signal_processing/experiment_spike_sorting.py
```

Several MATLAB exercises load course-supplied data that is not redistributable
(listed in `.gitignore`): `exercise6_data.mat` (the 22-neuron recording) and
`binaryClassData1–4.txt` (perceptron datasets). The Python experiments are
self-contained and generate their own synthetic ground truth.

---

## Privacy and provenance

- All personal identifiers (student IDs, in filenames and file contents) have
  been removed from this repository. Four assignment write-ups whose filenames
  contained student IDs were deleted outright; code files were renamed and
  their function signatures cleaned.
- **`contributors/ex7_contributed.m` is not the repo owner's work.** It arrived
  inside one of the coursework archives (a classmate's submission). It is kept
  isolated in `contributors/`, anonymised, and documented here.

## A note on this repository's history

The material arrived as **18 archives** — sixteen `.zip`, one `.rar`, and two
loose `.m` files. All 18 were extracted, de-duplicated and sorted. One archive
(`ex1.rar`) could not be extracted at the time; its contents are likely covered
by the other exercises.

## Related work

The Python side of the same interests lives in
[`hwilner/small--projects`](https://github.com/hwilner/small--projects).

## Licence

[MIT](LICENSE). Coursework written by Harel Wilner unless otherwise noted.
