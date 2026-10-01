# Computational neuroscience coursework (MATLAB)

Undergraduate coursework from a **neuroscience bachelor's at Bar-Ilan
University**, written in MATLAB.

This was originally a single `Matlab/` folder inside
[`small--projects`](https://github.com/hwilner/small--projects), where it sat
among Python machine-learning notebooks and did not belong. It is a different
subject, a different language, and a different stage of learning — so it lives
here now.

---

## What is in here

| Folder | Contents |
|---|---|
| [`src/information-theory/`](src/information-theory/) | Mutual information between spike direction and stimulus class; joint-distribution and firing-rate utilities |
| [`src/signal-processing/`](src/signal-processing/) | Fourier analysis, DFT/FFT, sampling, convolution, SNR — the computational neuroscience core sequence |
| [`src/machine-learning/`](src/machine-learning/) | Perceptron, k-nearest-neighbours with cross-validation |
| [`src/language/`](src/language/) | Language identification from raw text; letter extraction |
| [`data/`](data/) | Text corpora used by the language-identification exercise |
| [`assignments/`](assignments/) | Submitted write-ups and course PDFs |
| [`contributors/`](contributors/) | One file that is **not mine** — see below |

---

## The code

### Information theory — `src/information-theory/`

**`calculate_spike_direction_MI.m`** is the centrepiece. It computes, for each
neuron, the mutual information between the direction a spike travels and the
stimulus class that elicited it:

```
MI = Σₓ Σ_y  p(x, y) · log₂( p(x, y) / (p(x) · p(y)) )
```

It builds a joint histogram over spike-direction bins × stimulus class,
marginalises it, and accumulates the information term over bins with non-zero
probability. It then returns the index of the neuron with the highest MI — the
one whose firing most reliably distinguishes the stimulus conditions.

This is a real technique from systems neuroscience: MI quantifies how much a
neuron's response tells you about what it is responding to, and ranking neurons
by MI surfaces the most informative units in a population.

**`MINF.m`** and **`joint.m`** are the extracted utilities the above is built
from — MI from a joint distribution table, and the table itself from paired
stimulus/response vectors.

**`COMP3.m`** computes per-cell mean firing and normalised firing-rate
distributions across two recorded cells.

### Signal processing — `src/signal-processing/`

The standard computational neuroscience sequence, in order:

| File | Topic |
|---|---|
| `ex2.m` | Signal sampling and plotting |
| `ex3_Wilner_Harel_305571986.m` | Signal-to-noise ratio, z-score normalisation |
| `Assignment4.m` | Fourier representation of sinusoids, phase relationships |
| `Assignment5_harel_wilner_305571986.m` | Convolution and correlation |
| `Assignment6_harel_wilner_305571986.m`, `ass6.m` | Joint distributions over 22 recorded neurons, conditioned on stimulus class |
| `ex7.m`, `ex8_harel_wilner.m` | DFT of periodic functions, sampling rate and aliasing |
| `assign9_harel_wilner.m` | FFT, circular shift, and its effect in the frequency domain |
| `ex11_wilner_harel.m` | Tabular data and summary statistics |

`assign9` is the cleanest illustration: shifting a signal by *m* samples in time
multiplies its spectrum by a complex exponential — the same result derived two
ways.

### Machine learning — `src/machine-learning/`

- **`ex1_305571986.m`** — the perceptron, with the online update rule, a
  turn counter and an explicit `MAX_TURNS` bound so a non-separable dataset
  cannot loop forever. This is the MATLAB counterpart to
  [`ml_from_scratch.perceptron`](https://github.com/hwilner/small--projects/blob/master/src/ml_from_scratch/perceptron.py).
- **`get_kNN_classification.m`**, **`predict_kNN_multiclass_classification.m`** —
  k-NN with k-fold cross-validation, reporting held-out performance rather than
  training accuracy.
- **`ex1.m`** — a second, earlier perceptron exercise.

### Language — `src/language/`

`decide_text_language.m` and `extract_letters.m` classify text by language from
character statistics, over the corpora in [`data/`](data/).

---

## Running

Requires MATLAB (tested against the Octave-compatible subset).

```matlab
cd src/information-theory
calculate_spike_direction_MI(train_X, train_Y)
```

Several exercises load course-supplied data that is not redistributable and so
is not committed here. The ones affected, all listed in `.gitignore`:

- `exercise6_data.mat` — the 22-neuron recording for Assignment 6
- `binaryClassData1.txt` … `binaryClassData4.txt` — the perceptron datasets

Expect to adjust paths and filenames; the original file names were per-student
and some are inconsistent with each other.

### Notebooks

Some exercises are scripts rather than functions and will run top to bottom.
`ex7.m`, `ex11_wilner_harel.m` and `assign9_harel_wilner.m` follow that pattern.

---

## A note on this repository's history

The material arrived as **18 archives** — sixteen `.zip`, one `.rar`, and two
loose `.m` files. All 18 have been extracted, de-duplicated and sorted here.
Two findings worth recording:

- **`ex1_305571986.m` and `preceptron.m` were byte-identical** (same MD5). The
  duplicate is gone; one copy is kept.
- **`ex1.rar` could not be extracted** — no RAR decoder was available. Its
  contents are likely already covered by the other exercises, but if anything
  is missing, that archive is the place to look.

### Please read before publishing

**[`contributors/ex7_Efrat_Sofer_3048515.m`](contributors/) is not my work.**
It arrived inside one of the archives and is a classmate's submission
(`ex7_efrat_sofer_304855125.docx` sits beside it in `assignments/`). It is
isolated in `contributors/` rather than mixed into `src/`, but **delete both
files before making this repository public** unless you have that person's
permission.

Also worth knowing: several filenames contain what looks like a student ID
(`305571986`), and some assignment PDFs are graded submissions. Publishing a
public portfolio that links them may be more than you want. The `.gitignore`
does not cover this, because the files were already in the history of the
original repository.

---

## Related work

The Python side of the same interests — machine learning, algorithms from first
principles — lives in
[`hwilner/small--projects`](https://github.com/hwilner/small--projects), which
includes a tested NumPy perceptron and linear regression that correspond to
`ex1_305571986.m` here.

## Licence

[MIT](LICENSE). Coursework written by Harel Wilner unless otherwise noted.
