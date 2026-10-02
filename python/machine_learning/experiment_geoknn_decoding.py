"""Part 3 (2026 development): neural population decoding with Riemannian kNN.

Develops GeoKNN beyond the Part-2 toy benchmark into a neuroscience task
styled on the coursework's 22-neuron recordings (``ass6.m``): decode one of
8 stimuli from the population activity of 22 neurons.

Two coding regimes are simulated, mirroring a real dichotomy in systems
neuroscience:

- **Covariance-coded**: all stimuli share the same mean firing-rate vector;
  stimulus identity modulates the *correlation structure* via a
  stimulus-specific latent factor ``cov_s = I + g * v_s v_s^T``. Rate-based
  decoders (Euclidean kNN, LDA, SVM on mean rates) are at chance by
  construction.
- **Rate-coded** (control): stimuli shift the mean rate vector and share a
  common covariance. Rate-based decoders should win; covariance geometry is
  the wrong representation and we say so.

Methods: Euclidean kNN on rate vectors (the coursework baseline), GeoKNN
with log-Euclidean and affine-invariant (AIRM) metrics, MDM, LDA, and
RBF-SVM. AIRM is evaluated only for <= 50 trials/stimulus (pairwise
generalized eigenvalues scale poorly); the 200-trials cell is marked
deferred, per the repo's cheap-tests policy.

Appends captioned tables to ``RESULTS.md``. Deterministic seeds. Run::

    python3 experiment_geoknn_decoding.py
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from geoknn import GeoKNNClassifier, MDMClassifier

N_NEURONS = 22
N_STIM = 8
T_BINS = 64
GAIN_COV = 4.0  # latent-factor gain in the covariance-coded regime
GAIN_RATE = 1.2  # mean-shift magnitude in the rate-coded regime
TRIALS_GRID = [20, 50, 200]
N_SEEDS = 3
RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "RESULTS.md"


def make_population(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the random population parameters.

    Args:
        seed: Random seed.

    Returns:
        Tuple ``(base_rate, latent_vectors, rate_shifts)``: baseline mean
        rates ``(22,)``, per-stimulus latent factor directions
        ``(8, 22)``, and per-stimulus mean shift directions ``(8, 22)``.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(8.0, 25.0, size=N_NEURONS)
    v = rng.standard_normal((N_STIM, N_NEURONS))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.standard_normal((N_STIM, N_NEURONS))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return base, v, u


def generate_trials(
    n_trials: int, regime: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labeled population-activity trials.

    Each trial is a ``(T_BINS, N_NEURONS)`` window drawn from a Gaussian
    with stimulus-dependent covariance (covariance-coded) or mean
    (rate-coded), floored at zero to keep rates non-negative.

    Args:
        n_trials: Trials per stimulus.
        regime: ``"covariance"`` or ``"rate"``.
        seed: Random seed.

    Returns:
        Tuple ``(X, y, rates)``: windows ``(n, T_BINS, 22)``, labels
        ``(n,)``, and per-trial mean-rate vectors ``(n, 22)``.
    """
    base, v, u = make_population(seed)
    regime_code = {"covariance": 1, "rate": 2}[regime]
    rng = np.random.default_rng([seed, n_trials, regime_code])
    X, y = [], []
    for s in range(N_STIM):
        if regime == "covariance":
            cov = np.eye(N_NEURONS) + GAIN_COV * np.outer(v[s], v[s])
            mean = base.copy()
        elif regime == "rate":
            cov = np.eye(N_NEURONS) + GAIN_COV * np.outer(v[0], v[0])
            mean = base + GAIN_RATE * base * u[s]
        else:
            raise ValueError(f"unknown regime: {regime!r}")
        a = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_trials, T_BINS, N_NEURONS))
        trials = mean[None, None, :] + z @ a.T
        X.append(trials)
        y.append(np.full(n_trials, s))
    X = np.vstack(X)
    y = np.concatenate(y)
    # Shuffle trials so contiguous CV folds are class-balanced.
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    # GeoKNN works on true per-trial covariances: center each channel
    # within the trial so the SPD descriptor is not diluted by the
    # (stimulus-independent) mean-rate outer product.
    X_centered = X - X.mean(axis=1, keepdims=True)
    rates = X.mean(axis=1)
    return X_centered, y, rates


def cross_validate_geoknn(
    X: np.ndarray, y: np.ndarray, metric: str, k: int = 5, n_folds: int = 5
) -> float:
    """Cross-validated accuracy of GeoKNN/MDM on window data.

    Args:
        X: Windows ``(n, T, d)``.
        y: Labels ``(n,)``.
        metric: ``"logeuclid"``, ``"airm"``, or ``"mdm"``.
        k: Number of neighbors (ignored for MDM).
        n_folds: Number of contiguous folds.

    Returns:
        Mean accuracy across folds.
    """
    n = len(y)
    seg = n // n_folds
    scores = []
    for f in range(n_folds):
        te = np.arange(f * seg, (f + 1) * seg)
        tr = np.setdiff1d(np.arange(n), te)
        if metric == "mdm":
            clf = MDMClassifier()
        else:
            clf = GeoKNNClassifier(n_neighbors=k, metric=metric)
        clf.fit(X[tr], y[tr])
        scores.append(clf.score(X[te], y[te]))
    return float(np.mean(scores))


def run_regime(regime: str, n_trials: int, seed: int) -> dict:
    """Evaluate all decoders on one (regime, size, seed) configuration.

    Args:
        regime: ``"covariance"`` or ``"rate"``.
        n_trials: Trials per stimulus.
        seed: Random seed.

    Returns:
        Dict of mean cross-validated accuracies per method.
    """
    X, y, rates = generate_trials(n_trials, regime, seed)
    out = {"regime": regime, "n_trials": n_trials, "seed": seed}
    out["euc_knn"] = float(
        cross_val_score(KNeighborsClassifier(5), rates, y, cv=5).mean()
    )
    out["lda"] = float(cross_val_score(LinearDiscriminantAnalysis(), rates, y, cv=5).mean())
    out["svm"] = float(
        cross_val_score(SVC(kernel="rbf", gamma="scale"), rates, y, cv=5).mean()
    )
    out["geoknn_le"] = cross_validate_geoknn(X, y, "logeuclid")
    out["mdm"] = cross_validate_geoknn(X, y, "mdm")
    if n_trials <= 50:
        out["geoknn_airm"] = cross_validate_geoknn(X, y, "airm")
    else:
        out["geoknn_airm"] = float("nan")  # deferred: pairwise AIRM too slow
    return out


def append_results(rows: list[dict], runtime_s: float) -> None:
    """Append the Part-3 decoding section to RESULTS.md.

    Args:
        rows: Rows from :func:`run_regime`.
        runtime_s: Total wall-clock runtime in seconds.
    """
    def _fmt(regime: str, n: int, key: str) -> str:
        vals = [r[key] for r in rows if r["regime"] == regime and r["n_trials"] == n]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return "deferred"
        v = np.asarray(vals)
        return f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"

    methods = [
        ("Euclidean kNN (rates)", "euc_knn"),
        ("LDA (rates)", "lda"),
        ("SVM-RBF (rates)", "svm"),
        ("GeoKNN log-Euclidean", "geoknn_le"),
        ("GeoKNN AIRM", "geoknn_airm"),
        ("MDM", "mdm"),
    ]
    lines: list[str] = []
    lines.append("\n---\n\n## Part 3 (2026 retrospective) -- population decoding\n")
    lines.append(
        "GeoKNN developed into a neural-decoding task: 22 neurons, 8 stimuli, "
        "styled on the coursework's 22-neuron recording (ass6). In the "
        "covariance-coded regime, stimulus identity lives only in the "
        "correlation structure (mean rates identical); in the rate-coded "
        "control, identity lives in mean rates. AIRM cells at 200 "
        "trials/stimulus are deferred (pairwise generalized-eigenvalue cost), "
        "per the repo's cheap-tests policy.\n"
    )
    lines.append(
        "**Table 3.** Population decoding accuracy (5-fold CV, mean +/- std over "
        f"{N_SEEDS} seeds). Chance = {1.0 / N_STIM:.3f}.\n"
    )
    header = "| Method |"
    sep = "|---|"
    for n in TRIALS_GRID:
        header += f" cov-coded n={n} | rate-coded n={n} |"
        sep += "---:|---:|"
    lines.append(header)
    lines.append(sep)
    for name, key in methods:
        row = f"| {name} |"
        for n in TRIALS_GRID:
            row += f" {_fmt('covariance', n, key)} | {_fmt('rate', n, key)} |"
        lines.append(row)
    lines.append(f"\nTotal runtime {runtime_s:.0f} s.\n")
    with open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    """Run both regimes over the trials grid and append results."""
    t0 = time.time()
    rows: list[dict] = []
    for regime in ("covariance", "rate"):
        for n in TRIALS_GRID:
            for seed in range(N_SEEDS):
                rows.append(run_regime(regime, n, seed))
                print(f"done {regime} n={n} seed={seed}", flush=True)
    append_results(rows, time.time() - t0)
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
