"""Synthetic ground-truth experiment on small-sample MI bias.

Generates a population of 22 synthetic "neurons" driven by 8 stimulus
classes. Neuron ``i`` has reliability ``rho_i`` (spread over
``linspace(0, 0.9, 22)``) and its own number of response bins
(``N_BINS_PER_NEURON``, cycling over {4, 8, 12, 16}): with probability
``rho_i`` its response is a deterministic function of the stimulus class,
otherwise it is uniform noise over its response bins. Because the
generative model is known, the true MI of every neuron — and the true
synergy-redundancy index of every pair — are available analytically.

The heterogeneous bin counts are what makes small-sample bias dangerous
for neuron ranking: the plugin bias (~(m-1)/2N per entropy term) scales
with the number of response bins, so at small N a noisy many-binned
neuron can outrank an informative few-binned one.

The experiment sweeps trials-per-stimulus ``N`` over
``{10, 25, 50, 100, 400}`` with 200 repetitions and compares three
estimators (plugin, Miller-Madow, shuffle-corrected) on:

* bias and standard deviation of the estimated MI against the true MI;
* Spearman rank correlation between estimated and true neuron rankings;
* sign-flip rate of the synergy-redundancy index against the analytic
  reference.

Running this module as a script rewrites ``RESULTS.md`` next to it with
captioned markdown tables. All randomness is seeded, so results are
deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr

from .mi import MI_METHODS, mutual_information_plugin, neuron_ranking
from .synergy import synergy_matrix

N_NEURONS = 22
N_CLASSES = 8
RHOS = np.linspace(0.0, 0.9, N_NEURONS)
N_BINS_PER_NEURON = np.tile(np.array([4, 8, 12, 16]), N_NEURONS)[:N_NEURONS]
N_VALUES = [10, 25, 50, 100, 400]
REPS = 200
B_RANKING = 32  # surrogates for shuffle-corrected neuron ranking
B_SYNERGY = 10  # surrogates for shuffle-corrected synergy (cheaper; the
# surrogate-mean noise averages out over the 200 repetitions)
SEED = 20240517

RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


def analytic_joint(
    rho: float, n_bins: int, n_classes: int = N_CLASSES
) -> NDArray[np.float64]:
    """Exact joint distribution ``P(r, s)`` of one synthetic neuron.

    The response equals ``s mod n_bins`` with probability ``rho`` and is
    uniform over ``n_bins`` otherwise; stimulus classes are uniform.

    Args:
        rho: Reliability, between 0 (pure noise) and 1 (deterministic).
        n_bins: Number of response bins of the neuron.
        n_classes: Number of stimulus classes.

    Returns:
        Joint table of shape ``(n_bins, n_classes)`` summing to one.
    """
    joint = np.full((n_bins, n_classes), (1.0 - rho) / n_bins / n_classes)
    for s in range(n_classes):
        joint[s % n_bins, s] += rho / n_classes
    return joint


def true_mutual_information(
    rho: float, n_bins: int, n_classes: int = N_CLASSES
) -> float:
    """Analytic true MI of one synthetic neuron, in bits.

    Args:
        rho: Reliability of the neuron.
        n_bins: Number of response bins of the neuron.
        n_classes: Number of stimulus classes.

    Returns:
        The exact mutual information in bits.
    """
    return mutual_information_plugin(analytic_joint(rho, n_bins, n_classes))


def true_synergy_matrix(
    rhos: ArrayLike = RHOS, n_bins: ArrayLike = N_BINS_PER_NEURON
) -> NDArray[np.float64]:
    """Analytic true synergy-redundancy index for all neuron pairs.

    The synthetic neurons are conditionally independent given the
    stimulus, so ``I(R_i; R_j | S) = 0`` and the interaction information
    reduces to ``SR_true(i, j) = -I(R_i; R_j)``: every informative pair is
    truly redundant (negative SR), pairs involving a pure-noise neuron
    have ``SR_true = 0``.

    Args:
        rhos: Reliability of each neuron, shape ``(n_neurons,)``.
        n_bins: Response-bin count of each neuron, shape ``(n_neurons,)``.

    Returns:
        Symmetric ``(n_neurons, n_neurons)`` matrix of exact SR values in
        bits, with zeros on the diagonal.
    """
    rhos = np.asarray(rhos, dtype=float)
    n_bins = np.asarray(n_bins, dtype=int)
    # P_i(r | s) columns of the analytic joints.
    conditionals = [
        analytic_joint(rho, nb) * N_CLASSES for rho, nb in zip(rhos, n_bins)
    ]
    n = len(rhos)
    sr = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # P(r_i, r_j) = mean_s P_i(r_i | s) P_j(r_j | s).
            joint = (conditionals[i] @ conditionals[j].T) / N_CLASSES
            sr[i, j] = sr[j, i] = -mutual_information_plugin(joint)
    return sr


def simulate_population(
    rhos: NDArray[np.float64],
    trials_per_class: int,
    rng: np.random.Generator,
    n_bins: NDArray[np.int64] = N_BINS_PER_NEURON,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Draw one synthetic experiment (responses of all neurons).

    Args:
        rhos: Reliability of each neuron, shape ``(n_neurons,)``.
        trials_per_class: Number of trials ``N`` per stimulus class.
        rng: Random number generator.
        n_bins: Response-bin count of each neuron, shape ``(n_neurons,)``.

    Returns:
        A tuple ``(responses, stimuli)`` with ``responses`` of shape
        ``(n_neurons, n_classes * trials_per_class)`` and ``stimuli`` the
        matching class labels.
    """
    stimuli = np.tile(np.arange(N_CLASSES), trials_per_class)
    n_trials = stimuli.size
    responses = np.empty((len(rhos), n_trials), dtype=np.int64)
    for i, (rho, nb) in enumerate(zip(rhos, n_bins)):
        informative = rng.random(n_trials) < rho
        noise = rng.integers(0, nb, size=n_trials)
        responses[i] = np.where(informative, stimuli % nb, noise)
    return responses, stimuli


def run_experiment(
    seed: int = SEED, reps: int = REPS, verbose: bool = True
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Run the full sweep over trials-per-stimulus values.

    Args:
        seed: Master seed (all randomness derives from it).
        reps: Monte-Carlo repetitions per sample size.
        verbose: Print progress lines.

    Returns:
        Nested dict ``results[metric][N][method] -> value`` with the four
        metrics ``"bias"`` (mean estimate minus true MI, bits), ``"std"``
        (standard deviation of the estimate, bits), ``"spearman"`` (mean
        Spearman rank correlation of estimated vs true neuron rankings)
        and ``"sr_flip"`` (fraction of truly redundant pairs whose
        estimated SR sign flips to synergy).
    """
    rng = np.random.default_rng(seed)
    true_mi = np.array(
        [true_mutual_information(rho, nb) for rho, nb in zip(RHOS, N_BINS_PER_NEURON)]
    )
    sr_true = true_synergy_matrix()
    iu = np.triu_indices(N_NEURONS, k=1)
    redundant = sr_true[iu] < 0.0  # pairs whose true SR is strictly negative

    metrics = ("bias", "std", "spearman", "sr_flip")
    results: Dict[str, Dict[int, Dict[str, float]]] = {
        m: {n: {} for n in N_VALUES} for m in metrics
    }
    for n_per_class in N_VALUES:
        errors = {m: np.zeros((reps, N_NEURONS)) for m in MI_METHODS}
        estimates = {m: np.zeros((reps, N_NEURONS)) for m in MI_METHODS}
        spearmans = {m: np.zeros(reps) for m in MI_METHODS}
        flips = {m: np.zeros(reps) for m in MI_METHODS}
        for rep in range(reps):
            responses, stimuli = simulate_population(RHOS, n_per_class, rng)
            rep_seed = seed + 1000 * rep + n_per_class
            for method in MI_METHODS:
                _, mi_est = neuron_ranking(
                    responses,
                    stimuli,
                    method=method,
                    B=B_RANKING,
                    seed=rep_seed,
                )
                estimates[method][rep] = mi_est
                errors[method][rep] = mi_est - true_mi
                spearmans[method][rep] = spearmanr(mi_est, true_mi).statistic
                sr = synergy_matrix(
                    responses,
                    stimuli,
                    method=method,
                    B=B_SYNERGY,
                    seed=rep_seed,
                )
                flips[method][rep] = np.mean(sr[iu][redundant] > 0.0)
        for method in MI_METHODS:
            results["bias"][n_per_class][method] = float(errors[method].mean())
            results["std"][n_per_class][method] = float(estimates[method].std())
            results["spearman"][n_per_class][method] = float(
                spearmans[method].mean()
            )
            results["sr_flip"][n_per_class][method] = float(flips[method].mean())
        if verbose:
            print(f"N={n_per_class:4d} trials/stimulus done "
                  f"(bias plugin={results['bias'][n_per_class]['plugin']:+.3f} bits)")
    return results


def _table(
    title: str,
    caption: str,
    results: Dict[int, Dict[str, float]],
    scale: float = 1.0,
    fmt: str = "{:+.4f}",
) -> str:
    """Render one metric as a captioned markdown table."""
    lines = [f"### {title}", "", caption, ""]
    header = "| Estimator | " + " | ".join(f"N={n}" for n in N_VALUES) + " |"
    sep = "|" + "---|" * (len(N_VALUES) + 1)
    lines += [header, sep]
    for method in MI_METHODS:
        cells = [fmt.format(results[n][method] * scale) for n in N_VALUES]
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def format_results(results: Dict[str, Dict[int, Dict[str, float]]]) -> str:
    """Format the experiment results as a captioned markdown document."""
    true_mi = np.array(
        [true_mutual_information(rho, nb) for rho, nb in zip(RHOS, N_BINS_PER_NEURON)]
    )
    doc = [
        "# Small-sample bias of mutual-information estimators",
        "",
        "Synthetic ground truth: 22 neurons, 8 stimulus classes. Neuron *i* "
        "has reliability ρᵢ ∈ linspace(0, 0.9, 22) and its own number of "
        "response bins (cycling over {4, 8, 12, 16}): with probability ρᵢ "
        "its response is a deterministic function of the stimulus class, "
        "otherwise it is uniform noise over its bins. True MI per neuron "
        f"ranges from {true_mi.min():.3f} to {true_mi.max():.3f} bits (max "
        f"possible {np.log2(N_CLASSES):.1f} bits). Neurons are conditionally "
        "independent given the stimulus, so every informative pair is truly "
        "*redundant* (true SR < 0).",
        "",
        f"Sweep over trials/stimulus N ∈ {N_VALUES}, {REPS} repetitions per "
        f"N (shuffle corrections use B={B_RANKING} surrogates for ranking and "
        f"B={B_SYNERGY} for synergy). Everything is seeded and deterministic.",
        "",
        _table(
            "Table 1 — Mean bias of estimated MI (estimate − true MI, bits)",
            "The plugin estimator is strongly upward biased at small N "
            "(≈ (m−1)/2N per entropy term, growing with the neuron's number "
            "of response bins). The shuffle correction removes almost all of "
            "it; Miller–Madow removes most of it but under-corrects at tiny "
            "N because the observed nonzero-cell counts undershoot the true "
            "support.",
            results["bias"],
        ),
        _table(
            "Table 2 — Standard deviation of the MI estimate (bits)",
            "Variability across neurons and repetitions. The shuffle "
            "correction subtracts an empirical mean, slightly increasing "
            "variance at large N where the bias it removes is negligible.",
            results["std"],
            fmt="{:.4f}",
        ),
        _table(
            "Table 3 — Spearman ρ between estimated and true neuron ranking",
            "Rank correlation of the 22 estimated MIs against the analytic "
            "true MIs, averaged over repetitions. Because plugin bias grows "
            "with a neuron's bin count, the plugin estimator misranks "
            "many-binned noise neurons above few-binned informative ones at "
            "small N; the bias-corrected estimators fix this.",
            results["spearman"],
            fmt="{:.4f}",
        ),
        _table(
            "Table 4 — SR sign-flip rate vs the analytic reference (%)",
            "Fraction of truly redundant neuron pairs (true SR < 0) whose "
            "estimated SR is positive (apparent synergy). Plugin MI of the "
            "joint variable (Rᵢ, Rⱼ) has many bins, so its bias is large "
            "enough at small N to manufacture spurious synergy. "
            "Miller–Madow is unreliable here at small N: its correction of "
            "the joint term uses observed nonzero-cell counts, which "
            "undershoot the large joint support, so it under-corrects and "
            "flips *more* signs than plugin until N grows. The empirical "
            "shuffle correction is the trustworthy small-sample fix for SR.",
            results["sr_flip"],
            scale=100.0,
            fmt="{:.1f}",
        ),
        "## Notes",
        "",
        "* Miller–Madow and shuffle corrections are **small-sample fixes**: "
        "their correction terms shrink like 1/N (Miller–Madow analytically, "
        "shuffle empirically), and every table above shows the corrected "
        "estimators converging to the plugin estimator as N grows — at "
        "N=400 trials/stimulus all three are practically identical.",
        "* The plugin estimator is fine for comparing neurons at large N; "
        "at small N its bias depends on each neuron's number of effective "
        "response bins, which corrupts the ranking and inflates apparent "
        "synergy. The corrections fix exactly this.",
        "* Shuffle correction costs B extra histograms per estimate and adds "
        "a little variance; Miller–Madow is essentially free but assumes the "
        "first-order 1/N expansion is adequate and that the observed "
        "nonzero-cell counts approximate the true support — both fail for "
        "many-celled joint tables at tiny N (see Table 4), where only the "
        "empirical shuffle correction keeps SR signs reliable.",
        "",
    ]
    return "\n".join(doc)


def main() -> None:
    """Run the experiment and rewrite RESULTS.md next to this file."""
    results = run_experiment()
    doc = format_results(results)
    RESULTS_PATH.write_text(doc)
    print(f"\nWrote {RESULTS_PATH}\n")
    print(doc)


if __name__ == "__main__":
    main()
