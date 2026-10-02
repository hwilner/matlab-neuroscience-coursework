"""Tests for the information_theory package (MI estimators, ROC/AUC)."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from ..mi import (
    entropy,
    joint_histogram,
    mutual_information_miller_madow,
    mutual_information_plugin,
    mutual_information_shuffle_corrected,
    neuron_ranking,
)
from ..roc_auc import auc, roc_auc_score, roc_curve
from ..synergy import synergy_matrix
from ..experiment_mi_bias import (
    N_BINS_PER_NEURON,
    RHOS,
    simulate_population,
    true_mutual_information,
)

SEED = 12345

TRUE_MI = np.array(
    [true_mutual_information(rho, nb) for rho, nb in zip(RHOS, N_BINS_PER_NEURON)]
)


# --------------------------------------------------------------------------
# Plugin MI and entropy
# --------------------------------------------------------------------------

def test_plugin_mi_matches_analytic_value_on_constructed_table():
    """Perfectly correlated binary variables must give exactly 1 bit."""
    R = np.array([[0.25, 0.0], [0.0, 0.25]])
    assert mutual_information_plugin(R) == pytest.approx(1.0, abs=1e-12)
    # A uniform 4x4 independent table must give exactly 0 bits.
    R_ind = np.full((4, 4), 1.0 / 16.0)
    assert mutual_information_plugin(R_ind) == pytest.approx(0.0, abs=1e-12)
    # Counts and normalized tables must agree.
    counts = np.array([[10, 30], [5, 5]], dtype=float)
    assert mutual_information_plugin(counts) == pytest.approx(
        mutual_information_plugin(counts / counts.sum()), abs=1e-12
    )


def test_mi_of_variable_with_itself_equals_its_entropy():
    """MI(X; X) = H(X): the joint table collapses onto the diagonal."""
    rng = np.random.default_rng(SEED)
    x = rng.integers(0, 6, size=5000)
    R = joint_histogram(x, x)
    _, p_x = np.unique(x, return_counts=True)
    assert mutual_information_plugin(R) == pytest.approx(entropy(p_x / p_x.sum()), abs=1e-10)


def test_plugin_mi_near_zero_for_independent_variables_at_large_n():
    """At large N the plugin bias ~ (m-1)/(2N) is negligible."""
    rng = np.random.default_rng(SEED)
    n = 200_000
    x = rng.integers(0, 4, size=n)
    y = rng.integers(0, 4, size=n)
    mi = mutual_information_plugin(joint_histogram(x, y))
    assert 0.0 <= mi < 0.001


# --------------------------------------------------------------------------
# Bias corrections on the synthetic generator
# --------------------------------------------------------------------------

def test_miller_madow_reduces_bias_vs_plugin_at_small_n():
    """On the synthetic population, |mean bias| of MM < |mean bias| of plugin."""
    rng = np.random.default_rng(SEED)
    reps, n_per_class = 200, 10
    true_mi = TRUE_MI
    bias = {"plugin": np.zeros((reps, len(RHOS))),
            "miller_madow": np.zeros((reps, len(RHOS)))}
    for rep in range(reps):
        responses, stimuli = simulate_population(RHOS, n_per_class, rng)
        for method in bias:
            _, mi_est = neuron_ranking(responses, stimuli, method=method)
            bias[method][rep] = mi_est - true_mi
    bias_plugin = bias["plugin"].mean()
    bias_mm = bias["miller_madow"].mean()
    assert bias_plugin > 0.1  # sanity: the bias really exists at N=10
    assert abs(bias_mm) < abs(bias_plugin)


def test_shuffle_corrected_mi_near_zero_under_independence():
    """Under independence the shuffle-corrected MI averages to ~0 bits."""
    rng = np.random.default_rng(SEED)
    reps, n = 20, 640
    estimates = np.zeros(reps)
    for rep in range(reps):
        x = rng.integers(0, 8, size=n)
        y = rng.integers(0, 8, size=n)
        estimates[rep] = mutual_information_shuffle_corrected(x, y, B=100, seed=SEED + rep)
    assert abs(estimates.mean()) < 0.01
    assert np.abs(estimates).max() < 0.05


def test_bias_corrected_ranking_at_least_as_good_as_plugin_at_small_n():
    """Mean Spearman rho vs true ranking: corrected estimators >= plugin."""
    rng = np.random.default_rng(SEED)
    reps, n_per_class = 100, 10
    correlations = {m: np.zeros(reps)
                    for m in ("plugin", "miller_madow", "shuffle")}
    for rep in range(reps):
        responses, stimuli = simulate_population(RHOS, n_per_class, rng)
        for method in correlations:
            _, mi_est = neuron_ranking(
                responses, stimuli, method=method, B=50, seed=SEED + rep
            )
            correlations[method][rep] = spearmanr(mi_est, TRUE_MI).statistic
    mean_plugin = correlations["plugin"].mean()
    assert correlations["miller_madow"].mean() > mean_plugin
    assert correlations["shuffle"].mean() > mean_plugin


def test_synergy_matrix_sign_and_shuffle_correction():
    """Plugin SR is bias-inflated (many false synergies); shuffle fixes it."""
    rng = np.random.default_rng(SEED)
    responses, stimuli = simulate_population(RHOS, 400, rng)
    iu = np.triu_indices(len(RHOS), k=1)
    sr_plugin = synergy_matrix(responses, stimuli, method="plugin")
    assert sr_plugin.shape == (len(RHOS), len(RHOS))
    assert np.allclose(sr_plugin, sr_plugin.T)
    assert np.allclose(np.diag(sr_plugin), 0.0)
    sr_shuffle = synergy_matrix(responses, stimuli, method="shuffle", B=50, seed=SEED)
    frac_positive_plugin = np.mean(sr_plugin[iu] > 0.0)
    frac_positive_shuffle = np.mean(sr_shuffle[iu] > 0.0)
    # All pairs are truly redundant, so apparent synergy is spurious.
    assert frac_positive_plugin > 0.3  # plugin bias manufactures synergy
    assert frac_positive_shuffle < frac_positive_plugin
    assert frac_positive_shuffle < 0.2


# --------------------------------------------------------------------------
# ROC / AUC
# --------------------------------------------------------------------------

def test_roc_auc_for_separated_distributions_is_one():
    """Non-overlapping spike-rate distributions give AUC ~= 1."""
    rng = np.random.default_rng(SEED)
    signal = rng.integers(10, 21, size=500)  # high rates only
    noise = rng.integers(0, 10, size=500)  # low rates only
    assert roc_auc_score(signal, noise) == pytest.approx(1.0, abs=1e-12)


def test_roc_auc_for_overlapping_distributions_is_half():
    """Identical distributions give AUC ~= 0.5."""
    rng = np.random.default_rng(SEED)
    cond1 = rng.poisson(5, size=4000)
    cond2 = rng.poisson(5, size=4000)
    assert roc_auc_score(cond1, cond2) == pytest.approx(0.5, abs=0.05)


def test_roc_curve_endpoints_and_auc_consistency():
    """The ROC curve runs from (0, 0) to (1, 1) and auc() matches the score."""
    rng = np.random.default_rng(SEED)
    cond1 = rng.poisson(8, size=2000)
    cond2 = rng.poisson(3, size=2000)
    fpr, tpr = roc_curve(cond1, cond2)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert auc(fpr, tpr) == pytest.approx(roc_auc_score(cond1, cond2))
    assert 0.5 < auc(fpr, tpr) < 1.0  # separated but overlapping
