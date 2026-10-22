"""Tests for the language_id package (unigram and bigram SPRT)."""

import os

import numpy as np
import pytest

from python.language_id.sprt import (N_LETTERS, UnigramSPRT, expected_samples,
                                     extract_letters, train_unigram_model,
                                     wald_thresholds)
from python.language_id.bigram_sprt import BigramSPRT, train_bigram_model

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "..", "data"))
SEED = 1234


def _load(name: str) -> str:
    """Load a corpus file from the repo's data directory.

    Args:
        name: File name inside ``data/``.

    Returns:
        Raw text (latin-1 decoded, matching the legacy encodings).
    """
    with open(os.path.join(DATA_DIR, name), encoding="latin-1") as fh:
        return fh.read()


# ----------------------------------------------------------------------
# extract_letters
# ----------------------------------------------------------------------

def test_extract_letters_filters_and_lowercases() -> None:
    """Non-letters are dropped and uppercase letters are lowercased."""
    letters = extract_letters("Hello, World! 123 Zz.")
    assert letters.tolist() == ([ord(c) - ord("a") for c in "helloworldzz"])
    assert letters.dtype == np.int64


def test_extract_letters_empty_on_no_letters() -> None:
    """Text without a-z letters yields an empty index array."""
    assert extract_letters("12345 !? \n\t").size == 0


# ----------------------------------------------------------------------
# Wald thresholds
# ----------------------------------------------------------------------

@pytest.mark.parametrize("alpha,beta", [(0.05, 0.05), (0.01, 0.01),
                                        (0.05, 0.001), (1e-4, 1e-3)])
def test_wald_thresholds_match_formulas(alpha: float, beta: float) -> None:
    """Thresholds equal A=log10((1-a)/b), B=log10(a/(1-b)) exactly."""
    A, B = wald_thresholds(alpha, beta)
    assert A == pytest.approx(np.log10((1 - alpha) / beta))
    assert B == pytest.approx(np.log10(alpha / (1 - beta)))
    assert A > 0 > B


# ----------------------------------------------------------------------
# Unigram SPRT on the repo corpora
# ----------------------------------------------------------------------

def test_unigram_sprt_decides_corpus_languages() -> None:
    """English and French held-out snippets are classified correctly."""
    english = _load("English.txt")
    french = _load("French.txt")
    # Train on a quick subset (first 50k chars), test on later text.
    model_eng = train_unigram_model(english[:50_000])
    model_fr = train_unigram_model(french[:50_000])
    sprt = UnigramSPRT(model_eng, model_fr, alpha=0.01, beta=0.01,
                       label1="english", label0="french")
    label_eng, n_eng, traj_eng = sprt.decide(english[-5_000:])
    label_fr, n_fr, traj_fr = sprt.decide(french[-5_000:])
    assert label_eng == "english"
    assert label_fr == "french"
    # Both snippets decide well before exhausting the stream.
    assert n_eng < len(traj_eng) or n_eng <= 5_000
    assert n_fr < len(traj_fr) or n_fr <= 5_000
    assert n_eng < 2_000 and n_fr < 2_000
    # Trajectory crosses the decision threshold at the stopping point.
    assert traj_eng[-1] >= sprt.threshold_a
    assert traj_fr[-1] <= sprt.threshold_b


def test_expected_samples_uses_kl_divergence() -> None:
    """expected_samples returns base-10 KL divergences of the models."""
    p = train_unigram_model(_load("English.txt")[:20_000])
    q = train_unigram_model(_load("French.txt")[:20_000])
    est = expected_samples(p, q)
    assert est["dkl_p_q"] == pytest.approx(
        float(np.sum(p * np.log10(p / q))))
    assert est["dkl_q_p"] == pytest.approx(
        float(np.sum(q * np.log10(q / p))))
    assert est["dkl_p_q"] > 0 and est["dkl_q_p"] > 0


def test_expected_samples_reports_every_documented_key() -> None:
    """expected_samples returns all four keys its docstring promises.

    The function's docstring documents ``n_under_p`` / ``n_under_q`` next to
    the two KL divergences, so every caller following the documented
    contract must be able to read them.
    """
    p = train_unigram_model(_load("English.txt")[:20_000])
    q = train_unigram_model(_load("French.txt")[:20_000])
    est = expected_samples(p, q)
    assert set(est) == {"dkl_p_q", "dkl_q_p", "n_under_p", "n_under_q"}
    threshold_a, threshold_b = wald_thresholds(0.05, 0.05)
    assert est["n_under_p"] == pytest.approx(threshold_a / est["dkl_p_q"])
    assert est["n_under_q"] == pytest.approx(abs(threshold_b) / est["dkl_q_p"])


def test_expected_samples_reproduces_results_table_1() -> None:
    """The published Table 1 of language_id/RESULTS.md is reproducible.

    Table 1 is generated from the first 80% of each corpus by
    ``experiment_language_id.py``; this pins the KL-based expected sample
    sizes to the numbers the results document reports.
    """
    def _train_prefix(name: str) -> str:
        """Return the first 80% of a corpus, matching the experiment split."""
        return _load(name)[: int(len(_load(name)) * 0.8)]

    p = train_unigram_model(_train_prefix("English.txt"))
    q = train_unigram_model(_train_prefix("French.txt"))
    est = expected_samples(p, q, alpha=0.05, beta=0.05)
    assert est["dkl_p_q"] == pytest.approx(0.1287, abs=5e-5)
    assert est["dkl_q_p"] == pytest.approx(0.0670, abs=5e-5)
    assert est["n_under_p"] == pytest.approx(9.9, abs=0.05)
    assert est["n_under_q"] == pytest.approx(19.1, abs=0.05)


def test_expected_samples_rejects_non_positive_models() -> None:
    """A zero probability would make the divergence NaN, so it is rejected."""
    p = np.full(N_LETTERS, 1.0 / N_LETTERS)
    q = p.copy()
    q[0] = 0.0
    with pytest.raises(ValueError):
        expected_samples(p, q)


def test_expected_samples_is_infinite_for_identical_models() -> None:
    """Identical models never separate, so the expected size is infinite."""
    p = np.full(N_LETTERS, 1.0 / N_LETTERS)
    est = expected_samples(p, p)
    assert est["dkl_p_q"] == pytest.approx(0.0)
    assert est["n_under_p"] == float("inf")
    assert est["n_under_q"] == float("inf")


def test_unigram_sprt_expected_samples_uses_instance_thresholds() -> None:
    """UnigramSPRT.expected_samples uses the thresholds of its own instance."""
    p = train_unigram_model(_load("English.txt")[:20_000])
    q = train_unigram_model(_load("French.txt")[:20_000])
    sprt = UnigramSPRT(p, q, alpha=0.01, beta=0.01)
    est = sprt.expected_samples()
    threshold_a, threshold_b = wald_thresholds(0.01, 0.01)
    assert est["n_under_1"] == pytest.approx(threshold_a / est["dkl_1_0"])
    assert est["n_under_0"] == pytest.approx(abs(threshold_b) / est["dkl_0_1"])
    # The default-alpha estimate must not be reused for an alpha=0.01 SPRT.
    default = UnigramSPRT(p, q).expected_samples()
    assert est["n_under_1"] != pytest.approx(default["n_under_1"])


# ----------------------------------------------------------------------
# Bigram model
# ----------------------------------------------------------------------

def test_bigram_model_rows_sum_to_one() -> None:
    """Laplace-smoothed transition rows are proper distributions."""
    model = train_bigram_model(_load("English.txt")[:50_000], smoothing=1.0)
    assert model.transition.shape == (N_LETTERS, N_LETTERS)
    np.testing.assert_allclose(model.transition.sum(axis=1), 1.0,
                               rtol=1e-12)
    assert np.all(model.transition > 0)
    assert model.unigram.sum() == pytest.approx(1.0)
    assert np.all(model.unigram > 0)


def test_bigram_sprt_decides_on_short_windows() -> None:
    """The bigram SPRT stops within short held-out windows, correctly."""
    english = _load("English.txt")
    french = _load("French.txt")
    models = {"english": train_bigram_model(english[:100_000]),
              "french": train_bigram_model(french[:100_000])}
    sprt = BigramSPRT(models, alpha=0.05, beta=0.05)
    # Short windows from the held-out tails of both corpora.
    eng_window = english[int(0.8 * len(english)):][:150]
    fr_window = french[int(0.8 * len(french)):][:150]
    label_eng, n_eng, traj_eng = sprt.decide(eng_window)
    label_fr, n_fr, traj_fr = sprt.decide(fr_window)
    assert label_eng == "english"
    assert label_fr == "french"
    # Decisions are reached well inside the 150-character windows.
    assert n_eng < 150 and n_fr < 150
    assert len(traj_eng) == n_eng and len(traj_fr) == n_fr
    assert traj_eng[-1] >= sprt.threshold
    assert traj_fr[-1] >= sprt.threshold


# ----------------------------------------------------------------------
# Synthetic Markov chains: bigram SPRT beats unigram SPRT
# ----------------------------------------------------------------------

def _make_cyclic_transition(step: int, concentration: float = 0.7,
                            smoothing: float = 1.0) -> np.ndarray:
    """Build a row-stochastic matrix favouring ``next = (prev + step) % 26``.

    All such matrices (any ``step``) are doubly stochastic, so their
    stationary distribution is uniform: the two hypotheses share the exact
    same unigram marginal and differ only in transition structure.

    Args:
        step: Cyclic offset favoured by each row.
        concentration: Probability mass placed on the favoured successor.
        smoothing: Mass spread uniformly over all successors.

    Returns:
        Array of shape ``(26, 26)`` with rows summing to 1.
    """
    mat = np.full((N_LETTERS, N_LETTERS), smoothing / N_LETTERS)
    for prev in range(N_LETTERS):
        mat[prev, (prev + step) % N_LETTERS] += concentration
    return mat / mat.sum(axis=1, keepdims=True)


def _sample_chain(transition: np.ndarray, length: int, seed: int
                  ) -> np.ndarray:
    """Sample a letter chain from a first-order Markov model.

    Args:
        transition: Row-stochastic transition matrix, shape ``(26, 26)``.
        length: Number of letters to generate.
        seed: Generator seed.

    Returns:
        Integer array of letter indices of length ``length``.
    """
    rng = np.random.default_rng(seed)
    letters = np.empty(length, dtype=np.int64)
    letters[0] = rng.integers(0, N_LETTERS)
    for t in range(1, length):
        letters[t] = rng.choice(N_LETTERS, p=transition[letters[t - 1]])
    return letters


def test_bigram_sprt_no_worse_than_unigram_on_markov_chains() -> None:
    """On Markov chains with identical marginals, bigram error <= unigram.

    The two synthetic languages share a uniform unigram distribution, so a
    unigram SPRT has no discriminative signal, while the bigram SPRT sees
    the distinct transition structure.
    """
    trans1 = _make_cyclic_transition(step=1)
    trans0 = _make_cyclic_transition(step=3)
    # Train each bigram model once on a long independent chain.
    models_bi = {
        "H1": train_bigram_model("".join(
            chr(ord("a") + c) for c in
            _sample_chain(trans1, 20_000, seed=SEED + 1))),
        "H0": train_bigram_model("".join(
            chr(ord("a") + c) for c in
            _sample_chain(trans0, 20_000, seed=SEED + 2))),
    }
    bi = BigramSPRT(models_bi, alpha=0.05, beta=0.05)
    uni = UnigramSPRT(models_bi["H1"].unigram, models_bi["H0"].unigram,
                      alpha=0.05, beta=0.05, label1="H1", label0="H0")
    n_trials, length = 40, 60
    bigram_errors = unigram_errors = 0
    for truth, transition in (("H1", trans1), ("H0", trans0)):
        for trial in range(n_trials):
            chain = _sample_chain(transition, length,
                                  seed=SEED + 10_000 * (truth == "H0")
                                  + trial)
            text = "".join(chr(ord("a") + c) for c in chain)
            bigram_errors += int(bi.decide(text)[0] != truth)
            unigram_errors += int(uni.decide(text)[0] != truth)
    total = 2 * n_trials
    assert bigram_errors <= unigram_errors
    # Sanity: with uniform marginals the unigram test is near chance, while
    # the bigram test should be nearly perfect.
    assert bigram_errors <= 0.1 * total
    assert unigram_errors >= 0.2 * total
