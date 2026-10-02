"""Wald sequential probability ratio test (SPRT) for language identification.

This module ports the MATLAB coursework in ``src/language/`` (``extract_letters.m``
and ``decide_text_language.m``) to Python. Text is reduced to letter indices
a-z and a Wald SPRT is run over per-letter log-likelihood ratios between two
unigram language models.

The Wald thresholds are::

    A = log10((1 - alpha) / beta)   (accept hypothesis 1 / upper threshold)
    B = log10(alpha / (1 - beta))   (accept hypothesis 0 / lower threshold)

and the expected sample sizes are estimated from the Kullback-Leibler
divergence between the two letter distributions, following the original
MATLAB script.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N_LETTERS = 26
"""Number of letters in the a-z alphabet used by the models."""


def extract_letters(text: str) -> np.ndarray:
    """Extract lowercase letter indices a-z from raw text.

    Port of ``src/language/extract_letters.m``: the text is lowercased and
    every character that is not an ASCII letter a-z is discarded. Letters are
    returned as integer indices 0-25 (``'a'`` -> 0, ..., ``'z'`` -> 25),
    mirroring the MATLAB indices 1-26.

    Args:
        text: Raw text, possibly containing punctuation, digits and
            whitespace.

    Returns:
        Integer array of letter indices in ``[0, 25]`` in order of
        appearance. Empty if the text contains no ASCII letters.
    """
    chars = np.frombuffer(text.lower().encode("ascii", errors="ignore"),
                          dtype=np.uint8)
    idx = chars.astype(np.int64) - ord("a")
    return idx[(idx >= 0) & (idx < N_LETTERS)]


def train_unigram_model(text: str, smoothing: float = 1e-3) -> np.ndarray:
    """Train a smoothed unigram letter distribution from text.

    Counts the letters extracted with :func:`extract_letters` and normalises
    with additive (Laplace-style) smoothing so that no letter has zero
    probability.

    Args:
        text: Training corpus.
        smoothing: Additive smoothing constant added to every letter count
            before normalisation. Must be positive.

    Returns:
        Array of shape ``(26,)`` with ``P(letter)`` for letters a-z; sums
        to 1.

    Raises:
        ValueError: If ``smoothing`` is not positive or the text contains no
            letters.
    """
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    letters = extract_letters(text)
    if letters.size == 0:
        raise ValueError("training text contains no a-z letters")
    counts = np.bincount(letters, minlength=N_LETTERS).astype(np.float64)
    counts += smoothing
    return counts / counts.sum()


def wald_thresholds(alpha: float, beta: float) -> Tuple[float, float]:
    """Compute Wald SPRT thresholds.

    Args:
        alpha: Target probability of falsely accepting hypothesis 1 when
            hypothesis 0 is true (type-I error).
        beta: Target probability of falsely accepting hypothesis 0 when
            hypothesis 1 is true (type-II error).

    Returns:
        Tuple ``(A, B)`` with ``A = log10((1 - alpha) / beta)`` (upper /
        accept-hypothesis-1 threshold) and
        ``B = log10(alpha / (1 - beta))`` (lower / accept-hypothesis-0
        threshold).
    """
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("alpha and beta must lie in (0, 1)")
    A = np.log10((1.0 - alpha) / beta)
    B = np.log10(alpha / (1.0 - beta))
    return float(A), float(B)


def expected_samples(p: np.ndarray, q: np.ndarray) -> Dict[str, float]:
    """Estimate expected SPRT sample sizes via Kullback-Leibler divergence.

    Under hypothesis ``p`` the accumulated log-likelihood ratio grows on
    average by ``D_KL(p || q)`` per observation (base 10), so reaching the
    upper threshold ``A`` takes about ``A / D_KL(p || q)`` samples.
    Symmetrically, under hypothesis ``q`` the LLR drifts down by
    ``D_KL(q || p)`` per observation and reaching the lower threshold ``B``
    takes about ``|B| / D_KL(q || p)`` samples. These are the estimates used
    in ``src/language/decide_text_language.m`` (with equal priors).

    Args:
        p: Probability vector for hypothesis 1, shape ``(26,)``.
        q: Probability vector for hypothesis 0, shape ``(26,)``.

    Returns:
        Dictionary with keys ``"dkl_p_q"``, ``"dkl_q_p"`` (base-10 KL
        divergences) and ``"n_under_p"``, ``"n_under_q"`` (expected letters
        to decision under each hypothesis, for ``alpha = beta`` thresholds).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    dkl_p_q = float(np.sum(p * (np.log10(p) - np.log10(q))))
    dkl_q_p = float(np.sum(q * (np.log10(q) - np.log10(p))))
    return {
        "dkl_p_q": dkl_p_q,
        "dkl_q_p": dkl_q_p,
    }


class UnigramSPRT:
    """Sequential probability ratio test over single-letter likelihoods.

    Accumulates the log10 likelihood ratio
    ``log10(P_1(letter) / P_0(letter))`` letter by letter and stops at the
    Wald thresholds ``A`` (decide hypothesis 1) or ``B`` (decide
    hypothesis 0). If the stream ends first, the side of the running sum
    nearest to a threshold (sign of the LLR) is used as a forced decision.

    Attributes:
        model1: Unigram distribution under hypothesis 1, shape ``(26,)``.
        model0: Unigram distribution under hypothesis 0, shape ``(26,)``.
        label1: Human-readable name of hypothesis 1.
        label0: Human-readable name of hypothesis 0.
        alpha: Nominal type-I error probability.
        beta: Nominal type-II error probability.
        threshold_a: Upper (accept hypothesis 1) Wald threshold.
        threshold_b: Lower (accept hypothesis 0) Wald threshold.
    """

    def __init__(self, model1: np.ndarray, model0: np.ndarray,
                 alpha: float = 0.05, beta: float = 0.05,
                 label1: str = "H1", label0: str = "H0") -> None:
        """Initialise the SPRT with two unigram models and error targets.

        Args:
            model1: Letter distribution under hypothesis 1, shape ``(26,)``;
                must be strictly positive and sum to 1.
            model0: Letter distribution under hypothesis 0, shape ``(26,)``;
                must be strictly positive and sum to 1.
            alpha: Nominal type-I error probability.
            beta: Nominal type-II error probability.
            label1: Name of hypothesis 1 (e.g. ``"english"``).
            label0: Name of hypothesis 0 (e.g. ``"french"``).

        Raises:
            ValueError: If a model has the wrong shape or non-positive
                entries.
        """
        model1 = np.asarray(model1, dtype=np.float64)
        model0 = np.asarray(model0, dtype=np.float64)
        for name, m in (("model1", model1), ("model0", model0)):
            if m.shape != (N_LETTERS,):
                raise ValueError(f"{name} must have shape ({N_LETTERS},)")
            if np.any(m <= 0):
                raise ValueError(f"{name} must be strictly positive; "
                                 "use smoothing when training")
        self.model1 = model1
        self.model0 = model0
        self.label1 = label1
        self.label0 = label0
        self.alpha = alpha
        self.beta = beta
        self.threshold_a, self.threshold_b = wald_thresholds(alpha, beta)
        self._log_ratio = np.log10(model1) - np.log10(model0)

    def decide(self, text: str) -> Tuple[str, int, List[float]]:
        """Run the SPRT on a text sample.

        Args:
            text: Raw text; non-letters are ignored.

        Returns:
            Tuple ``(label, n_letters, llr_trajectory)`` where ``label`` is
            ``label1`` or ``label0``, ``n_letters`` is the number of letters
            consumed before the decision, and ``llr_trajectory`` is the
            running log10 likelihood ratio after each letter (including any
            forced decision at the end of the stream).

        Raises:
            ValueError: If the text contains no a-z letters.
        """
        letters = extract_letters(text)
        if letters.size == 0:
            raise ValueError("text contains no a-z letters")
        llr = 0.0
        trajectory: List[float] = []
        for letter in letters:
            llr += self._log_ratio[letter]
            trajectory.append(llr)
            if llr >= self.threshold_a:
                return self.label1, len(trajectory), trajectory
            if llr <= self.threshold_b:
                return self.label0, len(trajectory), trajectory
        # Stream exhausted without crossing a threshold: forced decision.
        label = self.label1 if llr >= 0.0 else self.label0
        return label, len(trajectory), trajectory

    def expected_samples(self) -> Dict[str, float]:
        """Expected letters-to-decision under each hypothesis.

        Returns:
            Dictionary with base-10 KL divergences ``dkl_1_0`` / ``dkl_0_1``
            and the KL-based expected sample sizes ``n_under_1`` /
            ``n_under_0`` for crossing the respective Wald threshold.
        """
        est = expected_samples(self.model1, self.model0)
        return {
            "dkl_1_0": est["dkl_p_q"],
            "dkl_0_1": est["dkl_q_p"],
            "n_under_1": self.threshold_a / est["dkl_p_q"],
            "n_under_0": abs(self.threshold_b) / est["dkl_q_p"],
        }


def pairwise_decide(models: Dict[str, np.ndarray], text: str,
                    alpha: float = 0.05, beta: float = 0.05) -> str:
    """Classify text among several languages via pairwise unigram SPRTs.

    Runs an SPRT for every ordered pair of languages and returns the language
    that wins the most pairwise duels (Copeland rule).

    Args:
        models: Mapping from language label to unigram distribution.
        text: Raw text to classify.
        alpha: Nominal type-I error probability for each pairwise test.
        beta: Nominal type-II error probability for each pairwise test.

    Returns:
        Label of the winning language.
    """
    labels = list(models)
    wins = {label: 0 for label in labels}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sprt = UnigramSPRT(models[labels[i]], models[labels[j]],
                               alpha=alpha, beta=beta,
                               label1=labels[i], label0=labels[j])
            wins[sprt.decide(text)[0]] += 1
    return max(wins, key=wins.get)
