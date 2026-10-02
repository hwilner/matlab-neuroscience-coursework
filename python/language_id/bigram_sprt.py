"""Bigram sequential probability ratio test for language identification.

Novel research component on top of the unigram SPRT port in ``sprt.py``.
Instead of treating letters as i.i.d. draws, each language is modelled as a
first-order Markov chain over the 26 letters: ``P(c_t | c_{t-1})``. The SPRT
accumulates the log-likelihood ratio over *transitions*; the very first
letter of a sample (which has no predecessor) contributes a smoothed unigram
likelihood.

Laplace (add-one-style) smoothing guarantees every conditional row is
strictly positive, so log-likelihood ratios are always finite.

Multi-language scheme (``K > 2``)
---------------------------------
Wald's SPRT is defined for two simple hypotheses. For ``K`` languages we
maintain the accumulated log-likelihood ``S_l`` of the observation stream
under each language model ``l`` and use a *max-rival* (generalised SPRT)
stopping rule: language ``l`` is accepted as soon as

``S_l - max_{m != l} S_m >= A`` with ``A = log10((1 - alpha) / beta)``,

i.e. its evidence exceeds the strongest competing hypothesis by the
pairwise Wald threshold. For ``K = 2`` this reduces exactly to the classic
Wald SPRT (the margin ``S_1 - S_0`` is the pairwise LLR, and the lower
threshold ``B = -A`` when ``alpha == beta``). If the stream ends before any
language crosses the threshold, the language with the largest accumulated
log-likelihood is returned as a forced (maximum-likelihood) decision.

Caveat: bigram observations (transitions) overlap and are only approximately
i.i.d., so Wald's optimality and the nominal error rates hold only
approximately. See ``RESULTS.md`` for the empirical assessment.
"""

from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from .sprt import N_LETTERS, extract_letters, wald_thresholds


class BigramModel(NamedTuple):
    """First-order Markov model over letters a-z.

    Attributes:
        transition: Array of shape ``(26, 26)`` with
            ``transition[prev, cur] = P(c_t = cur | c_{t-1} = prev)``; every
            row sums to 1.
        unigram: Array of shape ``(26,)`` with the smoothed marginal letter
            distribution, used to score the first letter of a sample.
    """

    transition: np.ndarray
    unigram: np.ndarray


def train_bigram_model(text: str, smoothing: float = 1.0) -> BigramModel:
    """Train a Laplace-smoothed bigram letter model from text.

    Counts consecutive letter pairs extracted with
    :func:`language_id.sprt.extract_letters`, adds ``smoothing`` to every
    cell of the 26x26 transition count matrix and normalises each row,
    yielding ``P(c_t | c_{t-1})``. The unigram marginal is smoothed the same
    way.

    Args:
        text: Training corpus.
        smoothing: Additive (Laplace) smoothing constant; ``smoothing=1.0``
            is classic Laplace smoothing. Must be positive.

    Returns:
        A :class:`BigramModel` with row-stochastic transition matrix and
        strictly positive unigram marginal.

    Raises:
        ValueError: If ``smoothing`` is not positive or the text contains
            fewer than two letters.
    """
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    letters = extract_letters(text)
    if letters.size < 2:
        raise ValueError("training text contains fewer than two a-z letters")
    counts = np.full((N_LETTERS, N_LETTERS), smoothing, dtype=np.float64)
    np.add.at(counts, (letters[:-1], letters[1:]), 1.0)
    transition = counts / counts.sum(axis=1, keepdims=True)
    uni_counts = np.bincount(letters, minlength=N_LETTERS).astype(np.float64)
    uni_counts += smoothing
    unigram = uni_counts / uni_counts.sum()
    return BigramModel(transition=transition, unigram=unigram)


class BigramSPRT:
    """Sequential probability ratio test over letter-transition likelihoods.

    Accumulates per-language log-likelihoods over bigram transitions (first
    letter scored with the smoothed unigram) and applies the max-rival
    stopping rule described in the module docstring.

    Attributes:
        models: Mapping from language label to :class:`BigramModel`.
        alpha: Nominal type-I error probability.
        beta: Nominal type-II error probability.
        threshold: Acceptance threshold ``A = log10((1 - alpha) / beta)`` for
            the evidence margin over the strongest rival.
    """

    def __init__(self, models: Dict[str, BigramModel], alpha: float = 0.05,
                 beta: float = 0.05) -> None:
        """Initialise the bigram SPRT.

        Args:
            models: Mapping from language label to trained
                :class:`BigramModel`. Must contain at least two languages.
            alpha: Nominal type-I error probability.
            beta: Nominal type-II error probability.

        Raises:
            ValueError: If fewer than two models are supplied.
        """
        if len(models) < 2:
            raise ValueError("BigramSPRT needs at least two language models")
        self.models = dict(models)
        self.labels: List[str] = list(self.models)
        self.alpha = alpha
        self.beta = beta
        self.threshold, lower = wald_thresholds(alpha, beta)
        # With alpha == beta the Wald interval is symmetric (B = -A); the
        # max-rival margin uses the upper threshold in both directions.
        self._lower = lower
        # Precompute per-language log-probability tensors.
        self._log_trans = np.stack(
            [np.log10(self.models[l].transition) for l in self.labels])
        self._log_uni = np.stack(
            [np.log10(self.models[l].unigram) for l in self.labels])

    def decide(self, text: str) -> Tuple[str, int, List[float]]:
        """Run the bigram SPRT on a text sample.

        Args:
            text: Raw text; non-letters are ignored.

        Returns:
            Tuple ``(label, n_letters, margin_trajectory)`` where ``label``
            is the decided language, ``n_letters`` the number of letters
            consumed before stopping, and ``margin_trajectory`` the running
            evidence margin (best language's log-likelihood minus its
            strongest rival's) after each letter. A positive margin at the
            end indicates a forced decision at end-of-stream.

        Raises:
            ValueError: If the text contains fewer than two a-z letters.
        """
        letters = extract_letters(text)
        if letters.size < 2:
            raise ValueError("text contains fewer than two a-z letters")
        n_lang = len(self.labels)
        scores = self._log_uni[:, letters[0]].copy()
        trajectory: List[float] = []
        margin = self._margin(scores)
        trajectory.append(margin)
        if margin >= self.threshold:
            return self.labels[int(np.argmax(scores))], 1, trajectory
        for t in range(1, letters.size):
            scores += self._log_trans[:, letters[t - 1], letters[t]]
            margin = self._margin(scores)
            trajectory.append(margin)
            if margin >= self.threshold:
                return self.labels[int(np.argmax(scores))], t + 1, trajectory
        # Forced decision: maximum accumulated likelihood.
        return self.labels[int(np.argmax(scores))], int(letters.size), trajectory

    @staticmethod
    def _margin(scores: np.ndarray) -> float:
        """Evidence margin of the best hypothesis over its strongest rival.

        Args:
            scores: Accumulated log-likelihoods, one per language.

        Returns:
            ``max_l S_l - max_{m != argmax} S_m`` (non-negative).
        """
        order = np.argsort(scores)
        return float(scores[order[-1]] - scores[order[-2]])
