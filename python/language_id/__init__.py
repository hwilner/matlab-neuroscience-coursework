"""Sequential language identification from raw text.

Ports the MATLAB coursework in ``src/language/`` (unigram Wald SPRT) to
Python and adds a novel bigram (first-order Markov) SPRT with Laplace
smoothing.
"""

from .sprt import (N_LETTERS, UnigramSPRT, expected_samples, extract_letters,
                   pairwise_decide, train_unigram_model, wald_thresholds)
from .bigram_sprt import BigramModel, BigramSPRT, train_bigram_model

__all__ = [
    "N_LETTERS",
    "UnigramSPRT",
    "extract_letters",
    "expected_samples",
    "pairwise_decide",
    "train_unigram_model",
    "wald_thresholds",
    "BigramModel",
    "BigramSPRT",
    "train_bigram_model",
]
