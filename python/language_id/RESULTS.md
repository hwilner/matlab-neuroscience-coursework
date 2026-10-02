# Language identification with sequential probability ratio tests

Port of the MATLAB coursework in `src/language/` plus a novel bigram (first-order Markov) SPRT with Laplace smoothing.

## Setup

- Training: first 80% of `data/English.txt` (643 KB) and `data/French.txt` (204 KB); the full `data/dutch.txt` (~4 KB, too small to split -- see leakage note below).
- Testing: 300 random windows per language, each of random length 20-200 characters, drawn from the held-out 20% (fixed seeds).
- Nominal error rates `alpha = beta` in {0.05, 0.01}; Wald thresholds `A = log10((1-alpha)/beta)`, `B = log10(alpha/(1-beta))`.
- Unigram smoothing `1e-3`; bigram Laplace smoothing `1.0`.


## Table 1. KL-divergence expected sample sizes (unigram, English vs French)

Port of the `DKL` block of `src/language/decide_text_language.m`: expected letters-to-decision ~ threshold / D_KL (base 10).

| Quantity | Value |
|---|---|
| D_KL(english \|\| french) | 0.1287 |
| D_KL(french \|\| english) | 0.0670 |
| E[letters \| english], alpha=beta=0.05 | 9.9 |
| E[letters \| french], alpha=beta=0.05 | 19.1 |


## Table 2. Pairwise English vs French: error rate and letters-to-decision

Empirical error rate vs nominal `alpha`, and mean/median letters consumed before the SPRT stopped (300 windows per condition).

| Classifier | nominal alpha | empirical error | mean letters | median letters |
|---|---|---|---|---|
| unigram, true=english | 0.05 | 0.0667 (20/300) | 15.7 | 13 |
| bigram, true=english | 0.05 | 0.0600 (18/300) | 7.6 | 6 |
| unigram, true=french | 0.05 | 0.0433 (13/300) | 21.8 | 18 |
| bigram, true=french | 0.05 | 0.0433 (13/300) | 8.7 | 7 |
| unigram, true=english | 0.01 | 0.0333 (10/300) | 20.6 | 17 |
| bigram, true=english | 0.01 | 0.0300 (9/300) | 10.3 | 9 |
| unigram, true=french | 0.01 | 0.0300 (9/300) | 31.1 | 28 |
| bigram, true=french | 0.01 | 0.0233 (7/300) | 13.2 | 12 |


## Table 3. Error-sample-size tradeoff (sorted by mean letters)

The bigram SPRT should sit strictly below-left of the unigram SPRT: fewer letters for equal or lower error.

| Condition | empirical error | mean letters | median letters |
|---|---|---|---|
| bigram (alpha=0.05, true=english) | 0.0600 | 7.6 | 6 |
| bigram (alpha=0.05, true=french) | 0.0433 | 8.7 | 7 |
| bigram (alpha=0.01, true=english) | 0.0300 | 10.3 | 9 |
| bigram (alpha=0.01, true=french) | 0.0233 | 13.2 | 12 |
| unigram (alpha=0.05, true=english) | 0.0667 | 15.7 | 13 |
| unigram (alpha=0.01, true=english) | 0.0333 | 20.6 | 17 |
| unigram (alpha=0.05, true=french) | 0.0433 | 21.8 | 18 |
| unigram (alpha=0.01, true=french) | 0.0300 | 31.1 | 28 |


## Table 4. Three-language task (English / French / Dutch)

Unigram: Copeland vote over all pairwise SPRTs (letters-to-decision = slowest duel). Bigram: max-rival generalised SPRT (see `bigram_sprt.py` docstring).

| Classifier | nominal alpha | empirical error | mean letters | median letters |
|---|---|---|---|---|
| unigram, true=dutch | 0.05 | 0.0867 (26/300) | 27.5 | 25 |
| bigram, true=dutch | 0.05 | 0.0267 (8/300) | 9.4 | 9 |
| unigram, true=english | 0.05 | 0.1167 (35/300) | 30.5 | 26 |
| bigram, true=english | 0.05 | 0.0500 (15/300) | 9.4 | 8 |
| unigram, true=french | 0.05 | 0.0733 (22/300) | 33.3 | 29 |
| bigram, true=french | 0.05 | 0.0467 (14/300) | 10.2 | 9 |
| unigram, true=dutch | 0.01 | 0.0467 (14/300) | 40.3 | 36 |
| bigram, true=dutch | 0.01 | 0.0100 (3/300) | 12.9 | 12 |
| unigram, true=english | 0.01 | 0.0667 (20/300) | 39.7 | 35 |
| bigram, true=english | 0.01 | 0.0233 (7/300) | 13.2 | 12 |
| unigram, true=french | 0.01 | 0.0533 (16/300) | 42.0 | 34 |
| bigram, true=french | 0.01 | 0.0267 (8/300) | 14.6 | 13 |


## Honest notes and caveats

1. **Wald optimality is conditional on the observation model.** The SPRT is optimal (minimal expected sample size at given error rates) only for simple-vs-simple hypotheses with i.i.d. observations. Unigram letters of real text are *not* independent, so even the baseline operates outside the theorem's assumptions; the gains in Tables 2-4 come from a better observation model (Wald-Wolfowitz: only the model can be improved, not the rule).
2. **Bigram transitions violate strict i.i.d. mildly.** Consecutive transitions share a letter, so the accumulated LLR increments are one-dependent. The Wald thresholds and nominal error rates therefore hold only approximately for the bigram SPRT; the empirical error rates above are the check.
3. **Dutch leakage.** `dutch.txt` is only ~4 KB, so the Dutch model is trained on the full text and the 3-language Dutch test windows overlap the training data. Dutch rows in Table 4 are optimistic and are included only to demonstrate the multi-language scheme.
4. **Smoothing sensitivity.** Unigram probabilities use additive smoothing `1e-3` (only to avoid zeros); the bigram model uses Laplace smoothing `1.0` over 676 cells, where smoothing matters more: too little smoothing makes rare transitions dominate the LLR and inflates error on atypical windows, too much flattens the models towards each other. `1.0` was stable across both alpha settings in this experiment, but the choice has not been tuned systematically.
5. **Corpus heterogeneity.** English.txt is Project-Gutenberg style prose (Plato's Republic) and French.txt is Voltaire; letter and bigram statistics differ from modern text, so absolute numbers do not transfer directly to other corpora.
6. **End-of-stream forced decisions.** Windows too short to reach a Wald threshold are decided by the sign/argmax of the accumulated evidence; those decisions do not enjoy the nominal error guarantees and contribute to the empirical error rate.
