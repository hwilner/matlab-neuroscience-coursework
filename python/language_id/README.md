# Language identification — Wald SPRT port + bigram SPRT

## Introduction

Python port of the MATLAB language-identification exercise (`extract_letters.m`,
`decide_text_language.m`): a Wald sequential probability ratio test (SPRT) over
single-letter statistics, plus a novel improvement — a **Laplace-smoothed
bigram SPRT** that accumulates log-likelihood ratios over letter *transitions*.

## Extended introduction

For simple-vs-simple hypotheses with i.i.d. observations, the Wald–Wolfowitz
theorem proves the SPRT minimises the expected number of observations among
all tests with the same error rates — the unigram SPRT is already optimal
*for its observation model*. The only honest way to beat it is a better
observation model: natural language is strongly Markov, so the conditional
distribution `P(c_t | c_{t−1})` carries far more per-letter information about
language identity than the marginal `P(c)`. Modern language ID uses n-gram or
neural models but usually as fixed-length batch classifiers; the sequential
setting (decide as early as possible, letter by letter) is where SPRT shines.

## Methods

- `sprt.py` — `extract_letters`, additive-smoothed unigram models, Wald
  thresholds `A = log10((1−α)/β)`, `B = log10(α/(1−β))`, KL-based expected
  sample sizes, `UnigramSPRT.decide(text)`.
- `bigram_sprt.py` — Laplace-smoothed bigram models `P(c_t | c_{t−1})`,
  `BigramSPRT` (first character scored by the smoothed unigram; K > 2
  languages handled by a max-rival generalised SPRT).
- `experiment_language_id.py` — 80/20 train/test split of the English and
  French corpora in `data/`, Dutch used whole (tiny corpus; leakage caveat
  disclosed); 300 random windows of 20–200 characters per language;
  α = β ∈ {0.05, 0.01}.

## Results

Full captioned tables in [`RESULTS.md`](RESULTS.md). Headline findings:

- **Pairwise (EN/FR):** the bigram SPRT needs roughly **half the letters** of
  the unigram SPRT at equal-or-lower empirical error (e.g., at α = 0.01,
  English: 10.3 vs 20.6 mean letters at 3.0% vs 3.3% error).
- **Three-language (EN/FR/NL):** the gap widens — bigram decides in ~9–15
  letters vs ~27–42 for unigram, at strictly lower error in every condition.
- **Honesty:** empirical error runs above the nominal α in both classifiers
  (text is not i.i.d.; forced end-of-window decisions), and the Dutch model
  saw its whole corpus — both disclosed in `RESULTS.md`.

## Tests

```bash
python -m pytest python/language_id -q   # 11 tests
```
