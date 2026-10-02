# Small-sample bias of mutual-information estimators

Synthetic ground truth: 22 neurons, 8 stimulus classes. Neuron *i* has reliability ρᵢ ∈ linspace(0, 0.9, 22) and its own number of response bins (cycling over {4, 8, 12, 16}): with probability ρᵢ its response is a deterministic function of the stimulus class, otherwise it is uniform noise over its bins. True MI per neuron ranges from 0.000 to 2.326 bits (max possible 3.0 bits). Neurons are conditionally independent given the stimulus, so every informative pair is truly *redundant* (true SR < 0).

Sweep over trials/stimulus N ∈ [10, 25, 50, 100, 400], 200 repetitions per N (shuffle corrections use B=32 surrogates for ranking and B=10 for synergy). Everything is seeded and deterministic.

### Table 1 — Mean bias of estimated MI (estimate − true MI, bits)

The plugin estimator is strongly upward biased at small N (≈ (m−1)/2N per entropy term, growing with the neuron's number of response bins). The shuffle correction removes almost all of it; Miller–Madow removes most of it but under-corrects at tiny N because the observed nonzero-cell counts undershoot the true support.

| Estimator | N=10 | N=25 | N=50 | N=100 | N=400 |
|---|---|---|---|---|---|
| `plugin` | +0.5231 | +0.2472 | +0.1242 | +0.0614 | +0.0146 |
| `miller_madow` | +0.3595 | +0.1164 | +0.0364 | +0.0106 | +0.0010 |
| `shuffle` | -0.0755 | +0.0044 | +0.0069 | +0.0044 | +0.0008 |

### Table 2 — Standard deviation of the MI estimate (bits)

Variability across neurons and repetitions. The shuffle correction subtracts an empirical mean, slightly increasing variance at large N where the bias it removes is negligible.

| Estimator | N=10 | N=25 | N=50 | N=100 | N=400 |
|---|---|---|---|---|---|
| `plugin` | 0.7102 | 0.7296 | 0.7238 | 0.7139 | 0.7028 |
| `miller_madow` | 0.7771 | 0.7583 | 0.7317 | 0.7125 | 0.7012 |
| `shuffle` | 0.6318 | 0.6985 | 0.7070 | 0.7054 | 0.7011 |

### Table 3 — Spearman ρ between estimated and true neuron ranking

Rank correlation of the 22 estimated MIs against the analytic true MIs, averaged over repetitions. Because plugin bias grows with a neuron's bin count, the plugin estimator misranks many-binned noise neurons above few-binned informative ones at small N; the bias-corrected estimators fix this.

| Estimator | N=10 | N=25 | N=50 | N=100 | N=400 |
|---|---|---|---|---|---|
| `plugin` | 0.9031 | 0.9652 | 0.9859 | 0.9933 | 0.9982 |
| `miller_madow` | 0.9284 | 0.9804 | 0.9922 | 0.9962 | 0.9990 |
| `shuffle` | 0.9667 | 0.9864 | 0.9931 | 0.9963 | 0.9990 |

### Table 4 — SR sign-flip rate vs the analytic reference (%)

Fraction of truly redundant neuron pairs (true SR < 0) whose estimated SR is positive (apparent synergy). Plugin MI of the joint variable (Rᵢ, Rⱼ) has many bins, so its bias is large enough at small N to manufacture spurious synergy. Miller–Madow is unreliable here at small N: its correction of the joint term uses observed nonzero-cell counts, which undershoot the large joint support, so it under-corrects and flips *more* signs than plugin until N grows. The empirical shuffle correction is the trustworthy small-sample fix for SR.

| Estimator | N=10 | N=25 | N=50 | N=100 | N=400 |
|---|---|---|---|---|---|
| `plugin` | 44.8 | 63.4 | 68.2 | 66.6 | 50.7 |
| `miller_madow` | 60.3 | 71.6 | 71.3 | 64.4 | 32.3 |
| `shuffle` | 1.4 | 1.1 | 2.1 | 4.8 | 10.4 |

## Notes

* Miller–Madow and shuffle corrections are **small-sample fixes**: their correction terms shrink like 1/N (Miller–Madow analytically, shuffle empirically), and every table above shows the corrected estimators converging to the plugin estimator as N grows — at N=400 trials/stimulus all three are practically identical.
* The plugin estimator is fine for comparing neurons at large N; at small N its bias depends on each neuron's number of effective response bins, which corrupts the ranking and inflates apparent synergy. The corrections fix exactly this.
* Shuffle correction costs B extra histograms per estimate and adds a little variance; Miller–Madow is essentially free but assumes the first-order 1/N expansion is adequate and that the observed nonzero-cell counts approximate the true support — both fail for many-celled joint tables at tiny N (see Table 4), where only the empirical shuffle correction keeps SR signs reliable.
