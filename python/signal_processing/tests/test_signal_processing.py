"""Tests for the signal_processing package (baseline ports + SpikeSPD)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from signal_processing import baseline as bl  # noqa: E402
from signal_processing import spike_spd as spd  # noqa: E402

FS = 30_000.0


# ------------------------------- helpers -----------------------------------
def biphasic_template(length: int = 61, width: float = 6.0, amp: float = 1.0) -> np.ndarray:
    """Simple biphasic test template with a negative trough and positive lobe."""
    t = np.arange(length) - length // 3
    w = -np.exp(-((t / width) ** 2)) + 0.5 * np.exp(-(((t - 2 * width) / (1.5 * width)) ** 2))
    return amp * w / np.max(np.abs(w))


def plant_spikes(
    n_samples: int, spike_idx: np.ndarray, template: np.ndarray, noise_std: float, seed: int
) -> np.ndarray:
    """Add a scaled template at each index of a Gaussian noise trace."""
    rng = np.random.default_rng(seed)
    signal = noise_std * rng.standard_normal(n_samples)
    pre = len(template) // 2
    for s in spike_idx:
        lo, hi = s - pre, s - pre + len(template)
        if lo >= 0 and hi <= n_samples:
            signal[lo:hi] += template
    return signal


def match_count(detected: np.ndarray, truth: np.ndarray, tol: int) -> int:
    """Number of ground-truth spikes with a detection within +/- tol samples."""
    return sum(int(np.any(np.abs(detected - t) <= tol)) for t in truth)


# --------------------------- circular convolution ---------------------------
def test_circular_convolution_known_result():
    """Direct definition matches a hand-computed circular convolution."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    expected = np.array([31.0, 31.0, 28.0])  # sum_j x[j] y[(i-j) mod 3]
    np.testing.assert_allclose(bl.circular_convolution(x, y), expected)


def test_circular_convolution_matches_fft_random():
    """Direct loop and FFT versions agree on random unequal-length inputs."""
    rng = np.random.default_rng(42)
    for n, m in [(7, 7), (13, 9), (32, 50)]:
        x = rng.standard_normal(n)
        y = rng.standard_normal(m)
        np.testing.assert_allclose(
            bl.circular_convolution(x, y), bl.circular_convolution_fft(x, y), atol=1e-10
        )


def test_circular_convolution_matches_coursework_example():
    """Convolving with an all-ones vector yields the (constant) total sum."""
    x = np.array([0, 0, 1, 2, 1, 0, 0], dtype=float)  # Assignment5_convolution.m
    y = np.ones(7)
    np.testing.assert_allclose(bl.circular_convolution(x, y), 4.0 * np.ones(7))
    np.testing.assert_allclose(bl.circular_convolution_fft(x, y), 4.0 * np.ones(7))


# ------------------------------ zscore / SNR --------------------------------
def test_zscore_identity():
    """Z-scored data has mean 0 and standard deviation 1."""
    rng = np.random.default_rng(0)
    z = bl.zscore(rng.normal(5.0, 3.0, size=1000))
    assert abs(z.mean()) < 1e-12
    assert abs(z.std() - 1.0) < 1e-12


def test_zscore_constant_raises():
    with pytest.raises(ValueError):
        bl.zscore(np.ones(10))


def test_snr_identities():
    """snr_power is the power ratio and snr_db is 10*log10 of it."""
    signal = np.array([2.0, -2.0, 2.0, -2.0])  # mean power 4
    noise = np.array([1.0, -1.0, 1.0, -1.0])  # mean power 1
    assert bl.snr_power(signal, noise) == pytest.approx(4.0)
    assert bl.snr_db(signal, noise) == pytest.approx(10.0 * np.log10(4.0))
    assert bl.snr_db(np.ones(8), np.ones(8)) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    s, n = rng.standard_normal(500), rng.standard_normal(500)
    assert bl.snr_db(s, n) == pytest.approx(10.0 * np.log10(bl.snr_power(s, n)))


# ------------------------------ spike detection ------------------------------
def test_mad_detector_recovers_planted_spikes_high_snr():
    """At high SNR the MAD detector recovers >95% of planted spikes."""
    n = int(10 * FS)
    rng = np.random.default_rng(7)
    truth = np.sort(rng.choice(np.arange(2000, n - 2000), size=60, replace=False))
    pairwise = np.abs(truth[:, None] - truth[None, :])
    np.fill_diagonal(pairwise, 10**9)  # ignore self-distance
    truth = truth[np.all(pairwise > 150, axis=1)]  # keep well-separated spikes
    template = biphasic_template(amp=8.0)  # peak = 8x noise std
    signal = plant_spikes(n, truth, template, noise_std=1.0, seed=11)
    detected = bl.detect_spikes(signal, k=5.0, fs=FS)
    recall = match_count(detected, truth, tol=15) / len(truth)
    assert recall > 0.95
    # Precision should also be high: almost no false positives.
    fp = sum(int(not np.any(np.abs(truth - d) <= 15)) for d in detected)
    assert fp / max(len(detected), 1) < 0.05


def test_mad_detector_beats_std_on_contaminated_noise():
    """On spike-contaminated noise the MAD threshold beats std/1.5 in F1.

    The spikes inflate the standard deviation, pushing the legacy
    std/1.5 threshold close to the noise floor: it fires constantly
    (precision collapse) while the robust MAD threshold stays selective.
    """
    n = int(10 * FS)
    # Dense 30 Hz-ish train with ~2 ms minimum spacing.
    isis_s = 0.002 + np.random.default_rng(3).exponential(1 / 30.0, 400)
    truth = np.cumsum((isis_s * FS).astype(int))
    truth = truth[(truth > 2000) & (truth < n - 2000)]
    template = biphasic_template(amp=6.0)  # peak = 6x noise std
    signal = plant_spikes(n, truth, template, noise_std=1.0, seed=5)

    det_mad = bl.detect_spikes(signal, k=5.0, fs=FS)
    det_std = bl.detect_spikes_std(signal, divisor=1.5, fs=FS)

    def pr(det: np.ndarray) -> tuple[float, float]:
        tp = match_count(det, truth, tol=15)
        return tp / max(len(det), 1), tp / len(truth)

    p_mad, r_mad = pr(det_mad)
    p_std, r_std = pr(det_std)
    f1_mad = 2 * p_mad * r_mad / max(p_mad + r_mad, 1e-12)
    f1_std = 2 * p_std * r_std / max(p_std + r_std, 1e-12)
    assert r_mad > 0.9
    assert p_mad > p_std
    assert f1_mad >= f1_std


def test_refractory_enforcement():
    """Of two spikes 1 ms apart, only the larger survives the refractory period."""
    n = int(1 * FS)
    template = biphasic_template()
    signal = np.zeros(n)
    big, small = 15000, 15030  # 1 ms apart, inside the 2 ms refractory
    for s, a in ((big, 8.0), (small, 5.0)):
        pre = len(template) // 2
        signal[s - pre : s - pre + len(template)] += a * template
    signal += 0.05 * np.random.default_rng(2).standard_normal(n)
    detected = bl.detect_spikes(signal, k=5.0, fs=FS, refractory_ms=2.0)
    assert len(detected) == 1
    assert abs(detected[0] - big) <= 15


def test_detected_spikes_respect_refractory():
    """No two detections are closer than the refractory period."""
    n = int(5 * FS)
    rng = np.random.default_rng(9)
    truth = np.sort(rng.choice(np.arange(1000, n - 1000), size=40, replace=False))
    template = biphasic_template(amp=8.0)
    signal = plant_spikes(n, truth, template, noise_std=1.0, seed=13)
    detected = bl.detect_spikes(signal, k=5.0, fs=FS, refractory_ms=2.0)
    min_gap_samples = 0.002 * FS
    assert np.all(np.diff(detected) > min_gap_samples)


# --------------------------- waveform extraction -----------------------------
def test_extract_waveforms_alignment():
    """Each extracted waveform is re-centered so its peak sits at `pre`."""
    n = 10_000
    template = biphasic_template(amp=5.0)
    pre = len(template) // 2
    signal = 0.05 * np.random.default_rng(4).standard_normal(n)
    spikes = [3000, 5000]
    for s in spikes:
        signal[s - pre : s - pre + len(template)] += template
    waveforms, kept = bl.extract_waveforms(signal, np.array(spikes), pre=30, post=30)
    assert waveforms.shape == (2, 61)
    peak_pos = np.argmax(np.abs(waveforms), axis=1)
    assert np.all(peak_pos == 30)  # aligned at sample `pre`


def test_extract_waveforms_drops_out_of_bounds():
    """Spikes whose window would leave the recording are skipped."""
    n = 10_000
    signal = 0.05 * np.random.default_rng(4).standard_normal(n)
    spikes = np.array([3000, 5000, 10, n - 5])  # two edge spikes too close
    waveforms, kept = bl.extract_waveforms(signal, spikes, pre=30, post=30, align=False)
    assert waveforms.shape == (2, 61)
    assert list(kept) == [3000, 5000]


# --------------------------- ISI / refractory / PSTH -------------------------
def test_isi_histogram_counts():
    times = np.cumsum([0.0, 0.5, 0.5, 1.5, 0.5])  # ISIs: 0.5, 0.5, 1.5, 0.5
    counts, edges = bl.isi_histogram(times, bin_width=1.0)
    assert counts.sum() == 4
    assert counts[0] == 3  # the three ISIs of 0.5 fall in the first bin [0, 1)
    assert counts[1] == 1  # the ISI of 1.5 falls in the second bin [1, 2]


def test_refractory_violations_counting():
    times = np.array([0.0, 0.5, 0.7, 3.0, 5.0])
    n_viol, idx = bl.refractory_violations(times, refractory=1.0)
    assert n_viol == 2  # spikes at 0.5 and 0.7 follow too closely
    assert list(times[idx]) == [0.5, 0.7]


def test_raster_psth():
    spikes = np.array([0.1, 1.1, 2.1, 1.3])
    events = np.array([0.0, 1.0, 2.0])
    trials, centers, psth = bl.raster_psth(spikes, events, window=(0.0, 0.5), bin_width=0.1)
    assert len(trials) == 3
    assert all(len(t) >= 1 for t in trials)  # each event sees its +0.1 s spike
    # The 0.1 s bin contains one spike per event, plus the 1.3 s spike for event 1... it falls at 0.3 s.
    assert psth[1] == pytest.approx(1.0)  # bin [0.1, 0.2): one spike per event
    assert psth[3] == pytest.approx(1.0 / 3.0)  # bin [0.3, 0.4): only one event


# ------------------------------ clustering -----------------------------------
def _make_waveforms(seed: int = 0, n_per: int = 60, noise: float = 0.1):
    """Three easy, well-separated waveform classes."""
    rng = np.random.default_rng(seed)
    t = np.arange(61) - 20
    templates = [
        -np.exp(-(t / 5.0) ** 2),
        np.exp(-((t - 5) / 4.0) ** 2),
        -np.exp(-((t + 5) / 8.0) ** 2) + 0.6 * np.exp(-((t - 12) / 6.0) ** 2),
    ]
    waves, truth = [], []
    for u, tpl in enumerate(templates):
        waves.append(tpl + noise * rng.standard_normal((n_per, 61)))
        truth += [u] * n_per
    return np.vstack(waves), np.array(truth)


def test_sort_spikes_pca_kmeans_easy_case():
    waves, truth = _make_waveforms()
    labels = bl.sort_spikes_pca_kmeans(waves, 3, random_state=0)
    assert adjusted_rand_score(truth, labels) > 0.8


def test_spike_spd_recovers_planted_clusters():
    """SpikeSPD recovers planted clusters on an easy synthetic case."""
    waves, truth = _make_waveforms()
    labels = spd.sort_spikes_spd(waves, 3, random_state=0)
    assert adjusted_rand_score(truth, labels) > 0.8


def test_spd_descriptor_is_positive_definite():
    w = np.sin(np.linspace(0, 3, 31))
    c = spd.spd_descriptor(w, eps=1e-3)
    assert np.all(np.linalg.eigvalsh(c) > 0)
    np.testing.assert_allclose(c, c.T, atol=1e-12)


def test_logm_expm_roundtrip():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 6))
    c = a @ a.T + np.eye(6)
    np.testing.assert_allclose(spd.expm_spd(spd.logm_spd(c)), c, atol=1e-8)


def test_log_euclidean_mean_identity_and_scale():
    """Log-Euclidean mean of scalars-as-matrices is the geometric mean."""
    a = np.array([[2.0]])
    b = np.array([[8.0]])
    m = spd.log_euclidean_mean(np.stack([a, b]))
    assert m[0, 0] == pytest.approx(4.0)
