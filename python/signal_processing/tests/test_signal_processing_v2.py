"""Tests for the developed SpikeSPD v2 pipeline (:mod:`spike_spd_v2`).

Run with::

    python3 -m pytest python/signal_processing/tests/test_signal_processing_v2.py -q
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import spike_spd  # noqa: E402
import spike_spd_v2 as v2  # noqa: E402


# ----------------------------- fixtures -------------------------------------
def _templates() -> np.ndarray:
    """Two distinct biphasic templates of length 91."""
    t = np.linspace(-1.0, 2.0, 91)
    a = -np.exp(-((t / 0.3) ** 2)) + 0.5 * np.exp(-(((t - 0.7) / 0.5) ** 2))
    b = -np.exp(-((t / 0.5) ** 2)) + 0.3 * np.exp(-(((t - 1.0) / 0.7) ** 2))
    return np.vstack([a / np.abs(a).max(), b / np.abs(b).max()])


def _waveforms(n: int = 60, noise: float = 0.08, jitter: int = 0, seed: int = 0):
    """Synthetic two-unit waveform set with ground-truth labels."""
    rng = np.random.default_rng(seed)
    tp = _templates()
    w, labels = [], []
    for u in range(2):
        x = tp[u] + noise * rng.standard_normal((n, tp.shape[1]))
        for s in range(n):
            if jitter:
                x[s] = np.roll(x[s], int(rng.integers(-jitter, jitter + 1)))
        w.append(x)
        labels.append(np.full(n, u))
    return np.vstack(w), np.concatenate(labels)


# --------------------------- TDE embedding ----------------------------------
def test_time_delay_embedding_shape():
    """Hankel matrix has (L - (dim-1)*lag) frames and dim columns."""
    h = v2.time_delay_embedding(np.arange(20.0), dim=4, lag=2)
    assert h.shape == (20 - 3 * 2, 4)
    assert np.allclose(h[0], [0, 2, 4, 6])


def test_tde_descriptor_is_spd_and_full_rank():
    """TDE descriptor is symmetric positive-definite and full rank."""
    x = _templates()[0] + 0.01 * np.random.default_rng(0).standard_normal(91)
    c = v2.tde_descriptor(x, dim=8, lag=2)
    assert np.allclose(c, c.T)
    eig = np.linalg.eigvalsh(c)
    assert np.all(eig > 0)
    assert np.linalg.matrix_rank(c) == 8


def test_tde_descriptor_captures_autocorrelation():
    """Smooth and white-noise waveforms of equal power get different descriptors."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 91)
    smooth = np.sin(t)
    white = rng.standard_normal(91)
    white *= np.linalg.norm(smooth) / np.linalg.norm(white)
    d_same = np.linalg.norm(
        v2.tde_descriptor(smooth) - v2.tde_descriptor(smooth + 0.01 * rng.standard_normal(91))
    )
    d_diff = np.linalg.norm(v2.tde_descriptor(smooth) - v2.tde_descriptor(white))
    assert d_diff > 5 * d_same


# --------------------------- denoise / align --------------------------------
def test_denoise_lowpass_reduces_hf_power():
    """Low-pass filtering attenuates high-frequency energy."""
    rng = np.random.default_rng(2)
    w = _templates()[0][None, :] + 0.5 * rng.standard_normal((1, 91))
    wf = v2.denoise_lowpass(w, fs=30000.0, cutoff=6000.0)
    spec_before = np.abs(np.fft.rfft(w[0]))[20:].sum()
    spec_after = np.abs(np.fft.rfft(wf[0]))[20:].sum()
    assert spec_after < 0.5 * spec_before


def test_align_waveforms_undoes_shifts():
    """Realignment recovers the template from randomly shifted copies."""
    rng = np.random.default_rng(3)
    tp = _templates()[0]
    w = np.stack([np.roll(tp, int(s)) for s in rng.integers(-4, 5, size=30)])
    wa = v2.align_waveforms(w, tp, max_shift=6)
    assert np.mean([np.linalg.norm(x - tp) for x in wa]) < 0.2 * np.mean(
        [np.linalg.norm(x - tp) for x in w]
    ) + 1e-9


# ------------------------------ clustering -----------------------------------
def test_v2_recovers_clusters_easy_case():
    """v2 clusters an easy two-unit set with ARI > 0.8."""
    w, labels = _waveforms(noise=0.05)
    pred, _ = v2.sort_spikes_spd_v2(w, 2, random_state=0)
    assert adjusted_rand_score(labels, pred) > 0.8


def test_v2_matches_v1_under_jitter():
    """With extraction jitter and amplitude drift, v2 (fusion) >= v1 - eps (ARI)."""
    rng = np.random.default_rng(7)
    w, labels = _waveforms(n=80, noise=0.10, jitter=3, seed=7)
    gains = 1.0 + 0.25 * np.sin(np.linspace(0, 3 * np.pi, len(w)))
    w = w * gains[:, None]  # amplitude drift
    lab_v1 = spike_spd.sort_spikes_spd(w, 2, random_state=0)
    lab_v2, _ = v2.sort_spikes_spd_v2(w, 2, random_state=0)
    assert adjusted_rand_score(labels, lab_v2) >= adjusted_rand_score(labels, lab_v1) - 0.05


def test_fusion_matches_tde_on_shape_coded_data():
    """Fusion descriptor clusters shape-coded data at least as well as TDE."""
    w, labels = _waveforms(noise=0.08)
    lab_tde, _ = v2.sort_spikes_spd_v2(w, 2, descriptor="tde", random_state=0)
    lab_f, _ = v2.sort_spikes_spd_v2(w, 2, descriptor="fusion", random_state=0)
    ari_tde = adjusted_rand_score(labels, lab_tde)
    ari_f = adjusted_rand_score(labels, lab_f)
    assert ari_f >= ari_tde - 0.05


# --------------------------- overlap screening -------------------------------
def test_screen_overlaps_flags_planted_overlaps():
    """Planted two-unit overlaps are flagged more often than clean spikes."""
    w, labels = _waveforms(n=80, noise=0.05)
    tp = _templates()
    rng = np.random.default_rng(5)
    idx = rng.choice(len(w), size=10, replace=False)
    for i in idx:
        w[i] = tp[0] + 0.8 * np.roll(tp[1], 5) + 0.05 * rng.standard_normal(91)
    pred = np.zeros(len(w), dtype=int)
    pred[81:] = 1  # rough truth: first half unit 0
    pred[:80] = 0
    is_ov, _ = v2.screen_overlaps(w, pred)
    planted_rate = is_ov[idx].mean()
    clean_rate = np.delete(is_ov, idx).mean()
    assert planted_rate > 3 * clean_rate


def test_screen_overlaps_low_false_positive_on_clean_data():
    """On clean well-separated data the false-flag rate stays below 15%."""
    w, labels = _waveforms(noise=0.05)
    is_ov, _ = v2.screen_overlaps(w, labels)
    assert is_ov.mean() < 0.15


# ------------------------------ input checks ---------------------------------
def test_invalid_inputs_raise():
    """Bad shapes and unknown descriptors raise ValueError."""
    with pytest.raises(ValueError):
        v2.sort_spikes_spd_v2(np.zeros((2, 91)), 3)
    with pytest.raises(ValueError):
        v2.sort_spikes_spd_v2(np.zeros((10, 91)), 2, descriptor="nope")
    with pytest.raises(ValueError):
        v2.tde_descriptor(np.zeros(10))
    with pytest.raises(ValueError):
        v2.time_delay_embedding(np.zeros(5), dim=8, lag=2)
