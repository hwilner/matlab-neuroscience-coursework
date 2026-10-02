"""Baseline neuroscience signal-processing utilities.

Clean Python/NumPy ports of the MATLAB coursework in
``src/signal-processing/`` and ``src/machine-learning/ex1.m``:
normalization and SNR, circular convolution (direct DFT definition and
FFT), threshold-based spike detection with a refractory period,
waveform extraction with peak alignment, ISI statistics, refractory
violation counting, raster/PSTH computation, and a classic
PCA + k-means spike-sorting baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

__all__ = [
    "zscore",
    "snr_power",
    "snr_db",
    "circular_convolution",
    "circular_convolution_fft",
    "detect_spikes",
    "detect_spikes_std",
    "extract_waveforms",
    "isi_histogram",
    "refractory_violations",
    "raster_psth",
    "sort_spikes_pca_kmeans",
]


# ---------------------------------------------------------------------------
# Normalization and SNR (port of ex3_snr.m)
# ---------------------------------------------------------------------------


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardize an array to zero mean and unit standard deviation.

    Uses the population standard deviation (ddof=0), matching MATLAB's
    ``zscore``.

    Args:
        x: Input array.

    Returns:
        Array of the same shape as ``x`` with mean 0 and standard
        deviation 1.

    Raises:
        ValueError: If ``x`` has zero standard deviation.

    Example:
        >>> z = zscore(np.array([1.0, 2.0, 3.0]))
        >>> float(np.abs(z.mean())) < 1e-12
        True
    """
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0.0:
        raise ValueError("zscore is undefined for constant input (std == 0).")
    return (x - x.mean()) / sd


def snr_power(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute the signal-to-noise power ratio.

    Defined as mean(signal**2) / mean(noise**2), i.e. the ratio of mean
    square powers. For zero-mean inputs this equals the ratio of
    variances, matching the coursework definition
    ``Ssignal**2 / Snoise**2``.

    Args:
        signal: Signal samples (or signal amplitude estimate).
        noise: Noise samples of compatible shape.

    Returns:
        The linear power SNR (>= 0).

    Raises:
        ValueError: If the noise power is zero.
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    p_noise = float(np.mean(noise**2))
    if p_noise == 0.0:
        raise ValueError("Noise power is zero; SNR is undefined.")
    return float(np.mean(signal**2)) / p_noise


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute the signal-to-noise ratio in decibels.

    Defined as ``10 * log10(snr_power(signal, noise))``.

    Args:
        signal: Signal samples.
        noise: Noise samples of compatible shape.

    Returns:
        SNR in dB.

    Example:
        >>> round(snr_db(np.ones(4), np.ones(4)), 6)
        0.0
    """
    return 10.0 * np.log10(snr_power(signal, noise))


# ---------------------------------------------------------------------------
# Circular convolution (port of Assignment5_convolution.m)
# ---------------------------------------------------------------------------


def _pad_to_equal_length(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Zero-pad two vectors to a common length ``n = max(len(x), len(y))``."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = max(x.size, y.size)
    x = np.pad(x, (0, n - x.size))
    y = np.pad(y, (0, n - y.size))
    return x, y, n


def circular_convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Circular convolution via the direct DFT-definition double loop.

    Both inputs are zero-padded to ``n = max(len(x), len(y))`` and the
    result is ``yh[i] = sum_j x[j] * y[(i - j) mod n]``, matching the
    MATLAB coursework loop and ``cconv(x, y, n)``.

    Args:
        x: First input vector.
        y: Second input vector.

    Returns:
        Length-``n`` circular convolution of the padded inputs.

    Example:
        >>> circular_convolution([0, 0, 1, 2, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1])
        array([4., 4., 4., 4., 4., 4., 4.])
    """
    x, y, n = _pad_to_equal_length(x, y)
    yh = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            k = (i - j) % n
            acc += x[j] * y[k]
        yh[i] = acc
    return yh


def circular_convolution_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Circular convolution via the convolution theorem (FFT).

    Computes ``ifft(fft(x) * fft(y))`` after zero-padding both inputs to
    the common length ``n = max(len(x), len(y))``. Numerically identical
    (up to floating-point error) to :func:`circular_convolution`.

    Args:
        x: First input vector.
        y: Second input vector.

    Returns:
        Length-``n`` circular convolution of the padded inputs.
    """
    x, y, n = _pad_to_equal_length(x, y)
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real


# ---------------------------------------------------------------------------
# Spike detection (robust MAD port of the threshold detector in ex1.m)
# ---------------------------------------------------------------------------


def _suprathreshold_candidates(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    """Return the index of the peak magnitude within each supra-threshold segment."""
    above = magnitude > threshold
    edges = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if above[0]:
        starts = np.concatenate(([0], starts))
    if above[-1]:
        ends = np.concatenate((ends, [above.size]))
    return np.array([s + np.argmax(magnitude[s:e]) for s, e in zip(starts, ends)], dtype=int)


def _enforce_refractory(candidates: np.ndarray, magnitude: np.ndarray, min_gap: int) -> np.ndarray:
    """Greedily keep the largest-amplitude candidate in each refractory window.

    Candidates are considered in decreasing magnitude; a candidate is
    accepted only if no already-accepted spike lies within ``min_gap``
    samples (tracked with a boolean blocked-sample mask, so the pass is
    linear in the number of candidates). The result is returned in
    increasing time order.
    """
    if candidates.size == 0:
        return candidates
    blocked = np.zeros(magnitude.size, dtype=bool)
    order = np.argsort(magnitude[candidates])[::-1]
    accepted: list[int] = []
    for idx in candidates[order]:
        if not blocked[idx]:
            accepted.append(int(idx))
            lo = max(0, idx - min_gap)
            hi = min(magnitude.size, idx + min_gap + 1)
            blocked[lo:hi] = True
    return np.array(sorted(accepted), dtype=int)


def detect_spikes(
    signal: np.ndarray,
    k: float = 5.0,
    fs: float = 30000.0,
    refractory_ms: float = 2.0,
) -> np.ndarray:
    """Detect spikes with a robust MAD-based threshold on |signal|.

    The noise level is estimated robustly as
    ``sigma = median(|x|) / 0.6745`` (Quiroga et al., 2004), and the
    detection threshold is ``k * sigma``. Within each supra-threshold
    segment the sample of maximum magnitude is taken as the spike time,
    and a refractory period is enforced by keeping the largest candidate
    within each refractory window.

    Args:
        signal: 1-D extracellular recording.
        k: Threshold multiplier on the robust sigma estimate.
        fs: Sampling rate in Hz (used to convert the refractory period
            to samples).
        refractory_ms: Refractory period in milliseconds; two detections
            closer than this are merged into one (largest magnitude).

    Returns:
        Sorted array of detected spike sample indices.

    Raises:
        ValueError: If the robust sigma estimate is zero.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> x = 0.1 * rng.standard_normal(30000)
        >>> x[15000] = 1.0
        >>> detect_spikes(x, k=5.0)
        array([15000])
    """
    x = np.asarray(signal, dtype=float).ravel()
    sigma = np.median(np.abs(x)) / 0.6745
    if sigma == 0.0:
        raise ValueError("Robust sigma estimate is zero; cannot set a threshold.")
    threshold = k * sigma
    magnitude = np.abs(x)
    candidates = _suprathreshold_candidates(magnitude, threshold)
    min_gap = int(round(refractory_ms * fs / 1000.0))
    return _enforce_refractory(candidates, magnitude, min_gap)


def detect_spikes_std(
    signal: np.ndarray,
    divisor: float = 1.5,
    fs: float = 30000.0,
    refractory_ms: float = 2.0,
) -> np.ndarray:
    """Detect spikes with the legacy coursework threshold ``std(x) / divisor``.

    Direct port of the detector in ``src/machine-learning/ex1.m``
    (``thr = std(d)/1.5``), generalized to |signal| and combined with
    the same segment-peak picking and refractory enforcement as
    :func:`detect_spikes` so the two detectors differ only in the
    threshold estimate. Because the standard deviation is inflated by
    the spikes themselves, this threshold is not robust on
    spike-contaminated recordings.

    Args:
        signal: 1-D extracellular recording.
        divisor: Value dividing the signal standard deviation to obtain
            the threshold (1.5 in the original MATLAB code).
        fs: Sampling rate in Hz.
        refractory_ms: Refractory period in milliseconds.

    Returns:
        Sorted array of detected spike sample indices.

    Raises:
        ValueError: If the signal standard deviation is zero.
    """
    x = np.asarray(signal, dtype=float).ravel()
    sd = x.std()
    if sd == 0.0:
        raise ValueError("Signal standard deviation is zero; cannot set a threshold.")
    threshold = sd / divisor
    magnitude = np.abs(x)
    candidates = _suprathreshold_candidates(magnitude, threshold)
    min_gap = int(round(refractory_ms * fs / 1000.0))
    return _enforce_refractory(candidates, magnitude, min_gap)


# ---------------------------------------------------------------------------
# Waveform extraction (port of the waveform loop in ex1.m)
# ---------------------------------------------------------------------------


def extract_waveforms(
    signal: np.ndarray,
    spike_idx: np.ndarray,
    pre: int,
    post: int,
    align: bool = True,
    align_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract spike waveforms around each detected index, with peak alignment.

    Each waveform spans ``[center - pre, center + post]`` (inclusive).
    With ``align=True`` the center is refined to the sample of maximum
    |signal| within ``center +/- align_window``, so waveform peaks line
    up at sample ``pre`` -- the standard alignment step of spike
    sorting. Spikes whose window would fall outside the recording are
    skipped.

    Args:
        signal: 1-D recording.
        spike_idx: Detected spike sample indices.
        pre: Number of samples before the peak.
        post: Number of samples after the peak.
        align: Whether to re-center each waveform on its local
            |signal| maximum.
        align_window: Half-width (in samples) of the alignment search
            window. Defaults to ``pre``.

    Returns:
        Tuple ``(waveforms, kept_idx)`` where ``waveforms`` has shape
        ``(n_kept, pre + post + 1)`` and ``kept_idx`` are the (possibly
        re-centered) sample indices of the kept spikes.

    Raises:
        ValueError: If ``pre`` or ``post`` are negative.
    """
    if pre < 0 or post < 0:
        raise ValueError("pre and post must be non-negative.")
    x = np.asarray(signal, dtype=float).ravel()
    spike_idx = np.asarray(spike_idx, dtype=int).ravel()
    if align_window is None:
        align_window = pre
    length = pre + post + 1
    waveforms = []
    kept = []
    for idx in spike_idx:
        center = int(idx)
        if align:
            lo = max(0, center - align_window)
            hi = min(x.size, center + align_window + 1)
            center = lo + int(np.argmax(np.abs(x[lo:hi])))
        if center - pre < 0 or center + post + 1 > x.size:
            continue
        waveforms.append(x[center - pre : center + post + 1])
        kept.append(center)
    if not waveforms:
        return np.empty((0, length)), np.empty(0, dtype=int)
    return np.vstack(waveforms), np.array(kept, dtype=int)


# ---------------------------------------------------------------------------
# ISI statistics, refractory violations, raster/PSTH (port of ex1.m)
# ---------------------------------------------------------------------------


def isi_histogram(
    spike_times: np.ndarray, bin_width: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the inter-spike-interval (ISI) histogram of a spike train.

    Args:
        spike_times: Spike times (any consistent time unit; sorted
            internally).
        bin_width: Histogram bin width in the same units. Defaults to
            ``max(isi) / 50``, or 1.0 for empty trains.

    Returns:
        Tuple ``(counts, bin_edges)`` as produced by
        ``numpy.histogram``.

    Example:
        >>> counts, edges = isi_histogram(np.cumsum(np.ones(10)))
        >>> int(counts.sum())
        9
    """
    times = np.sort(np.asarray(spike_times, dtype=float).ravel())
    isis = np.diff(times)
    if isis.size == 0:
        return np.zeros(0, dtype=int), np.zeros(1)
    if bin_width is None:
        bin_width = max(isis.max() / 50.0, np.finfo(float).eps)
    n_bins = max(1, int(np.ceil(isis.max() / bin_width)))
    return np.histogram(isis, bins=n_bins, range=(0.0, n_bins * bin_width))


def refractory_violations(spike_times: np.ndarray, refractory: float) -> tuple[int, np.ndarray]:
    """Count spikes occurring within the refractory period of the previous spike.

    Ports the coursework estimate
    ``refSpike = length(ISI) - length(diff(twv, 2))`` by explicitly
    counting ISIs shorter than ``refractory``.

    Args:
        spike_times: Spike times (sorted internally).
        refractory: Refractory period in the same time units.

    Returns:
        Tuple ``(n_violations, violation_idx)`` where ``violation_idx``
        holds the sorted-train indices of the offending (second) spikes.
    """
    times = np.sort(np.asarray(spike_times, dtype=float).ravel())
    isis = np.diff(times)
    bad = np.flatnonzero(isis < refractory) + 1
    return int(bad.size), bad


def raster_psth(
    spike_times: np.ndarray,
    events: np.ndarray,
    window: tuple[float, float] = (-0.5, 0.5),
    bin_width: float = 0.001,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build an event-aligned raster and peri-stimulus time histogram (PSTH).

    Ports the raster/PSTH section of ``src/machine-learning/ex1.m``:
    for every event, spikes within ``event + window`` are collected
    relative to the event; the PSTH is the mean spike count per bin per
    event.

    Args:
        spike_times: Spike times in seconds.
        events: Event (stimulus) onset times in seconds.
        window: ``(start, end)`` peri-event window in seconds, relative
            to each event.
        bin_width: PSTH bin width in seconds.

    Returns:
        Tuple ``(trials, bin_centers, psth)``:

        - ``trials``: list with one array of event-relative spike times
          per event (the raster).
        - ``bin_centers``: bin centers in seconds relative to the event.
        - ``psth``: mean spike count per bin averaged over events.

    Raises:
        ValueError: If ``window`` is not ``(start < end)``.
    """
    start, end = float(window[0]), float(window[1])
    if not start < end:
        raise ValueError("window must satisfy start < end.")
    times = np.sort(np.asarray(spike_times, dtype=float).ravel())
    events = np.asarray(events, dtype=float).ravel()
    edges = np.arange(start, end + bin_width, bin_width)
    bin_centers = edges[:-1] + bin_width / 2.0
    trials: list[np.ndarray] = []
    counts = np.zeros(bin_centers.size)
    for ev in events:
        rel = times[(times >= ev + start) & (times < ev + end)] - ev
        trials.append(rel)
        counts += np.histogram(rel, bins=edges)[0]
    psth = counts / max(len(events), 1)
    return trials, bin_centers, psth


# ---------------------------------------------------------------------------
# Classic PCA + k-means spike sorting baseline
# ---------------------------------------------------------------------------


def sort_spikes_pca_kmeans(
    waveforms: np.ndarray,
    n_clusters: int,
    n_components: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """Sort spike waveforms with the classic PCA(3) + k-means baseline.

    Waveforms are projected onto their first ``n_components`` principal
    components and clustered with k-means (Lloyd's algorithm, multiple
    restarts, deterministic seed).

    Args:
        waveforms: Array of shape ``(n_spikes, n_samples)``.
        n_clusters: Number of clusters (units) to find.
        n_components: Number of PCA features (default 3, the classic
            choice).
        random_state: Seed for k-means initialization (deterministic).

    Returns:
        Integer cluster labels of shape ``(n_spikes,)``.

    Raises:
        ValueError: If there are fewer waveforms than clusters.
    """
    w = np.asarray(waveforms, dtype=float)
    if w.ndim != 2:
        raise ValueError("waveforms must be a 2-D array (n_spikes, n_samples).")
    if w.shape[0] < n_clusters:
        raise ValueError("Need at least as many waveforms as clusters.")
    n_components = min(n_components, w.shape[0], w.shape[1])
    feats = PCA(n_components=n_components).fit_transform(w)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return km.fit_predict(feats)
