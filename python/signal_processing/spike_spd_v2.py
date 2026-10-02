"""SpikeSPD v2: developed SPD-manifold spike sorting (2026 retrospective).

Part 3 development of the Part-2 prototype :mod:`spike_spd`. The v1
descriptor ``C = x x^T + eps * I`` is a rank-1 outer product: at low SNR
it is dominated by noise, and it discards the *temporal correlation*
structure of a waveform. SpikeSPD v2 replaces it with a **time-delay
embedding (TDE) covariance descriptor**: the waveform is unfolded into a
Hankel (trajectory) matrix and the descriptor is the covariance of its
delay vectors,

    H = hankel(x, dim, lag),     C = H H^T / n + eps * I,

which is full-rank, encodes local temporal autocorrelation, and averages
noise over ``n`` delay frames. Three further developments address the
failure modes measured in Part 2:

1. **Denoising**: zero-phase low-pass filtering of waveforms before
   embedding, so the descriptor reflects signal rather than broadband
   noise.
2. **Template-adaptive realignment**: waveforms are aligned by
   cross-correlation to a running cluster template, removing the
   extraction jitter that corrupts covariance descriptors.
3. **Overlap screening**: spikes whose residual after subtracting their
   cluster template is large are greedily refit as a sum of two cluster
   templates; accepted two-template fits are flagged as overlaps and
   excluded from cluster statistics.

Clustering reuses the exact log-Euclidean k-means of :mod:`spike_spd`.

References:
    Broomhead & King (1986). "Extracting qualitative dynamics from
    experimental data." Physica D 20 -- the time-delay (Hankel)
    embedding.
    Arsigny et al. (2006). "Log-Euclidean metrics for fast and simple
    calculus on diffusion tensors." MRM 56(2).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from spike_spd import (
    log_euclidean_kmeans,
    spd_descriptors,
    tangent_features,
)

__all__ = [
    "time_delay_embedding",
    "tde_descriptor",
    "tde_descriptors",
    "denoise_lowpass",
    "align_waveforms",
    "cluster_templates",
    "screen_overlaps",
    "fusion_features",
    "sort_spikes_spd_v2",
]

_MIN_EIG = 1e-12


# ---------------------------------------------------------------------------
# Time-delay embedding descriptors
# ---------------------------------------------------------------------------


def time_delay_embedding(x: np.ndarray, dim: int = 8, lag: int = 2) -> np.ndarray:
    """Hankel (trajectory) matrix of a 1-D waveform.

    Row ``i`` of the embedding is ``[x[i], x[i+lag], ..., x[i+(dim-1)*lag]]``;
    consecutive rows are the delay vectors whose covariance forms the TDE
    descriptor.

    Args:
        x: 1-D waveform of length L.
        dim: Embedding dimension (number of delayed coordinates).
        lag: Delay between coordinates, in samples.

    Returns:
        Hankel matrix of shape ``(n_frames, dim)`` with
        ``n_frames = L - (dim - 1) * lag``.

    Raises:
        ValueError: If the waveform is too short for the embedding.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_frames = x.size - (dim - 1) * lag
    if n_frames < 2:
        raise ValueError(
            f"waveform of length {x.size} too short for dim={dim}, lag={lag}."
        )
    idx = np.arange(n_frames)[:, None] + lag * np.arange(dim)[None, :]
    return x[idx]


def tde_descriptor(
    waveform: np.ndarray, dim: int = 8, lag: int = 2, eps: float = 1e-3
) -> np.ndarray:
    """Regularized time-delay-embedding covariance descriptor of a waveform.

    Computes ``C = H H^T / n_frames + eps * (tr / dim) * I`` where ``H`` is
    the Hankel matrix from :func:`time_delay_embedding`. Unlike the rank-1
    outer-product descriptor of v1, this descriptor is full-rank and
    captures the waveform's local autocorrelation structure.

    Args:
        waveform: 1-D spike waveform of length L.
        dim: Embedding dimension.
        lag: Delay between coordinates, in samples.
        eps: Relative ridge regularization (units of mean diagonal power).

    Returns:
        SPD matrix of shape ``(dim, dim)``.

    Raises:
        ValueError: If the waveform is empty, all zeros, or too short.
    """
    x = np.asarray(waveform, dtype=float).ravel()
    if x.size == 0 or not np.any(x):
        raise ValueError("waveform must be non-empty and not all zeros.")
    h = time_delay_embedding(x, dim=dim, lag=lag)
    c = (h.T @ h) / h.shape[0]
    ridge = eps * np.trace(c) / dim
    return c + ridge * np.eye(dim)


def tde_descriptors(
    waveforms: np.ndarray, dim: int = 8, lag: int = 2, eps: float = 1e-3
) -> np.ndarray:
    """Map a set of waveforms to TDE covariance descriptors.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        dim: Embedding dimension.
        lag: Delay between coordinates, in samples.
        eps: Relative ridge regularization.

    Returns:
        Array of SPD matrices of shape ``(n_spikes, dim, dim)``.
    """
    w = np.asarray(waveforms, dtype=float)
    if w.ndim != 2:
        raise ValueError("waveforms must be a 2-D array (n_spikes, L).")
    return np.stack([tde_descriptor(x, dim=dim, lag=lag, eps=eps) for x in w])


# ---------------------------------------------------------------------------
# Denoising and alignment
# ---------------------------------------------------------------------------


def denoise_lowpass(
    waveforms: np.ndarray, fs: float = 30000.0, cutoff: float = 6000.0, order: int = 3
) -> np.ndarray:
    """Zero-phase low-pass filter a set of spike waveforms.

    Butterworth filter applied forward-backward (:func:`scipy.signal.filtfilt`)
    so spike shapes are smoothed without phase shift; this suppresses the
    broadband noise that dominates covariance descriptors at low SNR.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        fs: Sampling rate in Hz.
        cutoff: Low-pass cutoff in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered waveforms, same shape.
    """
    w = np.asarray(waveforms, dtype=float)
    nyq = fs / 2.0
    if cutoff >= nyq:
        return w.copy()
    b, a = butter(order, cutoff / nyq, btype="low")
    padlen = min(3 * max(len(a), len(b)), w.shape[1] - 1)
    return filtfilt(b, a, w, axis=1, padlen=padlen)


def align_waveforms(
    waveforms: np.ndarray, template: np.ndarray, max_shift: int = 6
) -> np.ndarray:
    """Align waveforms to a template by cross-correlation (circular shifts).

    Each waveform is shifted by the lag (within ``+/- max_shift`` samples)
    that maximizes its correlation with ``template``. This removes the
    extraction jitter that corrupts covariance descriptors.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        template: 1-D reference waveform of length L.
        max_shift: Maximum absolute circular shift, in samples.

    Returns:
        Aligned waveforms, same shape.
    """
    w = np.asarray(waveforms, dtype=float)
    t = np.asarray(template, dtype=float).ravel()
    aligned = np.empty_like(w)
    for i, x in enumerate(w):
        best_shift, best_corr = 0, -np.inf
        for s in range(-max_shift, max_shift + 1):
            c = float(np.dot(np.roll(x, s), t))
            if c > best_corr:
                best_corr, best_shift = c, s
        aligned[i] = np.roll(x, best_shift)
    return aligned


# ---------------------------------------------------------------------------
# Template-adaptive clustering and overlap screening
# ---------------------------------------------------------------------------


def cluster_templates(waveforms: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Median waveform template per cluster label.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        labels: Integer cluster labels of shape ``(n_spikes,)``.

    Returns:
        Array of shape ``(n_clusters, L)`` with the per-cluster median
        waveform, ordered by ascending label.
    """
    w = np.asarray(waveforms, dtype=float)
    labels = np.asarray(labels)
    return np.stack([np.median(w[labels == j], axis=0) for j in np.unique(labels)])


def screen_overlaps(
    waveforms: np.ndarray,
    labels: np.ndarray,
    residual_ratio: float = 2.5,
    outlier_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag spikes better explained by a sum of two cluster templates.

    A spike is flagged only if (a) its single-template residual energy is
    an outlier within its cluster (above ``outlier_factor`` times the
    cluster median residual -- so drifting or noisy but ordinary spikes
    are kept) and (b) the best greedy two-template fit (own-cluster
    template plus a second template fitted by least squares on the
    residual) reduces the residual energy by at least
    ``residual_ratio``.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        labels: Cluster labels of shape ``(n_spikes,)``.
        residual_ratio: Required residual-energy improvement factor.
        outlier_factor: Required factor above the cluster-median
            residual for a spike to be eligible.

    Returns:
        Tuple ``(is_overlap, clean_labels)``: boolean mask of shape
        ``(n_spikes,)``, and labels with overlapped spikes set to ``-1``.
    """
    w = np.asarray(waveforms, dtype=float)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    templates = cluster_templates(waveforms, labels)
    tmap = {lab: k for k, lab in enumerate(uniq)}

    # Per-cluster median single-template residual energy.
    resid = np.empty(w.shape[0])
    for i, x in enumerate(w):
        own = templates[tmap[labels[i]]]
        resid[i] = float(np.sum((x - own) ** 2))
    med = {lab: np.median(resid[labels == lab]) for lab in uniq}

    is_overlap = np.zeros(w.shape[0], dtype=bool)
    for i, x in enumerate(w):
        own = templates[tmap[labels[i]]]
        r1 = x - own
        e1 = float(r1 @ r1)
        if e1 <= max(outlier_factor * med[labels[i]], _MIN_EIG):
            continue
        best_e2 = e1
        for k, t2 in enumerate(templates):
            if k == tmap[labels[i]]:
                continue
            a = float(r1 @ t2) / max(float(t2 @ t2), _MIN_EIG)
            e2 = float(np.sum((r1 - a * t2) ** 2))
            best_e2 = min(best_e2, e2)
        if best_e2 * residual_ratio < e1:
            is_overlap[i] = True
    clean = labels.copy()
    clean[is_overlap] = -1
    return is_overlap, clean


# ---------------------------------------------------------------------------
# Hybrid (fusion) descriptors
# ---------------------------------------------------------------------------


def fusion_features(
    waveforms: np.ndarray,
    dim: int = 16,
    lag: int = 3,
    eps: float = 1e-3,
    weight_tde: float = 1.0,
    weight_outer: float = 1.0,
) -> np.ndarray:
    """Tangent features of the block-diagonal fusion descriptor.

    The fusion descriptor is ``blkdiag(C_tde, C_outer)``: the TDE
    covariance (temporal-autocorrelation structure) stacked
    block-diagonally with the v1 outer-product descriptor (amplitude
    profile). Because ``log(blkdiag(A, B)) = blkdiag(log A, log B)``,
    log-Euclidean distances of the fusion descriptor satisfy
    ``d^2 = d_tde^2 + d_outer^2``, so clustering its concatenated tangent
    features is exactly log-Euclidean clustering on the product manifold.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        dim: TDE embedding dimension.
        lag: TDE lag in samples.
        eps: Relative ridge regularization of both descriptor blocks.
        weight_tde: Multiplicative weight on the TDE tangent features.
        weight_outer: Multiplicative weight on the outer-product tangent
            features.

    Returns:
        Feature matrix of shape
        ``(n_spikes, dim*(dim+1)/2 + L*(L+1)/2)``.
    """
    w = np.asarray(waveforms, dtype=float)
    f_tde = tangent_features(tde_descriptors(w, dim=dim, lag=lag, eps=eps))
    f_out = tangent_features(spd_descriptors(w, eps=eps))
    return np.hstack([weight_tde * f_tde, weight_outer * f_out])


# ---------------------------------------------------------------------------
# Full SpikeSPD v2 pipeline
# ---------------------------------------------------------------------------


def sort_spikes_spd_v2(
    waveforms: np.ndarray,
    n_clusters: int,
    fs: float = 30000.0,
    cutoff: float = 6000.0,
    dim: int = 8,
    lag: int = 2,
    eps: float = 1e-3,
    descriptor: str = "fusion",
    max_shift: int = 6,
    realign_rounds: int = 1,
    overlap_screening: bool = True,
    max_iter: int = 50,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster spike waveforms with the developed SpikeSPD v2 pipeline.

    Pipeline: (1) zero-phase low-pass denoising; (2) global alignment to the
    median waveform; (3) covariance descriptors; (4) log-Euclidean k-means;
    (5) ``realign_rounds`` rounds of template-adaptive realignment (align
    each cluster to its median template, re-embed, re-cluster);
    (6) optional overlap screening (:func:`screen_overlaps`).

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        n_clusters: Number of units to find.
        fs: Sampling rate in Hz (used for the low-pass cutoff).
        cutoff: Low-pass cutoff in Hz.
        dim: TDE embedding dimension.
        lag: TDE lag in samples.
        eps: Relative ridge regularization of the descriptors.
        descriptor: ``"fusion"`` (default) for the block-diagonal
            fusion descriptor, or ``"tde"`` for the pure TDE covariance descriptor
            (:func:`fusion_features`); with ``"fusion"``, clustering is
            k-means on the concatenated tangent features, which equals
            log-Euclidean k-means on the product manifold.
        max_shift: Maximum alignment shift in samples.
        realign_rounds: Number of template-adaptive realignment rounds.
        overlap_screening: Whether to flag overlapping spikes.
        max_iter: Maximum k-means iterations.
        random_state: Seed for the k-means initialization.

    Returns:
        Tuple ``(labels, is_overlap)``: integer cluster labels of shape
        ``(n_spikes,)`` (overlaps keep their best-fit label), and a boolean
        overlap mask.

    Raises:
        ValueError: If there are fewer waveforms than clusters, or for an
            unknown ``descriptor``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> t = np.linspace(-1, 2, 91)
        >>> a = np.exp(-t**2 / 0.1) + 0.05 * rng.standard_normal((40, 91))
        >>> b = -np.exp(-(t - 0.5) ** 2 / 0.2) + 0.05 * rng.standard_normal((40, 91))
        >>> labels, ov = sort_spikes_spd_v2(np.vstack([a, b]), 2, random_state=0)
        >>> len(np.unique(labels))
        2
    """
    from sklearn.cluster import KMeans

    w = np.asarray(waveforms, dtype=float)
    if w.ndim != 2:
        raise ValueError("waveforms must be a 2-D array (n_spikes, L).")
    if w.shape[0] < n_clusters:
        raise ValueError("Need at least as many waveforms as clusters.")
    if descriptor not in ("tde", "fusion"):
        raise ValueError(f"unknown descriptor: {descriptor!r}")

    def _cluster(x: np.ndarray) -> np.ndarray:
        """Cluster waveforms with the selected descriptor."""
        if descriptor == "tde":
            return log_euclidean_kmeans(
                tde_descriptors(x, dim=dim, lag=lag, eps=eps),
                n_clusters,
                max_iter=max_iter,
                random_state=random_state,
            )
        feats = fusion_features(x, dim=max(dim, 16), lag=max(lag, 3), eps=eps)
        return KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state
        ).fit_predict(feats)

    w = denoise_lowpass(w, fs=fs, cutoff=cutoff)
    w = align_waveforms(w, np.median(w, axis=0), max_shift=max_shift)

    labels = _cluster(w)
    for _ in range(realign_rounds):
        templates = cluster_templates(w, labels)
        w_aligned = w.copy()
        for j, lab in enumerate(np.unique(labels)):
            idx = np.flatnonzero(labels == lab)
            w_aligned[idx] = align_waveforms(w[idx], templates[j], max_shift=max_shift)
        w = w_aligned
        new_labels = _cluster(w)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    is_overlap = np.zeros(w.shape[0], dtype=bool)
    if overlap_screening:
        is_overlap, _ = screen_overlaps(w, labels)
    return labels, is_overlap
