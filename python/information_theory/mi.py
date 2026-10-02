"""Mutual-information estimation with small-sample bias corrections.

Ports the MATLAB coursework in ``src/information-theory/`` (``joint.m``,
``MINF.m`` and ``calculate_spike_direction_MI.m``) to Python and adds the
novel research component of this package: bias-corrected mutual-information
(MI) estimation for ranking neurons from small numbers of trials.

The plugin (maximum-likelihood) MI estimator is upward biased at finite
sample size ``N``: each entropy term carries an expected bias of roughly
``(m - 1) / (2 N)`` nats, where ``m`` is the number of bins with nonzero
probability.  Two classical corrections are implemented here:

* Miller-Madow: add the first-order ``(m - 1) / (2 N)`` correction to each
  entropy term analytically, using the observed nonzero-cell counts.
* Shuffle correction: estimate the bias empirically as the mean plugin MI
  under label-permutation surrogates and subtract it.

Both corrections shrink to zero as ``N`` grows, so the corrected estimators
converge to the plugin estimator at large ``N``.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "entropy",
    "joint_histogram",
    "mutual_information_plugin",
    "mutual_information_miller_madow",
    "mutual_information_shuffle_corrected",
    "neuron_ranking",
    "MI_METHODS",
]

#: Estimator names accepted by :func:`neuron_ranking`.
MI_METHODS: Tuple[str, ...] = ("plugin", "miller_madow", "shuffle")

#: Method-name type alias.
Method = Literal["plugin", "miller_madow", "shuffle"]

_NATS_TO_BITS = 1.0 / np.log(2.0)


def entropy(p: ArrayLike) -> float:
    """Compute the Shannon entropy of a probability vector, in bits.

    Args:
        p: One-dimensional array of probabilities. Entries equal to zero are
            ignored (the ``0 * log(0)`` convention). The vector is
            renormalized if it does not sum exactly to one.

    Returns:
        The entropy ``H = -sum(p * log2(p))`` in bits.

    Raises:
        ValueError: If any entry is negative or the sum is not positive.
    """
    p = np.asarray(p, dtype=float).ravel()
    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative.")
    total = p.sum()
    if total <= 0:
        raise ValueError("Probability vector must have a positive sum.")
    p = p / total
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def joint_histogram(
    x: ArrayLike, y: ArrayLike, normalize: bool = True
) -> NDArray[np.float64]:
    """Build the joint distribution table of two discrete variables.

    Port of ``joint.m`` from the MATLAB coursework, generalized to arbitrary
    discrete labels (not only zero-based integers): distinct observed values
    are mapped to consecutive bins via :func:`numpy.unique`.

    Args:
        x: Samples of the first variable, shape ``(n_samples,)``.
        y: Samples of the second variable, shape ``(n_samples,)``.
        normalize: If True (default), return joint probabilities that sum to
            one; if False, return raw integer counts.

    Returns:
        Joint table ``R`` of shape ``(n_y_values, n_x_values)`` with
        ``R[j, i] = P(y == y_values[j], x == x_values[i])`` (or the
        corresponding count when ``normalize=False``).

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths or are empty.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of samples.")
    if x.size == 0:
        raise ValueError("x and y must be non-empty.")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    nx = int(xi.max()) + 1
    ny = int(yi.max()) + 1
    counts = np.bincount(yi * nx + xi, minlength=nx * ny).reshape(ny, nx)
    if not normalize:
        return counts.astype(float)
    return counts / counts.sum()


def mutual_information_plugin(R: ArrayLike) -> float:
    """Plugin (maximum-likelihood) mutual information of a joint table.

    Port of ``MINF.m``: computes ``I(X; Y) = sum p(x, y) log2(p(x, y) /
    (p(x) p(y)))`` directly from the empirical joint table.

    Args:
        R: Non-negative joint table, either normalized (rows index ``y``,
            columns index ``x``) or raw counts; it is renormalized
            internally.

    Returns:
        The plugin mutual-information estimate in bits. This estimator is
        upward biased at finite sample size.

    Raises:
        ValueError: If the table has a non-positive sum.
    """
    R = np.asarray(R, dtype=float)
    total = R.sum()
    if total <= 0:
        raise ValueError("Joint table must have a positive sum.")
    p = R / total
    px = p.sum(axis=0)
    py = p.sum(axis=1)
    denom = np.outer(py, px)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / denom[nz])))


def mutual_information_miller_madow(
    R: ArrayLike, n_samples: Optional[int] = None
) -> float:
    """Miller-Madow bias-corrected mutual information of a joint table.

    Applies the first-order ``(m - 1) / (2 N)`` correction (in nats) to each
    entropy term of ``I(X; Y) = H(X) + H(Y) - H(X, Y)``, using the observed
    nonzero-cell counts ``m``:

    ``I_MM = I_plugin + [(m_x - 1) + (m_y - 1) - (m_xy - 1)] / (2 N ln 2)``

    in bits. The correction vanishes as ``N`` grows, so this estimator
    converges to the plugin estimator for large samples.

    Args:
        R: Joint table of raw counts (preferred) or normalized probabilities.
            Rows index ``y``, columns index ``x``.
        n_samples: Total number of samples ``N``. If None (default), the
            table is assumed to contain raw counts and ``N`` is taken as its
            sum.

    Returns:
        The Miller-Madow corrected mutual-information estimate in bits. The
        value can be slightly negative when the true MI is zero.

    Raises:
        ValueError: If ``n_samples`` is not positive or the table is empty.
    """
    R = np.asarray(R, dtype=float)
    if n_samples is None:
        n_samples = int(round(R.sum()))
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    mi = mutual_information_plugin(R)
    m_x = int(np.count_nonzero(R.sum(axis=0)))
    m_y = int(np.count_nonzero(R.sum(axis=1)))
    m_xy = int(np.count_nonzero(R))
    correction = ((m_x - 1) + (m_y - 1) - (m_xy - 1)) * _NATS_TO_BITS / (2.0 * n_samples)
    return float(mi + correction)


def _plugin_mi_from_inverse(
    xi: NDArray[np.intp], yi: NDArray[np.intp], nx: int, ny: int
) -> float:
    """Plugin MI from integer label codes (internal helper).

    Args:
        xi: Integer codes of the first variable, shape ``(n_samples,)``.
        yi: Integer codes of the second variable, shape ``(n_samples,)``.
        nx: Number of distinct ``x`` values.
        ny: Number of distinct ``y`` values.

    Returns:
        Plugin mutual information in bits.
    """
    counts = np.bincount(yi * nx + xi, minlength=nx * ny).reshape(ny, nx)
    return mutual_information_plugin(counts)


def _joint_counts_batch(
    xi_batch: NDArray[np.intp], yi_batch: NDArray[np.intp], nx: int, ny: int
) -> NDArray[np.int64]:
    """Joint count tables for row-aligned batches of two coded variables.

    Args:
        xi_batch: Integer codes of the first variables, ``(R, n_samples)``.
        yi_batch: Integer codes of the second variables, ``(R, n_samples)``;
            row ``k`` is paired with row ``k`` of ``xi_batch``.
        nx: Number of distinct ``x`` values (shared by all rows).
        ny: Number of distinct ``y`` values (shared by all rows).

    Returns:
        Count tables of shape ``(R, ny, nx)``.
    """
    R, n = yi_batch.shape
    offsets = np.arange(R, dtype=np.intp)[:, None] * (nx * ny)
    flat = (offsets + yi_batch * nx + xi_batch).ravel()
    return np.bincount(flat, minlength=R * nx * ny).reshape(R, ny, nx)


def _plugin_mi_from_counts_batch(
    counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Plugin MI of each joint count table in a batch (internal helper).

    Args:
        counts: Joint count tables, shape ``(R, ny, nx)``.

    Returns:
        Array of ``R`` plugin mutual-information values in bits.
    """
    n = counts.sum(axis=(1, 2))
    p = counts / n[:, None, None]

    def _h(values: NDArray[np.float64], axis: tuple) -> NDArray[np.float64]:
        """Shannon entropy along ``axis`` of a probability array."""
        log_v = np.zeros_like(values)
        np.log2(values, out=log_v, where=values > 0)
        return -(values * log_v).sum(axis=axis)

    # I(X; Y) = H(X) + H(Y) - H(X, Y), avoiding an explicit outer product.
    return _h(p.sum(axis=1), (1,)) + _h(p.sum(axis=2), (1,)) - _h(p, (1, 2))


def _miller_madow_from_counts_batch(
    counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Miller-Madow corrected MI of each joint count table in a batch.

    Args:
        counts: Raw joint count tables, shape ``(R, ny, nx)``.

    Returns:
        Array of ``R`` Miller-Madow corrected MI values in bits.
    """
    n = counts.sum(axis=(1, 2)).astype(float)
    mi = _plugin_mi_from_counts_batch(counts)
    m_xy = np.count_nonzero(counts, axis=(1, 2))
    m_x = np.count_nonzero(counts.sum(axis=1) > 0, axis=1)
    m_y = np.count_nonzero(counts.sum(axis=2) > 0, axis=1)
    correction = ((m_x - 1) + (m_y - 1) - (m_xy - 1)) * _NATS_TO_BITS / (2.0 * n)
    return mi + correction


def _plugin_mi_batch(
    xi: NDArray[np.intp], yi_batch: NDArray[np.intp], nx: int, ny: int
) -> NDArray[np.float64]:
    """Vectorized plugin MI for several ``y`` labelings sharing one ``x``.

    Args:
        xi: Integer codes of the fixed variable, shape ``(n_samples,)``.
        yi_batch: Integer codes of ``B`` labelings of the second variable,
            shape ``(B, n_samples)``.
        nx: Number of distinct ``x`` values.
        ny: Number of distinct ``y`` values.

    Returns:
        Array of ``B`` plugin mutual-information values in bits.
    """
    xi_batch = np.broadcast_to(xi, yi_batch.shape)
    counts = _joint_counts_batch(xi_batch, yi_batch, nx, ny)
    return _plugin_mi_from_counts_batch(counts)


def mutual_information_shuffle_corrected(
    x: ArrayLike, y: ArrayLike, B: int = 200, seed: Optional[int] = None
) -> float:
    """Shuffle-corrected mutual information between two discrete variables.

    Estimates the finite-sample bias empirically: the mean plugin MI under
    ``B`` label-permutation surrogates (which destroy any genuine dependence
    while preserving the marginals) is subtracted from the observed plugin
    MI. Like Miller-Madow, the correction vanishes at large ``N``.

    Args:
        x: Samples of the first variable, shape ``(n_samples,)``.
        y: Samples of the second variable, shape ``(n_samples,)``.
        B: Number of permutation surrogates (default 200).
        seed: Seed for the surrogate permutations, for deterministic results.

    Returns:
        The shuffle-corrected mutual-information estimate in bits. The value
        can be slightly negative when the true MI is zero.

    Raises:
        ValueError: If inputs are invalid or ``B`` is not positive.
    """
    if B <= 0:
        raise ValueError("B must be positive.")
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape or x.size == 0:
        raise ValueError("x and y must be non-empty and equally long.")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    nx = int(xi.max()) + 1
    ny = int(yi.max()) + 1
    mi_observed = _plugin_mi_from_inverse(xi, yi, nx, ny)
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((B, x.size)), axis=1)
    mi_surrogates = _plugin_mi_batch(xi, yi[perms], nx, ny)
    return float(mi_observed - mi_surrogates.mean())


def neuron_ranking(
    responses: ArrayLike,
    stimuli: ArrayLike,
    method: Method = "plugin",
    B: int = 200,
    seed: Optional[int] = None,
) -> Tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Rank neurons by mutual information between response and stimulus.

    Port of ``calculate_spike_direction_MI.m``: each neuron's discrete
    response is jointly histogrammed against the stimulus class and scored
    by MI. Adds the bias-corrected estimators of this package, which give
    more faithful rankings at small trial counts.

    Args:
        responses: Discrete responses, shape ``(n_neurons, n_trials)`` (a
            one-dimensional array is treated as a single neuron).
        stimuli: Stimulus class labels, shape ``(n_trials,)``.
        method: One of ``"plugin"``, ``"miller_madow"`` or ``"shuffle"``.
        B: Number of surrogates when ``method="shuffle"``.
        seed: Base seed for deterministic shuffle corrections.

    Returns:
        A tuple ``(ranking, mi_values)`` where ``ranking`` contains neuron
            indices sorted by decreasing estimated MI (the most informative
            neuron first) and ``mi_values`` the per-neuron MI estimates in
            bits.

    Raises:
        ValueError: If ``method`` is unknown or shapes mismatch.
    """
    responses = np.asarray(responses)
    if responses.ndim == 1:
        responses = responses[None, :]
    stimuli = np.asarray(stimuli).ravel()
    if responses.shape[1] != stimuli.size:
        raise ValueError("responses and stimuli must agree on the trial axis.")
    if method not in MI_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {MI_METHODS}.")
    n_neurons = responses.shape[0]
    mi = np.zeros(n_neurons)

    _, si = np.unique(stimuli, return_inverse=True)
    ns = int(si.max()) + 1
    codes = [np.unique(responses[i], return_inverse=True)[1] for i in range(n_neurons)]
    sizes = [int(c.max()) + 1 for c in codes]

    si_perm = None
    if method == "shuffle":
        # One shared set of stimulus-label permutations for all neurons:
        # deterministic for a fixed seed and much faster than per-neuron
        # surrogate generation.
        rng = np.random.default_rng(seed)
        si_perm = si[np.argsort(rng.random((B, stimuli.size)), axis=1)]

    # All neurons are scored in a single vectorized batch call. Tables are
    # padded to the largest bin count; empty padded rows/columns never
    # occur, so plugin MI and the Miller-Madow nonzero-cell counts are
    # unaffected.
    nr_max = max(sizes)
    xb = np.stack(codes)
    if method == "shuffle":
        observed = _plugin_mi_from_counts_batch(
            _joint_counts_batch(np.tile(si, (n_neurons, 1)), xb, ns, nr_max)
        )
        counts_surr = _joint_counts_batch(
            np.tile(si_perm, (n_neurons, 1)),
            np.repeat(xb, B, axis=0),
            ns,
            nr_max,
        )
        surrogate = _plugin_mi_from_counts_batch(counts_surr).reshape(n_neurons, B)
        mi = observed - surrogate.mean(axis=1)
    else:
        counts = _joint_counts_batch(np.tile(si, (n_neurons, 1)), xb, ns, nr_max)
        if method == "plugin":
            mi = _plugin_mi_from_counts_batch(counts)
        else:
            mi = _miller_madow_from_counts_batch(counts)
    ranking = np.argsort(-mi, kind="stable")
    return ranking, mi
