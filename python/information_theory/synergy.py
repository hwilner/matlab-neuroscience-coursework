"""Synergy / redundancy analysis between pairs of neurons.

Port of the synergy part of ``ass6.m`` (``src/signal-processing/``): for
every pair of neurons ``(i, j)`` the synergy-redundancy index

``SR(i, j) = I(R_i, R_j; S) - I(R_i; S) - I(R_j; S)``

is computed against the stimulus ``S``. Negative values indicate
redundancy (the pair carries overlapping stimulus information), positive
values synergy (the pair carries more than the sum of its parts).

In addition to the plugin estimator of the original assignment, this
module offers the Miller-Madow and shuffle bias corrections from
:mod:`information_theory.mi`, which matter here because the joint
variable ``(R_i, R_j)`` has many bins and its plugin MI is strongly
upward biased at small trial counts — a bias that can flip the apparent
sign of ``SR``.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mi import (
    _joint_counts_batch,
    _miller_madow_from_counts_batch,
    _plugin_mi_from_counts_batch,
    neuron_ranking,
)

__all__ = ["synergy_matrix"]

Method = Literal["plugin", "miller_madow", "shuffle"]


def synergy_matrix(
    responses: ArrayLike,
    stimuli: ArrayLike,
    method: Method = "plugin",
    B: int = 100,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """Synergy-redundancy index over all pairs of neurons.

    Args:
        responses: Discrete responses, shape ``(n_neurons, n_trials)`` (a
            one-dimensional array is treated as a single neuron).
        stimuli: Stimulus class labels, shape ``(n_trials,)``.
        method: MI estimator for every term: ``"plugin"`` (the original
            ``ass6.m`` estimator), ``"miller_madow"`` or ``"shuffle"``.
        B: Number of permutation surrogates when ``method="shuffle"``. One
            shared set of stimulus-label permutations is used for all
            terms, which keeps the estimator deterministic for a fixed
            seed and reduces variance across pairs.
        seed: Seed for the surrogate permutations.

    Returns:
        Symmetric ``(n_neurons, n_neurons)`` matrix ``SR`` with zeros on
        the diagonal; ``SR[i, j] = I(R_i, R_j; S) - I(R_i; S) - I(R_j; S)``
        in bits.

    Raises:
        ValueError: If ``method`` is unknown, shapes mismatch, or ``B`` is
            not positive.
    """
    if method not in ("plugin", "miller_madow", "shuffle"):
        raise ValueError(f"Unknown method {method!r}.")
    if method == "shuffle" and B <= 0:
        raise ValueError("B must be positive.")
    responses = np.asarray(responses)
    if responses.ndim == 1:
        responses = responses[None, :]
    stimuli = np.asarray(stimuli).ravel()
    if responses.shape[1] != stimuli.size:
        raise ValueError("responses and stimuli must agree on the trial axis.")

    n_neurons = responses.shape[0]
    n_trials = stimuli.size
    _, si = np.unique(stimuli, return_inverse=True)
    ns = int(si.max()) + 1
    codes = [np.unique(responses[i], return_inverse=True)[1] for i in range(n_neurons)]
    sizes = [int(c.max()) + 1 for c in codes]

    # Single-neuron terms I(R_i; S), computed with the (batched) ranking
    # machinery using the same method, B and seed.
    _, mi_single = neuron_ranking(responses, stimuli, method=method, B=B, seed=seed)

    # Shared surrogate permutations of the stimulus labels for the
    # joint-variable terms I(R_i, R_j; S).
    si_perm = None
    if method == "shuffle":
        rng = np.random.default_rng(seed)
        si_perm = si[np.argsort(rng.random((B, n_trials)), axis=1)]

    # Group pairs by their exact combined table size so each group can be
    # scored with a few vectorized batch calls (the total number of table
    # cells — hence the cost — stays proportional to the true support).
    groups: dict = {}
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            groups.setdefault(sizes[i] * sizes[j], []).append((i, j))

    def mi_joint_batch(xb: NDArray[np.intp], n_vals: int) -> NDArray[np.float64]:
        """MI between each pair's combined response and the stimulus."""
        G = xb.shape[0]
        if method == "shuffle":
            observed = _plugin_mi_from_counts_batch(
                _joint_counts_batch(np.tile(si, (G, 1)), xb, ns, n_vals)
            )
            # Chunk the pair axis to bound the (pairs x B) surrogate memory.
            chunk = max(1, int(np.ceil(1500 / B)))
            surrogate_mean = np.zeros(G)
            for start in range(0, G, chunk):
                stop = min(start + chunk, G)
                g = stop - start
                counts_surr = _joint_counts_batch(
                    np.tile(si_perm, (g, 1)),
                    np.repeat(xb[start:stop], B, axis=0),
                    ns,
                    n_vals,
                )
                surrogate_mean[start:stop] = (
                    _plugin_mi_from_counts_batch(counts_surr)
                    .reshape(g, B)
                    .mean(axis=1)
                )
            return observed - surrogate_mean
        counts = _joint_counts_batch(np.tile(si, (G, 1)), xb, ns, n_vals)
        if method == "plugin":
            return _plugin_mi_from_counts_batch(counts)
        return _miller_madow_from_counts_batch(counts)

    SR = np.zeros((n_neurons, n_neurons))
    for n_vals, pair_list in groups.items():
        xb = np.stack([codes[i] * sizes[j] + codes[j] for i, j in pair_list])
        mi_joint = mi_joint_batch(xb, n_vals)
        for (i, j), mij in zip(pair_list, mi_joint):
            SR[i, j] = SR[j, i] = mij - mi_single[i] - mi_single[j]
    return SR
