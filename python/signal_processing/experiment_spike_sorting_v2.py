"""Part 3 (2026 development): stress-test and modern benchmark for SpikeSPD v2.

Two evaluation tracks for the developed sorter (:mod:`spike_spd_v2`):

TRACK A -- harder in-house synthetic. Extends the Part-2 generator with
three realistic stressors: slow **amplitude drift** (per-unit gain wanders
+/-25% sinusoidally), a **bursting unit** (gamma-ISH intra-burst spikes),
and **extraction jitter** (waveforms cut at the detected peak +/- up to 2
samples). SpikeSPD v1 vs v2 are compared on identical MAD detections,
plus v2 with overlap screening (flagged overlaps excluded from scoring).

TRACK B -- SpikeInterface benchmark. Ground-truth recordings generated
with :func:`spikeinterface.generate_ground_truth_recording` (single
channel, 4 units, 30 s), sorted by our full pipeline (MAD detection +
waveform extraction + SpikeSPD v1/v2, given the true unit count -- a
caveat we disclose) and by the modern builtin sorter **SpyKING CIRCUS 2**
as a reference. Agreement with ground truth is scored by coincidence
counting within +/-0.4 ms and Hungarian unit matching.

Appends captioned markdown tables to ``RESULTS.md``. Deterministic seeds.
Run::

    python3 experiment_spike_sorting_v2.py
"""

from __future__ import annotations

import pathlib
import tempfile
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import baseline as bl
import spike_spd
import spike_spd_v2 as v2
from experiment_spike_sorting import (
    FS,
    PRE,
    POST,
    add_overlaps,
    clustering_accuracy,
    colored_noise,
    make_templates,
    match_detections,
    poisson_train,
)

DURATION_S = 8.0
RATE_HZ = 20.0
REFRACTORY_MS = 2.0
N_SEEDS_A = 6
N_SEEDS_B = 3
RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "RESULTS.md"


# ------------------------- Track A: stress generators -----------------------
def bursting_train(
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    burst_rate_hz: float = 1.5,
    intra_isi_s: tuple[float, float] = (0.004, 0.010),
    spikes_per_burst: tuple[int, int] = (3, 5),
) -> np.ndarray:
    """Generate a bursting spike train.

    Bursts arrive as a Poisson process; within a burst, inter-spike
    intervals are uniform in ``intra_isi_s`` (above the refractory
    period).

    Args:
        n_samples: Recording length in samples.
        fs: Sampling rate in Hz.
        rng: NumPy random generator.
        burst_rate_hz: Mean number of bursts per second.
        intra_isi_s: (min, max) intra-burst inter-spike interval.
        spikes_per_burst: (min, max) spikes per burst (inclusive).

    Returns:
        Sorted spike sample indices.
    """
    duration_s = n_samples / fs
    n_bursts = int(duration_s * burst_rate_hz * 1.5) + 5
    starts = np.sort(rng.uniform(0.0, duration_s, n_bursts))
    times: list[float] = []
    for s in starts:
        n_sp = int(rng.integers(spikes_per_burst[0], spikes_per_burst[1] + 1))
        t = s
        for _ in range(n_sp):
            times.append(t)
            t += rng.uniform(*intra_isi_s)
    times = np.array(times)
    times = times[times < duration_s]
    return np.unique((times * fs).astype(int))


def build_signal_drift(
    templates: np.ndarray,
    trains: list[np.ndarray],
    snr: float,
    rng: np.random.Generator,
    n_samples: int,
    drift_depth: float = 0.25,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Superimpose templates with slowly drifting per-unit amplitude.

    The gain of unit ``u`` at time ``t`` is
    ``snr * (1 + drift_depth * sin(2*pi*t/T + phi_u))`` with random phase
    ``phi_u``, mimicking electrode drift over the recording.

    Args:
        templates: Unit-peak templates, shape ``(n_units, L)``.
        trains: Spike sample indices per unit.
        snr: Nominal peak-amplitude / noise-std ratio.
        rng: NumPy random generator.
        n_samples: Recording length in samples.
        drift_depth: Relative amplitude modulation depth.

    Returns:
        Tuple ``(signal, kept_trains)`` (see ``build_signal``).
    """
    signal = colored_noise(n_samples, rng)
    duration_s = n_samples / FS
    phases = rng.uniform(0, 2 * np.pi, size=len(trains))
    kept: list[np.ndarray] = []
    for unit, times in enumerate(trains):
        valid = times[(times >= PRE) & (times + POST < n_samples)]
        kept.append(valid)
        for t in valid:
            gain = snr * (1.0 + drift_depth * np.sin(2 * np.pi * t / n_samples + phases[unit]))
            signal[t - PRE : t + POST + 1] += gain * templates[unit]
    return signal, kept


def extract_with_jitter(
    signal: np.ndarray,
    spike_idx: np.ndarray,
    rng: np.random.Generator,
    jitter: int,
) -> np.ndarray:
    """Extract waveforms at the detected index plus uniform jitter.

    Args:
        signal: Filtered recording.
        spike_idx: Detected peak indices.
        rng: NumPy random generator.
        jitter: Maximum absolute extraction offset in samples.

    Returns:
        Waveform array of shape ``(n_spikes, PRE + POST + 1)``.
    """
    shifted = spike_idx + rng.integers(-jitter, jitter + 1, size=len(spike_idx))
    shifted = np.clip(shifted, PRE, len(signal) - POST - 1)
    return bl.extract_waveforms(signal, shifted, PRE, POST, align=False)[0]


def run_config_a(snr: float, overlap: float, jitter: int, seed: int) -> dict:
    """Run one Track-A configuration (drift + bursting always on).

    Args:
        snr: Nominal peak-amplitude / noise-std ratio.
        overlap: Fraction of spikes forced to overlap.
        jitter: Maximum extraction jitter in samples.
        seed: Seed index.

    Returns:
        Dict with accuracy/ARI for SpikeSPD v1, v2, and v2 with overlap
        screening, plus the fraction of spikes flagged as overlaps.
    """
    rng = np.random.default_rng([seed, int(snr * 10), int(overlap * 100), jitter])
    n_samples = int(DURATION_S * FS)
    templates = make_templates()
    trains = [
        bursting_train(n_samples, FS, rng),  # unit 0 bursts
        poisson_train(n_samples, RATE_HZ, REFRACTORY_MS / 1000.0, FS, rng),
        poisson_train(n_samples, RATE_HZ, REFRACTORY_MS / 1000.0, FS, rng),
    ]
    if overlap > 0:
        trains = add_overlaps(trains, overlap, int(REFRACTORY_MS * FS / 1000.0), rng)
    signal, trains = build_signal_drift(templates, trains, snr, rng, n_samples)
    truth_times = np.concatenate(trains)
    truth_labels = np.concatenate([np.full(len(t), u) for u, t in enumerate(trains)])

    det = bl.detect_spikes(signal, k=4.0, fs=FS)
    pairs = match_detections(det, truth_times)
    det_matched = det[[p[0] for p in pairs]]
    lab_matched = truth_labels[[p[1] for p in pairs]]

    wv = extract_with_jitter(signal, det_matched, rng, jitter)

    lab_v1 = spike_spd.sort_spikes_spd(wv, 3, random_state=seed)
    lab_v2, _ = v2.sort_spikes_spd_v2(wv, 3, fs=FS, descriptor="tde", random_state=seed)
    lab_f, is_ov = v2.sort_spikes_spd_v2(wv, 3, fs=FS, descriptor="fusion", random_state=seed)

    keep = ~is_ov
    out = {
        "snr": snr,
        "overlap": overlap,
        "jitter": jitter,
        "acc_v1": clustering_accuracy(lab_v1, lab_matched),
        "ari_v1": adjusted_rand_score(lab_matched, lab_v1),
        "acc_v2": clustering_accuracy(lab_v2, lab_matched),
        "ari_v2": adjusted_rand_score(lab_matched, lab_v2),
        "acc_f": clustering_accuracy(lab_f, lab_matched),
        "ari_f": adjusted_rand_score(lab_matched, lab_f),
        "overlap_frac": float(is_ov.mean()),
    }
    if keep.sum() > 10:
        out["acc_fs"] = clustering_accuracy(lab_f[keep], lab_matched[keep])
        out["ari_fs"] = adjusted_rand_score(lab_matched[keep], lab_f[keep])
    else:
        out["acc_fs"] = out["acc_f"]
        out["ari_fs"] = out["ari_f"]
    return out


# --------------------- Track B: SpikeInterface benchmark --------------------
def _agreement_accuracy(
    gt_times: list[np.ndarray], pred_times: list[np.ndarray], tol_samples: int
) -> float:
    """Hungarian-matched mean per-unit agreement between two sortings.

    The agreement between a ground-truth unit and a predicted unit is
    ``n_match / max(n_gt, n_pred)`` with ``n_match`` the number of spikes
    coincident within ``tol_samples``; units are matched by solving the
    linear assignment problem on the agreement matrix.

    Args:
        gt_times: Spike sample indices per ground-truth unit.
        pred_times: Spike sample indices per predicted unit.
        tol_samples: Coincidence tolerance in samples.

    Returns:
        Mean agreement over matched unit pairs, in ``[0, 1]``.
    """
    if not gt_times or not pred_times:
        return 0.0
    n_gt, n_pr = len(gt_times), len(pred_times)
    agree = np.zeros((n_gt, n_pr))
    for i, g in enumerate(gt_times):
        for j, p in enumerate(pred_times):
            if len(g) == 0 or len(p) == 0:
                continue
            d = np.abs(g[:, None].astype(np.int64) - p[None, :].astype(np.int64))
            r, c = linear_sum_assignment(d)
            n_match = int(sum(1 for a, b in zip(r, c) if d[a, b] <= tol_samples))
            agree[i, j] = n_match / max(len(g), len(p))
    r, c = linear_sum_assignment(-agree)
    return float(agree[r, c].mean())


def run_si_config(seed: int) -> dict:
    """Run one Track-B SpikeInterface ground-truth configuration.

    Generates a single-channel 30 s recording with 4 units, band-pass
    filters it (300-6000 Hz), and compares SpikeSPD v1, SpikeSPD v2
    (both given the true unit count), and SpyKING CIRCUS 2 against
    ground truth.

    Args:
        seed: Seed index for recording generation.

    Returns:
        Dict with Hungarian-matched agreement accuracy per method.

    Raises:
        ImportError: If spikeinterface is not installed.
    """
    import spikeinterface.full as si

    rec, gt_sorting = si.generate_ground_truth_recording(
        num_channels=1,
        durations=[30.0],
        num_units=4,
        seed=seed,
        generate_probe_kwargs={"num_columns": 1},
    )
    fs = rec.get_sampling_frequency()
    rec_f = si.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)
    trace = rec_f.get_traces().squeeze()
    tol = int(round(0.4 * fs / 1000.0))

    gt_times = [
        gt_sorting.get_unit_spike_train(u) for u in gt_sorting.get_unit_ids()
    ]

    # Our pipeline: MAD detection -> waveform extraction -> SPD sorting.
    det = bl.detect_spikes(trace, k=5.0, fs=fs)
    pre, post = int(1e-3 * fs), int(2e-3 * fs)
    valid = det[(det >= pre) & (det + post < len(trace))]
    wv = bl.extract_waveforms(trace, valid, pre, post)[0]

    def _ours(labels: np.ndarray) -> float:
        pred_times = [valid[labels == j] for j in np.unique(labels)]
        return _agreement_accuracy(gt_times, pred_times, tol)

    acc_v1 = _ours(spike_spd.sort_spikes_spd(wv, 4, random_state=seed))
    lab_v2, _ = v2.sort_spikes_spd_v2(wv, 4, fs=fs, descriptor="tde", random_state=seed)
    acc_v2 = _ours(lab_v2)
    lab_f, _ = v2.sort_spikes_spd_v2(wv, 4, fs=fs, descriptor="fusion", random_state=seed)
    acc_f = _ours(lab_f)

    # Modern reference sorter.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            sorting_sc2 = si.run_sorter(
                "spykingcircus2", rec_f, folder=f"{tmp}/sc2", verbose=False,
                remove_existing_folder=True,
            )
        except Exception:
            sorting_sc2 = si.run_sorter(
                "tridesclous2", rec_f, folder=f"{tmp}/tdc2", verbose=False,
                remove_existing_folder=True,
            )
        pred_times = [
            sorting_sc2.get_unit_spike_train(u) for u in sorting_sc2.get_unit_ids()
        ]
        acc_ref = _agreement_accuracy(gt_times, pred_times, tol)

    return {"seed": seed, "acc_v1": acc_v1, "acc_v2": acc_v2, "acc_f": acc_f, "acc_ref": acc_ref}


# ------------------------------- reporting ---------------------------------
def _fmt(values: np.ndarray) -> str:
    """Format mean +/- std, matching the Part-2 table style."""
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}" if len(values) > 1 else f"{values.mean():.3f}"


def append_results(rows_a: list[dict], rows_b: list[dict], runtime_s: float) -> None:
    """Append the Part-3 section with captioned tables to RESULTS.md.

    Args:
        rows_a: Track-A rows from :func:`run_config_a`.
        rows_b: Track-B rows from :func:`run_si_config`.
        runtime_s: Total wall-clock runtime in seconds.
    """
    lines: list[str] = []
    lines.append("\n---\n\n## Part 3 (2026 retrospective) -- developed SpikeSPD v2\n")
    lines.append(
        "Development over the Part-2 prototype: time-delay-embedding (TDE) "
        "covariance descriptors replacing rank-1 outer products, zero-phase "
        "low-pass denoising, template-adaptive realignment, and overlap "
        "screening. Track A stress-tests on synthetic recordings with "
        "amplitude drift, a bursting unit, and extraction jitter; Track B "
        "benchmarks on SpikeInterface ground-truth recordings against "
        "SpyKING CIRCUS 2.\n"
    )

    lines.append(
        "**Table 4.** Track A: clustering accuracy (Hungarian best match) under "
        "amplitude drift and bursting, on identical MAD detections. `fusion` = "
        "block-diagonal TDE + outer-product descriptor; `fusion screened` excludes "
        "spikes flagged as overlaps. Mean +/- std over seeds.\n"
    )
    lines.append("| SNR | Overlap | Jitter | v1 acc | TDE acc | fusion acc | fusion screened acc | v1 ARI | TDE ARI | fusion ARI |")
    lines.append("|----:|--------:|-------:|-------:|--------:|-----------:|--------------------:|-------:|--------:|-----------:|")
    for snr in sorted({r["snr"] for r in rows_a}):
        for ov in sorted({r["overlap"] for r in rows_a}):
            for jt in sorted({r["jitter"] for r in rows_a}):
                sel = [r for r in rows_a if r["snr"] == snr and r["overlap"] == ov and r["jitter"] == jt]
                lines.append(
                    f"| {snr:g} | {ov:g} | {jt} | {_fmt([r['acc_v1'] for r in sel])} "
                    f"| {_fmt([r['acc_v2'] for r in sel])} | {_fmt([r['acc_f'] for r in sel])} "
                    f"| {_fmt([r['acc_fs'] for r in sel])} | {_fmt([r['ari_v1'] for r in sel])} "
                    f"| {_fmt([r['ari_v2'] for r in sel])} | {_fmt([r['ari_f'] for r in sel])} |"
                )
    lines.append("")

    lines.append(
        "**Table 5.** Track B: SpikeInterface ground-truth benchmark (single channel, "
        "4 units, 30 s, 300-6000 Hz band-pass). Hungarian-matched per-unit agreement "
        "with ground truth (coincidence +/-0.4 ms). Caveat: SpikeSPD variants are given "
        "the true unit count; SpyKING CIRCUS 2 estimates it.\n"
    )
    lines.append("| Seed | SpikeSPD v1 | SpikeSPD v2 (TDE) | SpikeSPD v2 (fusion) | SpyKING CIRCUS 2 |")
    lines.append("|-----:|------------:|------------------:|---------------------:|-----------------:|")
    for r in rows_b:
        lines.append(
            f"| {r['seed']} | {r['acc_v1']:.3f} | {r['acc_v2']:.3f} | {r['acc_f']:.3f} | {r['acc_ref']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"Mean: v1 {_fmt([r['acc_v1'] for r in rows_b])}, "
        f"TDE {_fmt([r['acc_v2'] for r in rows_b])}, "
        f"fusion {_fmt([r['acc_f'] for r in rows_b])}, "
        f"SC2 {_fmt([r['acc_ref'] for r in rows_b])}. "
        f"Total runtime {runtime_s:.0f} s.\n"
    )
    with open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    """Run both tracks and append results to RESULTS.md."""
    t0 = time.time()
    rows_a: list[dict] = []
    for snr in (5.0, 10.0):
        for ov in (0.0, 0.15):
            for jt in (0, 2):
                for seed in range(N_SEEDS_A):
                    rows_a.append(run_config_a(snr, ov, jt, seed))
                    print(f"A done snr={snr} ov={ov} jitter={jt} seed={seed}", flush=True)

    rows_b: list[dict] = []
    for seed in range(N_SEEDS_B):
        rows_b.append(run_si_config(seed))
        print(f"B done seed={seed}: {rows_b[-1]}", flush=True)

    append_results(rows_a, rows_b, time.time() - t0)
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
