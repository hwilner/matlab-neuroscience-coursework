"""Python port of MATLAB neuroscience signal-processing coursework.

Modules:
    baseline: Z-scoring, SNR, circular convolution, spike detection,
        waveform extraction, ISI statistics, raster/PSTH, and a classic
        PCA + k-means spike-sorting baseline.
    spike_spd: SpikeSPD -- SPD-manifold waveform clustering using
        log-Euclidean geometry.
"""

from . import baseline, spike_spd

__all__ = ["baseline", "spike_spd"]
