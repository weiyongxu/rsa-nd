"""Tests for the sequence-of-sensor-objects RSA input adapters.

``Sequence[mne.Evoked]`` and ``Sequence[mne.time_frequency.AverageTFR]`` are
per-item sensor inputs: each element of the sequence is one RSA item, with
sensor geometry taken from the first element.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from rsa_nd import prepare_rsa_data, rsa_nd

_CH_NAMES = ["Fz", "Cz", "Pz", "Oz"]


def _evokeds(n_items=5, n_times=30, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info(_CH_NAMES, sfreq=100.0, ch_types="eeg")
    return [
        mne.EvokedArray(
            rng.standard_normal((len(_CH_NAMES), n_times)),
            info,
            tmin=0.0,
            comment=f"c{i}",
        )
        for i in range(n_items)
    ]


def _avg_tfrs(n_items=5, n_freq=3, n_times=20, seed=1):
    rng = np.random.default_rng(seed)
    info = mne.create_info(_CH_NAMES, sfreq=100.0, ch_types="eeg")
    times = np.arange(n_times) / 100.0
    freqs = np.array([8.0, 12.0, 20.0])[:n_freq]
    return [
        mne.time_frequency.AverageTFRArray(
            info=info,
            data=np.abs(rng.standard_normal((len(_CH_NAMES), n_freq, n_times))),
            times=times,
            freqs=freqs,
            nave=1,
        )
        for _ in range(n_items)
    ]


def _spectra(n_items=5, n_times=64, seed=2):
    """One ``Spectrum`` per item, via ``Evoked.compute_psd()``."""
    rng = np.random.default_rng(seed)
    info = mne.create_info(_CH_NAMES, sfreq=100.0, ch_types="eeg")
    return [
        mne.EvokedArray(
            rng.standard_normal((len(_CH_NAMES), n_times)), info, tmin=0.0
        ).compute_psd()
        for _ in range(n_items)
    ]


def _model(n_items):
    v = np.arange(n_items, dtype=float)
    return np.abs(v[:, None] - v[None, :])


def test_evokeds_dispatch_and_prepare():
    evokeds = _evokeds()

    prepared = prepare_rsa_data(evokeds)
    assert prepared.input_kind == "evokeds"
    assert prepared.rsa_data.axes == ("item", "space", "time")
    assert prepared.rsa_data.data.shape[0] == len(evokeds)
    # Sensor geometry from the first element.
    assert prepared.space is not None
    assert prepared.space.assembly_role() == "per_channel"


def test_evokeds_end_to_end_rsa():
    evokeds = _evokeds(n_items=6)
    result = rsa_nd(
        prepare_rsa_data(evokeds),
        _model(6),
        search_axes=("time",),
        radii={"time": 0.0},
    )
    assert np.asarray(result.rsa).shape == (30,)


def test_evokeds_with_morlet_tfr_adds_freq_axis():
    evokeds = _evokeds()
    prepared = prepare_rsa_data(
        evokeds, tfr={"freqs": np.array([10.0, 20.0]), "n_cycles": 1.0}
    )
    assert prepared.rsa_data.axes == ("item", "space", "freq", "time")


def test_evokeds_sensor_groups():
    evokeds = _evokeds()
    prepared = prepare_rsa_data(
        evokeds, space={"frontal": ["Fz", "Cz"], "posterior": ["Pz", "Oz"]}
    )
    assert prepared.space.is_grouped is True
    assert set(prepared.space.patches.keys()) == {"frontal", "posterior"}


def test_average_tfrs_dispatch_and_prepare():
    tfrs = _avg_tfrs()

    prepared = prepare_rsa_data(tfrs)
    assert prepared.input_kind == "average_tfrs"
    assert prepared.rsa_data.axes == ("item", "space", "freq", "time")
    assert prepared.rsa_data.data.shape[0] == len(tfrs)


def test_average_tfrs_end_to_end_rsa():
    tfrs = _avg_tfrs(n_items=6)
    result = rsa_nd(
        prepare_rsa_data(tfrs),
        _model(6),
        search_axes=("time",),
        radii={"time": 0.0},
    )
    assert np.asarray(result.rsa).shape == (20,)


def test_spectra_dispatch_and_prepare():
    spectra = _spectra()

    # A Spectrum sequence must dispatch to the sequence path, not the
    # within-object EpochsSpectrum path.
    prepared = prepare_rsa_data(spectra)
    assert prepared.input_kind == "spectrums"
    assert prepared.rsa_data.axes == ("item", "space", "freq")
    assert prepared.rsa_data.data.shape[0] == len(spectra)
    assert prepared.space is not None
    assert prepared.space.assembly_role() == "per_channel"


def test_spectra_end_to_end_rsa():
    spectra = _spectra(n_items=6)
    n_freq = spectra[0].get_data().shape[-1]
    result = rsa_nd(
        prepare_rsa_data(spectra),
        _model(6),
        search_axes=("freq",),
        radii={"freq": 0.0},
    )
    assert np.asarray(result.rsa).shape == (n_freq,)


def test_spectra_reject_tfr_kwarg_and_band_aggregation():
    spectra = _spectra()
    with pytest.raises(ValueError, match="already-spectral"):
        prepare_rsa_data(spectra, tfr={"freqs": [10.0]})
    # Named-band aggregation should flow through the spectral feature pipeline.
    prepared = prepare_rsa_data(spectra, freq={"alpha": (8.0, 13.0)})
    assert prepared.rsa_data.axes == ("item", "space", "band")


def test_average_tfrs_rejects_tfr_kwarg():
    tfrs = _avg_tfrs()
    with pytest.raises(ValueError, match="already-spectral"):
        prepare_rsa_data(tfrs, tfr={"freqs": [10.0]})


def test_sequence_requires_consistent_channels():
    evokeds = _evokeds()
    odd_info = mne.create_info(["A", "B", "C", "D"], sfreq=100.0, ch_types="eeg")
    evokeds[2] = mne.EvokedArray(np.asarray(evokeds[2].data), odd_info, tmin=0.0)
    with pytest.raises(ValueError, match="share channel names"):
        prepare_rsa_data(evokeds)


def test_list_of_source_estimates_still_routes_to_source_adapter():
    """The sequence dispatch must not capture source-estimate lists."""
    verts = [np.arange(4), np.arange(4, 8)]
    stcs = [
        mne.SourceEstimate(
            np.random.default_rng(i).standard_normal((8, 5)),
            vertices=verts,
            tmin=0.0,
            tstep=0.01,
            subject="sample",
        )
        for i in range(3)
    ]
    assert prepare_rsa_data(stcs).input_kind == "source_estimates"
