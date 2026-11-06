"""Tests for ``RsaResult.to_mne()`` (Layer 3).

``to_mne`` reads only the result -- its ``axes`` / ``coords`` and typed
``space`` geometry -- and returns an MNE container when the geometry maps
honestly, otherwise raising ``ValueError``. There is no conservative fallback
to ``RsaResult`` and no aggregate back-fill onto a native grid.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from rsa_nd import prepare_rsa_data, rsa_nd


def _ordinal_rdm(values):
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def _toy_array_input():
    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    rng = np.random.default_rng(1)
    data = rng.standard_normal((len(item_values), 3, 5))
    coords = {
        "space": ["c0", "c1", "c2"],
        "time": np.linspace(0.0, 0.4, 5),
    }
    return data, item_values, coords


def _toy_epochs(n_items=6, n_ch=4, n_time=20, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"][:n_ch], sfreq=100.0, ch_types="eeg")
    events = np.column_stack(
        [np.arange(n_items), np.zeros(n_items, int), np.arange(n_items)]
    )
    epochs = mne.EpochsArray(
        rng.standard_normal((n_items, n_ch, n_time)),
        info,
        events=events,
        tmin=0.0,
        verbose=False,
    )
    return epochs, _ordinal_rdm(np.arange(n_items))


def _toy_stcs(n_items=4, seed=0):
    times = np.linspace(0.0, 0.49, 50)
    vertices = [np.arange(40, dtype=int), np.arange(20, 40, dtype=int)]
    item_values = np.arange(n_items, dtype=float)
    rng = np.random.default_rng(seed)
    stcs = [
        mne.SourceEstimate(
            rng.standard_normal((60, 50)) + iv,
            vertices=vertices,
            tmin=0.0,
            tstep=times[1] - times[0],
            subject="sample",
        )
        for iv in item_values
    ]
    return stcs, item_values


def test_singleton_misc_to_evoked_array():
    data, item_values, coords = _toy_array_input()
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        _ordinal_rdm(item_values),
        search_axes=("time",),
    )
    assembled = result.to_mne()
    assert isinstance(assembled, mne.EvokedArray)
    assert assembled.data.shape[0] == 1
    assert assembled.ch_names == ["rsa"]


def test_scalar_result_to_mne_raises():
    data, item_values, coords = _toy_array_input()
    # search_axes=() -> scalar RSA; no time/freq grid -> unsupported.
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        _ordinal_rdm(item_values),
    )
    with pytest.raises(ValueError, match="cannot represent"):
        result.to_mne()


def test_model_axis_to_mne_raises():
    data, item_values, coords = _toy_array_input()
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        [_ordinal_rdm(item_values), _ordinal_rdm(item_values)],
        search_axes=("time",),
    )
    assert result.axes == ("model", "time")
    with pytest.raises(ValueError, match="model"):
        result.to_mne()


def test_per_channel_sensor_to_evoked():
    epochs, model = _toy_epochs()
    result = rsa_nd(
        prepare_rsa_data(epochs),
        model,
        search_axes=("space", "time"),
        radii={"space": 0.0, "time": 0.0},
        data_rdm_metric="euclidean",
    )
    out = result.to_mne()
    assert isinstance(out, mne.EvokedArray)
    assert out.ch_names == ["Fz", "Cz", "Pz", "Oz"]


def test_per_group_sensor_to_evoked_with_group_names():
    epochs, model = _toy_epochs()
    prepared = prepare_rsa_data(
        epochs, space={"front": ["Fz", "Cz"], "back": ["Pz", "Oz"]}
    )
    result = rsa_nd(prepared, model, search_axes=("space", "time"), radii={"time": 0.0})
    out = result.to_mne()
    assert isinstance(out, mne.EvokedArray)
    assert out.ch_names == ["front", "back"]


def test_per_group_roi_aggregate_to_evoked():
    stcs, item_values = _toy_stcs()
    rois = [
        mne.Label(np.array([1, 5, 10]), hemi="lh", name="a"),
        mne.Label(np.array([22, 25]), hemi="rh", name="b"),
    ]
    prepared = prepare_rsa_data(stcs, space=rois, space_mode="aggregate")
    result = rsa_nd(
        prepared,
        _ordinal_rdm(item_values),
        search_axes=("roi", "time"),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
    )
    assert result.axes == ("roi", "time")
    out = result.to_mne()
    assert isinstance(out, mne.EvokedArray)
    assert out.ch_names == ["a", "b"]


def test_native_source_to_source_estimate():
    stcs, item_values = _toy_stcs()
    result = rsa_nd(
        prepare_rsa_data(stcs),
        _ordinal_rdm(item_values),
        search_axes=("space", "time"),
        radii={"space": 0.0, "time": 0.0},
        data_rdm_metric="euclidean",
    )
    out = result.to_mne()
    assert isinstance(out, mne.SourceEstimate)
    assert sum(len(v) for v in out.vertices) == 60


def test_grouped_window_axis_to_mne_raises():
    epochs, model = _toy_epochs()
    prepared = prepare_rsa_data(epochs, time={"early": (0.0, 0.1), "late": (0.1, 0.2)})
    result = rsa_nd(prepared, model, search_axes=("window",))
    assert result.axes == ("window",)
    with pytest.raises(ValueError, match="cannot represent"):
        result.to_mne()
