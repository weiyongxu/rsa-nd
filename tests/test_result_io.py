"""Persistence round-trip and model-axis tests for `RsaResult`."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from rsa_nd import RsaResult, prepare_rsa_data, rsa_dynamic, rsa_nd


def _ordinal_rdm(values):
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def _toy_array():
    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(item_values), 3, 5))
    coords = {"space": ["c0", "c1", "c2"], "time": np.linspace(0.0, 0.4, 5)}
    return data, item_values, coords


def _toy_epochs(n_items=6, n_time=20, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=100.0, ch_types="eeg")
    events = np.column_stack(
        [np.arange(n_items), np.zeros(n_items, int), np.arange(n_items)]
    )
    epochs = mne.EpochsArray(
        rng.standard_normal((n_items, 4, n_time)),
        info,
        events=events,
        tmin=0.0,
        verbose=False,
    )
    return epochs, _ordinal_rdm(np.arange(n_items))


# --------------------------------------------------------------------------- #
# Save / load round trip
# --------------------------------------------------------------------------- #
def test_save_load_array_result_roundtrip(tmp_path):
    data, item_values, coords = _toy_array()
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        _ordinal_rdm(item_values),
        search_axes=("time",),
        data_rdm_metric="euclidean",
    )
    fname = tmp_path / "result.h5"
    result.save(fname)
    loaded = RsaResult.load(fname)

    assert isinstance(loaded, RsaResult)
    assert loaded.axes == result.axes
    np.testing.assert_allclose(loaded.rsa, result.rsa)
    np.testing.assert_allclose(loaded.coords["time"], result.coords["time"])
    assert loaded.dynamic is None
    assert loaded.info["run"]["rsa_metric"] == "spearman"


def test_load_reads_files_written_by_mne_ez(tmp_path):
    from h5io import read_hdf5, write_hdf5

    data, item_values, coords = _toy_array()
    result = rsa_nd(
        data,
        _ordinal_rdm(item_values),
        axes=("item", "space", "time"),
        coords=coords,
        search_axes=("space",),
    )
    fname = tmp_path / "new.h5"
    result.save(fname)
    payload = read_hdf5(str(fname), title="rsa_nd_result", slash="replace")
    payload["kind"] = "mne_ez.rsa_result"
    legacy = tmp_path / "legacy.h5"
    write_hdf5(str(legacy), payload, title="mne_ez_rsa_result", slash="replace")

    loaded = RsaResult.load(legacy)

    np.testing.assert_array_equal(loaded.rsa, result.rsa)
    assert loaded.axes == result.axes


def test_load_rejects_other_h5_files(tmp_path):
    from h5io import write_hdf5

    fname = tmp_path / "other.h5"
    write_hdf5(str(fname), {"kind": "other"}, title="rsa_nd_result")

    with pytest.raises(ValueError, match="not an rsa-nd result file"):
        RsaResult.load(fname)


def test_save_load_preserves_sensor_info(tmp_path):
    epochs, model = _toy_epochs()
    epochs.info["bads"] = ["Cz"]
    result = rsa_nd(
        # picks=None keeps every channel (incl. the bad one) so bads survives.
        prepare_rsa_data(epochs, picks=None),
        model,
        search_axes=("space", "time"),
        radii={"space": 0.0, "time": 0.0},
        data_rdm_metric="euclidean",
    )
    fname = tmp_path / "sensor.h5"
    result.save(fname)
    loaded = RsaResult.load(fname)

    # Channel names, types, bads, and sfreq survive the round trip.
    assert loaded.space.info["ch_names"] == ["Fz", "Cz", "Pz", "Oz"]
    assert loaded.space.info["bads"] == ["Cz"]
    assert loaded.space.info["sfreq"] == 100.0
    out = loaded.to_mne()
    assert isinstance(out, mne.EvokedArray)
    assert out.ch_names == ["Fz", "Cz", "Pz", "Oz"]


def test_result_space_info_is_defensively_copied():
    epochs, model = _toy_epochs()
    prepared = prepare_rsa_data(epochs)
    result = rsa_nd(
        prepared,
        model,
        search_axes=("space", "time"),
        radii={"space": 0.0, "time": 0.0},
        data_rdm_metric="euclidean",
    )
    # Mutating the prepared geometry must not leak into the result.
    prepared.space.info["bads"] = ["Pz"]
    assert result.space.info["bads"] == []


def test_save_without_overwrite_then_with_overwrite(tmp_path):
    data, item_values, coords = _toy_array()
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        _ordinal_rdm(item_values),
        search_axes=("time",),
        data_rdm_metric="euclidean",
    )
    fname = tmp_path / "r.h5"
    result.save(fname)
    with pytest.raises(OSError):
        result.save(fname)  # exists, overwrite=False
    result.save(fname, overwrite=True)  # ok


def test_save_load_dynamic_result_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 2, 6))
    coords = {"time": np.linspace(0.0, 0.5, 6)}
    model_times = np.linspace(0.0, 0.5, 6)
    model_rdms = [_ordinal_rdm(np.arange(4)) for _ in model_times]
    result = rsa_dynamic(
        data,
        model_rdms,
        model_times,
        axes=("item", "space", "time"),
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
    )
    fname = tmp_path / "dyn.h5"
    result.save(fname)
    loaded = RsaResult.load(fname)

    assert loaded.dynamic is not None
    assert loaded.dynamic.mode == "generalization"
    assert loaded.axes == ("model_time", "time")
    np.testing.assert_allclose(loaded.dynamic.full_rsa, result.dynamic.full_rsa)
    np.testing.assert_allclose(loaded.dynamic.model_times, model_times)


# --------------------------------------------------------------------------- #
# Model axis
# --------------------------------------------------------------------------- #
def test_unnamed_model_list_uses_arange_coord():
    data, item_values, coords = _toy_array()
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        [_ordinal_rdm(item_values), _ordinal_rdm(item_values)],
        search_axes=("time",),
        data_rdm_metric="euclidean",
    )
    assert result.axes == ("model", "time")
    assert list(result.coords["model"]) == [0, 1]


def test_model_names_set_model_coord():
    data, item_values, coords = _toy_array()
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        [_ordinal_rdm(item_values), _ordinal_rdm(item_values)],
        search_axes=("time",),
        model_names=["m0", "m1"],
        data_rdm_metric="euclidean",
    )
    assert list(result.coords["model"]) == ["m0", "m1"]


def test_3d_stack_adds_model_axis():
    data, item_values, coords = _toy_array()
    stack = np.stack([_ordinal_rdm(item_values), _ordinal_rdm(item_values)])
    result = rsa_nd(
        prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords),
        stack,
        search_axes=("time",),
        data_rdm_metric="euclidean",
    )
    assert result.axes == ("model", "time")
    assert result.rsa.shape == (2, 5)


def test_model_names_rejected_for_single_square():
    data, item_values, coords = _toy_array()
    prepared = prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords)
    with pytest.raises(ValueError, match="multi-model"):
        rsa_nd(
            prepared,
            _ordinal_rdm(item_values),
            search_axes=("time",),
            model_names=["only"],
            data_rdm_metric="euclidean",
        )


def test_model_names_length_mismatch_raises():
    data, item_values, coords = _toy_array()
    prepared = prepare_rsa_data(data, axes=("item", "space", "time"), coords=coords)
    with pytest.raises(ValueError, match="model_names"):
        rsa_nd(
            prepared,
            [_ordinal_rdm(item_values), _ordinal_rdm(item_values)],
            search_axes=("time",),
            model_names=["only-one"],
            data_rdm_metric="euclidean",
        )
