import typing

import mne
import numpy as np
import pytest
from mne.time_frequency import EpochsTFRArray

from rsa_nd import (
    RsaData,
    RsaResult,
    compute_rsa,
    prepare_rsa_data,
    rsa_dynamic,
    rsa_dynamic_windows,
    rsa_nd,
    safe_log10_power,
)
from rsa_nd._blocks import (
    _flat_feature_indices_from_vectors,
    build_array_layout,
)
from rsa_nd._features import _zscore_features
from rsa_nd._source import source_roi_patches


def _prepare_rsa_data_only(*args, **kwargs):
    return prepare_rsa_data(*args, **kwargs).rsa_data


def _events(n_epochs):
    return np.column_stack(
        [
            np.arange(n_epochs),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    )


def _info(n_channels=2, *, with_locs=False):
    info = mne.create_info(
        ch_names=[f"EEG {idx:03d}" for idx in range(n_channels)],
        sfreq=100.0,
        ch_types="eeg",
    )
    if with_locs:
        for idx, ch in enumerate(info["chs"]):
            ch["loc"][:3] = [0.01 * idx, 0.0, 0.0]
    return info


def _toy_tfr(n_epochs=3, n_channels=2):
    n_freqs, n_times = 4, 6
    data = np.arange(n_epochs * n_channels * n_freqs * n_times, dtype=float).reshape(
        n_epochs, n_channels, n_freqs, n_times
    )
    return EpochsTFRArray(
        _info(n_channels),
        data,
        times=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        freqs=np.array([4.0, 6.0, 10.0, 20.0]),
        events=_events(n_epochs),
        event_id={"stim": 1},
        method="toy",
    )


def _toy_array():
    data = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    axes = ("item", "space", "freq", "time")
    coords = {
        "space": ["V1", "AMY", "PFC"],
        "freq": [4.0, 6.0, 10.0, 20.0],
        "time": [0.0, 0.1, 0.2, 0.3, 0.4],
    }
    return data, axes, coords


def _ordinal_rdm(values):
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def _dynamic_spectral_array():
    labels = np.array(["a", "b", "c", "d"])
    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    space_values = np.array([0.0, 0.5])
    freq_values = np.array([4.0, 8.0, 12.0])
    time_values = np.array([0.0, 0.1, 0.2, 0.3])
    data = (
        item_values[:, np.newaxis, np.newaxis, np.newaxis]
        + space_values[np.newaxis, :, np.newaxis, np.newaxis]
        + 0.01 * freq_values[np.newaxis, np.newaxis, :, np.newaxis]
        + 0.1 * time_values[np.newaxis, np.newaxis, np.newaxis, :]
    )
    axes = ("item", "space", "freq", "time")
    coords = {
        "space": ["P1", "P2"],
        "freq": freq_values,
        "time": time_values,
    }
    model_rdms = np.stack(
        [
            _ordinal_rdm(item_values),
            _ordinal_rdm(item_values**2),
            _ordinal_rdm(np.array([0.0, 1.0, 3.0, 4.0])),
            _ordinal_rdm(np.array([0.0, 2.0, 3.0, 5.0])),
        ]
    )
    return data, axes, coords, labels, model_rdms, time_values


def _toy_stcs_and_rois():
    vertices = [np.array([10, 20, 30]), np.array([11, 21])]
    times = np.arange(60) / 100.0
    item_values = np.array([0.0, 1.0, 2.5, 4.0])
    vertex_weights = np.array([0.2, 0.7, -0.3, 1.1, -0.8])
    stcs = []
    for item_value in item_values:
        data = (
            np.sin(2 * np.pi * 10.0 * times)[np.newaxis, :]
            + item_value * vertex_weights[:, np.newaxis]
            + 0.01 * np.arange(len(vertices[0]) + len(vertices[1]))[:, np.newaxis]
        )
        stcs.append(
            mne.SourceEstimate(
                data,
                vertices=vertices,
                tmin=0.0,
                tstep=0.01,
                subject="sample",
            )
        )
    rois = [
        mne.Label(np.array([10, 30]), hemi="lh", name="roi-a-lh"),
        mne.Label(np.array([21]), hemi="rh", name="roi-b-rh"),
        mne.Label(np.array([999]), hemi="lh", name="empty-lh"),
    ]
    model_rdm = _ordinal_rdm(item_values)
    labels = np.arange(len(stcs))
    return stcs, rois, model_rdm, labels


def test_safe_log10_power_clips_zero_values():
    logged = safe_log10_power(np.array([0.0, 1.0, 100.0]))

    assert np.isfinite(logged).all()
    assert logged[1] == pytest.approx(0.0)
    assert logged[2] == pytest.approx(2.0)


def test_safe_log10_power_uses_median_scaled_default_floor():
    logged = safe_log10_power(np.array([0.0, 1e-30, 1.0, 100.0]))

    assert logged[0] == pytest.approx(-6.0)
    assert logged[1] == pytest.approx(-6.0)


def test_prepare_rsa_data_exposes_explicit_log_floor():
    data = np.array([[[[0.0, 1.0]]]])

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [10.0], "time": [0.0, 0.1]},
        log=True,
        log_floor=0.01,
    )

    assert named.data[0, 0, 0, 0] == pytest.approx(-2.0)


def test_public_type_hints_are_resolvable_after_lazy_mne_import():
    hints = typing.get_type_hints(RsaResult.search_coords.fget)

    assert "return" in hints


def test_unhashable_agg_raises_clear_value_error():
    data = np.ones((1, 1, 2, 1), dtype=float)

    with pytest.raises(ValueError, match="agg must be"):
        _prepare_rsa_data_only(
            data,
            axes=("item", "space", "freq", "time"),
            coords={"freq": [8.0, 10.0], "time": [0.0]},
            freq={"alpha": (8.0, 13.0)},
            agg=["mean"],
        )


def test_unknown_coordinate_keys_raise_clear_error():
    data = np.ones((2, 3), dtype=float)

    with pytest.raises(ValueError, match="Unknown coordinate keys"):
        _prepare_rsa_data_only(
            data,
            axes=("item", "freq"),
            coords={"freq": [8.0, 12.0, 20.0], "frq": [1, 2, 3]},
        )


def test_float32_data_is_preserved_through_named_and_layout_conversion():
    data = np.ones((2, 3, 4), dtype=np.float32)

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq"),
        coords={"space": ["a", "b", "c"], "freq": [8.0, 10.0, 12.0, 14.0]},
    )
    layout = build_array_layout(named, feature_axes=("space",), sample_axis="freq")

    assert named.data.dtype == np.float32
    assert layout.X.dtype == np.float32


def test_prepare_rsa_data_can_zscore_features_across_items():
    data = np.array(
        [
            [[1.0, 10.0], [5.0, 2.0]],
            [[2.0, 12.0], [5.0, 4.0]],
            [[3.0, 14.0], [5.0, 6.0]],
        ]
    )

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq"),
        standardize="features",
    )

    assert named.data[:, 0, 0] == pytest.approx([-1.22474487, 0.0, 1.22474487])
    assert named.data[:, 0, 1] == pytest.approx([-1.22474487, 0.0, 1.22474487])
    assert named.data[:, 1, 0] == pytest.approx([0.0, 0.0, 0.0])


def test_prepare_rsa_data_rejects_unknown_standardize_mode():
    data = np.ones((2, 3), dtype=float)

    with pytest.raises(ValueError, match="standardize"):
        _prepare_rsa_data_only(
            data,
            axes=("item", "space"),
            standardize="items",
        )


def test_prepare_rsa_data_rejects_boolean_false_standardize_mode():
    data = np.ones((2, 3), dtype=float)

    with pytest.raises(ValueError, match="standardize"):
        _prepare_rsa_data_only(
            data,
            axes=("item", "space"),
            standardize=False,
        )


def test_prepare_rsa_data_warns_when_standardizing_single_item():
    data = np.ones((1, 3), dtype=float)

    with pytest.warns(RuntimeWarning, match="fewer than 2 items"):
        named = _prepare_rsa_data_only(
            data,
            axes=("item", "space"),
            standardize="features",
        )

    assert np.allclose(named.data, 0.0)


def test_prepare_rsa_data_projects_complex_tfr_feature_modes():
    data = np.array([[[[1.0 + 1.0j, 0.0 + 0.0j], [0.0 + 2.0j, -3.0 + 0.0j]]]])
    axes = ("item", "space", "freq", "time")
    coords = {"freq": [4.0, 8.0], "time": [0.0, 0.1]}

    power = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        tfr_feature="power",
    )
    amplitude = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        tfr_feature="amplitude",
    )
    real_imag = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        tfr_feature="real_imag",
    )
    amp_phase = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        tfr_feature="amp_phase",
    )
    phase = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        tfr_feature="phase",
    )

    assert power.axes == axes
    assert np.allclose(power.data, np.abs(data) ** 2)
    assert np.allclose(amplitude.data, np.abs(data))

    assert real_imag.axes == ("item", "space", "freq", "time", "component")
    assert real_imag.coords["component"].tolist() == ["real", "imag"]
    assert np.allclose(real_imag.data[..., 0], data.real)
    assert np.allclose(real_imag.data[..., 1], data.imag)

    assert amp_phase.coords["component"].tolist() == [
        "amplitude",
        "sin_phase",
        "cos_phase",
    ]
    assert np.allclose(amp_phase.data[..., 0], np.abs(data))
    assert amp_phase.data[0, 0, 0, 1, 1] == pytest.approx(0.0)
    assert amp_phase.data[0, 0, 0, 1, 2] == pytest.approx(0.0)

    assert phase.coords["component"].tolist() == ["sin_phase", "cos_phase"]
    assert phase.data[0, 0, 0, 1, 0] == pytest.approx(0.0)
    assert phase.data[0, 0, 0, 1, 1] == pytest.approx(0.0)


def test_complex_derived_features_reject_log_and_named_range_aggregation():
    data = np.ones((2, 1, 2, 2), dtype=complex)
    axes = ("item", "space", "freq", "time")
    coords = {"freq": [4.0, 8.0], "time": [0.0, 0.1]}

    with pytest.raises(ValueError, match="log=True"):
        _prepare_rsa_data_only(
            data,
            axes=axes,
            coords=coords,
            tfr_feature="real_imag",
            log=True,
        )

    with pytest.raises(ValueError, match="named frequency"):
        _prepare_rsa_data_only(
            data,
            axes=axes,
            coords=coords,
            tfr_feature="phase",
            freq={"theta": (4.0, 8.0)},
        )

    selected = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        tfr_feature="phase",
        freq=(4.0, 8.0),
    )
    assert selected.axes == ("item", "space", "freq", "time", "component")
    assert selected.coords["freq"].tolist() == [4.0, 8.0]


def test_amplitude_feature_allows_log_transform_on_complex_input():
    data = np.array([[[[1.0 + 1.0j, 3.0 + 4.0j]]]], dtype=complex)

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [8.0], "time": [0.1, 0.2]},
        tfr_feature="amplitude",
        log=True,
    )

    assert named.axes == ("item", "space", "freq", "time")
    assert np.allclose(named.data, np.log10(np.abs(data)))


def test_phase_feature_rejects_real_input():
    data = np.ones((2, 1, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="requires complex input"):
        _prepare_rsa_data_only(
            data,
            axes=("item", "space", "freq", "time"),
            tfr_feature="phase",
        )


def test_complex_derived_feature_rejects_existing_component_axis():
    data = np.ones((2, 1, 2, 2, 1), dtype=complex)

    with pytest.raises(ValueError, match="already contain 'component'"):
        _prepare_rsa_data_only(
            data,
            axes=("item", "space", "freq", "time", "component"),
            coords={"component": ["old"]},
            tfr_feature="real_imag",
        )


def test_prepare_rsa_data_keeps_frequency_points_with_tuple_selection():
    data, axes, coords = _toy_array()

    named = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        freq=(8.0, 13.0),
    )

    assert isinstance(named, RsaData)
    assert named.axes == ("item", "space", "freq", "time")
    assert np.allclose(named.coords["freq"], [10.0])
    assert named.data.shape == (2, 3, 1, 5)
    assert named.data[0, 1, 0, 2] == data[0, 1, 2, 2]


def test_tuple_frequency_selection_includes_both_edges_by_default():
    data = np.arange(1 * 1 * 4 * 1, dtype=float).reshape(1, 1, 4, 1)

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [7.0, 8.0, 13.0, 14.0], "time": [0.0]},
        freq=(8.0, 13.0),
    )

    assert np.allclose(named.coords["freq"], [8.0, 13.0])


def test_tuple_frequency_selection_can_request_half_open_ranges():
    data = np.arange(1 * 1 * 4 * 1, dtype=float).reshape(1, 1, 4, 1)

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [7.0, 8.0, 13.0, 14.0], "time": [0.0]},
        freq=(8.0, 13.0),
        range_closed="left",
    )

    assert np.allclose(named.coords["freq"], [8.0])


def test_named_frequency_bands_are_half_open_by_default_to_avoid_overlap():
    data = np.arange(1 * 1 * 4 * 1, dtype=float).reshape(1, 1, 4, 1)

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [4.0, 8.0, 13.0, 20.0], "time": [0.0]},
        freq={"theta": (4.0, 8.0), "alpha": (8.0, 13.0)},
    )

    assert named.axes == ("item", "space", "band", "time")
    assert named.data[0, 0, 0, 0] == data[0, 0, [0], 0].mean()
    assert named.data[0, 0, 1, 0] == data[0, 0, [1], 0].mean()


def test_named_frequency_bands_can_request_closed_ranges_explicitly():
    data = np.arange(1 * 1 * 4 * 1, dtype=float).reshape(1, 1, 4, 1)

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [4.0, 8.0, 13.0, 20.0], "time": [0.0]},
        freq={"alpha": (8.0, 13.0)},
        range_closed="both",
    )

    assert named.axes == ("item", "space", "band", "time")
    assert named.data[0, 0, 0, 0] == data[0, 0, [1, 2], 0].mean()


def test_duplicate_named_range_labels_raise_clear_error():
    data = np.arange(1 * 1 * 4 * 1, dtype=float).reshape(1, 1, 4, 1)

    with pytest.raises(ValueError, match="Duplicate range label"):
        _prepare_rsa_data_only(
            data,
            axes=("item", "space", "freq", "time"),
            coords={"freq": [4.0, 8.0, 13.0, 20.0], "time": [0.0]},
            freq=[("alpha", (8.0, 13.0)), ("alpha", (13.0, 30.0))],
        )


def test_prepare_rsa_data_replaces_frequency_with_named_bands():
    data, axes, coords = _toy_array()

    named = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        freq={"low": (4.0, 8.0), "high": (8.0, 30.0)},
    )

    assert named.axes == ("item", "space", "band", "time")
    assert list(named.coords["band"]) == ["low", "high"]
    assert named.data.shape == (2, 3, 2, 5)
    assert named.data[0, 0, 0, 0] == data[0, 0, [0, 1], 0].mean()
    assert named.data[0, 0, 1, 0] == data[0, 0, [2, 3], 0].mean()


def test_prepare_rsa_data_replaces_time_with_named_windows():
    data, axes, coords = _toy_array()

    named = _prepare_rsa_data_only(
        data,
        axes=axes,
        coords=coords,
        time={"early": (0.0, 0.25), "late": (0.25, 0.5)},
    )

    assert named.axes == ("item", "space", "freq", "window")
    assert list(named.coords["window"]) == ["early", "late"]
    assert named.data.shape == (2, 3, 4, 2)
    assert named.data[0, 0, 0, 0] == data[0, 0, 0, [0, 1, 2]].mean()
    assert named.data[0, 0, 0, 1] == data[0, 0, 0, [3, 4]].mean()


def test_mne_spectral_extraction_keeps_data_and_channel_coords_aligned_with_bads():
    info = _info(2)
    info["bads"] = ["EEG 001"]
    spectrum = mne.time_frequency.EpochsSpectrumArray(
        np.ones((3, 2, 4)),
        info,
        freqs=np.array([5.0, 10.0, 15.0, 20.0]),
        events=_events(3),
        event_id={"stim": 1},
    )

    named = _prepare_rsa_data_only(spectrum)
    layout = build_array_layout(named, feature_axes=("space",), sample_axis="freq")

    assert named.axes == ("item", "space", "freq")
    assert layout.X.shape == (3, 1, 4)
    assert list(named.coords["space"]) == ["EEG 000"]
    assert layout.feature_axes == ("space",)


def test_psd_does_not_expose_fake_time_axis():
    spectrum = mne.time_frequency.EpochsSpectrumArray(
        np.ones((3, 2, 4)),
        _info(2),
        freqs=np.array([5.0, 10.0, 15.0, 20.0]),
        events=_events(3),
        event_id={"stim": 1},
    )

    with pytest.raises(ValueError, match="Search axes not present"):
        rsa_nd(
            _prepare_rsa_data_only(spectrum),
            _ordinal_rdm([0.0, 1.0, 2.0]),
            search_axes=("time",),
            data_rdm_metric="euclidean",
            labels=["a", "b", "c"],
            labels_rdm_model=["a", "b", "c"],
        )


def test_tfr_can_be_flattened_to_full_spectrotemporal_features():
    tfr = _toy_tfr()

    named = _prepare_rsa_data_only(tfr)
    layout = build_array_layout(
        named,
        feature_axes=("space", "freq", "time"),
        sample_axis=None,
    )

    assert layout.X.shape == (3, 2 * 4 * 6, 1)
    assert layout.feature_axes == ("space", "freq", "time")
    assert layout.sample_axis is None
    assert layout.X[0, 0, 0] == tfr.get_data()[0, 0, 0, 0]
    assert layout.X[0, -1, 0] == tfr.get_data()[0, -1, -1, -1]


def test_rsa_nd_returns_frequency_time_map_for_generic_array():
    labels = np.array(["a", "b", "c", "d"])
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    item_values = np.array([1.0, 2.0, 3.0, 5.0])
    space_values = np.array([1.0, 2.0])
    freq_values = np.array([6.0, 10.0, 20.0])
    time_values = np.array([0.0, 0.1, 0.2, 0.3])
    data = (
        item_values[:, None, None, None] * space_values[None, :, None, None]
        + item_values[:, None, None, None] ** 2 * freq_values[None, None, :, None]
        + time_values[None, None, None, :]
    )

    result = rsa_nd(
        data,
        model_rdm,
        axes=("item", "space", "freq", "time"),
        coords={
            "space": ["P1", "P2"],
            "freq": freq_values,
            "time": time_values,
        },
        search_axes=("freq", "time"),
        radii={"freq": 0.0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
    )

    assert result.rsa.shape == (3, 4)
    assert result.search_axes == ("freq", "time")
    assert np.allclose(result.search_coords["freq"], freq_values)
    assert np.allclose(result.search_coords["time"], time_values)


def test_prep_then_rsa_nd_can_switch_to_band_time_map_with_freq_mapping():
    labels = np.array(["a", "b", "c", "d"])
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    data = np.arange(4 * 2 * 4 * 3, dtype=float).reshape(4, 2, 4, 3)

    prepared = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={
            "space": ["P1", "P2"],
            "freq": [4.0, 6.0, 10.0, 20.0],
            "time": [0.0, 0.1, 0.2],
        },
        freq={"low": (4.0, 8.0), "high": (8.0, 30.0)},
    )
    result = rsa_nd(
        prepared,
        model_rdm,
        search_axes=("band", "time"),
        radii={"band": 0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
    )

    assert result.rsa.shape == (2, 3)
    assert prepared.axes == ("item", "space", "band", "time")
    assert list(result.search_coords["band"]) == ["low", "high"]


def test_band_time_values_match_direct_rdm_computation():
    from scipy.spatial import distance

    labels = np.array(["a", "b", "c", "d"])
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    model_vec = distance.squareform(model_rdm, checks=False)
    rng = np.random.default_rng(42)
    data = rng.normal(size=(4, 3, 4, 5))
    data += (
        np.arange(4, dtype=float)[:, None, None, None]
        * np.array([0.0, 0.2, 0.7, 1.1])[None, None, :, None]
    )

    prepared = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={
            "space": ["P1", "P2", "P3"],
            "freq": [4.0, 6.0, 10.0, 20.0],
            "time": [0.0, 0.1, 0.2, 0.3, 0.4],
        },
        freq={"low": (4.0, 8.0), "high": (8.0, 30.0)},
    )
    result = rsa_nd(
        prepared,
        model_rdm,
        search_axes=("band", "time"),
        radii={"band": 0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    named = prepared

    expected = np.empty((2, 5), dtype=float)
    for band_idx in range(2):
        for time_idx in range(5):
            features = named.data[:, :, band_idx, time_idx]
            data_vec = distance.pdist(features, metric="euclidean")
            expected[band_idx, time_idx] = np.corrcoef(data_vec, model_vec)[0, 1]

    assert np.allclose(result.rsa, expected)


def test_standardize_features_in_prepare_matches_manual_named_data():
    labels = np.array(["a", "b", "c", "d"])
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    rng = np.random.default_rng(13)
    data = rng.normal(size=(4, 3, 4, 5))
    coords = {
        "space": ["P1", "P2", "P3"],
        "freq": [4.0, 6.0, 10.0, 20.0],
        "time": [0.0, 0.1, 0.2, 0.3, 0.4],
    }

    standardized_prep = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords=coords,
        freq={"low": (4.0, 8.0), "high": (8.0, 30.0)},
        standardize="features",
    )
    standardized = rsa_nd(
        standardized_prep,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="correlation",
    )

    named = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords=coords,
        freq={"low": (4.0, 8.0), "high": (8.0, 30.0)},
    )
    manual = RsaData(
        data=_zscore_features(named.data, item_axis=named.axis_index("item")),
        axes=named.axes,
        coords=named.coords,
    )
    expected = rsa_nd(
        manual,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="correlation",
    )

    assert np.allclose(standardized.rsa, expected.rsa, equal_nan=True)


def test_prep_then_rsa_nd_matches_rsa_array_for_tfr_time_searchlight():
    import mne_rsa

    tfr = _toy_tfr()
    labels = ["a", "b", "c"]
    model_rdm = _ordinal_rdm([0.0, 1.0, 3.0])

    prepared = _prepare_rsa_data_only(tfr)
    result = rsa_nd(
        prepared,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
        labels=labels,
        labels_rdm_model=labels,
    )
    layout = build_array_layout(
        prepared,
        feature_axes=("space", "freq"),
        sample_axis="time",
    )
    expected = mne_rsa.rsa_array(
        layout.X,
        model_rdm,
        patches=mne_rsa.searchlight(layout.X.shape, temporal_radius=0),
        data_rdm_metric="euclidean",
        labels_X=labels,
        labels_rdm_model=labels,
    )

    assert result.rsa.shape == (6,)
    assert np.allclose(result.rsa, expected, equal_nan=True)


def test_rsa_nd_passes_documented_3d_array_to_mne_rsa(monkeypatch):
    import mne_rsa

    labels = np.array(["a", "b", "c", "d"])
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    data = np.arange(4 * 2 * 3 * 4, dtype=float).reshape(4, 2, 3, 4)
    seen = {}

    def fake_rsa_array(X, rdm_model, patches=None, **kwargs):
        seen["X_shape"] = X.shape
        seen["patch_shape"] = patches.shape
        first_patch = next(iter(patches))
        seen["first_patch"] = first_patch
        return np.arange(np.prod(patches.shape), dtype=float).reshape(patches.shape)

    monkeypatch.setattr(mne_rsa, "rsa_array", fake_rsa_array)

    result = rsa_nd(
        data,
        model_rdm,
        axes=("item", "space", "freq", "time"),
        coords={
            "space": ["P1", "P2"],
            "freq": [6.0, 10.0, 20.0],
            "time": [0.0, 0.1, 0.2, 0.3],
        },
        search_axes=("freq", "time"),
        radii={"freq": 0.0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
    )

    assert seen["X_shape"] == (4, 2 * 3 * 4, 1)
    assert seen["patch_shape"] == (3, 4)
    assert len(seen["first_patch"]) == 3
    assert np.array_equal(seen["first_patch"][1], [0, 12])
    assert result.rsa.shape == (3, 4)


def test_flat_feature_indices_match_meshgrid_oracle_for_2d_feature_volume():
    index_vectors = (np.arange(3), np.array([1, 3]))
    grids = np.meshgrid(*index_vectors, indexing="ij")
    expected = np.ravel_multi_index(
        [grid.ravel() for grid in grids],
        dims=(3, 4),
    )

    actual = _flat_feature_indices_from_vectors(index_vectors, (3, 4))

    assert np.array_equal(actual, expected)


def test_flat_feature_indices_match_meshgrid_oracle_for_3d_feature_volume():
    index_vectors = (np.arange(2), np.array([0, 2]), np.array([1, 3]))
    grids = np.meshgrid(*index_vectors, indexing="ij")
    expected = np.ravel_multi_index(
        [grid.ravel() for grid in grids],
        dims=(2, 3, 4),
    )

    actual = _flat_feature_indices_from_vectors(index_vectors, (2, 3, 4))

    assert np.array_equal(actual, expected)


def test_rsa_nd_uses_abstract_space_axis_as_coordinate_searchlight():
    labels = np.array(["a", "b", "c", "d"])
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    data = np.arange(4 * 3 * 2 * 4, dtype=float).reshape(4, 3, 2, 4)

    result = rsa_nd(
        data,
        model_rdm,
        axes=("item", "space", "freq", "time"),
        coords={
            "space": ["P1", "P2", "P3"],
            "freq": [8.0, 12.0],
            "time": [0.0, 0.1, 0.2, 0.3],
        },
        search_axes=("space", "time"),
        radii={"space": 0.0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
    )

    assert result.rsa.shape == (3, 4)
    assert list(result.search_coords["space"]) == ["P1", "P2", "P3"]
    assert np.allclose(result.search_coords["time"], [0.0, 0.1, 0.2, 0.3])


def test_source_roi_patches_convert_labels_to_stc_vertex_indices():
    stcs, rois, _, _ = _toy_stcs_and_rois()

    patches, kept_rois = source_roi_patches(stcs[0], rois)

    assert list(patches) == ["roi-a-lh", "roi-b-rh"]
    assert [roi.name for roi in kept_rois] == ["roi-a-lh", "roi-b-rh"]
    assert np.array_equal(patches["roi-a-lh"], np.array([0, 2]))
    assert np.array_equal(patches["roi-b-rh"], np.array([4]))


def test_compute_rsa_source_roi_tfr_matches_manual_roi_rsa():
    from mne.time_frequency import tfr_array_morlet

    stcs, rois, model_rdm, labels = _toy_stcs_and_rois()
    result = compute_rsa(
        stcs,
        model_rdm,
        space=rois,
        tfr={"freqs": [10.0], "n_cycles": 1.0, "n_jobs": 1, "verbose": False},
        tfr_feature="power",
        log=False,
        standardize=None,
        search_axes=("space", "time"),
        radii={"time": 0.02},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
        n_jobs=1,
    )

    patches, kept_rois = source_roi_patches(stcs[0], rois)
    first_patch = patches[kept_rois[0].name]
    roi_data = np.stack([stc.data[first_patch] for stc in stcs])
    complex_tfr = tfr_array_morlet(
        roi_data,
        sfreq=100.0,
        freqs=[10.0],
        n_cycles=1.0,
        output="complex",
        n_jobs=1,
        verbose=False,
    )
    manual_prep = _prepare_rsa_data_only(
        complex_tfr,
        axes=("item", "space", "freq", "time"),
        coords={
            "space": np.arange(len(first_patch)),
            "freq": [10.0],
            "time": stcs[0].times,
        },
        tfr_feature="power",
        log=False,
        standardize=None,
    )
    manual = rsa_nd(
        manual_prep,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0.02},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
        n_jobs=1,
    )

    assert result.rsa.shape == (2, len(result.search_coords["time"]))
    assert result.search_coords["space"].tolist() == ["roi-a-lh", "roi-b-rh"]
    assert np.allclose(result.search_coords["time"], manual.search_coords["time"])
    assert np.allclose(result.rsa[0], manual.rsa, equal_nan=True)


def test_prepared_sensor_space_drives_mne_aware_searchlight():
    labels = ["a", "b", "c", "d"]
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    data = np.arange(4 * 3 * 2 * 3, dtype=float).reshape(4, 3, 2, 3)
    tfr = EpochsTFRArray(
        _info(3, with_locs=True),
        data,
        times=np.array([0.0, 0.1, 0.2]),
        freqs=np.array([8.0, 12.0]),
        events=_events(4),
        event_id={"stim": 1},
        method="toy",
    )

    prepared = prepare_rsa_data(tfr)
    result = rsa_nd(
        prepared,
        model_rdm,
        search_axes=("space", "time"),
        radii={"space": 0.011, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
    )

    assert result.rsa.shape == (3, 3)
    assert list(result.search_coords["space"]) == tfr.ch_names


def test_duplicate_numeric_search_coords_raise_clear_error():
    data = np.ones((3, 1, 3), dtype=float)

    with pytest.raises(ValueError, match="Duplicate coordinate values"):
        rsa_nd(
            data,
            _ordinal_rdm([0.0, 1.0, 2.0]),
            axes=("item", "space", "freq"),
            coords={"freq": [8.0, 8.0, 10.0]},
            search_axes=("freq",),
            data_rdm_metric="euclidean",
            labels=["a", "b", "c"],
            labels_rdm_model=["a", "b", "c"],
        )


def test_unknown_radius_keys_raise_clear_error():
    data = np.ones((3, 2, 4), dtype=float)

    with pytest.raises(ValueError, match="Unknown radii keys"):
        rsa_nd(
            data,
            _ordinal_rdm([0.0, 1.0, 2.0]),
            axes=("item", "space", "freq"),
            coords={"freq": [8.0, 10.0, 12.0, 14.0]},
            search_axes=("freq",),
            radii={"frq": 0.0},
            data_rdm_metric="euclidean",
            labels=["a", "b", "c"],
            labels_rdm_model=["a", "b", "c"],
        )


def test_array_labels_must_be_explicit_when_model_labels_are_named():
    data = np.ones((3, 2), dtype=float)

    with pytest.raises(ValueError, match="labels must be provided"):
        rsa_nd(
            data,
            _ordinal_rdm([0.0, 1.0, 2.0]),
            axes=("item", "space"),
            search_axes=(),
            data_rdm_metric="euclidean",
            labels_rdm_model=["a", "b", "c"],
        )


def test_mne_event_labels_must_match_named_model_labels_or_be_explicit():
    tfr = _toy_tfr()
    prepared = _prepare_rsa_data_only(tfr)

    with pytest.raises(ValueError, match="labels must be provided"):
        rsa_nd(
            prepared,
            _ordinal_rdm([0.0, 1.0, 2.0]),
            search_axes=("time",),
            radii={"time": 0.0},
            data_rdm_metric="euclidean",
            labels_rdm_model=["a", "b", "c"],
        )


def test_rsa_data_rejects_raw_complex_values():
    data = np.ones((2, 1), dtype=complex)

    with pytest.raises(ValueError, match="Complex spectral data"):
        RsaData(data, axes=("item", "space"), coords={})


def test_prep_then_rsa_nd_flattens_complex_derived_features_to_real_3d_array(
    monkeypatch,
):
    import mne_rsa

    labels = np.array(["a", "b", "c", "d"])
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    data = np.arange(4 * 2 * 2 * 3, dtype=float).reshape(4, 2, 2, 3) + 1j * np.ones(
        (4, 2, 2, 3), dtype=float
    )
    seen = {}

    def fake_rsa_array(X, rdm_model, patches, **kwargs):
        patch_list = list(patches)
        seen["X_shape"] = X.shape
        seen["is_complex"] = np.iscomplexobj(X)
        seen["patch_shape"] = patches.shape
        seen["first_patch_feature_count"] = len(patch_list[0][1])
        return np.zeros(patches.shape)

    monkeypatch.setattr(mne_rsa, "rsa_array", fake_rsa_array)

    prepared = _prepare_rsa_data_only(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [4.0, 8.0], "time": [0.0, 0.1, 0.2]},
        tfr_feature="real_imag",
    )
    result = rsa_nd(
        prepared,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
    )

    assert seen == {
        "X_shape": (4, 24, 1),
        "is_complex": False,
        "patch_shape": (3,),
        "first_patch_feature_count": 8,
    }
    assert result.rsa.shape == (3,)


def test_rsa_dynamic_generalization_matches_static_model_list():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    dynamic = rsa_dynamic(
        data,
        model_rdms,
        model_times,
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    direct = rsa_nd(
        data,
        list(model_rdms),
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert dynamic.dynamic.mode == "generalization"
    assert dynamic.rsa.shape == (len(model_times), len(coords["time"]))
    assert np.allclose(dynamic.rsa, direct.rsa)
    assert np.allclose(dynamic.dynamic.full_rsa, direct.rsa)


def test_rsa_dynamic_aligned_extracts_matching_time_diagonal():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    dynamic = rsa_dynamic(
        data,
        model_rdms,
        model_times,
        mode="aligned",
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    full = rsa_nd(
        data,
        list(model_rdms),
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    ).rsa

    assert dynamic.rsa.shape == (len(coords["time"]),)
    assert np.allclose(dynamic.rsa, np.diag(full))
    assert dynamic.dynamic.model_indices.tolist() == [0, 1, 2, 3]


def test_rsa_dynamic_lagged_uses_model_time_before_neural_time():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    dynamic = rsa_dynamic(
        data,
        model_rdms,
        model_times,
        mode="lagged",
        lags=[0.0, 0.1],
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    full = dynamic.dynamic.full_rsa
    expected = np.array(
        [
            [full[0, 0], full[1, 1], full[2, 2], full[3, 3]],
            [np.nan, full[0, 1], full[1, 2], full[2, 3]],
        ]
    )

    assert dynamic.rsa.shape == (2, len(coords["time"]))
    assert np.allclose(dynamic.rsa, expected, equal_nan=True)
    assert dynamic.dynamic.lags.tolist() == [0.0, 0.1]


def test_rsa_dynamic_lagged_supports_frequency_time_searchlights():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    dynamic = rsa_dynamic(
        data,
        model_rdms,
        model_times,
        mode="lagged",
        lags=[0.0, 0.1],
        dynamic_axis="time",
        axes=axes,
        coords=coords,
        search_axes=("freq", "time"),
        radii={"freq": 0.0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    full = dynamic.dynamic.full_rsa
    expected = np.full((2, 3, 4), np.nan)
    for freq_idx in range(3):
        for time_idx in range(4):
            expected[0, freq_idx, time_idx] = full[time_idx, freq_idx, time_idx]
            if time_idx > 0:
                expected[1, freq_idx, time_idx] = full[
                    time_idx - 1,
                    freq_idx,
                    time_idx,
                ]

    assert dynamic.rsa.shape == (2, 3, 4)
    assert np.allclose(dynamic.rsa, expected, equal_nan=True)


def test_dynamic_gather_helper_handles_layered_invalid_model_indices():
    from rsa_nd._dynamic import (
        gather_dynamic_along_axis as _gather_dynamic_along_axis,
    )

    full_rsa = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    indices = np.array(
        [
            [0, -1, 2],
            [1, 0, -1],
        ]
    )

    rsa, model_indices = _gather_dynamic_along_axis(
        full_rsa,
        search_axes=("freq", "time"),
        dynamic_axis="time",
        indices_for_axis_layers=indices,
    )

    expected = np.full((2, 2, 3), np.nan)
    expected[0, :, 0] = full_rsa[0, :, 0]
    expected[0, :, 2] = full_rsa[2, :, 2]
    expected[1, :, 0] = full_rsa[1, :, 0]
    expected[1, :, 1] = full_rsa[0, :, 1]

    assert np.allclose(rsa, expected, equal_nan=True)
    assert model_indices[:, 0, :].tolist() == [[0, -1, 2], [1, 0, -1]]
    assert model_indices[:, 1, :].tolist() == [[0, -1, 2], [1, 0, -1]]


def test_rsa_dynamic_validates_inputs_and_marks_out_of_tolerance_nan():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    with pytest.raises(ValueError, match="model_times"):
        rsa_dynamic(
            data,
            model_rdms,
            model_times[:2],
            axes=axes,
            coords=coords,
            search_axes=("time",),
            labels=labels,
            labels_rdm_model=labels,
        )

    with pytest.raises(ValueError, match="dynamic_axis"):
        rsa_dynamic(
            data,
            model_rdms,
            model_times,
            dynamic_axis="time",
            axes=axes,
            coords=coords,
            search_axes=("freq",),
            labels=labels,
            labels_rdm_model=labels,
        )

    with pytest.raises(ValueError, match="lags"):
        rsa_dynamic(
            data,
            model_rdms,
            model_times,
            mode="lagged",
            axes=axes,
            coords=coords,
            search_axes=("time",),
            labels=labels,
            labels_rdm_model=labels,
        )

    sparse = rsa_dynamic(
        data,
        model_rdms[[0, 2]],
        model_times[[0, 2]],
        mode="aligned",
        lag_tolerance=0.01,
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert np.isfinite(sparse.rsa[[0, 2]]).all()
    assert np.isnan(sparse.rsa[[1, 3]]).all()


def test_rsa_dynamic_rejects_static_rdm_model_kwarg():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    with pytest.raises(TypeError, match="Use 'model_rdms', not 'rdm_model'"):
        rsa_dynamic(
            data,
            model_rdms,
            model_times,
            axes=axes,
            coords=coords,
            search_axes=("time",),
            labels=labels,
            labels_rdm_model=labels,
            rdm_model=model_rdms[0],
        )


def test_rsa_dynamic_accepts_single_square_rdm_array():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    dynamic = rsa_dynamic(
        data,
        model_rdms[0],
        model_times[:1],
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    static = rsa_nd(
        data,
        model_rdms[0],
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert dynamic.rsa.shape == (1, len(coords["time"]))
    assert np.allclose(dynamic.rsa[0], static.rsa)


def test_rsa_dynamic_accepts_single_condensed_rdm_array():
    from scipy.spatial.distance import squareform

    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()
    condensed = squareform(model_rdms[0], checks=False)

    dynamic = rsa_dynamic(
        data,
        condensed,
        model_times[:1],
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert dynamic.rsa.shape == (1, len(coords["time"]))


def test_rsa_dynamic_rejects_ambiguous_square_2d_stack():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    with pytest.raises(ValueError, match="ambiguous"):
        rsa_dynamic(
            data,
            model_rdms[0][:3, :3],
            model_times[:3],
            axes=axes,
            coords=coords,
            search_axes=("time",),
            labels=labels[:3],
            labels_rdm_model=labels[:3],
        )


def test_rsa_dynamic_validates_model_label_count_early():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()

    with pytest.raises(ValueError, match="labels_rdm_model"):
        rsa_dynamic(
            data,
            model_rdms,
            model_times,
            axes=axes,
            coords=coords,
            search_axes=("time",),
            labels=labels,
            labels_rdm_model=labels[:3],
        )


def test_rsa_dynamic_forwards_complex_feature_options():
    data, axes, coords, labels, model_rdms, model_times = _dynamic_spectral_array()
    complex_data = data + 1j * (data + 0.5)

    dynamic = rsa_dynamic(
        complex_data,
        model_rdms[:2],
        model_times[:2],
        axes=axes,
        coords=coords,
        search_axes=("time",),
        radii={"time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        tfr_feature="real_imag",
        log=False,
        standardize="features",
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert dynamic.rsa.shape == (2, len(coords["time"]))
    assert tuple(dynamic.info["prepared"]["input_axes"]) == (
        "item",
        "space",
        "freq",
        "time",
        "component",
    )


# ---------------------------------------------------------------------------
# rsa_dynamic_windows
# ---------------------------------------------------------------------------


def _stationary_window_inputs():
    """Neural + model signals whose item structure is constant over time.

    Because nothing varies along ``time``, every random window yields the
    same RSA, so :func:`rsa_dynamic_windows` must collapse exactly to a
    single :func:`rsa_dynamic` call over one window.
    """

    rng = np.random.default_rng(0)
    n_items, n_space, n_time = 5, 4, 40
    sfreq = 100.0
    times = np.arange(n_time) / sfreq
    base = rng.normal(size=(n_items, n_space))
    data = np.repeat(base[:, :, np.newaxis], n_time, axis=2)
    rsa_data = RsaData(
        data=data, axes=("item", "space", "time"), coords={"time": times}
    )
    model_values = np.array([0.0, 1.0, 2.0, 4.0, 7.0])
    model_signal = np.repeat(model_values[:, np.newaxis], n_time, axis=1)
    return rsa_data, model_signal, times, sfreq


def test_rsa_dynamic_windows_matches_rsa_dynamic_for_stationary_signal():
    rsa_data, model_signal, times, sfreq = _stationary_window_inputs()
    window_s = 0.1
    win = int(round(window_s * sfreq))

    result = rsa_dynamic_windows(
        rsa_data,
        model_signal,
        times,
        window_s=window_s,
        n_windows=5,
        random_state=0,
        mode="generalization",
        model_metric="cityblock",
        search_axes=("time",),
        radii={"time": 0.0},
    )

    rel_times = times[:win] - times[0]
    window_data = RsaData(
        data=rsa_data.data[:, :, :win],
        axes=rsa_data.axes,
        coords={"time": rel_times},
    )
    model_rdm = np.abs(model_signal[:, 0][:, None] - model_signal[:, 0][None, :])
    expected = rsa_dynamic(
        window_data,
        np.stack([model_rdm] * win),
        rel_times,
        mode="generalization",
        search_axes=("time",),
        radii={"time": 0.0},
    )

    assert result.dynamic.mode == "generalization"
    assert result.rsa.shape == expected.rsa.shape == (win, win)
    assert np.allclose(result.rsa, expected.rsa, equal_nan=True)
    assert np.allclose(
        result.dynamic.full_rsa, expected.dynamic.full_rsa, equal_nan=True
    )


def test_rsa_dynamic_windows_is_reproducible_and_seed_sensitive():
    rng = np.random.default_rng(1)
    n_items, n_space, n_time = 5, 4, 80
    times = np.arange(n_time) / 100.0
    rsa_data = RsaData(
        data=rng.normal(size=(n_items, n_space, n_time)),
        axes=("item", "space", "time"),
        coords={"time": times},
    )
    model_signal = rng.normal(size=(n_items, n_time))
    kw = dict(
        window_s=0.15,
        n_windows=6,
        search_axes=("time",),
        radii={"time": 0.0},
    )

    a = rsa_dynamic_windows(rsa_data, model_signal, times, random_state=7, **kw)
    b = rsa_dynamic_windows(rsa_data, model_signal, times, random_state=7, **kw)
    c = rsa_dynamic_windows(rsa_data, model_signal, times, random_state=8, **kw)

    assert np.allclose(a.rsa, b.rsa, equal_nan=True)
    assert not np.allclose(a.rsa, c.rsa, equal_nan=True)


def test_rsa_dynamic_windows_validates_arguments():
    rsa_data, model_signal, times, _ = _stationary_window_inputs()

    with pytest.raises(ValueError, match="n_windows"):
        rsa_dynamic_windows(rsa_data, model_signal, times, window_s=0.1, n_windows=0)
    with pytest.raises(ValueError, match="window_s must be positive"):
        rsa_dynamic_windows(rsa_data, model_signal, times, window_s=0.0, n_windows=1)
    with pytest.raises(ValueError, match="exceeds"):
        rsa_dynamic_windows(rsa_data, model_signal, times, window_s=10.0, n_windows=1)


# ---------------------------------------------------------------------------
# Model-RDM shape handling
# ---------------------------------------------------------------------------


def test_3_item_square_rdm_is_treated_as_single_model():
    """A (3, 3) square model RDM is one model, not three condensed RDMs.

    With ``n_items == 3`` the condensed length also equals 3, so a (3, 3) array
    is shape-ambiguous. The runner defers to mne-rsa, which reads it as a single
    square RDM -> one RSA score per searchlight position, never silently
    reinterpreted as three condensed models.
    """
    pytest.importorskip("mne_rsa")

    rng = np.random.default_rng(5)
    n_items, n_space, n_time = 3, 4, 5
    data = rng.normal(size=(n_items, n_space, n_time))
    v = np.arange(n_items, dtype=float)
    square_rdm = np.abs(v[:, None] - v[None, :])  # (3, 3): ambiguous shape

    result = rsa_nd(
        data,
        square_rdm,
        axes=("item", "space", "time"),
        coords={"time": np.arange(n_time) / 100.0},
        search_axes=("time",),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )

    assert result.rsa.shape == (n_time,)


def test_epochs_time_domain_input_through_adapter():
    """A bare time-domain ``mne.Epochs`` exercises the epochs branch end to
    end: item axis from epochs, labels from events[:, 2], and the on-the-fly
    Morlet ``tfr=`` branch.
    """
    rng = np.random.default_rng(0)
    n_ep, n_ch, n_t = 5, 3, 20
    info = mne.create_info(["C1", "C2", "C3"], sfreq=100.0, ch_types="eeg")
    # Distinct event codes -> one condition per epoch, so the labels inferred
    # from events[:, 2] line up with the 5-item model RDM below.
    events = np.column_stack(
        [np.arange(n_ep), np.zeros(n_ep, int), np.arange(1, n_ep + 1)]
    )
    epochs = mne.EpochsArray(
        rng.standard_normal((n_ep, n_ch, n_t)) * 1e-6,
        info,
        events=events,
        tmin=0.0,
        verbose=False,
    )

    prepared = prepare_rsa_data(epochs)
    assert prepared.input_kind == "epochs"
    assert prepared.rsa_data.axes == ("item", "space", "time")
    assert prepared.rsa_data.data.shape == (n_ep, n_ch, n_t)
    np.testing.assert_array_equal(prepared.item_labels, events[:, 2])
    assert type(prepared.space).__name__ == "SensorSpace"

    model = np.abs(np.arange(n_ep)[:, None] - np.arange(n_ep)[None, :]).astype(float)
    result = rsa_nd(prepared, model, search_axes=("time",), radii={"time": 0.0})
    assert np.asarray(result.rsa).shape == (n_t,)

    prepared_tfr = prepare_rsa_data(
        epochs, tfr={"freqs": np.array([10.0, 20.0]), "n_cycles": 1.0}
    )
    assert prepared_tfr.rsa_data.axes == ("item", "space", "freq", "time")


def _grouping_fixture(seed=0):
    rng = np.random.default_rng(seed)
    n_item, n_space, n_time = 6, 5, 9
    times = np.linspace(0.0, 1.0, n_time)
    arr = rng.standard_normal((n_item, n_space, n_time))
    v = np.arange(n_item, dtype=float)
    model = np.abs(v[:, None] - v[None, :])
    data = RsaData(arr, axes=("item", "space", "time"), coords={"time": times})
    return data, arr, times, model


def test_time_mode_elements_one_rsa_per_window():
    data, arr, times, model = _grouping_fixture()

    prepared = prepare_rsa_data(
        data, time={"w1": (0.0, 0.5), "w2": (0.5, 2.0)}, time_mode="elements"
    )
    # Elements mode keeps the native time axis and records the groups.
    assert prepared.rsa_data.axes == ("item", "space", "time")
    assert set(prepared.axis_groups) == {"time"}
    assert prepared.axis_groups["time"].mode == "elements"

    res = rsa_nd(prepared, model, search_axes=("time",))
    assert np.asarray(res.rsa).shape == (2,)
    assert list(res.search_coords["time"]) == ["w1", "w2"]

    # Each group keeps all its member time points as joint features, so it must
    # equal RSA over the explicitly sliced window (left-closed range rule).
    for pos, (lo, hi) in enumerate([(0.0, 0.5), (0.5, 2.0)]):
        mask = (times >= lo) & (times < hi)
        sliced = RsaData(
            arr[:, :, mask],
            axes=("item", "space", "time"),
            coords={"time": times[mask]},
        )
        ref = rsa_nd(sliced, model)
        np.testing.assert_allclose(
            np.squeeze(np.asarray(res.rsa)[pos]), np.squeeze(np.asarray(ref.rsa))
        )


def test_time_mode_elements_vs_aggregate_differ():
    """elements keeps member points; aggregate (default) averages them to a scalar."""
    data, _, _, model = _grouping_fixture()
    windows = {"w1": (0.0, 0.5), "w2": (0.5, 2.0)}

    elements = rsa_nd(
        prepare_rsa_data(data, time=windows, time_mode="elements"),
        model,
        search_axes=("time",),
    )
    # Aggregate twin: named windows collapse each range to its mean -> 'window'
    # axis; a radius-0 search then yields one RSA per window value.
    aggregated = rsa_nd(
        prepare_rsa_data(data, time=windows),  # time_mode="aggregate" (default)
        model,
        search_axes=("window",),
        radii={"window": 0.0},
    )
    assert np.asarray(elements.rsa).shape == np.asarray(aggregated.rsa).shape == (2,)
    assert not np.allclose(np.asarray(elements.rsa), np.asarray(aggregated.rsa))


def test_mixed_grouping_modes_aggregate_freq_elements_time():
    """One prepare call: aggregate freq into a band, keep time as elements."""
    rng = np.random.default_rng(3)
    times = np.linspace(0.0, 1.0, 9)
    freqs = np.array([8.0, 12.0, 20.0])
    arr = rng.standard_normal((6, 4, 3, 9))
    v = np.arange(6, dtype=float)
    model = np.abs(v[:, None] - v[None, :])
    data = RsaData(
        arr,
        axes=("item", "space", "freq", "time"),
        coords={"freq": freqs, "time": times},
    )

    prepared = prepare_rsa_data(
        data,
        freq={"alpha": (8.0, 13.0)},
        freq_mode="aggregate",
        time={"early": (0.0, 0.5), "late": (0.5, 2.0)},
        time_mode="elements",
    )
    assert "band" in prepared.rsa_data.axes  # freq collapsed
    assert "time" in prepared.rsa_data.axes  # time kept native
    # Array input has no template, so aggregate (band) back-fill metadata is not
    # recorded; only the elements-mode time grouping is.
    assert set(prepared.axis_groups) == {"time"}
    assert prepared.axis_groups["time"].mode == "elements"

    res = rsa_nd(prepared, model, search_axes=("time",))
    assert np.asarray(res.rsa).shape == (2,)
    assert list(res.search_coords["time"]) == ["early", "late"]


def test_grouping_mode_validation_errors():
    data, _, _, model = _grouping_fixture()

    # elements mode requires named ranges, not a plain (start, stop) range
    with pytest.raises(ValueError, match="requires named ranges"):
        prepare_rsa_data(data, time=(0.0, 0.5), time_mode="elements")

    # unknown mode value
    with pytest.raises(ValueError, match="must be 'aggregate' or 'elements'"):
        prepare_rsa_data(data, time={"w": (0.0, 1.0)}, time_mode="nope")

    # empty group (no coordinate values in range) is caught at prepare time
    with pytest.raises(ValueError, match="no coordinate values"):
        prepare_rsa_data(data, time={"none": (5.0, 6.0)}, time_mode="elements")

    # an element-grouped axis must be searched
    prepared = prepare_rsa_data(data, time={"w": (0.0, 1.0)}, time_mode="elements")
    with pytest.raises(ValueError, match="must be in search_axes"):
        rsa_nd(prepared, model, search_axes=())

    # an axis cannot take both a radius and element groups
    with pytest.raises(ValueError, match="both a radius and element groups"):
        rsa_nd(prepared, model, search_axes=("time",), radii={"time": 0.1})


def _grouped_sensor_epochs(seed=0, n_item=6, n_t=12):
    rng = np.random.default_rng(seed)
    ch = ["Fz", "Cz", "Pz", "Oz"]
    info = mne.create_info(ch, sfreq=100.0, ch_types="eeg")
    X = rng.standard_normal((n_item, 4, n_t))
    events = np.column_stack(
        [np.arange(n_item), np.zeros(n_item, int), np.arange(n_item)]
    )
    epochs = mne.EpochsArray(X, info, events=events, tmin=0.0, verbose=False)
    groups = {"frontal": ["Fz", "Cz"], "posterior": ["Pz", "Oz"]}
    v = np.arange(n_item, dtype=float)
    model = np.abs(v[:, None] - v[None, :])
    return epochs, X, groups, model


def test_space_aggregate_mode_collapses_channels_to_roi():
    epochs, X, groups, model = _grouped_sensor_epochs()

    prepared = prepare_rsa_data(epochs, space=groups, space_mode="aggregate")
    assert prepared.rsa_data.axes == ("item", "roi", "time")
    assert list(prepared.rsa_data.coords["roi"]) == ["frontal", "posterior"]
    assert prepared.space is None  # per-element geometry dropped after collapse
    np.testing.assert_allclose(
        prepared.rsa_data.data[:, 0, :], X[:, [0, 1], :].mean(axis=1)
    )

    res = rsa_nd(prepared, model, search_axes=("roi",))
    assert np.asarray(res.rsa).shape == (2,)

    # equals RSA on manually channel-averaged data
    man = np.stack([X[:, [0, 1], :].mean(1), X[:, [2, 3], :].mean(1)], axis=1)
    man_data = RsaData(
        man,
        axes=("item", "roi", "time"),
        coords={"roi": np.array(["frontal", "posterior"], dtype=object)},
    )
    ref = rsa_nd(man_data, model, search_axes=("roi",))
    np.testing.assert_allclose(np.asarray(res.rsa), np.asarray(ref.rsa))


def test_space_aggregate_vs_elements_mode_differ():
    """Aggregate averages members to one signal; elements keeps them as features."""
    epochs, _, groups, model = _grouped_sensor_epochs(seed=1)

    aggregated = rsa_nd(
        prepare_rsa_data(epochs, space=groups, space_mode="aggregate"),
        model,
        search_axes=("roi",),
    )
    elements = rsa_nd(
        prepare_rsa_data(epochs, space=groups),  # default elements
        model,
        search_axes=("space",),
    )
    assert np.asarray(aggregated.rsa).shape == np.asarray(elements.rsa).shape == (2,)
    assert not np.allclose(np.asarray(aggregated.rsa), np.asarray(elements.rsa))


def test_space_aggregate_mode_requires_grouped_space():
    epochs, _, _, _ = _grouped_sensor_epochs(seed=2)
    with pytest.raises(ValueError, match="requires space= groups"):
        prepare_rsa_data(epochs, space_mode="aggregate")
