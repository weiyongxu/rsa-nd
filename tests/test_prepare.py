"""Tests for reading inputs and building features."""

import mne
import numpy as np
import pytest
from mne.time_frequency import AverageTFRArray, EpochsTFRArray, SpectrumArray
from numpy.testing import assert_allclose, assert_array_equal

from rsa_nd import RsaData, prepare_rsa_data, source_roi_patches

CH_NAMES = ["Fz", "Cz", "Pz", "Oz"]
SFREQ = 100.0
FREQS = np.array([4.0, 8.0, 13.0, 30.0])
AXES = ("item", "freq")


def _epochs(n_items=4, n_times=40):
    rng = np.random.default_rng(0)
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    return mne.EpochsArray(
        rng.standard_normal((n_items, len(CH_NAMES), n_times)), info, verbose=False
    )


def _array(*shape):
    return np.random.default_rng(0).uniform(0.5, 1.5, shape)


def test_float32_is_kept():
    """Large tensors should not be silently doubled in size."""
    prepared = prepare_rsa_data(
        _array(3, 4).astype(np.float32), axes=AXES, coords={"freq": FREQS}
    )
    assert prepared.rsa_data.data.dtype == np.float32


def test_rsa_data_rejects_complex_and_bad_coords():
    with pytest.raises(ValueError, match="real-valued"):
        RsaData(np.ones((2, 2), complex), ("item", "time"))
    with pytest.raises(ValueError, match="Unknown coordinate"):
        RsaData(np.ones((2, 2)), ("item", "time"), {"freq": [1, 2]})
    with pytest.raises(ValueError, match="coordinates"):
        RsaData(np.ones((2, 2)), ("item", "time"), {"time": [1, 2, 3]})
    with pytest.raises(ValueError, match="unique"):
        RsaData(np.ones((2, 2)), ("item", "item"))


def test_array_needs_axes_and_item_axis():
    with pytest.raises(ValueError, match="axes is required"):
        prepare_rsa_data(np.ones((2, 2)))
    with pytest.raises(ValueError, match="'item' axis"):
        prepare_rsa_data(np.ones((2, 2)), axes=("a", "b"))


def test_array_rejects_mne_only_options():
    with pytest.raises(ValueError, match="space= needs MNE"):
        prepare_rsa_data(np.ones((2, 2)), axes=AXES, space={"a": [0]})
    with pytest.raises(ValueError, match="tfr= needs MNE"):
        prepare_rsa_data(np.ones((2, 2)), axes=AXES, tfr={"freqs": FREQS})


@pytest.mark.parametrize(
    "closed, expected",
    [(None, [8, 13]), ("left", [8]), ("right", [13]), ("neither", [])],
)
def test_range_ends(closed, expected):
    kwargs = dict(axes=AXES, coords={"freq": FREQS}, freq=(8, 13), range_closed=closed)
    if not expected:
        with pytest.raises(ValueError, match="contains no values"):
            prepare_rsa_data(_array(2, 4), **kwargs)
        return
    prepared = prepare_rsa_data(_array(2, 4), **kwargs)
    assert_array_equal(prepared.rsa_data.coords["freq"], expected)


def test_named_ranges_do_not_share_edges():
    """Adjacent named groups must not count the shared edge twice."""
    data = _array(2, 4)
    prepared = prepare_rsa_data(
        data,
        axes=AXES,
        coords={"freq": FREQS},
        freq={"low": (4, 13), "high": (13, 31)},
    )
    assert prepared.rsa_data.axes == ("item", "band")
    assert_allclose(prepared.rsa_data.data[:, 0], data[:, :2].mean(axis=1))
    assert_allclose(prepared.rsa_data.data[:, 1], data[:, 2:].mean(axis=1))


def test_bad_ranges_raise():
    kwargs = dict(axes=AXES, coords={"freq": FREQS})
    with pytest.raises(ValueError, match="stop > start"):
        prepare_rsa_data(_array(2, 4), freq=(13, 8), **kwargs)
    with pytest.raises(ValueError, match="at least one range"):
        prepare_rsa_data(_array(2, 4), freq={}, **kwargs)
    with pytest.raises(ValueError, match="no such axis"):
        prepare_rsa_data(_array(2, 4), time=(0, 1), **kwargs)
    with pytest.raises(ValueError, match="needs named ranges"):
        prepare_rsa_data(_array(2, 4), freq=(8, 13), freq_mode="elements", **kwargs)


@pytest.mark.parametrize("agg", ["median", "max", np.mean])
def test_agg(agg):
    data = _array(2, 4)
    prepared = prepare_rsa_data(
        data, axes=AXES, coords={"freq": FREQS}, freq={"all": (0, 31)}, agg=agg
    )
    func = getattr(np, agg) if isinstance(agg, str) else agg
    assert_allclose(prepared.rsa_data.data[:, 0], func(data, axis=1))


def test_invalid_options_raise():
    kwargs = dict(axes=AXES, coords={"freq": FREQS})
    with pytest.raises(ValueError, match="agg must"):
        prepare_rsa_data(_array(2, 4), freq={"a": (0, 31)}, agg="nope", **kwargs)
    with pytest.raises(ValueError, match="standardize"):
        prepare_rsa_data(_array(2, 4), standardize=False, **kwargs)
    with pytest.raises(ValueError, match="Modes"):
        prepare_rsa_data(_array(2, 4), time_mode="nope", **kwargs)
    with pytest.raises(ValueError, match="tfr_feature"):
        prepare_rsa_data(_array(2, 4), tfr_feature="nope", **kwargs)
    with pytest.raises(ValueError, match="range_closed"):
        prepare_rsa_data(_array(2, 4), freq=(4, 8), range_closed="nope", **kwargs)


def test_log_floor():
    """Zeros are clipped, so the log never gives -inf."""
    data = np.array([[0.0, 1.0], [10.0, 100.0]])
    kwargs = dict(axes=("item", "x"), log=True)
    logged = prepare_rsa_data(data, **kwargs).rsa_data.data
    assert np.isfinite(logged).all()
    assert_allclose(logged[0, 0], np.log10(10 * 1e-6))
    floored = prepare_rsa_data(data, log_floor=0.1, **kwargs).rsa_data.data
    assert_allclose(floored[0, 0], -1)
    with pytest.raises(ValueError, match="positive"):
        prepare_rsa_data(data, log_floor=0, **kwargs)


def test_standardize_handles_constant_features():
    data = np.array([[1.0, 5.0], [3.0, 5.0]])
    prepared = prepare_rsa_data(data, axes=("item", "x"), standardize="features")
    assert_allclose(prepared.rsa_data.data, [[-1, 0], [1, 0]])


def test_complex_features():
    data = np.array([[3 + 4j, 0j]])
    kwargs = dict(axes=("item", "x"))
    for feature, expected in (("power", [25, 0]), ("amplitude", [5, 0])):
        prepared = prepare_rsa_data(data, tfr_feature=feature, **kwargs)
        assert_allclose(prepared.rsa_data.data[0], expected)
    prepared = prepare_rsa_data(data, tfr_feature="amp_phase", **kwargs)
    assert prepared.rsa_data.axes == ("item", "x", "component")
    assert list(prepared.rsa_data.coords["component"]) == [
        "amplitude",
        "sin_phase",
        "cos_phase",
    ]
    # Zero amplitude has no phase, so its sin and cos stay zero.
    assert_allclose(prepared.rsa_data.data[0], [[5, 0.8, 0.6], [0, 0, 0]])


def test_complex_feature_errors():
    data = np.ones((2, 2), complex)
    with pytest.raises(ValueError, match="needs complex"):
        prepare_rsa_data(data.real, axes=AXES, tfr_feature="phase")
    with pytest.raises(ValueError, match="log=True"):
        prepare_rsa_data(data, axes=AXES, tfr_feature="phase", log=True)
    with pytest.raises(ValueError, match="does not work with"):
        prepare_rsa_data(
            data,
            axes=AXES,
            coords={"freq": [1, 2]},
            tfr_feature="phase",
            freq={"a": (0, 3)},
        )
    with pytest.raises(ValueError, match="component"):
        prepare_rsa_data(data, axes=("item", "component"), tfr_feature="phase")


def test_bad_channels_are_excluded_by_default():
    epochs = _epochs()
    epochs.info["bads"] = ["Cz"]
    prepared = prepare_rsa_data(epochs)
    assert list(prepared.rsa_data.coords["space"]) == ["Fz", "Pz", "Oz"]
    assert_array_equal(prepared.rsa_data.data, epochs.get_data(picks=[0, 2, 3]))
    assert prepared.space.channel_names == ("Fz", "Pz", "Oz")

    tfr = EpochsTFRArray(
        epochs.info, _array(4, 4, 2, 5), np.arange(5) / SFREQ, FREQS[:2]
    )
    assert prepare_rsa_data(tfr, picks=None).rsa_data.data.shape[1] == 4


def test_event_codes_become_item_labels():
    epochs = _epochs()
    epochs.events[:, 2] = [3, 3, 7, 7]
    assert_array_equal(prepare_rsa_data(epochs).item_labels, [3, 3, 7, 7])


def test_spectrum_has_no_time_axis():
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    spectra = [SpectrumArray(_array(4, 4), info, FREQS) for _ in range(3)]
    assert prepare_rsa_data(spectra).rsa_data.axes == ("item", "space", "freq")


def test_morlet_tfr_from_time_domain():
    prepared = prepare_rsa_data(
        _epochs(),
        tfr={"freqs": np.array([10.0, 20.0]), "n_cycles": 2.0, "verbose": False},
    )
    assert prepared.rsa_data.axes == ("item", "space", "freq", "time")
    assert prepared.rsa_data.data.shape == (4, 4, 2, 40)
    assert np.isrealobj(prepared.rsa_data.data)


@pytest.mark.parametrize("decim", (2, slice(0, 40, 4)))
def test_morlet_decimation_keeps_times_aligned(decim):
    epochs = _epochs()
    tfr = {"freqs": [20.0], "n_cycles": 2.0, "decim": decim, "verbose": False}
    prepared = prepare_rsa_data(epochs, tfr=tfr).rsa_data
    step = decim if isinstance(decim, int) else decim.step
    assert_allclose(prepared.coords["time"], epochs.times[::step])


def test_tfr_option_errors():
    with pytest.raises(ValueError, match="include 'freqs'"):
        prepare_rsa_data(_epochs(), tfr={"n_cycles": 2.0})
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    average = AverageTFRArray(info, _array(4, 2, 5), np.arange(5) / SFREQ, FREQS[:2])
    with pytest.raises(ValueError, match="time-domain"):
        prepare_rsa_data([average, average], tfr={"freqs": FREQS})


def test_tapered_spectra_are_rejected_clearly():
    spectrum = _epochs(n_times=100).compute_psd(
        method="multitaper", output="complex", verbose=False
    )
    with pytest.raises(ValueError, match="tapers"):
        prepare_rsa_data(spectrum)


def test_items_must_match():
    evokeds = [epochs.average() for epochs in (_epochs(), _epochs())]
    other = evokeds[1].copy().rename_channels({"Oz": "O1"})
    with pytest.raises(ValueError, match="does not match item 0"):
        prepare_rsa_data([evokeds[0], other])
    shifted = evokeds[1].copy().shift_time(0.5)
    with pytest.raises(ValueError, match="does not match item 0"):
        prepare_rsa_data([evokeds[0], shifted])


def _stcs(vertices=None):
    vertices = vertices or [np.array([0, 2, 5]), np.array([1, 3])]
    return [
        mne.SourceEstimate(_array(5, 4), vertices, tmin=0, tstep=0.01, subject="s")
        for _ in range(3)
    ]


def test_source_vertices_must_match():
    first = _stcs()[0]
    second = _stcs([np.array([0, 2, 6]), np.array([1, 3])])[0]
    with pytest.raises(ValueError, match="does not match item 0"):
        prepare_rsa_data([first, second])


def test_vector_source_estimates_are_rejected():
    vector = mne.VectorSourceEstimate(
        _array(5, 3, 4), [np.arange(3), np.arange(2)], 0, 0.01
    )
    with pytest.raises(TypeError, match="SourceEstimate"):
        prepare_rsa_data([vector, vector])


def test_unsupported_input():
    with pytest.raises(TypeError, match="inst must be"):
        prepare_rsa_data("data.fif")
    with pytest.raises(TypeError, match="inst must be"):
        prepare_rsa_data([])


def test_sensor_group_errors():
    with pytest.raises(ValueError, match="unknown channels"):
        prepare_rsa_data(_epochs(), space={"a": ["Fz", "X1"]})
    with pytest.raises(TypeError, match="map group names"):
        prepare_rsa_data(_epochs(), space=["Fz"])
    with pytest.raises(ValueError, match="needs groups"):
        prepare_rsa_data(_epochs(), space_mode="aggregate")


def test_source_roi_patches():
    stc = _stcs()[0]
    rois = [
        mne.Label(np.array([0, 5, 9]), hemi="lh", name="a-lh"),
        mne.Label(np.array([3]), hemi="rh", name="b-rh"),
        mne.Label(np.array([8]), hemi="rh", name="empty-rh"),
    ]
    patches, kept = source_roi_patches(stc, rois)
    assert list(patches) == ["a-lh", "b-rh"]
    assert kept == rois[:2]
    assert_array_equal(patches["a-lh"], [0, 2])
    assert_array_equal(patches["b-rh"], [4])  # rh rows follow the 3 lh rows
    with pytest.raises(ValueError, match="Duplicate ROI"):
        source_roi_patches(stc, [rois[0], rois[0]])


def test_bihemi_label_covers_both_hemispheres():
    lh = mne.Label(np.array([0]), hemi="lh", name="a-lh")
    rh = mne.Label(np.array([1]), hemi="rh", name="a-rh")
    prepared = prepare_rsa_data(_stcs(), space={"both": lh + rh})
    assert_array_equal(prepared.space.patches["both"], [0, 3])


def test_labels_without_vertices_raise():
    label = mne.Label(np.array([99]), hemi="lh", name="a-lh")
    with pytest.raises(ValueError, match="No group"):
        prepare_rsa_data(_stcs(), space=[label])
