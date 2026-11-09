"""Tests for dynamic RSA."""

import mne
import numpy as np
import pytest
from _helpers import make_info
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import pdist, squareform

from rsa_nd import RsaData, prepare_rsa_data, rsa_dynamic, rsa_dynamic_windows, rsa_nd

N_ITEMS = 7
TIMES = np.arange(10) / 10
AXES = ("item", "space", "time")
METRIC = dict(data_rdm_metric="euclidean")


def _data():
    rng = np.random.default_rng(0)
    return RsaData(rng.standard_normal((N_ITEMS, 3, 10)), AXES, {"time": TIMES})


def _rdms(n_times=10):
    rng = np.random.default_rng(1)
    return [pdist(rng.standard_normal((N_ITEMS, 1))) for _ in range(n_times)]


def test_generalization_is_static_rsa_per_model():
    result = rsa_dynamic(_data(), _rdms(), TIMES, **METRIC)
    static = rsa_nd(_data(), _rdms(), search_axes=("time",), **METRIC)
    assert result.axes == ("model_time", "time")
    assert_allclose(result.rsa, static.rsa)
    assert result.dynamic.mode == "generalization"
    assert_allclose(result.dynamic.full_rsa, result.rsa)


def test_aligned_takes_the_diagonal():
    result = rsa_dynamic(_data(), _rdms(), TIMES, mode="aligned", **METRIC)
    assert result.axes == ("time",)
    assert_allclose(result.rsa, np.diag(result.dynamic.full_rsa))
    assert_array_equal(result.dynamic.model_indices, np.arange(10))


def test_lagged_uses_earlier_model_times():
    result = rsa_dynamic(
        _data(), _rdms(), TIMES, mode="lagged", lags=[0.1, 0.2], **METRIC
    )
    full = result.dynamic.full_rsa
    assert result.axes == ("lag", "time")
    assert_array_equal(result.coords["lag"], [0.1, 0.2])
    assert_allclose(result.rsa[0, 1:], np.diag(full, 1))
    assert_allclose(result.rsa[1, 2:], np.diag(full, 2))
    assert np.isnan(result.rsa[1, :2]).all()
    assert_array_equal(result.dynamic.model_indices[1, :2], -1)


def test_dynamic_axis_can_be_any_search_position():
    result = rsa_dynamic(
        _data(),
        _rdms(),
        TIMES,
        mode="aligned",
        search_axes=("space", "time"),
        **METRIC,
    )
    assert result.axes == ("space", "time")
    full = result.dynamic.full_rsa
    assert_allclose(result.rsa[2], np.diag(full[:, 2, :]))


def test_lag_tolerance():
    model_times = TIMES[::2]
    kwargs = dict(mode="aligned", **METRIC)
    auto = rsa_dynamic(_data(), _rdms(5), model_times, **kwargs)
    assert np.isfinite(auto.rsa).all()
    strict = rsa_dynamic(_data(), _rdms(5), model_times, lag_tolerance=0.01, **kwargs)
    assert np.isnan(strict.rsa[1::2]).all()
    assert np.isfinite(strict.rsa[::2]).all()
    with pytest.raises(ValueError, match="lag_tolerance"):
        rsa_dynamic(_data(), _rdms(5), model_times, lag_tolerance=-1, **kwargs)


@pytest.mark.parametrize("condensed", (True, False))
def test_single_rdm_array(condensed):
    single = pdist(np.arange(7.0)[:, None])
    if not condensed:
        single = squareform(single)
    result = rsa_dynamic(_data(), single, [0.0], **METRIC)
    assert result.rsa.shape == (1, 10)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(mode="nope"), "mode must"),
        (dict(mode="lagged"), "needs lags"),
        (dict(mode="lagged", lags=[]), "non-empty"),
        (dict(dynamic_axis="space"), "dynamic_axis must"),
        (dict(radii={"time": None}), "radius of None"),
    ],
)
def test_option_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rsa_dynamic(_data(), _rdms(), TIMES, **kwargs)


def test_model_input_errors():
    with pytest.raises(ValueError, match="strictly increasing"):
        rsa_dynamic(_data(), _rdms(), TIMES[::-1])
    with pytest.raises(ValueError, match="model RDMs"):
        rsa_dynamic(_data(), _rdms(3), TIMES)
    with pytest.raises(ValueError, match="same shape"):
        rsa_dynamic(_data(), [*_rdms(9), np.ones(3)], TIMES)
    with pytest.raises(ValueError, match="ambiguous"):
        rsa_dynamic(_data(), np.eye(10), TIMES)


def test_dynamic_passes_preparation_options():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_ITEMS, 3, 4, 10)) + 1j
    result = rsa_dynamic(
        data,
        _rdms(),
        TIMES,
        axes=("item", "space", "freq", "time"),
        coords={"time": TIMES},
        tfr_feature="phase",
        **METRIC,
    )
    assert result.info["input_axes"][-1] == "component"


def _windows(data=None, **kwargs):
    signal = np.random.default_rng(2).standard_normal((N_ITEMS, 10))
    defaults = dict(window_s=0.4, n_windows=4, random_state=0, **METRIC)
    return rsa_dynamic_windows(
        _data() if data is None else data, signal, TIMES, **{**defaults, **kwargs}
    )


def test_windows_use_relative_time():
    result = _windows()
    assert result.axes == ("model_time", "time")
    assert result.rsa.shape == (4, 4)
    assert_allclose(result.coords["time"], TIMES[:4])
    assert_allclose(result.dynamic.full_rsa, result.rsa)


def test_windows_average_over_runs():
    assert_allclose(_windows().rsa, _windows().rsa)
    assert not np.allclose(_windows().rsa, _windows(random_state=1).rsa)


def test_windows_keep_geometry_and_labels():
    """Windows must not lose what prepare_rsa_data() read from MNE objects."""
    rng = np.random.default_rng(0)
    info = make_info(["Fz", "Cz", "Pz"], 10.0)
    codes = np.array([7, 6, 5, 4, 3, 2, 1])
    events = np.column_stack([np.arange(7) * 20, np.zeros(7, int), codes])
    epochs = mne.EpochsArray(
        rng.standard_normal((N_ITEMS, 3, 10)), info, events=events, verbose=False
    )
    result = _windows(
        prepare_rsa_data(epochs),
        search_axes=("space", "time"),
        radii={"space": 0.09},
    )
    assert result.space is not None
    assert list(result.coords["space"]) == ["Fz", "Cz", "Pz"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_windows=0), "n_windows"),
        (dict(window_s=0), "window_s must"),
        (dict(window_s=0.1), "too short"),
        (dict(window_s=5.0), "longer than"),
        (dict(dynamic_axis="nope"), "no 'nope' axis"),
    ],
)
def test_window_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _windows(**kwargs)


def test_window_signal_shape():
    with pytest.raises(ValueError, match="model_signal"):
        rsa_dynamic_windows(_data(), np.ones(10), TIMES, window_s=0.4, n_windows=1)
    with pytest.raises(ValueError, match="one value per"):
        rsa_dynamic_windows(
            _data(), np.ones((N_ITEMS, 9)), TIMES, window_s=0.4, n_windows=1
        )
