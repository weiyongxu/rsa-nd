"""Tests for running RSA, including agreement with mne-rsa."""

import mne
import mne_rsa
import numpy as np
import pytest
from _helpers import make_info
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from rsa_nd import (
    PreparedRsaData,
    RsaData,
    compute_rsa,
    prepare_rsa_data,
    rsa_nd,
)

N_ITEMS = 8
CH_NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4"]
SFREQ = 100.0
MODEL = pdist(np.arange(N_ITEMS, dtype=float)[:, np.newaxis])


def _data(*shape):
    return np.random.default_rng(0).standard_normal((N_ITEMS, *shape))


def _info():
    return make_info(CH_NAMES, SFREQ)


def _epochs():
    codes = np.arange(N_ITEMS) + 1
    events = np.column_stack([codes * 30, np.zeros(N_ITEMS, int), codes])
    return mne.EpochsArray(_data(6, 30), _info(), events=events, verbose=False)


def test_no_search_gives_one_score():
    data = _data(3, 4)
    result = rsa_nd(data, MODEL, axes=("item", "a", "b"))
    assert result.axes == ()
    expected = spearmanr(pdist(data.reshape(N_ITEMS, -1), "correlation"), MODEL)[0]
    assert_allclose(result.rsa, expected)


def test_any_axis_names_work():
    """Nothing is special about time, frequency or space."""
    data = _data(3, 4, 5)
    axes = ("subject", "item", "voxel", "lag")
    moved = np.moveaxis(data, 0, 1)
    coords = {"subject": ["a", "b", "c"][: moved.shape[0]], "lag": np.arange(5) * 0.1}
    result = rsa_nd(
        moved,
        MODEL,
        axes=axes,
        coords=coords,
        search_axes=("subject", "lag"),
        radii={"lag": 0.1},
    )
    assert result.axes == ("subject", "lag")
    assert result.rsa.shape == (3, 5)
    assert list(result.coords["subject"]) == ["a", "b", "c"]
    expected = spearmanr(
        pdist(data[:, 1, :, :2].reshape(N_ITEMS, -1), "correlation"), MODEL
    )[0]
    assert_allclose(result.rsa[1, 0], expected)


def test_search_axes_set_the_output_order():
    data = _data(3, 4)
    kwargs = dict(axes=("item", "a", "b"), data_rdm_metric="euclidean")
    first = rsa_nd(data, MODEL, search_axes=("a", "b"), **kwargs)
    second = rsa_nd(data, MODEL, search_axes=("b", "a"), **kwargs)
    assert second.axes == ("b", "a")
    assert_allclose(first.rsa, second.rsa.T)


def test_radius_includes_the_edge_despite_rounding():
    """A radius of 2 samples means 2 samples at every time point."""
    times = np.arange(20) / 50
    result = rsa_nd(
        _data(2, 20),
        MODEL,
        axes=("item", "x", "time"),
        coords={"time": times},
        search_axes=("time",),
        radii={"time": 0.04},
    )
    edges = rsa_nd(
        _data(2, 20),
        MODEL,
        axes=("item", "x", "time"),
        coords={"time": np.arange(20)},
        search_axes=("time",),
        radii={"time": 2},
    )
    assert_allclose(result.rsa, edges.rsa)


def test_radius_none_pools_the_axis():
    data = _data(3, 4)
    kwargs = dict(axes=("item", "a", "b"))
    pooled = rsa_nd(data, MODEL, search_axes=("a", "b"), radii={"b": None}, **kwargs)
    plain = rsa_nd(data, MODEL, search_axes=("a",), **kwargs)
    assert pooled.axes == ("a",)
    assert_allclose(pooled.rsa, plain.rsa)


def test_sensor_radius_is_strict():
    """As in mne-rsa, a channel exactly at the radius is excluded."""
    epochs = _epochs()
    pos = np.array([ch["loc"][:3] for ch in epochs.info["chs"]])
    radius = np.linalg.norm(pos[0] - pos[1])
    result = rsa_nd(
        prepare_rsa_data(epochs),
        MODEL,
        search_axes=("space",),
        radii={"space": radius},
    )
    expected = np.flatnonzero(np.linalg.norm(pos - pos[0], axis=1) < radius)
    assert 1 not in expected
    rdm = pdist(epochs.get_data()[:, expected].reshape(N_ITEMS, -1), "correlation")
    assert_allclose(result.rsa[0], spearmanr(rdm, MODEL)[0])


def test_space_radius_needs_channel_locations():
    epochs = _epochs()
    for ch in epochs.info["chs"]:
        ch["loc"][:3] = 0
    with pytest.raises(ValueError, match="location"):
        compute_rsa(epochs, MODEL, search_axes=("space",), radii={"space": 0.1})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(search_axes=("item",)), "'item' cannot"),
        (dict(search_axes=("nope",)), "not present"),
        (dict(search_axes=("a", "a")), "unique"),
        (dict(search_axes=("a",), radii={"b": 1}), "not searched"),
        (dict(search_axes=("a",), radii={"a": -1}), "non-negative"),
    ],
)
def test_search_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rsa_nd(_data(3, 4), MODEL, axes=("item", "a", "b"), **kwargs)


def test_named_coordinates_only_take_radius_zero():
    coords = {"a": ["x", "y", "z"]}
    with pytest.raises(ValueError, match="radius of 0"):
        rsa_nd(
            _data(3),
            MODEL,
            axes=("item", "a"),
            coords=coords,
            search_axes=("a",),
            radii={"a": 1},
        )


def test_duplicate_coordinates_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        rsa_nd(
            _data(3),
            MODEL,
            axes=("item", "a"),
            coords={"a": [1, 1, 2]},
            search_axes=("a",),
        )


def test_named_groups_must_be_searched_without_radius():
    groups = {"a": ["Fz", "Cz"], "b": ["Pz", "Oz"]}
    prepared = prepare_rsa_data(_epochs(), space=groups)
    with pytest.raises(ValueError, match="must be in search_axes"):
        rsa_nd(prepared, MODEL)
    with pytest.raises(ValueError, match="take no radius"):
        rsa_nd(prepared, MODEL, search_axes=("space",), radii={"space": 0.1})
    windows = {"early": (0, 0.1), "late": (0.1, 0.3)}
    prepared = prepare_rsa_data(_epochs(), time=windows, time_mode="elements")
    with pytest.raises(ValueError, match="must be in search_axes"):
        rsa_nd(prepared, MODEL)
    with pytest.raises(ValueError, match="take no radius"):
        rsa_nd(prepared, MODEL, search_axes=("time",), radii={"time": 0.1})


def test_several_models():
    data = _data(3)
    models = [MODEL, MODEL**2]
    kwargs = dict(axes=("item", "a"), search_axes=("a",), data_rdm_metric="euclidean")
    single = rsa_nd(data, models[1], **kwargs)
    result = rsa_nd(data, models, model_names=["lin", "sq"], **kwargs)
    assert result.axes == ("model", "a")
    assert result.search_axes == ("a",)
    assert list(result.coords["model"]) == ["lin", "sq"]
    assert_allclose(result.rsa[1], single.rsa)

    square = np.abs(np.subtract.outer(np.arange(N_ITEMS), np.arange(N_ITEMS)))
    stacked = rsa_nd(data, np.stack([square, square]), **kwargs)
    assert stacked.rsa.shape == (2, 3)
    assert_array_equal(stacked.coords["model"], [0, 1])

    one = rsa_nd(data, [MODEL], **kwargs)
    assert one.axes == ("model", "a")
    assert one.rsa.shape == (1, 3)


def test_model_errors():
    kwargs = dict(axes=("item", "a"))
    with pytest.raises(ValueError, match="must be square"):
        rsa_nd(_data(3), np.ones((3, 4)), **kwargs)
    with pytest.raises(ValueError, match="at least one"):
        rsa_nd(_data(3), [], **kwargs)
    with pytest.raises(ValueError, match="needs several"):
        rsa_nd(_data(3), MODEL, model_names=["a"], **kwargs)
    with pytest.raises(ValueError, match="model_names"):
        rsa_nd(_data(3), [MODEL, MODEL], model_names=["a"], **kwargs)


def test_repeated_labels_are_averaged():
    data = np.repeat(_data(5), 2, axis=0)[: 2 * N_ITEMS]
    data += 0.01 * np.random.default_rng(1).standard_normal(data.shape)
    labels = np.repeat(np.arange(N_ITEMS), 2)
    result = rsa_nd(data, MODEL, axes=("item", "a"), labels=labels)
    averaged = data.reshape(N_ITEMS, 2, 5).mean(axis=1)
    expected = spearmanr(pdist(averaged, "correlation"), MODEL)[0]
    assert_allclose(result.rsa, expected)


def test_folds_use_repeated_labels():
    data = np.repeat(_data(5), 2, axis=0)
    labels = np.repeat(np.arange(N_ITEMS), 2)
    result = rsa_nd(data, MODEL, axes=("item", "a"), labels=labels, n_folds=2)
    assert np.isfinite(result.rsa)


def test_model_labels_align_rows():
    data = _data(5)
    order = np.arange(N_ITEMS)[::-1]
    square = np.abs(np.subtract.outer(np.arange(N_ITEMS), np.arange(N_ITEMS)) ** 2.0)
    expected = rsa_nd(data, square, axes=("item", "a"))
    shuffled = rsa_nd(
        data,
        square[np.ix_(order, order)],
        axes=("item", "a"),
        labels=np.arange(N_ITEMS),
        labels_rdm_model=order,
    )
    assert_allclose(shuffled.rsa, expected.rsa)


def test_mismatched_labels_raise():
    with pytest.raises(ValueError, match="labels"):
        rsa_nd(
            _data(5),
            MODEL,
            axes=("item", "a"),
            labels=list("abcdefgh"),
            labels_rdm_model=list("abcdefgz"),
        )


def test_compute_rsa_splits_arguments():
    direct = compute_rsa(_epochs(), MODEL, time=(0, 0.1), search_axes=("space",))
    prepared = prepare_rsa_data(_epochs(), time=(0, 0.1))
    assert_allclose(direct.rsa, rsa_nd(prepared, MODEL, search_axes=("space",)).rsa)
    assert_allclose(
        direct.rsa, compute_rsa(prepared, MODEL, search_axes=("space",)).rsa
    )
    with pytest.raises(ValueError, match="cannot be used"):
        compute_rsa(prepared, MODEL, time=(0, 0.1))


def test_rsa_nd_input_types():
    data = _data(3)
    rsa_data = RsaData(data, ("item", "a"))
    assert_allclose(
        rsa_nd(rsa_data, MODEL).rsa, rsa_nd(PreparedRsaData(rsa_data), MODEL).rsa
    )
    with pytest.raises(TypeError, match="data must be"):
        rsa_nd("data", MODEL)
    with pytest.raises(ValueError, match="axes is required"):
        rsa_nd(data, MODEL)
    with pytest.raises(ValueError, match="not used"):
        rsa_nd(rsa_data, MODEL, axes=("item", "a"))
    with pytest.raises(ValueError, match="real-valued"):
        rsa_nd(data.astype(complex), MODEL, axes=("item", "a"))


def test_metrics_and_jobs_reach_mne_rsa():
    data = _data(4)
    kwargs = dict(axes=("item", "a"), data_rdm_metric="cityblock", rsa_metric="pearson")
    result = rsa_nd(data, MODEL, **kwargs)
    assert_allclose(result.rsa, np.corrcoef(pdist(data, "cityblock"), MODEL)[0, 1])
    parallel = rsa_nd(data, MODEL, search_axes=("a",), n_jobs=2, **kwargs)
    assert_allclose(parallel.rsa, rsa_nd(data, MODEL, search_axes=("a",), **kwargs).rsa)


def test_ignore_nan():
    data = _data(4)
    model = MODEL.copy()
    model[0] = np.nan
    result = rsa_nd(data, model, axes=("item", "a"), ignore_nan=True)
    assert np.isfinite(result.rsa)


def test_matches_mne_rsa_epochs():
    epochs = _epochs()
    expected = mne_rsa.rsa_epochs(
        epochs, MODEL, spatial_radius=0.09, temporal_radius=0.02, verbose=False
    )
    result = compute_rsa(
        epochs,
        MODEL,
        search_axes=("space", "time"),
        radii={"space": 0.09, "time": 0.02},
    )
    keep = np.isin(np.round(result.coords["time"], 6), np.round(expected.times, 6))
    assert_allclose(result.rsa[:, keep], expected.data)


def test_matches_mne_rsa_evokeds():
    evokeds = [mne.EvokedArray(x, _info(), verbose=False) for x in _data(6, 30)]
    expected = mne_rsa.rsa_evokeds(
        evokeds, MODEL, spatial_radius=0.09, temporal_radius=0.02, verbose=False
    )
    result = compute_rsa(
        evokeds,
        MODEL,
        search_axes=("space", "time"),
        radii={"space": 0.09, "time": 0.02},
    )
    keep = np.isin(np.round(result.coords["time"], 6), np.round(expected.times, 6))
    assert_allclose(result.rsa[:, keep], expected.data)


def test_source_radius_needs_matching_src():
    stcs = [
        mne.VolSourceEstimate(x, [np.arange(4)], tmin=0, tstep=0.01)
        for x in _data(4, 3)
    ]
    with pytest.raises(ValueError, match="src="):
        compute_rsa(stcs, MODEL, search_axes=("space",), radii={"space": 0.02})


@pytest.mark.slowtest
@pytest.mark.parametrize("radius", [0.015, 0.025])
def test_matches_mne_rsa_surface_source(radius):
    data_path = mne.datasets.sample.data_path()
    fname = data_path / "subjects" / "sample" / "bem" / "sample-oct-6-src.fif"
    src = mne.read_source_spaces(fname, verbose=False)
    vertices = [hemi["vertno"] for hemi in src]
    stcs = [
        mne.SourceEstimate(x, vertices, tmin=0, tstep=0.01, subject="sample")
        for x in _data(sum(map(len, vertices)), 4)
    ]
    expected = mne_rsa.rsa_stcs(
        stcs, MODEL, src=src, spatial_radius=radius, temporal_radius=None, verbose=False
    )
    result = compute_rsa(
        stcs, MODEL, src=src, search_axes=("space",), radii={"space": radius}
    )
    assert_allclose(result.rsa, expected.data.ravel(), atol=1e-6)


@pytest.mark.slowtest
def test_matches_mne_rsa_volume_source():
    data_path = mne.datasets.sample.data_path()
    fname = data_path / "subjects" / "sample" / "bem" / "volume-7mm-src.fif"
    src = mne.read_source_spaces(fname, verbose=False)
    vertices = [src[0]["vertno"]]
    stcs = [
        mne.VolSourceEstimate(x, vertices, tmin=0, tstep=0.01, subject="sample")
        for x in _data(len(vertices[0]), 2)
    ]
    expected = mne_rsa.rsa_stcs(
        stcs, MODEL, src=src, spatial_radius=0.02, temporal_radius=None, verbose=False
    )
    result = compute_rsa(
        stcs, MODEL, src=src, search_axes=("space",), radii={"space": 0.02}
    )
    assert_allclose(result.rsa, expected.data.ravel(), atol=1e-6)
