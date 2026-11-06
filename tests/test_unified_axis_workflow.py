"""End-to-end workflow tests for the unified-axis RSA refactor.

Complements ``test_unified_axis_end_to_end.py`` (which checks every supported
MNE input kind resolves and scores correctly) by exercising three workflow
properties the refactor promises but the per-input tests do not pin down:

1. efficiency -- a searchlight is a single batched pass, and the radius-0
   temporal fast path avoids the mne-rsa backend entirely;
2. one-call conversion -- ``compute_rsa(...).to_mne()`` returns native MNE
   containers for representable sensor results (per-channel maps) and raises
   for non-MNE grouped axes (window / band);
3. group-mode consistency -- ``elements`` and ``aggregate`` behave the same way
   across the ``time``, ``freq``, and ``space`` axes of one EpochsTFR.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from rsa_nd import compute_rsa, prepare_rsa_data, rsa_nd
from rsa_nd._types import RsaData


def _model(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def _sensor_info() -> mne.Info:
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=100.0, ch_types="eeg")
    locs = np.array(
        [
            [-0.03, 0.04, 0.0],
            [0.00, 0.02, 0.0],
            [0.00, -0.02, 0.0],
            [0.03, -0.04, 0.0],
        ]
    )
    for ch, loc in zip(info["chs"], locs):
        ch["loc"][:3] = loc
    return info


def _events(n_items: int) -> np.ndarray:
    return np.column_stack(
        [np.arange(n_items), np.zeros(n_items, dtype=int), np.arange(1, n_items + 1)]
    )


def _epochs(item_values: np.ndarray, *, n_times: int = 24) -> mne.EpochsArray:
    times = np.arange(n_times) / 100.0
    channel_weights = np.array([0.6, 1.0, 1.4, 1.8])
    time_weights = 1.0 + 0.2 * np.sin(2 * np.pi * 6.0 * times)
    data = (
        item_values[:, None, None]
        * channel_weights[None, :, None]
        * time_weights[None, None, :]
    ) * 1e-6
    return mne.EpochsArray(
        data,
        _sensor_info(),
        events=_events(len(item_values)),
        tmin=0.0,
        baseline=None,
        verbose=False,
    )


def _epochs_spectrum(item_values: np.ndarray):
    freqs = np.array([6.0, 10.0, 18.0])
    channel_weights = np.array([0.7, 1.1, 1.5, 1.9])
    freq_weights = np.array([0.8, 1.2, 1.7])
    data = 1.0 + (
        item_values[:, None, None]
        * channel_weights[None, :, None]
        * freq_weights[None, None, :]
    )
    return mne.time_frequency.EpochsSpectrumArray(
        data,
        _sensor_info(),
        freqs=freqs,
        events=_events(len(item_values)),
        event_id={f"i{idx}": idx + 1 for idx in range(len(item_values))},
        verbose=False,
    )


def _epochs_tfr(item_values: np.ndarray):
    freqs = np.array([6.0, 10.0, 18.0])
    times = np.arange(20) / 100.0
    channel_weights = np.array([0.7, 1.1, 1.5, 1.9])
    freq_weights = np.array([0.8, 1.2, 1.7])
    time_weights = 1.0 + 0.1 * np.cos(2 * np.pi * 5.0 * times)
    data = (
        1.0
        + item_values[:, None, None, None]
        * channel_weights[None, :, None, None]
        * freq_weights[None, None, :, None]
        * time_weights[None, None, None, :]
    )
    return mne.time_frequency.EpochsTFRArray(
        _sensor_info(),
        data,
        times=times,
        freqs=freqs,
        events=_events(len(item_values)),
        event_id={f"i{idx}": idx + 1 for idx in range(len(item_values))},
        method="synthetic",
    )


def _named_time_data(item_values: np.ndarray, n_times: int = 30) -> RsaData:
    times = np.arange(n_times) / 100.0
    space_weights = np.array([0.5, 1.0, 1.5])
    time_weights = 1.0 + times
    data = (
        item_values[:, None, None]
        * space_weights[None, :, None]
        * time_weights[None, None, :]
    )
    return RsaData(
        data,
        axes=("item", "space", "time"),
        coords={"time": times},
    )


# --------------------------------------------------------------------------- #
# 1. Efficiency
# --------------------------------------------------------------------------- #
def test_searchlight_is_a_single_batched_backend_call(monkeypatch):
    """A 30-position searchlight is one batched backend call, not 30.

    The default dense loader serves the whole plan as one chunk, so the backend
    ``mne_rsa.rsa_array`` is invoked exactly once for the entire searchlight
    rather than once per patch.
    """

    import mne_rsa

    calls = {"n": 0}
    original = mne_rsa.rsa_array

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mne_rsa, "rsa_array", counting)

    item_values = np.linspace(0.0, 2.0, 6)
    rsa_data = _named_time_data(item_values)
    result = compute_rsa(
        rsa_data,
        _model(item_values),
        search_axes=("time",),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )

    assert result.rsa.shape == (30,)
    assert calls["n"] == 1
    assert float(np.nanmean(result.rsa)) > 0.95


def test_overlapping_radius_patches_reuse_one_prepared_array(monkeypatch):
    """Overlapping radius patches slice one shared array, not one per patch.

    Scope: this pins down the *dense* execution model only -- ``rsa_nd`` builds
    a single array layout from the already-prepared data and slices every
    searchlight patch from it, so overlapping neighbours cost no extra layout
    passes. It does NOT test lazy / chunked TFR loading (the unified path is
    dense; per-ROI source chunking lives in ``_source``). We assert the
    layout is built exactly once for a wide radius where every patch overlaps
    its neighbours.
    """

    import rsa_nd._core as core

    calls = {"n": 0}
    original = core.build_array_layout

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "build_array_layout", counting)

    item_values = np.linspace(0.0, 2.0, 6)
    prepared = prepare_rsa_data(_named_time_data(item_values))
    result = rsa_nd(
        prepared,
        _model(item_values),
        search_axes=("time",),
        radii={"time": 0.1},  # wide: neighbouring patches overlap
        data_rdm_metric="euclidean",
        rsa_metric="kendall-tau-a",  # backend path, so build_array_layout is hit once
    )

    assert result.rsa.shape == (30,)
    # One layout build for the whole searchlight -- features are not
    # re-extracted per patch.
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# 2. One-call assembly to native MNE containers
# --------------------------------------------------------------------------- #
def test_compute_rsa_to_mne_returns_evoked_for_per_channel_searchlight():
    item_values = np.linspace(0.0, 2.0, 6)
    epochs = _epochs(item_values)
    result = compute_rsa(
        epochs,
        _model(item_values),
        search_axes=("space", "time"),
        radii={"space": 0.0, "time": 0.0},
        data_rdm_metric="euclidean",
    )
    out = result.to_mne()
    assert isinstance(out, mne.EvokedArray)
    assert out.ch_names == epochs.ch_names
    assert len(out.times) == len(epochs.times)
    assert np.isfinite(out.data).any()


def test_compute_rsa_window_axis_to_mne_raises():
    item_values = np.linspace(0.0, 2.0, 6)
    epochs = _epochs(item_values)
    result = compute_rsa(
        epochs,
        _model(item_values),
        time={"early": (0.0, 0.1), "late": (0.1, 0.23)},
        search_axes=("window",),
        data_rdm_metric="euclidean",
    )
    # Grouped "window" is not a native MNE axis; to_mne does not back-fill.
    assert result.axes == ("window",)
    with pytest.raises(ValueError, match="cannot represent"):
        result.to_mne()


def test_compute_rsa_band_axis_to_mne_raises():
    item_values = np.linspace(0.0, 2.0, 6)
    spectrum = _epochs_spectrum(item_values)
    result = compute_rsa(
        spectrum,
        _model(item_values),
        freq={"alpha": (8.0, 13.0)},
        search_axes=("band",),
        data_rdm_metric="euclidean",
    )
    assert result.axes == ("band",)
    with pytest.raises(ValueError, match="cannot represent"):
        result.to_mne()


# --------------------------------------------------------------------------- #
# 3. Group-mode consistency across time / freq / space
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "search_axes", "n_groups"),
    [
        # time: two windows
        (dict(time={"early": (0.0, 0.1), "late": (0.1, 0.2)}), "time", 2),
        # freq: two bands
        (dict(freq={"low": (5.0, 8.0), "high": (12.0, 20.0)}), "freq", 2),
        # space: two sensor groups
        (dict(space={"front": ["Fz", "Cz"], "back": ["Pz", "Oz"]}), "space", 2),
    ],
)
def test_elements_and_aggregate_modes_are_consistent_across_axes(
    kwargs: dict, search_axes: str, n_groups: int
):
    """``elements`` and ``aggregate`` produce one score per group on every axis.

    The unified vocabulary promises that ``elements`` (members kept as joint
    features) and ``aggregate`` (members reduced to one value) behave the same
    way regardless of whether the grouped axis is time, frequency, or space.
    Both must yield exactly ``n_groups`` RSA scores with the group names as
    coordinates, on each of the three axes.
    """

    item_values = np.linspace(0.0, 2.0, 6)
    tfr = _epochs_tfr(item_values)
    model = _model(item_values)
    axis_key = next(iter(kwargs))  # "time" / "freq" / "space"

    # elements mode keeps members as features; the searched axis is the native
    # name (time / freq / space).
    elements = compute_rsa(
        tfr,
        model,
        **kwargs,
        **{f"{axis_key}_mode": "elements"},
        search_axes=(search_axes,),
        data_rdm_metric="euclidean",
    )
    # aggregate mode reduces members; time/freq collapse to window/band, space
    # stays "space" (a grouped SpaceDescription is searched as one axis).
    aggregate_axis = {"time": "window", "freq": "band", "space": "roi"}[search_axes]
    aggregate = compute_rsa(
        tfr,
        model,
        **kwargs,
        **{f"{axis_key}_mode": "aggregate"},
        search_axes=(aggregate_axis,),
        data_rdm_metric="euclidean",
    )

    assert elements.rsa.shape == (n_groups,)
    assert aggregate.rsa.shape == (n_groups,)
    assert list(elements.search_coords[search_axes]) == list(kwargs[axis_key].keys())
    assert list(aggregate.search_coords[aggregate_axis]) == list(
        kwargs[axis_key].keys()
    )
    # Monotone synthetic data: every group recovers the item structure.
    assert float(np.nanmin(elements.rsa)) > 0.9
    assert float(np.nanmin(aggregate.rsa)) > 0.9
