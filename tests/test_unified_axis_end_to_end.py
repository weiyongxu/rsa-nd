from __future__ import annotations

from collections.abc import Callable

import mne
import numpy as np
import pytest

from rsa_nd import (
    SensorSpace,
    SourceSpace,
    compute_rsa,
    prepare_rsa_data,
    rsa_nd,
)
from rsa_nd._blocks import build_searchlight_plan
from rsa_nd._types import RsaData


def _model(values):
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def _events(n_items):
    return np.column_stack(
        [np.arange(n_items), np.zeros(n_items, dtype=int), np.arange(1, n_items + 1)]
    )


def _sensor_info():
    ch_names = ["Fz", "Cz", "Pz", "Oz"]
    info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
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


def _sensor_time_data(item_values, *, n_times=24):
    times = np.arange(n_times) / 100.0
    channel_weights = np.array([0.6, 1.0, 1.4, 1.8])
    time_weights = 1.0 + 0.2 * np.sin(2 * np.pi * 6.0 * times)
    data = (
        item_values[:, None, None]
        * channel_weights[None, :, None]
        * time_weights[None, None, :]
    )
    return data * 1e-6, times


def _sensor_freq_data(item_values):
    freqs = np.array([6.0, 10.0, 18.0])
    channel_weights = np.array([0.7, 1.1, 1.5, 1.9])
    freq_weights = np.array([0.8, 1.2, 1.7])
    data = (
        1.0
        + item_values[:, None, None]
        * channel_weights[None, :, None]
        * freq_weights[None, None, :]
    )
    return data, freqs


def _sensor_tfr_data(item_values):
    spectral, freqs = _sensor_freq_data(item_values)
    times = np.arange(20) / 100.0
    time_weights = 1.0 + 0.1 * np.cos(2 * np.pi * 5.0 * times)
    data = spectral[..., None] * time_weights[None, None, None, :]
    return data, freqs, times


def _epochs(item_values):
    data, times = _sensor_time_data(item_values)
    return mne.EpochsArray(
        data,
        _sensor_info(),
        events=_events(len(item_values)),
        tmin=float(times[0]),
        baseline=None,
        verbose=False,
    )


def _epochs_spectrum(item_values):
    data, freqs = _sensor_freq_data(item_values)
    return mne.time_frequency.EpochsSpectrumArray(
        data,
        _sensor_info(),
        freqs=freqs,
        events=_events(len(item_values)),
        event_id={f"i{idx}": idx + 1 for idx in range(len(item_values))},
        verbose=False,
    )


def _epochs_tfr(item_values):
    data, freqs, times = _sensor_tfr_data(item_values)
    return mne.time_frequency.EpochsTFRArray(
        _sensor_info(),
        data,
        times=times,
        freqs=freqs,
        events=_events(len(item_values)),
        event_id={f"i{idx}": idx + 1 for idx in range(len(item_values))},
        method="synthetic",
    )


def _evokeds(item_values):
    data, times = _sensor_time_data(item_values)
    return [
        mne.EvokedArray(epoch, _sensor_info(), tmin=float(times[0]), comment=f"i{idx}")
        for idx, epoch in enumerate(data)
    ]


def _spectrums(item_values):
    data, freqs = _sensor_freq_data(item_values)
    return [
        mne.time_frequency.SpectrumArray(epoch, _sensor_info(), freqs=freqs)
        for epoch in data
    ]


def _average_tfrs(item_values):
    data, freqs, times = _sensor_tfr_data(item_values)
    return [
        mne.time_frequency.AverageTFRArray(
            info=_sensor_info(),
            data=epoch,
            times=times,
            freqs=freqs,
            nave=1,
            method="synthetic",
        )
        for epoch in data
    ]


def _source_estimates(item_values):
    times = np.arange(18) / 100.0
    vertices = [np.array([10, 20, 30]), np.array([11, 21])]
    vertex_weights = np.array([0.5, 0.9, 1.2, 1.6, 2.0])
    time_weights = 1.0 + 0.15 * np.sin(2 * np.pi * 4.0 * times)
    return [
        mne.SourceEstimate(
            value * vertex_weights[:, None] * time_weights[None, :],
            vertices=vertices,
            tmin=float(times[0]),
            tstep=float(times[1] - times[0]),
            subject="sample",
        )
        for value in item_values
    ]


def _vol_source_estimates(item_values):
    times = np.arange(18) / 100.0
    vertices = [np.arange(5)]
    vertex_weights = np.array([0.5, 0.9, 1.2, 1.6, 2.0])
    time_weights = 1.0 + 0.15 * np.cos(2 * np.pi * 4.0 * times)
    return [
        mne.VolSourceEstimate(
            value * vertex_weights[:, None] * time_weights[None, :],
            vertices=vertices,
            tmin=float(times[0]),
            tstep=float(times[1] - times[0]),
            subject="sample",
        )
        for value in item_values
    ]


@pytest.mark.parametrize(
    ("factory", "search_axes", "radii"),
    [
        (_epochs, ("time",), {"time": 0.0}),
        (_epochs_spectrum, ("freq",), {"freq": 0.0}),
        (_epochs_tfr, ("freq", "time"), {"freq": 0.0, "time": 0.0}),
        (_evokeds, ("time",), {"time": 0.0}),
        (_spectrums, ("freq",), {"freq": 0.0}),
        (_average_tfrs, ("freq", "time"), {"freq": 0.0, "time": 0.0}),
        (_source_estimates, ("space",), {"space": 0.0}),
        (_vol_source_estimates, ("space",), {"space": 0.0}),
    ],
)
def test_compute_rsa_end_to_end_for_all_supported_mne_inputs(
    factory: Callable[[np.ndarray], object],
    search_axes: tuple[str, ...],
    radii: dict[str, float],
):
    item_values = np.linspace(0.0, 2.0, 6)
    inst = factory(item_values)
    model = _model(item_values)

    direct = compute_rsa(
        inst,
        model,
        search_axes=search_axes,
        radii=radii,
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
        n_jobs=1,
    )
    prepared = prepare_rsa_data(inst)
    two_step = rsa_nd(
        prepared,
        model,
        search_axes=search_axes,
        radii=radii,
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
        n_jobs=1,
    )

    assert direct.search_axes == search_axes
    np.testing.assert_allclose(direct.rsa, two_step.rsa, atol=1e-9, equal_nan=True)
    assert np.isfinite(np.asarray(direct.rsa)).any()
    assert float(np.nanmean(direct.rsa)) > 0.95


def test_unified_axis_grouped_searchlight_patches():
    item_values = np.linspace(0.0, 1.0, 5)
    tfr = _epochs_tfr(item_values)
    prepared = prepare_rsa_data(
        tfr,
        freq={"alpha": (8.0, 13.0)},
        time={"early": (0.0, 0.1), "late": (0.1, 0.3)},
        time_mode="elements",
        space={"front": ["Fz", "Cz"], "back": ["Pz", "Oz"]},
    )

    # The grouped axes resolve into the searchlight patches the engine
    # consumes: one patch per space group x time window (the alpha band is
    # aggregated to a single feature, so it is not a search axis).
    plan = build_searchlight_plan(prepared, ("space", "time"), {})
    assert plan.shape == (2, 2)  # 2 sensor groups x 2 time windows
    assert plan.n_patches == 4
    front, back = plan.axis_patches["space"]
    assert front.tolist() == [0, 1]  # Fz, Cz
    assert back.tolist() == [2, 3]  # Pz, Oz
    early, late = plan.axis_patches["time"]
    assert early.size == 10  # early window: 10 samples at 100 Hz
    assert late.size == 10
    assert list(plan.search_coords["space"]) == ["front", "back"]
    assert list(plan.search_coords["time"]) == ["early", "late"]


def test_prepared_rsa_data_rejects_incoherent_explicit_axis_groups():
    from rsa_nd._types import AxisGroups, PreparedRsaData

    rsa_data = RsaData(
        np.zeros((4, 3, 5)),
        axes=("item", "space", "time"),
        coords={"time": np.arange(5) / 100.0},
    )

    # Unknown axis key.
    with pytest.raises(ValueError, match="axis_groups keys must be axes"):
        PreparedRsaData(
            rsa_data=rsa_data,
            input_kind="rsa_data",
            axis_groups={
                "freq": AxisGroups("freq", "elements", (("a", np.array([0])),))
            },
        )

    # Group member index past the axis length (time has 5 points; index 9 invalid).
    with pytest.raises(ValueError, match="outside the axis length"):
        PreparedRsaData(
            rsa_data=rsa_data,
            input_kind="rsa_data",
            axis_groups={
                "time": AxisGroups("time", "elements", (("late", np.array([3, 9])),))
            },
        )


def test_searchlight_modes_singleton_radius_group_aggregate_and_pool():
    rng = np.random.default_rng(0)
    item_values = np.linspace(0.0, 2.0, 6)
    space_weights = np.array([0.5, 1.0, 1.5])
    freq_weights = np.array([0.8, 1.3, 1.9])
    times = np.arange(6) / 10.0
    data = (
        item_values[:, None, None, None]
        * space_weights[None, :, None, None]
        * freq_weights[None, None, :, None]
        * (1.0 + times[None, None, None, :])
    )
    data += 1e-3 * rng.standard_normal(data.shape)
    rsa_data = RsaData(
        data,
        axes=("item", "space", "freq", "time"),
        coords={"freq": [6.0, 10.0, 18.0], "time": times},
    )
    model = _model(item_values)

    singleton = compute_rsa(
        rsa_data,
        model,
        search_axes=("time",),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )
    assert singleton.rsa.shape == (len(times),)
    assert float(np.nanmean(singleton.rsa)) > 0.95

    radius = compute_rsa(
        rsa_data,
        model,
        search_axes=("time",),
        radii={"time": 0.11},
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )
    assert radius.rsa.shape == (len(times),)
    assert float(np.nanmean(radius.rsa)) > 0.95

    windows = {"early": (0.0, 0.3), "late": (0.3, 0.7)}
    elements = compute_rsa(
        rsa_data,
        model,
        time=windows,
        time_mode="elements",
        search_axes=("time",),
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )
    assert elements.rsa.shape == (2,)
    assert list(elements.search_coords["time"]) == ["early", "late"]

    aggregate = compute_rsa(
        rsa_data,
        model,
        time=windows,
        search_axes=("window",),
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )
    assert aggregate.rsa.shape == (2,)
    assert list(aggregate.search_coords["window"]) == ["early", "late"]

    pooled = compute_rsa(
        rsa_data,
        model,
        search_axes=("time",),
        radii={"time": None},
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )
    scalar = compute_rsa(
        rsa_data,
        model,
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )
    assert pooled.rsa.shape == (1,)
    np.testing.assert_allclose(pooled.rsa[0], scalar.rsa)


def test_space_radius_none_pools_native_sensor_and_source_space():
    info = _sensor_info()
    sensor = SensorSpace(channel_names=tuple(info.ch_names), info=info)
    patches, centers = sensor.searchlight_patches(None)
    assert len(patches) == 1
    assert patches[0].tolist() == [0, 1, 2, 3]
    assert centers.tolist() == ["all"]

    source = SourceSpace(vertices=(np.array([10, 20]), np.array([11, 21, 31])))
    patches, centers = source.searchlight_patches(None)
    assert len(patches) == 1
    assert patches[0].tolist() == [0, 1, 2, 3, 4]
    assert centers.tolist() == ["all"]


@pytest.mark.slowtest
def test_mne_sample_sensor_inputs_when_dataset_is_available():
    try:
        sample_root = mne.datasets.sample.data_path(download=False, verbose=False)
    except Exception as exc:
        pytest.skip(f"MNE sample dataset is not available locally: {exc}")

    raw_path = sample_root / "MEG" / "sample" / "sample_audvis_raw.fif"
    events_path = sample_root / "MEG" / "sample" / "sample_audvis_raw-eve.fif"
    if not raw_path.exists() or not events_path.exists():
        pytest.skip("MNE sample raw/events files are not available locally.")

    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")[:4]
    if len(picks) < 2:
        picks = mne.pick_types(raw.info, meg=True, eeg=False, exclude="bads")[:4]
    events = mne.read_events(events_path, verbose=False)
    event_id = {"aud_l": 1, "aud_r": 2, "vis_l": 3, "vis_r": 4}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.05,
        tmax=0.15,
        picks=picks,
        baseline=None,
        preload=True,
        reject=None,
        verbose=False,
    )

    chosen = []
    for code in event_id.values():
        matches = np.flatnonzero(epochs.events[:, 2] == code)
        if matches.size < 2:
            pytest.skip("MNE sample subset does not have two epochs per condition.")
        chosen.extend(matches[:2].tolist())
    epochs = epochs[chosen]

    labels_model = np.asarray(list(event_id.values()))
    model = _model(labels_model)
    common = dict(
        labels_rdm_model=labels_model,
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
        n_folds=2,
        n_jobs=1,
    )

    time_result = compute_rsa(
        epochs,
        model,
        search_axes=("time",),
        radii={"time": 0.0},
        **common,
    )
    assert time_result.rsa.shape == (len(epochs.times),)
    assert np.isfinite(time_result.rsa).any()

    spectrum = epochs.compute_psd(
        method="welch",
        fmin=4.0,
        fmax=30.0,
        n_fft=64,
        n_per_seg=64,
        verbose=False,
    )
    freq_result = compute_rsa(
        spectrum,
        model,
        search_axes=("freq",),
        radii={"freq": 0.0},
        **common,
    )
    assert freq_result.rsa.shape == (len(spectrum.freqs),)
    assert np.isfinite(freq_result.rsa).any()

    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=np.array([8.0, 12.0]),
        n_cycles=1.0,
        average=False,
        output="power",
        verbose=False,
    )
    tfr_result = compute_rsa(
        tfr,
        model,
        search_axes=("freq", "time"),
        radii={"freq": 0.0, "time": 0.0},
        **common,
    )
    assert tfr_result.rsa.shape == (2, len(tfr.times))
    assert np.isfinite(tfr_result.rsa).any()
