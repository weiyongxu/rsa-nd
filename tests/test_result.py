"""Tests for results: MNE conversion and saving."""

import mne
import numpy as np
import pytest
from _helpers import make_info
from h5io import read_hdf5, write_hdf5
from mne.time_frequency import AverageTFR, EpochsTFRArray, Spectrum
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold

from rsa_nd import (
    RsaResult,
    compute_rsa,
    prepare_rsa_data,
    rsa_dynamic,
    rsa_nd,
)

N_ITEMS = 6
CH_NAMES = ["Fz", "Cz", "Pz", "Oz"]
SFREQ = 50.0
TIMES = np.arange(10) / SFREQ
FREQS = np.array([8.0, 12.0, 20.0])
MODEL = pdist(np.arange(N_ITEMS, dtype=float)[:, np.newaxis])
METRIC = dict(data_rdm_metric="euclidean")


def _data(*shape):
    return np.random.default_rng(0).standard_normal((N_ITEMS, *shape))


def _info():
    return make_info(CH_NAMES, SFREQ)


def _events():
    codes = np.arange(N_ITEMS) + 1
    return np.column_stack([codes * 20, np.zeros(N_ITEMS, int), codes])


def _epochs():
    return mne.EpochsArray(
        _data(4, 10), _info(), tmin=0.1, events=_events(), verbose=False
    )


def _tfr():
    return EpochsTFRArray(
        _info(), np.abs(_data(4, 3, 10)), TIMES, FREQS, events=_events()
    )


def _stcs(cls=mne.SourceEstimate):
    vertices = [np.arange(4)] if cls is mne.VolSourceEstimate else [np.arange(2)] * 2
    return [
        cls(x, vertices, tmin=0.1, tstep=1 / SFREQ, subject="s") for x in _data(4, 10)
    ]


def test_sensor_time_to_evoked():
    result = compute_rsa(_epochs(), MODEL, search_axes=("space", "time"), **METRIC)
    evoked = result.to_mne()
    assert isinstance(evoked, mne.Evoked)
    assert evoked.ch_names == CH_NAMES
    assert_allclose(evoked.times, result.coords["time"])
    assert_allclose(evoked.data, result.rsa)


def test_axis_order_does_not_matter():
    result = compute_rsa(_tfr(), MODEL, search_axes=("time", "space", "freq"), **METRIC)
    tfr = result.to_mne()
    assert isinstance(tfr, AverageTFR)
    assert_allclose(tfr.get_data(), np.transpose(result.rsa, (1, 2, 0)))


def test_sensor_freq_to_spectrum():
    result = compute_rsa(
        _tfr(), MODEL, search_axes=("space", "freq"), time=(0, 0.1), **METRIC
    )
    spectrum = result.to_mne()
    assert isinstance(spectrum, Spectrum)
    assert_allclose(spectrum.freqs, FREQS)


def test_space_only_gives_one_time_point():
    evoked = compute_rsa(_epochs(), MODEL, search_axes=("space",), **METRIC).to_mne()
    assert evoked.data.shape == (4, 1)


def test_time_only_uses_one_channel():
    evoked = compute_rsa(_epochs(), MODEL, search_axes=("time",), **METRIC).to_mne()
    assert evoked.ch_names == ["rsa"]
    assert evoked.info["sfreq"] == pytest.approx(SFREQ)


@pytest.mark.parametrize(
    "space_mode, axis", [("elements", "space"), ("aggregate", "roi")]
)
def test_groups_become_channels(space_mode, axis):
    result = compute_rsa(
        _epochs(),
        MODEL,
        space={"mid": ["Fz", "Cz"], "back": ["Pz", "Oz"]},
        space_mode=space_mode,
        search_axes=(axis, "time"),
        **METRIC,
    )
    assert result.to_mne().ch_names == ["mid", "back"]


@pytest.mark.parametrize("cls", (mne.SourceEstimate, mne.VolSourceEstimate))
def test_source_time_to_stc(cls):
    result = compute_rsa(_stcs(cls), MODEL, search_axes=("space", "time"), **METRIC)
    stc = result.to_mne()
    assert isinstance(stc, cls)
    assert stc.subject == "s"
    assert_allclose(stc.times, result.coords["time"])
    assert_allclose(stc.data, result.rsa)


def test_source_space_only_to_stc():
    stc = compute_rsa(_stcs(), MODEL, search_axes=("space",), **METRIC).to_mne()
    assert stc.data.shape == (4, 1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(), "at least one"),
        (dict(rdm_model=[MODEL, MODEL], search_axes=("space",)), "'model'"),
        (dict(time={"a": (0.1, 0.2)}, search_axes=("window",)), "'window'"),
        (
            dict(
                time={"a": (0.1, 0.2)},
                time_mode="elements",
                search_axes=("time",),
            ),
            "named 'time'",
        ),
    ],
)
def test_to_mne_errors(kwargs, match):
    rdm_model = kwargs.pop("rdm_model", MODEL)
    result = compute_rsa(_epochs(), rdm_model, **kwargs, **METRIC)
    with pytest.raises(ValueError, match=match):
        result.to_mne()


def test_to_mne_needs_geometry():
    result = rsa_nd(
        _data(4), MODEL, axes=("item", "space"), search_axes=("space",), **METRIC
    )
    with pytest.raises(ValueError, match="geometry"):
        result.to_mne()


def test_source_with_freq_has_no_mne_container():
    result = compute_rsa(
        _stcs(),
        MODEL,
        tfr={"freqs": FREQS, "n_cycles": 1.0, "verbose": False},
        search_axes=("space", "freq"),
        time=(0, 0.2),
        **METRIC,
    )
    with pytest.raises(ValueError, match="'freq'"):
        result.to_mne()


def test_repr():
    result = compute_rsa(_epochs(), [MODEL, MODEL], search_axes=("time",), **METRIC)
    assert repr(result) == "<RsaResult | model: 2, time: 10>"
    assert repr(compute_rsa(_epochs(), MODEL, **METRIC)) == "<RsaResult | scalar>"


def test_result_info_is_a_copy():
    prepared = prepare_rsa_data(_epochs())
    result = rsa_nd(prepared, MODEL, search_axes=("space",), **METRIC)
    prepared.space.info["bads"] = ["Pz"]
    assert result.space.info["bads"] == []


def test_save_load_keeps_everything(tmp_path):
    epochs = _epochs()
    epochs.info["bads"] = ["Cz"]
    result = compute_rsa(
        epochs,
        [MODEL, MODEL**2],
        model_names=["a", "b"],
        search_axes=("space", "time"),
        radii={"time": 0.04},
        **METRIC,
    )
    result.save(tmp_path / "r.h5")
    loaded = RsaResult.load(tmp_path / "r.h5")
    assert_array_equal(loaded.rsa, result.rsa)
    assert loaded.axes == result.axes
    assert list(loaded.coords["model"]) == ["a", "b"]
    assert list(loaded.coords["space"]) == ["Fz", "Pz", "Oz"]
    assert loaded.info["rsa_metric"] == "spearman"
    assert loaded.space.channel_names == ("Fz", "Pz", "Oz")
    assert loaded.space.info["sfreq"] == SFREQ


def test_save_load_source_groups(tmp_path):
    label = mne.Label(np.arange(2), hemi="lh", name="a-lh")
    result = compute_rsa(
        _stcs(), MODEL, space=[label], search_axes=("space",), **METRIC
    )
    result.save(tmp_path / "r.h5")
    loaded = RsaResult.load(tmp_path / "r.h5")
    assert_array_equal(loaded.space.patches["a-lh"], [0, 1])
    assert loaded.space.subject == "s"
    assert loaded.to_mne().ch_names == ["a-lh"]


def test_save_load_dynamic(tmp_path):
    rdms = [MODEL, MODEL**2, MODEL**3]
    result = rsa_dynamic(
        _epochs(),
        rdms,
        [0.1, 0.12, 0.14],
        mode="lagged",
        lags=[0.0, 0.02],
        **METRIC,
    )
    result.save(tmp_path / "r.h5")
    loaded = RsaResult.load(tmp_path / "r.h5")
    assert_array_equal(loaded.rsa, result.rsa)
    assert_array_equal(loaded.dynamic.full_rsa, result.dynamic.full_rsa)
    assert_array_equal(loaded.dynamic.lags, [0, 0.02])
    assert loaded.dynamic.full_rsa_axes == ("model_time", "time")


def test_save_records_splitters_as_text(tmp_path):
    data = np.repeat(_data(4), 4, axis=0)
    result = rsa_nd(
        data,
        MODEL,
        axes=("item", "x"),
        labels=np.repeat(np.arange(N_ITEMS), 4),
        n_folds=KFold(2),
        **METRIC,
    )
    result.save(tmp_path / "r.h5")
    assert "KFold" in RsaResult.load(tmp_path / "r.h5").info["n_folds"]


def test_save_needs_overwrite(tmp_path):
    result = compute_rsa(_epochs(), MODEL, **METRIC)
    result.save(tmp_path / "r.h5")
    with pytest.raises(OSError, match="overwrite"):
        result.save(tmp_path / "r.h5")
    result.save(tmp_path / "r.h5", overwrite=True)


def test_loads_files_written_by_mne_ez(tmp_path):
    result = compute_rsa(_epochs(), MODEL, search_axes=("time",), **METRIC)
    result.save(tmp_path / "new.h5")
    payload = read_hdf5(tmp_path / "new.h5", title="rsa_nd_result", slash="replace")
    payload["kind"] = "mne_ez.rsa_result"
    payload["space"]["is_grouped"] = False  # no longer written
    write_hdf5(tmp_path / "old.h5", payload, title="mne_ez_rsa_result", slash="replace")
    assert_array_equal(RsaResult.load(tmp_path / "old.h5").rsa, result.rsa)


def test_load_rejects_other_files(tmp_path):
    write_hdf5(tmp_path / "other.h5", {"kind": "other"}, title="rsa_nd_result")
    with pytest.raises(ValueError, match="not an rsa-nd result"):
        RsaResult.load(tmp_path / "other.h5")
