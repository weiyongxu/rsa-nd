"""Check input types and dimension setups against a brute-force reference.

The reference uses only NumPy and SciPy loops, so it shares no code with the
package: it rebuilds each searchlight patch, computes the data RDM with
``pdist``, and correlates it with the model RDM with ``spearmanr``.
"""

import itertools
import os

import mne
import numpy as np
import pytest
from _helpers import make_info
from mne.time_frequency import (
    AverageTFRArray,
    EpochsSpectrumArray,
    EpochsTFRArray,
    SpectrumArray,
    tfr_array_morlet,
)
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from rsa_nd import RsaData, RsaResult, prepare_rsa_data, rsa_nd

N_ITEMS = 6
CH_NAMES = ["Fz", "Cz", "Pz", "Oz", "C3"]
SFREQ = 50.0
TIMES = np.arange(16) / SFREQ
FREQS = np.array([8.0, 12.0, 16.0, 24.0])
TFR = {"freqs": FREQS, "n_cycles": 1.0}
RANGES = {"time": (0.05, 0.21), "freq": (10.0, 20.0)}
NAMED = {
    "time": {"early": (0.0, 0.16), "late": (0.16, 0.32)},
    "freq": {"low": (8.0, 14.0), "high": (14.0, 25.0)},
}
SENSOR_GROUPS = {"mid": ["Fz", "Cz", "Pz"], "back": ["Oz", "C3"]}
GROUP_INDICES = {"mid": [0, 1, 2], "back": [3, 4]}
RADII = {"space": 0.09, "time": 0.04, "freq": 4.0}
COARSE = {"time": "window", "freq": "band", "space": "roi"}
MODEL_RDM = pdist(np.arange(N_ITEMS, dtype=float)[:, np.newaxis] ** 1.5)

INPUTS = {
    "array": ("item", "space", "freq", "time"),
    "rsa_data": ("item", "space", "freq", "time"),
    "epochs": ("item", "space", "time"),
    "epochs+tfr": ("item", "space", "freq", "time"),
    "evokeds": ("item", "space", "time"),
    "evokeds+tfr": ("item", "space", "freq", "time"),
    "stcs": ("item", "space", "time"),
    "stcs+tfr": ("item", "space", "freq", "time"),
    "vol_stcs": ("item", "space", "time"),
    "epochs_tfr": ("item", "space", "freq", "time"),
    "epochs_tfr_complex": ("item", "space", "freq", "time"),
    "average_tfrs": ("item", "space", "freq", "time"),
    "average_tfrs_complex": ("item", "space", "freq", "time"),
    "epochs_spectrum": ("item", "space", "freq"),
    "epochs_spectrum_complex": ("item", "space", "freq"),
    "spectrums": ("item", "space", "freq"),
    "spectrums_complex": ("item", "space", "freq"),
}
COMPLEX_FEATURES = ("real_imag", "amp_phase", "phase")


def _make_info():
    return make_info(CH_NAMES, SFREQ)


def _make_events():
    codes = np.arange(N_ITEMS) + 1
    return np.column_stack([codes * 20, np.zeros(N_ITEMS, int), codes])


def _make_data(*shape, complex_=False):
    rng = np.random.default_rng(len(shape))
    data = np.abs(rng.standard_normal((N_ITEMS, *shape))) + 0.1
    return data * np.exp(1j * rng.uniform(-3, 3, data.shape)) if complex_ else data


def _make_stcs(data, kind):
    if kind == "volume":
        cls, vertices = mne.VolSourceEstimate, [np.arange(5)]
    else:
        cls, vertices = mne.SourceEstimate, [np.array([0, 2, 5]), np.array([1, 3])]
    return [cls(x, vertices, tmin=0.0, tstep=1 / SFREQ, subject="s") for x in data]


def make_input(name):
    """Return ``(inst, data)``, with ``data`` the matching plain array."""
    info, n_ch, n_time, n_freq = _make_info(), 5, len(TIMES), len(FREQS)
    complex_ = name.endswith("complex")
    if name in ("array", "rsa_data"):
        data = _make_data(n_ch, n_freq, n_time)
        if name == "array":
            return data, data
        coords = {"freq": FREQS, "time": TIMES}
        return RsaData(data, INPUTS[name], coords), data
    if name.startswith(("epochs_tfr", "average_tfrs")):
        data = _make_data(n_ch, n_freq, n_time, complex_=complex_)
        if name.startswith("epochs"):
            return EpochsTFRArray(info, data, TIMES, FREQS, events=_make_events()), data
        return [AverageTFRArray(info, x, TIMES, FREQS) for x in data], data
    if name.startswith(("epochs_spectrum", "spectrums")):
        data = _make_data(n_ch, n_freq, complex_=complex_)
        if name.startswith("epochs"):
            return EpochsSpectrumArray(data, info, FREQS, events=_make_events()), data
        return [SpectrumArray(x, info, FREQS) for x in data], data

    data = _make_data(n_ch, n_time)
    if name.startswith("epochs"):
        inst = mne.EpochsArray(data, info, events=_make_events(), verbose=False)
    elif name.startswith("evokeds"):
        inst = [mne.EvokedArray(x, info, verbose=False) for x in data]
    else:
        inst = _make_stcs(data, "volume" if name == "vol_stcs" else "surface")
    if name.endswith("+tfr"):
        data = tfr_array_morlet(
            data, SFREQ, FREQS, n_cycles=1.0, output="complex", verbose=False
        )
    return inst, data


def _space_arg(name):
    if name == "vol_stcs":
        return {"mid": np.array([0, 1, 2]), "back": np.array([3, 4])}
    if name in ("stcs", "stcs+tfr"):
        return {
            "mid": [mne.Label(np.array([0, 2, 5]), hemi="lh", name="mid-lh")],
            "back": [mne.Label(np.array([1, 3]), hemi="rh", name="back-rh")],
        }
    return SENSOR_GROUPS


def _groups_for(name):
    if name in ("stcs", "stcs+tfr"):
        return {"mid": [0, 1, 2], "back": [3, 4]}
    return GROUP_INDICES


def _project(data, feature):
    if not np.iscomplexobj(data):
        return data
    amp, phase = np.abs(data), np.angle(data)
    if feature == "power":
        return amp**2
    if feature == "amplitude":
        return amp
    parts = {
        "real_imag": (data.real, data.imag),
        "amp_phase": (amp, np.sin(phase), np.cos(phase)),
        "phase": (np.sin(phase), np.cos(phase)),
    }[feature]
    return np.stack(parts, axis=-1)


def _reduce(data, pos, members):
    pieces = [np.take(data, idx, axis=pos).mean(axis=pos) for idx in members]
    return np.stack(pieces, axis=pos)


def _members(values, ranges):
    return [
        np.flatnonzero((values >= start) & (values < stop))
        for start, stop in ranges.values()
    ]


def _radius_patches(values, radius):
    if radius is None:
        return [np.arange(len(values))]
    return [np.flatnonzero(np.abs(values - v) <= radius + 1e-9) for v in values]


def _space_patches(case, n_space):
    radius = case["search"]["space"]
    if radius is None:
        return [np.arange(n_space)]
    if radius == 0:
        return [np.array([idx]) for idx in range(n_space)]
    if case["input"] in ("array", "rsa_data"):
        return _radius_patches(np.arange(n_space, dtype=float), radius)
    pos = np.array([ch["loc"][:3] for ch in _make_info()["chs"]])
    dist = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis], axis=-1)
    return [np.flatnonzero(row < radius) for row in dist]


def reference(case):
    """Return ``(scores, axes)`` computed by brute force."""
    data = _project(make_input(case["input"])[1], case["feature"])
    axes = list(INPUTS[case["input"]])
    if data.ndim > len(axes):
        axes.append("component")
    if case["log"]:
        data = np.log10(data)
    values = {"time": TIMES, "freq": FREQS}
    patches = {}
    for axis in ("freq", "time"):
        if axis not in axes:
            continue
        pos, how = axes.index(axis), case[axis]
        if how == "range":
            keep = (values[axis] >= RANGES[axis][0]) & (values[axis] <= RANGES[axis][1])
            data = np.compress(keep, data, axis=pos)
            values[axis] = values[axis][keep]
        elif how == "named":
            data = _reduce(data, pos, _members(values[axis], NAMED[axis]))
            axes[pos] = COARSE[axis]
            values[COARSE[axis]] = None
        elif how == "elements":
            patches[axis] = _members(values[axis], NAMED[axis])
    if case["space"] == "named":
        groups = list(_groups_for(case["input"]).values())
        data = _reduce(data, axes.index("space"), groups)
        axes[axes.index("space")] = "roi"
    elif case["space"] == "elements":
        patches["space"] = [np.array(x) for x in _groups_for(case["input"]).values()]
    if case["standardize"]:
        std = data.std(axis=0, keepdims=True)
        data = (data - data.mean(axis=0, keepdims=True)) / np.where(std > 0, std, 1)

    search = {}
    for axis, radius in case["search"].items():
        n = data.shape[axes.index(axis)]
        if axis in patches:
            search[axis] = patches[axis]
        elif axis == "space":
            search[axis] = _space_patches(case, n)
        elif values.get(axis) is None or radius == 0:
            search[axis] = (
                [np.arange(n)] if radius is None else [np.array([i]) for i in range(n)]
            )
        else:
            search[axis] = _radius_patches(values[axis], radius)

    scores = []
    for combo in itertools.product(*search.values()):
        sub = data
        for axis, idx in zip(search, combo):
            sub = np.take(sub, idx, axis=axes.index(axis))
        rdm = pdist(sub.reshape(N_ITEMS, -1), "euclidean")
        scores.append(spearmanr(rdm, MODEL_RDM)[0])
    pooled = {axis for axis, radius in case["search"].items() if radius is None}
    kept = {axis: grid for axis, grid in search.items() if axis not in pooled}
    return np.array(scores).reshape([len(g) for g in kept.values()]), tuple(kept)


def run_case(case):
    """Run one case through the package."""
    inst = make_input(case["input"])[0]
    kwargs = {}
    if case["input"] == "array":
        kwargs = dict(axes=INPUTS["array"], coords={"freq": FREQS, "time": TIMES})
    if case["input"].endswith("+tfr"):
        kwargs["tfr"] = {**TFR, "verbose": False}
    for axis in ("time", "freq"):
        if case[axis] == "range":
            kwargs[axis] = RANGES[axis]
        elif case[axis] in ("named", "elements"):
            kwargs[axis] = NAMED[axis]
            kwargs[f"{axis}_mode"] = (
                "aggregate" if case[axis] == "named" else "elements"
            )
    if case["space"] != "native":
        kwargs["space"] = _space_arg(case["input"])
        kwargs["space_mode"] = "aggregate" if case["space"] == "named" else "elements"
    prepared = prepare_rsa_data(
        inst,
        tfr_feature=case["feature"],
        log=case["log"],
        standardize="features" if case["standardize"] else None,
        **kwargs,
    )
    radii = {a: r for a, r in case["search"].items() if r != 0}
    return rsa_nd(
        prepared,
        MODEL_RDM,
        search_axes=tuple(case["search"]),
        radii=radii,
        data_rdm_metric="euclidean",
    )


def _axis_choices(axes):
    """Yield per-axis ``(time, freq, space)`` preparation choices."""
    time = ("native", "range", "named", "elements") if "time" in axes else ("native",)
    freq = ("native", "range", "named", "elements") if "freq" in axes else ("native",)
    return itertools.product(time, freq, ("native", "named", "elements"))


def _final_axes(axes, time, freq, space):
    out = [a for a in axes if a != "item"]
    for axis, how in (("time", time), ("freq", freq), ("space", space)):
        if how == "named":
            out[out.index(axis)] = COARSE[axis]
    return out


def _search_choices(final, time, freq, space, feature):
    """Yield every valid ``{axis: radius}`` search for the prepared axes."""
    how = {"time": time, "freq": freq, "space": space}
    options = []
    for axis in final:
        if how.get(axis) == "elements":
            options.append([(axis, 0)])  # elements mode must be searched
        elif axis in RADII:
            options.append([None, (axis, 0), (axis, RADII[axis]), (axis, None)])
        else:
            options.append([None, (axis, 0), (axis, None)])
    if feature in COMPLEX_FEATURES:
        options.append([None, ("component", 0)])
    for combo in itertools.product(*options):
        yield {axis: radius for axis, radius in filter(None, combo)}


def all_cases():
    """Yield every valid case."""
    for name, axes in INPUTS.items():
        complex_input = name.endswith(("complex", "+tfr"))
        features = (
            ("power", "amplitude", *COMPLEX_FEATURES) if complex_input else ("power",)
        )
        for feature in features:
            for time, freq, space in _axis_choices(axes):
                if space != "native" and name in ("array", "rsa_data"):
                    continue
                if feature in COMPLEX_FEATURES and "named" in (time, freq):
                    continue
                final = _final_axes(axes, time, freq, space)
                for search in _search_choices(final, time, freq, space, feature):
                    # A space radius needs geometry that surface stcs lack here.
                    if search.get("space") not in (0, None) and "stcs" in name:
                        continue
                    yield dict(
                        input=name,
                        feature=feature,
                        time=time,
                        freq=freq,
                        space=space,
                        search=search,
                        log=False,
                        standardize=False,
                    )


def _coverage_cases():
    """Pick a sample that tries every input, setup and search at least once.

    Set ``RSA_ND_ALL_CASES=1`` to run all of the roughly 35,000 cases.
    """
    if os.environ.get("RSA_ND_ALL_CASES"):
        return list(all_cases())
    seen, cases = set(), []
    for case in all_cases():
        how = (case["time"], case["freq"], case["space"])
        keys = [
            (case["input"], case["feature"], case["space"]),
            (case["input"], *how),
            ("search", str(case["search"]))
            if how[:2] == ("native", "native")
            else None,
        ]
        if not seen.issuperset(filter(None, keys)):
            seen.update(filter(None, keys))
            cases.append(case)
    return cases


def _case_id(case):
    return "-".join(
        [
            case["input"],
            case["feature"],
            f"t.{case['time']}",
            f"f.{case['freq']}",
            f"s.{case['space']}",
            *(f"{axis}.{radius}" for axis, radius in case["search"].items()),
        ]
    )


@pytest.mark.parametrize("case", _coverage_cases(), ids=_case_id)
def test_matches_reference(case, tmp_path):
    """Every setup gives the brute-force scores and a usable result."""
    expected, axes = reference(case)
    result = run_case(case)
    assert result.axes == axes
    np.testing.assert_allclose(result.rsa, expected, atol=1e-8)

    try:
        assert result.to_mne().get_data().size == result.rsa.size
    except ValueError as err:
        assert "to_mne()" in str(err)

    result.save(tmp_path / "result.h5")
    loaded = RsaResult.load(tmp_path / "result.h5")
    assert loaded.axes == result.axes
    np.testing.assert_allclose(loaded.rsa, result.rsa)


@pytest.mark.parametrize("name", INPUTS)
@pytest.mark.parametrize("log", (False, True))
@pytest.mark.parametrize("standardize", (False, True))
def test_log_and_standardize(name, log, standardize):
    """The log and z-score steps work for every input, in the right order."""
    case = dict(
        input=name,
        feature="power",
        time="named" if "time" in INPUTS[name] else "native",
        freq="native",
        space="native" if name in ("array", "rsa_data") else "named",
        search={},
        log=log,
        standardize=standardize,
    )
    case["search"] = {"roi": 0} if case["space"] == "named" else {"space": 0}
    expected, _ = reference(case)
    np.testing.assert_allclose(run_case(case).rsa, expected, atol=1e-8)
