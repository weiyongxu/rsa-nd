"""Turn MNE objects and arrays into the named feature tensor used for RSA."""

from collections.abc import Mapping
from numbers import Number

import mne
import numpy as np
from mne.time_frequency import (
    AverageTFR,
    EpochsSpectrum,
    EpochsTFR,
    Spectrum,
    tfr_array_morlet,
)

from ._data import PreparedRsaData, RsaData, SensorSpace, SourceSpace

_SINGLE_TYPES = (mne.BaseEpochs, EpochsSpectrum, EpochsTFR)
_SOURCE_TYPES = (mne.SourceEstimate, mne.VolSourceEstimate)
_ITEM_TYPES = (mne.Evoked, Spectrum, AverageTFR, *_SOURCE_TYPES)
_AGGREGATORS = ("mean", "median", "sum", "min", "max")
_MODES = ("aggregate", "elements")
_COMPONENTS = {
    "real_imag": ("real", "imag"),
    "amp_phase": ("amplitude", "sin_phase", "cos_phase"),
    "phase": ("sin_phase", "cos_phase"),
}


def prepare_rsa_data(
    inst,
    *,
    axes=None,
    coords=None,
    time=None,
    time_mode="aggregate",
    freq=None,
    freq_mode="aggregate",
    space=None,
    space_mode="elements",
    picks="data",
    tfr=None,
    tfr_feature="power",
    log=False,
    log_floor=None,
    standardize=None,
    range_closed=None,
    agg="mean",
    src=None,
):
    """Build the feature tensor for RSA.

    Parameters
    ----------
    inst : instance of Epochs | EpochsTFR | EpochsSpectrum | RsaData | ndarray | list
        The data. A list holds one ``Evoked``, ``Spectrum``, ``AverageTFR``,
        ``SourceEstimate`` or ``VolSourceEstimate`` per item.
    axes : sequence of str | None
        Axis names. Required when ``inst`` is an array.
    coords : dict | None
        Axis coordinates for an array ``inst``.
    time : tuple | dict | None
        Either ``(start, stop)`` to keep part of the axis, or
        ``{"name": (start, stop), ...}`` for named groups (see ``time_mode``).
    time_mode : str
        How named groups are used, for ``time``, ``freq`` and ``space`` alike.
        ``"aggregate"`` reduces each group to one value with ``agg``, turning the
        axis into ``"window"``, ``"band"`` or ``"roi"``. ``"elements"`` keeps every
        member as a feature, so each group is one searchlight position; the axis
        must then be searched.
    freq : tuple | dict | None
        Same as ``time``, for frequency.
    freq_mode : str
        See ``time_mode``.
    space : dict | list of Label | None
        Named groups. For sensors, a mapping from group name to channel names.
        For source estimates, a list of labels, or a mapping from group name to
        labels (or to vertex numbers, for volume source spaces).
    space_mode : str
        See ``time_mode``.
    picks : str | list | None
        Channels to use for sensor input. Bad channels are excluded unless
        ``picks`` is ``None``.
    tfr : dict | None
        Arguments for :func:`mne.time_frequency.tfr_array_morlet`, to compute a
        time-frequency representation of time-domain MNE input. Must hold
        ``"freqs"``.
    tfr_feature : str
        Real-valued feature for complex data: ``"power"``, ``"amplitude"``,
        ``"real_imag"``, ``"amp_phase"`` or ``"phase"``. The last three add a
        ``"component"`` axis.
    log : bool
        Whether to take the log10 before reducing ranges and groups.
    log_floor : float | None
        Smallest value to take the log of. By default, one millionth of the
        median positive value.
    standardize : str | None
        Use ``"features"`` to z-score each feature across items, as the last step.
    range_closed : str | None
        Which ends of a range to include: ``"both"``, ``"left"``, ``"right"`` or
        ``"neither"``. By default ``"both"`` for ``(start, stop)`` and ``"left"``
        for named groups, so that adjacent groups share no points.
    agg : str | callable
        Reduction for ``"aggregate"`` mode: ``"mean"``, ``"median"``, ``"sum"``,
        ``"min"``, ``"max"``, or a function taking ``(data, axis=...)``.
    src : instance of SourceSpaces | None
        Source space, needed for a spatial radius search in source space.

    Returns
    -------
    prepared : PreparedRsaData
        The prepared data.
    """
    _check_options(time_mode, freq_mode, space_mode, standardize, tfr_feature, log)

    if isinstance(inst, (RsaData, np.ndarray)):
        data, axes, coords = _read_array(inst, axes, coords, space, tfr)
        kind, labels, geometry = "array", None, None
    else:
        data, axes, coords, kind, labels, geometry, sfreq = _read_mne(
            inst, picks, space, src
        )
        if tfr is not None:
            data, axes, coords = _compute_tfr(data, axes, coords, sfreq, tfr)
    if "item" not in axes:
        raise ValueError("The data needs an 'item' axis.")

    data, axes, coords = _project_feature(data, axes, coords, tfr_feature)
    if log:
        data = _safe_log10(data, log_floor)

    element_groups = {}
    for axis, coarse, spec, mode in (
        ("freq", "band", freq, freq_mode),
        ("time", "window", time, time_mode),
    ):
        if spec is None:
            continue
        if axis not in axes:
            raise ValueError(f"Cannot select {axis!r}: the data has no such axis.")
        values = np.asarray(coords[axis], dtype=float)
        if _is_plain_range(spec):
            if mode == "elements":
                raise ValueError(f"{axis}_mode='elements' needs named ranges.")
            keep = _in_range(values, spec, range_closed or "both", axis)
            data = np.take(data, keep, axis=axes.index(axis))
            coords[axis] = values[keep]
            continue
        groups = _named_ranges(values, spec, range_closed or "left", axis)
        if mode == "elements":
            element_groups[axis] = groups
            continue
        if tfr_feature in _COMPONENTS:
            raise ValueError(
                f"Named {axis} aggregation does not work with tfr_feature="
                f"{tfr_feature!r}. Use a (start, stop) range or {axis}_mode='elements'."
            )
        data, axes, coords = _collapse(data, axes, coords, axis, coarse, groups, agg)

    if space_mode == "aggregate":
        if getattr(geometry, "patches", None) is None:
            raise ValueError("space_mode='aggregate' needs groups in space=.")
        groups, geometry = geometry.patches, None
        data, axes, coords = _collapse(data, axes, coords, "space", "roi", groups, agg)

    if standardize == "features":
        data = _zscore(data, axes.index("item"))

    return PreparedRsaData(
        RsaData(data, axes, coords), kind, labels, geometry, element_groups
    )


def source_roi_patches(stc, rois):
    """Find the rows of a source estimate that belong to each label.

    Parameters
    ----------
    stc : instance of SourceEstimate | VolSourceEstimate
        Its vertices define the available source space.
    rois : list of Label
        The labels.

    Returns
    -------
    patches : dict
        Row indices of ``stc.data`` per label name.
    kept_rois : list of Label
        The labels with at least one vertex in ``stc``, in input order.
    """
    patches, kept_rois = {}, []
    for idx, roi in enumerate(rois):
        name = getattr(roi, "name", None) or f"roi_{idx}"
        if name in patches:
            raise ValueError(f"Duplicate ROI name {name!r}.")
        indices = _label_indices(stc, [roi])
        if indices.size:
            patches[name] = indices
            kept_rois.append(roi)
    return patches, kept_rois


def _check_options(time_mode, freq_mode, space_mode, standardize, feature, log):
    for mode in (time_mode, freq_mode, space_mode):
        if mode not in _MODES:
            raise ValueError(f"Modes must be one of {_MODES}, got {mode!r}.")
    if standardize not in (None, "features"):
        raise ValueError("standardize must be None or 'features'.")
    if feature not in ("power", "amplitude", *_COMPONENTS):
        raise ValueError(f"Unknown tfr_feature {feature!r}.")
    if log and feature in _COMPONENTS:
        raise ValueError("log=True needs tfr_feature='power' or 'amplitude'.")


def _read_array(inst, axes, coords, space, tfr):
    if space is not None:
        raise ValueError("space= needs MNE input, which carries the geometry.")
    if tfr is not None:
        raise ValueError(
            "tfr= needs MNE input, which carries the timing. Compute the TFR with "
            "mne.time_frequency.tfr_array_morlet() and pass the result instead."
        )
    if isinstance(inst, RsaData):
        if axes is not None or coords is not None:
            raise ValueError("axes and coords are not used with RsaData.")
        return inst.data, list(inst.axes), dict(inst.coords)
    if axes is None:
        raise ValueError("axes is required when the data is an array.")
    # RsaData checks the axes; complex values are projected to real ones later.
    checked = RsaData(np.zeros(inst.shape), axes, coords)
    return inst, list(checked.axes), dict(checked.coords)


def _read_mne(inst, picks, space, src):
    single = isinstance(inst, _SINGLE_TYPES)
    if not single and not (
        isinstance(inst, (list, tuple))
        and len(inst) > 0
        and isinstance(inst[0], _ITEM_TYPES)
    ):
        raise TypeError(
            "inst must be an array, RsaData, Epochs, EpochsTFR, EpochsSpectrum, or a "
            "list of Evoked, Spectrum, AverageTFR, SourceEstimate or "
            f"VolSourceEstimate. Got {type(inst).__name__}."
        )
    items = [inst] if single else list(inst)
    is_source = isinstance(items[0], _SOURCE_TYPES)
    if picks is not None and not is_source:
        items = [item.copy().pick(picks, exclude="bads") for item in items]
    _check_compatible(items)
    ref = items[0]

    axes, coords = ["item", "space"], {}
    if hasattr(ref, "freqs"):
        axes.append("freq")
        coords["freq"] = np.asarray(ref.freqs, dtype=float)
    if hasattr(ref, "times"):
        axes.append("time")
        coords["time"] = np.asarray(ref.times, dtype=float)

    if is_source:
        data = np.stack([stc.data for stc in items])
        sfreq = 1 / ref.tstep
        patches = None if space is None else _source_patches(ref, space)
        kind = "volume" if isinstance(ref, mne.VolSourceEstimate) else "surface"
        vertices = tuple(np.asarray(verts) for verts in ref.vertices)
        geometry = SourceSpace(vertices, src, patches, kind, ref.subject)
    else:
        kwargs = {"exclude": ()} if hasattr(ref, "freqs") else {}
        if single:
            data = ref.get_data(**kwargs)
        else:
            data = np.stack([item.get_data(**kwargs) for item in items])
        sfreq = ref.info["sfreq"]
        patches = None if space is None else _sensor_patches(ref.ch_names, space)
        geometry = SensorSpace(tuple(ref.ch_names), ref.info, patches)
        coords["space"] = np.asarray(ref.ch_names, dtype=object)
    if data.ndim != len(axes):
        raise ValueError(
            "Spectral data with tapers or segments is not supported. Use "
            "output='power', or a method that gives one estimate per item."
        )
    labels = ref.events[:, 2] if single else None
    return data, axes, coords, type(ref).__name__, labels, geometry, sfreq


def _check_compatible(items):
    ref = items[0]
    for idx, item in enumerate(items[1:], 1):
        same = True
        if hasattr(ref, "ch_names"):
            same &= item.ch_names == ref.ch_names
        if hasattr(ref, "vertices"):
            same &= len(item.vertices) == len(ref.vertices) and all(
                np.array_equal(a, b) for a, b in zip(item.vertices, ref.vertices)
            )
        for attr in ("times", "freqs"):
            if hasattr(ref, attr):
                a, b = getattr(item, attr), getattr(ref, attr)
                same &= len(a) == len(b) and np.allclose(a, b)
        if not same:
            raise ValueError(
                f"Item {idx} does not match item 0: all items need the same "
                "channels or vertices, and the same times and frequencies."
            )


def _sensor_patches(ch_names, space):
    if not isinstance(space, Mapping):
        raise TypeError("space= for sensor input must map group names to channels.")
    patches = {}
    for name, members in space.items():
        unknown = sorted(set(members) - set(ch_names))
        if unknown:
            raise ValueError(f"Group {name!r} has unknown channels: {unknown}.")
        patches[str(name)] = np.array([ch_names.index(ch) for ch in members])
    return patches


def _source_patches(stc, space):
    if isinstance(space, Mapping):
        entries = [
            (name, labels if isinstance(labels, (list, tuple)) else [labels])
            for name, labels in space.items()
        ]
        patches = {str(name): _label_indices(stc, labels) for name, labels in entries}
        patches = {name: idx for name, idx in patches.items() if idx.size}
    else:
        patches, _ = source_roi_patches(stc, space)
    if not patches:
        raise ValueError("No group in space= has vertices in the source estimates.")
    return patches


def _label_indices(stc, labels):
    """Return the rows of ``stc.data`` covered by labels or vertex numbers."""
    offsets = np.cumsum([0, *(len(verts) for verts in stc.vertices)])
    indices = [np.zeros(0, int)]
    for label in labels:
        if isinstance(label, mne.BiHemiLabel):
            indices.append(_label_indices(stc, [label.lh, label.rh]))
        elif isinstance(label, mne.Label):
            hemi = 1 if label.hemi == "rh" else 0
            if hemi < len(stc.vertices):
                verts = stc.vertices[hemi]
                used = label.get_vertices_used(verts)
                indices.append(np.searchsorted(verts, used) + offsets[hemi])
        else:
            used = np.intersect1d(label, stc.vertices[0])
            indices.append(np.searchsorted(stc.vertices[0], used))
    return np.unique(np.concatenate(indices))


def _compute_tfr(data, axes, coords, sfreq, tfr):
    if "time" not in axes or "freq" in axes:
        raise ValueError("tfr= needs time-domain input.")
    if "freqs" not in tfr:
        raise ValueError("tfr= must include 'freqs'.")
    data = tfr_array_morlet(data, sfreq, output="complex", **tfr)
    decim = tfr.get("decim", 1)
    times = coords["time"][
        slice(None, None, decim) if isinstance(decim, int) else decim
    ]
    freqs = np.asarray(tfr["freqs"], dtype=float)
    return data, axes[:2] + ["freq", "time"], {**coords, "freq": freqs, "time": times}


def _project_feature(data, axes, coords, feature):
    """Project complex data onto real values."""
    if not np.iscomplexobj(data):
        if feature != "power":
            raise ValueError(f"tfr_feature={feature!r} needs complex data.")
        return data, axes, coords
    amplitude = np.abs(data)
    if feature == "power":
        return amplitude**2, axes, coords
    if feature == "amplitude":
        return amplitude, axes, coords
    if "component" in axes:
        raise ValueError(f"tfr_feature={feature!r} adds a 'component' axis.")
    # Unit phasors give sin and cos of the phase; zero amplitude has no phase.
    phasor = np.divide(data, amplitude, out=np.zeros_like(data), where=amplitude > 0)
    parts = {
        "real": data.real,
        "imag": data.imag,
        "amplitude": amplitude,
        "sin_phase": phasor.imag,
        "cos_phase": phasor.real,
    }
    names = _COMPONENTS[feature]
    data = np.stack([parts[name] for name in names], axis=-1)
    return data, [*axes, "component"], {**coords, "component": np.array(names, object)}


def _safe_log10(data, floor=None):
    """Take the log10, clipping values below ``floor`` to avoid ``-inf``."""
    if floor is None:
        positive = data[np.isfinite(data) & (data > 0)]
        floor = np.median(positive) * 1e-6 if positive.size else 0
        floor = max(floor, np.finfo(float).tiny)
    if floor <= 0:
        raise ValueError("log_floor must be positive.")
    return np.log10(np.maximum(data, floor))


def _is_plain_range(spec):
    return (
        isinstance(spec, (tuple, list))
        and len(spec) == 2
        and all(isinstance(limit, Number) for limit in spec)
    )


def _in_range(values, limits, closed, name):
    """Return the indices of ``values`` within ``limits``."""
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError("range_closed must be 'both', 'left', 'right' or 'neither'.")
    start, stop = limits
    if stop <= start:
        raise ValueError(f"Range {name!r} needs stop > start, got {limits}.")
    lower = values >= start if closed in ("both", "left") else values > start
    upper = values <= stop if closed in ("both", "right") else values < stop
    indices = np.flatnonzero(lower & upper)
    if indices.size == 0:
        raise ValueError(f"Range {name!r} {limits} contains no values.")
    return indices


def _named_ranges(values, spec, closed, axis):
    groups = {
        str(name): _in_range(values, limits, closed, name)
        for name, limits in dict(spec).items()
    }
    if not groups:
        raise ValueError(f"{axis}= needs at least one range.")
    return groups


def _collapse(data, axes, coords, axis, new_axis, groups, agg):
    """Reduce each group of an axis to one value and rename the axis."""
    if not callable(agg) and agg not in _AGGREGATORS:
        raise ValueError(f"agg must be callable or one of {_AGGREGATORS}.")
    reduce = agg if callable(agg) else getattr(np, agg)
    pos = axes.index(axis)
    pieces = [reduce(np.take(data, idx, axis=pos), axis=pos) for idx in groups.values()]
    coords = {key: value for key, value in coords.items() if key != axis}
    coords[new_axis] = np.array(list(groups), dtype=object)
    return np.stack(pieces, axis=pos), [*axes[:pos], new_axis, *axes[pos + 1 :]], coords


def _zscore(data, item_axis):
    mean = np.nanmean(data, axis=item_axis, keepdims=True)
    std = np.nanstd(data, axis=item_axis, keepdims=True)
    return (data - mean) / np.where(std > 0, std, 1)
