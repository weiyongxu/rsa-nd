"""Spectral feature preparation for named RSA data."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._types import (
    _AGGREGATORS,
    _ALL_TFR_FEATURES,
    _COMPLEX_DERIVED_FEATURES,
    AggFunc,
    AxisSpec,
    EpochsSpectrum,
    EpochsTFR,
    RangeSpec,
    RsaData,
    RsaPrepOptions,
    StandardizeMode,
    TfrFeatureMode,
    _as_real_float,
    _normalize_coords,
    _validate_axes,
)

_RANGE_CLOSED = {"both", "left", "right", "neither"}


def prepare_time_domain_rsa(
    data: np.ndarray,
    *,
    item_coords: Sequence[Any],
    space_coords: Sequence[Any],
    times: Sequence[float],
    sfreq: float,
    opts: RsaPrepOptions,
) -> RsaData:
    """Finalize ``item x space x time`` data into ``RsaData``.

    Shared finalizer for the time-domain Adapters (Epochs / Evokeds /
    SourceEstimates): with ``opts.tfr`` set, compute an on-the-fly Morlet TFR;
    otherwise feed the raw time-domain array straight into the spectral feature
    pipeline. The Adapter owns only the MNE extraction that produces ``data``
    and its coordinates.
    """
    times = np.asarray(times, dtype=float)
    if opts.tfr is None:
        return _prepare_spectral_core(
            data,
            axes=("item", "space", "time"),
            coords={
                "item": np.asarray(item_coords),
                "space": np.asarray(space_coords),
                "time": times,
            },
            picks=None,
            opts=opts,
        )
    return _prepare_morlet_rsa_core(
        data,
        sfreq=float(sfreq),
        times=times,
        space_coords=space_coords,
        item_coords=item_coords,
        opts=opts,
    )


def finalize_spectral_rsa(
    inst_or_data: RsaData | EpochsSpectrum | EpochsTFR | np.ndarray,
    *,
    axes: Sequence[str] | None,
    coords: Mapping[str, Sequence[Any]] | None,
    picks: Any,
    opts: RsaPrepOptions,
) -> RsaData:
    """Finalize already-spectral input via the spectral feature pipeline.

    Shared finalizer for the already-spectral Adapters (EpochsSpectrum /
    AverageTFRs / array / RsaData), which differ only in how ``axes`` /
    ``coords`` / ``picks`` are sourced.
    """
    return _prepare_spectral_core(
        inst_or_data,
        axes=axes,
        coords=coords,
        picks=picks,
        opts=opts,
    )


def safe_log10_power(data: np.ndarray, floor: float | None = None) -> np.ndarray:
    """Return log10 power without producing ``-inf`` for zeros.

    If ``floor`` is not supplied, values are clipped to
    ``median(positive_values) * 1e-6``. This avoids letting tiny numerical
    positives set an extremely small floor for the whole array.
    """

    arr = _as_real_float(data)
    if floor is None:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        floor = (
            float(np.median(positive) * 1e-6) if positive.size else np.finfo(float).tiny
        )
        floor = max(floor, np.finfo(float).tiny)
    if floor <= 0:
        raise ValueError("floor must be positive.")
    return np.log10(np.maximum(arr, floor))


def _standardize_data(
    data: np.ndarray,
    axes: Sequence[str],
    standardize: StandardizeMode,
) -> np.ndarray:
    if standardize is None:
        return data
    if standardize != "features":
        raise ValueError("standardize must be None or 'features'.")
    if "item" not in axes:
        raise ValueError("standardize='features' requires an 'item' axis.")
    return _zscore_features(data, item_axis=tuple(axes).index("item"))


def _zscore_features(data: np.ndarray, *, item_axis: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.shape[item_axis] < 2:
        warnings.warn(
            "standardize='features' received fewer than 2 items; all "
            "standardized feature values will be zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    mean = np.nanmean(arr, axis=item_axis, keepdims=True)
    std = np.nanstd(arr, axis=item_axis, keepdims=True)
    return (arr - mean) / np.where(std > 0, std, 1.0)


def _normalize_ranges(ranges: RangeSpec) -> list[tuple[str, tuple[float, float]]]:
    items = ranges.items() if isinstance(ranges, Mapping) else ranges
    normalized = []
    seen_labels: set[str] = set()
    for name, limits in items:
        label = str(name)
        if label in seen_labels:
            raise ValueError(f"Duplicate range label {label!r}.")
        seen_labels.add(label)
        if len(limits) != 2:
            raise ValueError(f"Range {name!r} must be a (start, stop) pair.")
        start, stop = float(limits[0]), float(limits[1])
        if stop <= start:
            raise ValueError(f"Range {name!r} stop must be greater than start.")
        normalized.append((label, (start, stop)))
    if not normalized:
        raise ValueError("At least one range must be supplied.")
    return normalized


def _range_mask(
    values: np.ndarray,
    start: float,
    stop: float,
    closed: str,
) -> np.ndarray:
    if closed not in _RANGE_CLOSED:
        raise ValueError("range_closed must be 'both', 'left', 'right', or 'neither'.")
    lower = values >= start if closed in ("both", "left") else values > start
    upper = values <= stop if closed in ("both", "right") else values < stop
    return lower & upper


def _format_range(start: float, stop: float, closed: str) -> str:
    left = "[" if closed in ("both", "left") else "("
    right = "]" if closed in ("both", "right") else ")"
    return f"{left}{start}, {stop}{right}"


def _replace_range_axis(
    data: np.ndarray,
    axes: list[str],
    coords: dict[str, np.ndarray],
    *,
    source_axis: str,
    target_axis: str,
    ranges: RangeSpec,
    range_closed: str,
    agg: AggFunc,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if source_axis not in axes:
        raise ValueError(f"Cannot create {target_axis!r}: no {source_axis!r} axis.")

    axis_idx = axes.index(source_axis)
    values = np.asarray(coords[source_axis], dtype=float)
    pieces = []
    labels = []
    for label, (start, stop) in _normalize_ranges(ranges):
        mask = _range_mask(values, start, stop, range_closed)
        if not np.any(mask):
            raise ValueError(
                f"{target_axis.title()} {label!r} "
                f"{_format_range(start, stop, range_closed)} contains no "
                f"{source_axis} values."
            )
        selected = np.take(data, np.where(mask)[0], axis=axis_idx)
        pieces.append(_aggregate(selected, axis_idx, agg))
        labels.append(label)

    data = np.stack(pieces, axis=axis_idx)
    axes = list(axes)
    axes[axis_idx] = target_axis
    coords = dict(coords)
    coords.pop(source_axis)
    coords[target_axis] = np.asarray(labels, dtype=object)
    return data, axes, coords


def resolve_element_groups(
    values: Sequence[Any],
    ranges: RangeSpec,
    *,
    range_closed: str | None = None,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Resolve named coordinate ranges into ordered ``(name, member_indices)``.

    Prep-time home of elements-mode grouping for numeric axes (``time`` /
    ``freq``): each named range selects the member indices kept as one
    searchlight patch later. Mirrors the band / window endpoint rule -- the
    default is ``"left"`` so adjacent groups tile without sharing a boundary
    point. Raises if a group selects no coordinate values.
    """

    numeric = np.asarray(values)
    if not np.issubdtype(numeric.dtype, np.number):
        raise ValueError(
            "elements-mode grouping requires a numeric axis (e.g. time / freq)."
        )
    numeric = numeric.astype(float)
    closed = "left" if range_closed is None else range_closed
    groups: list[tuple[str, np.ndarray]] = []
    for label, (start, stop) in _normalize_ranges(ranges):
        indices = np.flatnonzero(_range_mask(numeric, start, stop, closed))
        if indices.size == 0:
            raise ValueError(
                f"Group {label!r} contains no coordinate values on this axis."
            )
        groups.append((label, indices.astype(int)))
    return tuple(groups)


def aggregate_space_groups(
    rsa_data: RsaData,
    groups: Mapping[str, Sequence[int]],
    *,
    agg: AggFunc,
    source_axis: str = "space",
    target_axis: str = "roi",
) -> RsaData:
    """Collapse ``source_axis`` into ``target_axis`` by aggregating member indices.

    Membership analogue of the prep-time band / window aggregation
    (:func:`_replace_range_axis`): each group's member indices along
    ``source_axis`` are reduced with ``agg`` (default mean) into one element,
    and the axis is renamed to ``target_axis`` with the group names as
    coordinates. Backs aggregate-mode source ROI / channel-group RSA -- one
    representative (e.g. mean) signal per group instead of keeping every member
    as a joint feature.
    """

    axes = list(rsa_data.axes)
    if source_axis not in axes:
        raise ValueError(f"Cannot aggregate groups: no {source_axis!r} axis.")
    if not groups:
        raise ValueError("space groups must contain at least one group.")

    axis_idx = axes.index(source_axis)
    pieces = []
    labels = []
    for name, indices in groups.items():
        idx = np.atleast_1d(np.asarray(indices, dtype=int))
        if idx.size == 0:
            raise ValueError(f"Group {name!r} has no members to aggregate.")
        selected = np.take(rsa_data.data, idx, axis=axis_idx)
        pieces.append(_aggregate(selected, axis_idx, agg))
        labels.append(str(name))

    new_data = np.stack(pieces, axis=axis_idx)
    axes[axis_idx] = target_axis
    coords = dict(rsa_data.coords)
    coords.pop(source_axis, None)
    coords[target_axis] = np.asarray(labels, dtype=object)
    return RsaData(data=new_data, axes=tuple(axes), coords=coords)


def _aggregate(data: np.ndarray, axis: int, agg: AggFunc) -> np.ndarray:
    if callable(agg):
        return np.asarray(agg(data, axis=axis))
    try:
        aggregator = _AGGREGATORS[agg]
    except (KeyError, TypeError):
        raise ValueError(
            "agg must be 'mean', 'median', 'sum', 'min', 'max', or callable."
        ) from None
    return aggregator(data, axis=axis)


def _prepare_morlet_rsa_core(
    data: np.ndarray,
    *,
    sfreq: float,
    times: Sequence[float],
    space_coords: Sequence[Any],
    item_coords: Sequence[Any] | None = None,
    opts: RsaPrepOptions,
) -> RsaData:
    """Compute Morlet TFR on item x space x time data and return ``RsaData``.

    The caller owns MNE-object extraction. This helper owns the array-level
    ``opts.tfr={...}`` Morlet grammar and the handoff into the spectral feature
    pipeline. ``opts.tfr`` must be set: the caller takes this branch only when a
    Morlet TFR is requested. The post-spectral feature options on ``opts``
    (``tfr_feature``, ``log``, ``freq``, ``time``, ...) then apply to the
    computed TFR; see :class:`RsaPrepOptions` for the field semantics.
    """

    if opts.tfr is None:
        raise ValueError("_prepare_morlet_rsa_core requires opts.tfr to be set.")
    morlet_kwargs = dict(opts.tfr)
    if "freqs" not in morlet_kwargs:
        raise ValueError("tfr={...} must include 'freqs'.")
    freqs = np.asarray(morlet_kwargs.pop("freqs"), dtype=float)
    n_cycles = morlet_kwargs.pop("n_cycles", 1.0)

    from mne.time_frequency import tfr_array_morlet

    complex_tfr = tfr_array_morlet(
        np.asarray(data),
        sfreq=float(sfreq),
        freqs=freqs,
        n_cycles=n_cycles,
        output="complex",
        **morlet_kwargs,
    )
    item_values = (
        np.arange(np.asarray(data).shape[0])
        if item_coords is None
        else np.asarray(item_coords)
    )
    return _prepare_spectral_core(
        complex_tfr,
        axes=("item", "space", "freq", "time"),
        coords={
            "item": item_values,
            "space": np.asarray(space_coords),
            "freq": freqs,
            "time": np.asarray(times, dtype=float),
        },
        picks=None,
        opts=opts,
    )


def _prepare_spectral_core(
    inst_or_data: RsaData | EpochsSpectrum | EpochsTFR | np.ndarray,
    *,
    axes: Sequence[str] | None = None,
    coords: Mapping[str, Sequence[Any]] | None = None,
    picks: Any = "data",
    opts: RsaPrepOptions,
) -> RsaData:
    """Convert MNE spectral data or a NumPy array to ``RsaData``.

    ``axes`` / ``coords`` describe a raw NumPy array (and are unused for MNE /
    ``RsaData`` input); ``picks`` selects channels for MNE spectral objects.
    Every other preparation choice -- ``tfr_feature``, ``log``, ``log_floor``,
    ``freq``, ``time``, ``standardize``, ``range_closed``, ``agg`` -- is read
    from ``opts``; see :class:`RsaPrepOptions` for the field semantics.
    """

    data, axes_list, coord_map = _coerce_to_named_parts(
        inst_or_data,
        axes=axes,
        coords=coords,
        picks=picks,
    )

    if "item" not in axes_list:
        raise ValueError("axes must include an 'item' axis.")
    data, axes_list, coord_map = _project_tfr_feature(
        data,
        axes_list,
        coord_map,
        opts.tfr_feature,
    )
    if opts.log and opts.tfr_feature in _COMPLEX_DERIVED_FEATURES:
        raise ValueError(
            "log=True is only valid for tfr_feature='power' or 'amplitude'."
        )
    if opts.tfr_feature in _COMPLEX_DERIVED_FEATURES:
        if _is_named_range_spec(opts.freq):
            raise ValueError(
                "named frequency aggregation is not supported for "
                f"tfr_feature={opts.tfr_feature!r}; use a tuple range such as "
                "freq=(4.0, 30.0) to keep frequency points."
            )
        if _is_named_range_spec(opts.time):
            raise ValueError(
                "named time aggregation is not supported for "
                f"tfr_feature={opts.tfr_feature!r}; use a tuple range to keep "
                "time points."
            )
    if opts.log:
        data = safe_log10_power(data, floor=opts.log_floor)

    data, axes_list, coord_map = _apply_axis_spec(
        data,
        axes_list,
        coord_map,
        axis="freq",
        aggregate_axis="band",
        spec=opts.freq,
        range_closed=opts.range_closed,
        agg=opts.agg,
    )
    data, axes_list, coord_map = _apply_axis_spec(
        data,
        axes_list,
        coord_map,
        axis="time",
        aggregate_axis="window",
        spec=opts.time,
        range_closed=opts.range_closed,
        agg=opts.agg,
    )
    data = _standardize_data(data, axes_list, opts.standardize)

    return RsaData(
        data=data,
        axes=tuple(axes_list),
        coords=coord_map,
    )


def _coerce_to_named_parts(
    inst_or_data: RsaData | EpochsSpectrum | EpochsTFR | np.ndarray,
    *,
    axes: Sequence[str] | None,
    coords: Mapping[str, Sequence[Any]] | None,
    picks: Any,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if isinstance(inst_or_data, RsaData):
        if axes is not None or coords is not None:
            raise ValueError("axes/coords are not used when passing RsaData.")
        return (
            _as_real_float(inst_or_data.data),
            list(inst_or_data.axes),
            dict(inst_or_data.coords),
        )

    if isinstance(inst_or_data, np.ndarray):
        if axes is None:
            raise ValueError(
                "axes must be provided when inst_or_data is a NumPy array."
            )
        data = np.asarray(inst_or_data)
        axes_tuple = tuple(axes)
        _validate_axes(data, axes_tuple)
        return data, list(axes_tuple), _normalize_coords(data, axes_tuple, coords)

    return _extract_mne_spectral(inst_or_data, picks=picks)


def _extract_mne_spectral(
    inst: EpochsSpectrum | EpochsTFR,
    *,
    picks: Any,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    try:
        from mne.time_frequency import EpochsSpectrum, EpochsTFR
    except ImportError as exc:
        raise ImportError("MNE spectral objects require mne-python.") from exc

    if picks is None:
        obj = inst
    else:
        obj = inst.copy()
        obj.pick(picks, exclude="bads")

    if not isinstance(obj, (EpochsSpectrum, EpochsTFR)):
        raise TypeError(
            "inst_or_data must be RsaData, an MNE EpochsSpectrum/"
            f"EpochsTFR object, or a NumPy array; got {type(inst).__name__}."
        )

    selection = getattr(obj, "selection", None)
    if selection is not None and len(selection) == len(obj):
        item_values = np.asarray(selection)
    else:
        item_values = np.arange(len(obj))

    coords = {
        "item": item_values,
        "space": np.asarray(obj.ch_names, dtype=object),
    }
    axes = ["item", "space", "freq"]

    if isinstance(obj, EpochsTFR):
        data, times, freqs = obj.get_data(
            exclude=(),
            return_times=True,
            return_freqs=True,
        )
        coords["freq"] = np.asarray(freqs, dtype=float)
        axes.append("time")
        coords["time"] = np.asarray(times, dtype=float)
    else:
        data, freqs = obj.get_data(exclude=(), return_freqs=True)
        coords["freq"] = np.asarray(freqs, dtype=float)
    return np.asarray(data), axes, coords


def _project_tfr_feature(
    data: np.ndarray,
    axes: list[str],
    coords: dict[str, np.ndarray],
    tfr_feature: TfrFeatureMode,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if tfr_feature not in _ALL_TFR_FEATURES:
        raise ValueError(
            "tfr_feature must be 'power', 'amplitude', 'real_imag', "
            "'amp_phase', or 'phase'."
        )

    arr = np.asarray(data)
    if not np.iscomplexobj(arr):
        if tfr_feature == "power":
            return _as_real_float(arr), list(axes), dict(coords)
        raise ValueError(
            f"tfr_feature={tfr_feature!r} requires complex input. Compute TFR "
            "with output='complex' or pass tfr_feature='power' for real-valued data."
        )

    if tfr_feature == "power":
        return np.abs(arr) ** 2, list(axes), dict(coords)
    if tfr_feature == "amplitude":
        return np.abs(arr), list(axes), dict(coords)

    if "component" in axes:
        raise ValueError(
            "Complex-derived tfr_feature modes add a 'component' axis, but the "
            "input axes already contain 'component'. Rename that axis before "
            "using tfr_feature='real_imag', 'amp_phase', or 'phase'."
        )

    amplitude = np.abs(arr)
    if tfr_feature == "real_imag":
        components = ("real", "imag")
        projected = np.stack((arr.real, arr.imag), axis=arr.ndim)
    else:
        phase = np.angle(arr)
        nonzero = amplitude > 0
        sin_phase = np.where(nonzero, np.sin(phase), 0.0)
        cos_phase = np.where(nonzero, np.cos(phase), 0.0)
        if tfr_feature == "amp_phase":
            components = ("amplitude", "sin_phase", "cos_phase")
            projected = np.stack((amplitude, sin_phase, cos_phase), axis=arr.ndim)
        else:
            components = ("sin_phase", "cos_phase")
            projected = np.stack((sin_phase, cos_phase), axis=arr.ndim)

    coord_map = dict(coords)
    coord_map["component"] = np.asarray(components, dtype=object)
    return projected, [*axes, "component"], coord_map


def _is_named_range_spec(spec: AxisSpec) -> bool:
    return spec is not None and not _is_plain_range(spec)


def _is_plain_range(spec: Any) -> bool:
    if isinstance(spec, Mapping):
        return False
    if not isinstance(spec, (tuple, list, np.ndarray)) or len(spec) != 2:
        return False
    return isinstance(spec[0], (int, float, np.integer, np.floating)) and isinstance(
        spec[1], (int, float, np.integer, np.floating)
    )


def _apply_axis_spec(
    data: np.ndarray,
    axes: list[str],
    coords: dict[str, np.ndarray],
    *,
    axis: str,
    aggregate_axis: str,
    spec: AxisSpec,
    range_closed: str | None,
    agg: AggFunc,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if spec is None:
        return data, axes, coords
    if _is_plain_range(spec):
        closed = "both" if range_closed is None else range_closed
        return _select_axis_range(data, axes, coords, axis, spec, closed)
    closed = "left" if range_closed is None else range_closed
    return _replace_range_axis(
        data,
        axes,
        coords,
        source_axis=axis,
        target_axis=aggregate_axis,
        ranges=spec,
        range_closed=closed,
        agg=agg,
    )


def _select_axis_range(
    data: np.ndarray,
    axes: list[str],
    coords: dict[str, np.ndarray],
    axis_name: str,
    limits: Sequence[float],
    range_closed: str,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if axis_name not in axes:
        raise ValueError(f"Cannot select {axis_name!r}: axis is not present.")

    start, stop = float(limits[0]), float(limits[1])
    if stop <= start:
        raise ValueError(f"{axis_name!r} range stop must be greater than start.")

    axis_idx = axes.index(axis_name)
    values = np.asarray(coords[axis_name], dtype=float)
    keep = np.where(_range_mask(values, start, stop, range_closed))[0]
    if keep.size == 0:
        raise ValueError(
            f"{axis_name!r} range {_format_range(start, stop, range_closed)} "
            "contains no coordinate values."
        )

    selected = np.take(data, keep, axis=axis_idx)
    coord_map = dict(coords)
    coord_map[axis_name] = values[keep]
    return selected, list(axes), coord_map


__all__ = [
    "safe_log10_power",
]
