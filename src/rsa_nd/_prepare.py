"""Public Layer 1 entry point: ``prepare_rsa_data``.

Converts MNE input or already-abstract data into a typed ``PreparedRsaData``
that downstream callers (``rsa_nd``, ``rsa_dynamic``) consume as a single
positional.

``prepare_rsa_data`` detects the input kind and extracts the named-axis data,
MNE template, item labels, and spatial geometry in one branch per input family
(:func:`_extract_input`); the shared finalizers in :mod:`._features` absorb the
common feature pipeline. ``picks`` is applied once, by the matching sensor
branch.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import numpy as np

from ._features import (
    _is_named_range_spec,
    aggregate_space_groups,
    finalize_spectral_rsa,
    prepare_time_domain_rsa,
    resolve_element_groups,
)
from ._source import (
    build_source_space,
    check_source_estimates_compatible,
)
from ._types import (
    AggFunc,
    AxisGroups,
    AxisSpec,
    EpochsSpectrum,
    EpochsTFR,
    GroupReduceMode,
    PreparedRsaData,
    RsaData,
    RsaInputKind,
    RsaPrepOptions,
    RsaTemplate,
    SensorSpace,
    SpaceDescription,
    StandardizeMode,
    TfrFeatureMode,
)


# --------------------------------------------------------------------------- #
# Input extraction helpers
# --------------------------------------------------------------------------- #
def _pick(obj: Any, picks: Any) -> Any:
    """Apply a within-object channel pick once (identity when ``picks`` is None)."""
    if picks is None:
        return obj
    return obj.copy().pick(picks, exclude="bads")


def _pick_sequence(objs: Any, picks: Any) -> list[Any]:
    """Pick each element of a per-item sensor sequence (identity when None)."""
    objs = list(objs)
    if picks is None:
        return objs
    return [obj.copy().pick(picks, exclude="bads") for obj in objs]


def _reject_spectral_tfr(tfr: Mapping[str, Any] | None, kind: str) -> None:
    """Raise if ``tfr={...}`` is combined with already-spectral input."""
    if tfr is not None:
        raise ValueError(
            "tfr= cannot be combined with already-spectral input "
            f"({kind}); the input is already in the frequency domain."
        )


def _check_sensor_sequence_compatible(objs: list[Any]) -> None:
    """Raise unless every sequence element shares the first's channel names."""
    ref_names = list(objs[0].ch_names)
    for idx, obj in enumerate(objs[1:], 1):
        if list(obj.ch_names) != ref_names:
            raise ValueError(
                "All sequence elements must share channel names; element "
                f"{idx} differs from element 0."
            )


def _epochs_item_values(obj: Any) -> np.ndarray:
    """Per-epoch item coordinates: the original selection when intact, else
    a plain index range."""
    selection = getattr(obj, "selection", None)
    if selection is not None and len(selection) == len(obj):
        return np.asarray(selection)
    return np.arange(len(obj))


def _events_to_item_labels(inst_or_data: Any) -> np.ndarray | None:
    """Read item labels from MNE events at prepare time."""
    events = getattr(inst_or_data, "events", None)
    if events is None:
        return None
    events_arr = np.asarray(events)
    if events_arr.ndim != 2 or events_arr.shape[1] < 3:
        return None
    return events_arr[:, 2]


def _build_sensor_space(
    obj: Any,
    space_arg: Any,
) -> SensorSpace:
    channel_names = tuple(obj.ch_names)
    info = obj.info

    if space_arg is None:
        return SensorSpace(
            channel_names=channel_names,
            info=info,
            patches=None,
            is_grouped=False,
        )

    if isinstance(space_arg, Mapping):
        name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
        patches: dict[str, np.ndarray] = {}
        for group_name, members in space_arg.items():
            unknown = [m for m in members if m not in name_to_idx]
            if unknown:
                raise ValueError(
                    f"Group {group_name!r} references unknown channels: {unknown}."
                )
            patches[str(group_name)] = np.asarray(
                [name_to_idx[m] for m in members], dtype=int
            )
        return SensorSpace(
            channel_names=channel_names,
            info=info,
            patches=patches,
            is_grouped=True,
        )

    raise TypeError(
        "space= for sensor input must be None or Mapping[str, Sequence[str]]; "
        f"got {type(space_arg).__name__}."
    )


def _raise_unsupported_input(inst_or_data: Any) -> NoReturn:
    """Raise ``TypeError`` with a targeted hint for near-miss source inputs."""
    if isinstance(inst_or_data, (list, tuple)) and inst_or_data:
        try:
            from mne import (
                MixedSourceEstimate,
                VectorSourceEstimate,
                VolVectorSourceEstimate,
            )

            vector_mixed = (
                VectorSourceEstimate,
                VolVectorSourceEstimate,
                MixedSourceEstimate,
            )
        except ImportError:
            vector_mixed = ()
        if vector_mixed and isinstance(inst_or_data[0], vector_mixed):
            raise TypeError(
                "Vector and mixed source estimates are not supported; RSA "
                "expects scalar per-vertex data. Convert first, e.g. "
                "stc.magnitude() for a VectorSourceEstimate."
            )
    raise TypeError(
        "inst_or_data must be RsaData, ndarray, EpochsSpectrum, EpochsTFR, "
        "Epochs, Sequence[Evoked], Sequence[Spectrum], Sequence[AverageTFR], "
        f"or Sequence[SourceEstimate]; got {type(inst_or_data).__name__}."
    )


# --------------------------------------------------------------------------- #
# Input-kind dispatch
# --------------------------------------------------------------------------- #
def _extract_input(
    inst_or_data: Any,
    opts: RsaPrepOptions,
    *,
    picks: Any,
    space_arg: Any,
    src: Any,
) -> tuple[
    RsaInputKind,
    RsaData,
    RsaTemplate,
    np.ndarray | None,
    SpaceDescription | None,
]:
    """Detect the input kind and extract everything Layer 2 needs.

    Returns ``(input_kind, rsa_data, template, item_labels, space)``. One
    branch per input family; the shared finalizers ``prepare_time_domain_rsa``
    and ``finalize_spectral_rsa`` absorb the common feature pipeline. ``tfr=``
    is rejected for inputs already in the frequency domain (and for abstract
    arrays, which carry no MNE timing); time-domain inputs (Epochs / Evokeds /
    source estimates) accept it and compute an on-the-fly Morlet TFR. ``picks``
    is applied exactly once, by the sensor branches only.
    """

    # Abstract array / RsaData: no MNE template, picks is a no-op, no geometry.
    # An ndarray is just an RsaData the caller did not wrap by hand.
    if isinstance(inst_or_data, (RsaData, np.ndarray)):
        if opts.tfr is not None:
            raise ValueError(
                "tfr= is not supported for abstract array / RsaData input "
                "in this refactor: arrays do not carry MNE timing semantics. "
                "Compute TFR with mne.time_frequency.tfr_array_morlet first "
                "and pass the result with tfr_feature= instead."
            )
        if space_arg is not None:
            raise ValueError(
                "space= is not supported for abstract array / RsaData input. "
                "Wrap the array in an MNE container first when spatial "
                "Searchlight geometry should be interpreted as sensor or source "
                "space."
            )
        rsa_data = finalize_spectral_rsa(
            inst_or_data, axes=opts.axes, coords=opts.coords, picks=None, opts=opts
        )
        return "rsa_data", rsa_data, None, None, None

    try:
        from mne import BaseEpochs, Evoked, SourceEstimate, VolSourceEstimate
        from mne.time_frequency import (
            AverageTFR,
            EpochsSpectrum,
            EpochsTFR,
            Spectrum,
        )
    except ImportError:
        _raise_unsupported_input(inst_or_data)

    # Within-object sensor input: items are the first axis of one MNE object.
    if isinstance(inst_or_data, BaseEpochs):
        working = _pick(inst_or_data, picks)
        rsa_data = prepare_time_domain_rsa(
            np.asarray(working.get_data()),
            item_coords=_epochs_item_values(working),
            space_coords=np.asarray(working.ch_names, dtype=object),
            times=np.asarray(working.times, dtype=float),
            sfreq=float(working.info["sfreq"]),
            opts=opts,
        )
        return (
            "epochs",
            rsa_data,
            working,
            _events_to_item_labels(working),
            _build_sensor_space(working, space_arg),
        )

    if isinstance(inst_or_data, (EpochsSpectrum, EpochsTFR)):
        kind: RsaInputKind = (
            "epochs_spectrum"
            if isinstance(inst_or_data, EpochsSpectrum)
            else "epochs_tfr"
        )
        _reject_spectral_tfr(opts.tfr, kind)
        working = _pick(inst_or_data, picks)
        rsa_data = finalize_spectral_rsa(
            working, axes=opts.axes, coords=opts.coords, picks=None, opts=opts
        )
        return (
            kind,
            rsa_data,
            working,
            _events_to_item_labels(working),
            _build_sensor_space(working, space_arg),
        )

    # Sequence inputs: each element of the sequence is one item.
    if isinstance(inst_or_data, (list, tuple)) and len(inst_or_data) > 0:
        first = inst_or_data[0]

        if isinstance(first, Evoked):
            working = _pick_sequence(inst_or_data, picks)
            _check_sensor_sequence_compatible(working)
            ref = working[0]
            rsa_data = prepare_time_domain_rsa(
                np.stack([np.asarray(ev.data) for ev in working], axis=0),
                item_coords=np.arange(len(working)),
                space_coords=np.asarray(ref.ch_names, dtype=object),
                times=np.asarray(ref.times, dtype=float),
                sfreq=float(ref.info["sfreq"]),
                opts=opts,
            )
            return "evokeds", rsa_data, ref, None, _build_sensor_space(ref, space_arg)

        if isinstance(first, Spectrum):
            _reject_spectral_tfr(opts.tfr, "spectrums")
            working = _pick_sequence(inst_or_data, picks)
            _check_sensor_sequence_compatible(working)
            ref = working[0]
            rsa_data = finalize_spectral_rsa(
                np.stack([np.asarray(sp.get_data()) for sp in working], axis=0),
                axes=("item", "space", "freq"),
                coords={
                    "item": np.arange(len(working)),
                    "space": np.asarray(ref.ch_names, dtype=object),
                    "freq": np.asarray(ref.freqs, dtype=float),
                },
                picks=None,
                opts=opts,
            )
            return "spectrums", rsa_data, ref, None, _build_sensor_space(ref, space_arg)

        if isinstance(first, AverageTFR):
            _reject_spectral_tfr(opts.tfr, "average_tfrs")
            working = _pick_sequence(inst_or_data, picks)
            _check_sensor_sequence_compatible(working)
            ref = working[0]
            rsa_data = finalize_spectral_rsa(
                np.stack([np.asarray(tfr.data) for tfr in working], axis=0),
                axes=("item", "space", "freq", "time"),
                coords={
                    "item": np.arange(len(working)),
                    "space": np.asarray(ref.ch_names, dtype=object),
                    "freq": np.asarray(ref.freqs, dtype=float),
                    "time": np.asarray(ref.times, dtype=float),
                },
                picks=None,
                opts=opts,
            )
            return (
                "average_tfrs",
                rsa_data,
                ref,
                None,
                _build_sensor_space(ref, space_arg),
            )

        if isinstance(first, (SourceEstimate, VolSourceEstimate)):
            stcs = list(inst_or_data)
            check_source_estimates_compatible(stcs)
            ref = stcs[0]
            data = np.stack([np.asarray(stc.data) for stc in stcs], axis=0)
            rsa_data = prepare_time_domain_rsa(
                data,
                item_coords=np.arange(len(stcs)),
                space_coords=np.arange(data.shape[1]),
                times=np.asarray(ref.times, dtype=float),
                sfreq=1.0 / float(ref.tstep),
                opts=opts,
            )
            space = build_source_space(stcs, space_arg=space_arg, src=src)
            return "source_estimates", rsa_data, ref, None, space

    _raise_unsupported_input(inst_or_data)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def prepare_rsa_data(
    inst_or_data: RsaData | EpochsSpectrum | EpochsTFR | np.ndarray,
    *,
    axes: Sequence[str] | None = None,
    coords: Mapping[str, Sequence[Any]] | None = None,
    time: AxisSpec = None,
    time_mode: GroupReduceMode = "aggregate",
    freq: AxisSpec = None,
    freq_mode: GroupReduceMode = "aggregate",
    space: Any | None = None,
    space_mode: GroupReduceMode = "elements",
    picks: Any = "data",
    tfr: Mapping[str, Any] | None = None,
    tfr_feature: TfrFeatureMode = "power",
    log: bool = False,
    log_floor: float | None = None,
    standardize: StandardizeMode = None,
    range_closed: str | None = None,
    agg: AggFunc = "mean",
    src: Any | None = None,
) -> PreparedRsaData:
    """Convert an MNE input or array into a typed `PreparedRsaData`.

    Parameters
    ----------
    inst_or_data
        One of: ``RsaData``, ``np.ndarray``, ``mne.Epochs``,
        ``EpochsSpectrum``, ``EpochsTFR``, ``Sequence[mne.Evoked]``,
        ``Sequence[mne.time_frequency.Spectrum]``,
        ``Sequence[mne.time_frequency.AverageTFR]``, or
        ``Sequence[SourceEstimate | VolSourceEstimate]``. For the sequence
        inputs each element is one item.
    time_mode, freq_mode, space_mode
        How a grouped axis is reduced -- the unified grouping vocabulary shared
        by all three axes. ``"aggregate"`` averages each named group's members
        (with ``agg``) into one value, collapsing the native axis into a coarse
        categorical axis at prep time: ``time`` -> ``"window"``, ``freq`` ->
        ``"band"``, ``space`` -> ``"roi"`` (and ``space`` then drops its
        per-element geometry). ``"elements"`` instead keeps every member as a
        joint feature so each group is one searchlight patch with a multivariate
        RDM (one RSA per group when the axis is searched). Defaults are
        history-driven: ``time_mode``/``freq_mode`` default ``"aggregate"`` and
        ``space_mode`` defaults ``"elements"``. ``time_mode``/``freq_mode`` only
        apply to a named-range mapping (e.g. ``time={"early": (0, 0.2)}``);
        ``space_mode`` only applies to a grouped ``space=``.
    src
        Optional ``mne.SourceSpaces``. Carried on the source
        `SpaceDescription` so a spatial radius search can measure geodesic
        distance in meters; only needed when ``inst_or_data`` is a sequence of
        source estimates and a non-grouped spatial searchlight is requested.
    """

    # Elements-mode specs are kept out of the aggregate feature pipeline (the
    # native axis is preserved) and resolved into element_groups once the data,
    # and thus the axis coordinates, exist.
    time_pipeline, time_is_elements = _split_group_spec(time, time_mode, "time")
    freq_pipeline, freq_is_elements = _split_group_spec(freq, freq_mode, "freq")

    opts = RsaPrepOptions(
        axes=axes,
        coords=coords,
        time=time_pipeline,
        freq=freq_pipeline,
        tfr=tfr,
        tfr_feature=tfr_feature,
        log=log,
        log_floor=log_floor,
        standardize=standardize,
        range_closed=range_closed,
        agg=agg,
    )

    # Detect the input kind and extract the named-axis data, MNE template,
    # item labels, and spatial geometry. ``picks`` is applied here
    # exactly once, inside the matching sensor branch.
    input_kind, rsa_data, template, item_labels, space_description = _extract_input(
        inst_or_data, opts, picks=picks, space_arg=space, src=src
    )

    if space_mode not in ("elements", "aggregate"):
        raise ValueError("space_mode must be 'elements' or 'aggregate'.")

    axis_groups: dict[str, AxisGroups] = {}

    if space_mode == "aggregate":
        if space is None:
            raise ValueError("space_mode='aggregate' requires space= groups.")
        if space_description is None or space_description.patches is None:
            raise ValueError(
                "space_mode='aggregate' requires grouped space= input "
                "(a mapping of group name to members / Labels)."
            )
        # Collapse each group's members to one signal: the space axis becomes a
        # categorical 'roi' axis and the per-element geometry no longer applies.
        # The membership is kept so the result can be back-filled across the
        # original channels / vertices at assembly.
        space_members = tuple(
            (str(name), np.asarray(idx, dtype=int))
            for name, idx in space_description.patches.items()
        )
        rsa_data = aggregate_space_groups(rsa_data, space_description.patches, agg=agg)
        space_description = None
        axis_groups["roi"] = AxisGroups("space", "aggregate", space_members)

    # Elements-mode numeric axes keep their native grid; record the member
    # indices for the searchlight (one RSA per group).
    if time_is_elements:
        axis_groups["time"] = AxisGroups(
            "time",
            "elements",
            _resolve_axis_groups(rsa_data, time, "time", range_closed),
        )
    if freq_is_elements:
        axis_groups["freq"] = AxisGroups(
            "freq",
            "elements",
            _resolve_axis_groups(rsa_data, freq, "freq", range_closed),
        )

    # Aggregate-mode numeric axes were collapsed by the feature pipeline; record
    # the membership over the original grid so the result can be back-filled to
    # the input MNE format at assembly.
    _record_aggregate_groups(
        axis_groups,
        "time",
        "window",
        time,
        time_mode,
        template,
        tfr,
        range_closed,
        rsa_data,
    )
    _record_aggregate_groups(
        axis_groups,
        "freq",
        "band",
        freq,
        freq_mode,
        template,
        tfr,
        range_closed,
        rsa_data,
    )

    time_axis, time_is_grouped = _detect_axis(rsa_data.axes, "time", "window")
    freq_axis, freq_is_grouped = _detect_axis(rsa_data.axes, "freq", "band")

    return PreparedRsaData(
        rsa_data=rsa_data,
        input_kind=input_kind,
        template=template,
        item_labels=item_labels,
        space=space_description,
        time_axis=time_axis,
        time_is_grouped=time_is_grouped,
        freq_axis=freq_axis,
        freq_is_grouped=freq_is_grouped,
        axis_groups=axis_groups or None,
    )


def _split_group_spec(
    spec: AxisSpec, mode: GroupReduceMode, axis: str
) -> tuple[AxisSpec, bool]:
    """Return ``(spec_for_pipeline, is_elements)``.

    Aggregate specs flow into the feature pipeline unchanged. Elements specs are
    diverted (``None`` to the pipeline so the native axis is preserved) and
    flagged for later resolution into ``axis_groups``.
    """

    if mode not in ("aggregate", "elements"):
        raise ValueError(f"{axis}_mode must be 'aggregate' or 'elements'.")
    if mode == "aggregate":
        return spec, False
    if not _is_named_range_spec(spec):
        raise ValueError(
            f"{axis}_mode='elements' requires named ranges, e.g. "
            f"{axis}={{'early': (0, 0.2)}}."
        )
    return None, True


def _resolve_axis_groups(
    rsa_data: RsaData, spec: AxisSpec, axis: str, range_closed: str | None
) -> tuple[tuple[str, np.ndarray], ...]:
    if axis not in rsa_data.axes:
        raise ValueError(
            f"{axis}_mode='elements' was set but the prepared data has no "
            f"{axis!r} axis to group."
        )
    return resolve_element_groups(
        rsa_data.coords[axis], spec, range_closed=range_closed
    )


def _original_axis_grid(
    axis: str, template: RsaTemplate, tfr: Mapping[str, Any] | None
) -> np.ndarray | None:
    """Recover the original ``time`` / ``freq`` grid for aggregate back-fill.

    Returns ``None`` when the grid is not recoverable (e.g. an abstract input
    with no template, or a Morlet freq axis with no ``tfr['freqs']``), in which
    case the aggregate result simply stays coarse and assembly falls back.
    """

    if axis == "time":
        times = getattr(template, "times", None)
        return None if times is None else np.asarray(times, dtype=float)
    freqs = getattr(template, "freqs", None)
    if freqs is None and tfr is not None:
        freqs = tfr.get("freqs")
    return None if freqs is None else np.asarray(freqs, dtype=float)


def _record_aggregate_groups(
    axis_groups: dict[str, AxisGroups],
    original_axis: str,
    coarse_axis: str,
    spec: AxisSpec,
    mode: GroupReduceMode,
    template: RsaTemplate,
    tfr: Mapping[str, Any] | None,
    range_closed: str | None,
    rsa_data: RsaData,
) -> None:
    if mode != "aggregate" or not _is_named_range_spec(spec):
        return
    if coarse_axis not in rsa_data.axes:
        return
    grid = _original_axis_grid(original_axis, template, tfr)
    if grid is None:
        return
    members = resolve_element_groups(grid, spec, range_closed=range_closed)
    axis_groups[coarse_axis] = AxisGroups(original_axis, "aggregate", members, grid)


def _detect_axis(
    axes: Sequence[str], native: str, grouped: str
) -> tuple[str | None, bool]:
    if grouped in axes:
        return grouped, True
    if native in axes:
        return native, False
    return None, False


_PREP_KWARG_NAMES: frozenset[str] = frozenset(
    inspect.signature(prepare_rsa_data).parameters
) - {"inst_or_data"}


def prepare_rsa_data_kwarg_names() -> frozenset[str]:
    """Return the keyword names ``prepare_rsa_data`` accepts.

    Derived once from the ``prepare_rsa_data`` signature (minus the
    ``inst_or_data`` positional). Single source of truth for the prep / RSA
    kwarg split shared by ``compute_rsa`` and the dynamic frontends, so none of
    them hand-maintain a parallel grammar.
    """
    return _PREP_KWARG_NAMES


__all__ = [
    "prepare_rsa_data",
    "prepare_rsa_data_kwarg_names",
    "PreparedRsaData",
    "SpaceDescription",
    "RsaInputKind",
    "RsaTemplate",
]
