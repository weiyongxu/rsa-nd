"""Layer 3: convert an `RsaResult` into an MNE container (`result.to_mne()`).

`rsa_result_to_mne` reads only the result -- its `axes`, `coords`, and typed
`space` geometry -- and returns an MNE container when the result geometry maps
honestly onto one, otherwise raising `ValueError`. Unlike the earlier
`assemble_rsa_result`, it does not consume `PreparedRsaData`, does not silently
fall back to the `RsaResult`, and does not back-fill aggregate windows / bands /
ROIs onto a native grid -- only exact result axes are supported.

Supported exact result-axis patterns (any leading ``model`` / ``model_time``
axis is rejected -- select a single model first):

- sensor ``("space", "time")`` -> ``EvokedArray``
- sensor ``("space", "freq")`` -> ``SpectrumArray``
- sensor ``("space", "freq", "time")`` -> ``AverageTFRArray``
- native source ``("space", "time")`` -> ``SourceEstimate`` / ``VolSourceEstimate``
- grouped ``("space"/"roi", ...)`` -> ROI/group-named carrier
- time/freq only (no space) -> one-misc-channel carrier
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import (
    _MODEL_AXES,
    RsaResult,
    SensorSpace,
    SourceSpace,
)


def rsa_result_to_mne(result: RsaResult) -> Any:
    """Convert ``result`` to an MNE container, or raise ``ValueError``.

    Backs :meth:`RsaResult.to_mne`. Decides the container from the result's
    ``axes`` / ``coords`` and its typed ``space`` geometry.
    """

    axes = result.axes
    bad = [axis for axis in axes if axis in _MODEL_AXES]
    if bad:
        raise ValueError(
            f"to_mne() does not support a {bad[0]!r} axis; select or average a "
            "single model first."
        )

    role = _resolve_role(result)
    if role == "per_channel":
        return _assemble_per_channel(result)
    if role == "per_vertex":
        return _assemble_per_vertex(result)
    if role == "per_group":
        return _assemble_per_group(result)
    if role == "singleton_misc":
        return _assemble_singleton_misc(result)
    raise ValueError(
        f"to_mne() cannot represent result axes {axes!r} as an MNE container."
    )


def _resolve_role(result: RsaResult) -> str:
    axes = result.axes
    if "space" in axes:
        if result.space is None:
            raise ValueError(
                "result has a 'space' axis but no geometry to convert; "
                "to_mne() needs SensorSpace / SourceSpace."
            )
        return result.space.assembly_role()
    if "roi" in axes:
        return "per_group"
    other = set(axes)
    if other and other.issubset({"time", "freq"}):
        return "singleton_misc"
    return "unsupported"


def _assemble_per_channel(result: RsaResult) -> Any:
    """sensor space x time -> EvokedArray; x freq -> SpectrumArray;
    x freq x time -> AverageTFRArray."""

    space = result.space
    assert isinstance(space, SensorSpace)
    axes = result.axes
    data = np.moveaxis(np.asarray(result.rsa), axes.index("space"), 0)
    other_axes = tuple(axis for axis in axes if axis != "space")
    return _build_container(data, other_axes, result, space.info)


def _assemble_per_group(result: RsaResult) -> Any:
    """grouped sensor / source / ROI -> carrier with group-named channels."""

    axes = result.axes
    if "space" in axes:
        space = result.space
        assert space is not None and space.patches is not None
        centers = [str(name) for name in space.patches]
        space_axis = "space"
    else:
        space_axis = "roi"
        centers = [str(name) for name in result.coords["roi"]]
    info = _carrier_info(centers, sfreq=_sfreq(result))
    data = np.moveaxis(np.asarray(result.rsa), axes.index(space_axis), 0)
    other_axes = tuple(axis for axis in axes if axis != space_axis)
    return _build_container(data, other_axes, result, info)


def _assemble_per_vertex(result: RsaResult) -> Any:
    """native source space x time -> SourceEstimate / VolSourceEstimate."""

    space = result.space
    assert isinstance(space, SourceSpace)
    axes = result.axes
    if "time" not in axes:
        raise ValueError("only space x time native source assembly is supported")

    data = np.moveaxis(
        np.asarray(result.rsa), (axes.index("space"), axes.index("time")), (0, 1)
    )
    if data.ndim != 2:
        raise ValueError("native source assembly expects a 2-D space x time result")

    times = np.asarray(result.coords["time"], dtype=float)
    tmin = float(times[0])
    tstep = float(times[1] - times[0]) if len(times) > 1 else 1.0

    try:
        from mne import SourceEstimate, VolSourceEstimate
    except ImportError as exc:
        raise ImportError("Source assembly requires mne-python.") from exc

    estimate_cls = (
        VolSourceEstimate if space.estimate_kind == "volume" else SourceEstimate
    )
    return estimate_cls(
        data,
        vertices=list(space.vertices),
        tmin=tmin,
        tstep=tstep,
        subject=space.subject,
    )


def _assemble_singleton_misc(result: RsaResult) -> Any:
    """No space axis: build a one-channel carrier over time and/or freq."""

    info = _carrier_info(["rsa"], sfreq=_sfreq(result))
    data = np.asarray(result.rsa)[np.newaxis, ...]
    return _build_container(data, result.axes, result, info)


def _build_container(
    data: np.ndarray,
    other_axes: tuple[str, ...],
    result: RsaResult,
    info: Any,
) -> Any:
    """Build the MNE container for the non-space axes; ``data[0]`` is channels."""

    try:
        from mne import EvokedArray
        from mne.time_frequency import AverageTFRArray, SpectrumArray
    except ImportError as exc:
        raise ImportError("Output assembly requires mne-python.") from exc

    coords = result.coords
    if other_axes == ("time",):
        times = np.asarray(coords["time"], dtype=float)
        tmin = float(times[0]) if len(times) else 0.0
        return EvokedArray(data, info, tmin=tmin)
    if other_axes == ("freq",):
        freqs = np.asarray(coords["freq"], dtype=float)
        return SpectrumArray(data, info, freqs=freqs)
    if other_axes in (("freq", "time"), ("time", "freq")):
        freqs = np.asarray(coords["freq"], dtype=float)
        times = np.asarray(coords["time"], dtype=float)
        if other_axes == ("time", "freq"):
            data = np.moveaxis(data, 1, 2)
        return AverageTFRArray(info=info, data=data, times=times, freqs=freqs)
    if other_axes == ():
        # space-only: one scalar per channel/group; EvokedArray at t=0.
        return EvokedArray(data[..., np.newaxis], info, tmin=0.0)
    raise ValueError(f"Unsupported non-space result axes: {other_axes!r}")


def _carrier_info(channel_names: list[str], *, sfreq: float) -> Any:
    """Info for synthetic RSA-carrier channels.

    Uses a generic *data* channel type rather than ``"misc"``: ``SpectrumArray``
    and ``AverageTFRArray`` validate against good/bad *data* channels and reject
    misc-only info, while ``EvokedArray`` accepts either. The channel is only a
    carrier for RSA scores, so the type is cosmetic.
    """
    try:
        from mne import create_info
    except ImportError as exc:
        raise ImportError("Output assembly requires mne-python.") from exc
    return create_info(ch_names=list(channel_names), sfreq=float(sfreq), ch_types="eeg")


def _sfreq(result: RsaResult) -> float:
    """Carrier sampling rate: sensor Info, else the time grid, else 1.0."""
    info = getattr(result.space, "info", None)
    if info is not None:
        sfreq = info.get("sfreq", None)
        if sfreq:
            return float(sfreq)
    times = result.coords.get("time")
    if times is not None and len(times) > 1:
        diffs = np.diff(np.asarray(times, dtype=float))
        if diffs.size and diffs[0] > 0:
            return 1.0 / float(diffs[0])
    return 1.0


__all__ = ["rsa_result_to_mne"]
