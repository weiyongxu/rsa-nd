"""Shared types and result containers for rsa_nd."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from mne import BaseEpochs, Label, SourceEstimate, VolSourceEstimate
    from mne.time_frequency import EpochsSpectrum, EpochsTFR
else:
    BaseEpochs = Any
    Label = Any
    SourceEstimate = Any
    VolSourceEstimate = Any
    EpochsSpectrum = Any
    EpochsTFR = Any


AggFunc = str | Callable[..., np.ndarray]
RangeSpec = (
    Mapping[str, tuple[float, float]] | Sequence[tuple[str, tuple[float, float]]]
)
AxisSpec = tuple[float, float] | RangeSpec | None
StandardizeMode = Literal["features"] | None
GroupReduceMode = Literal["aggregate", "elements"]
TfrFeatureMode = Literal["power", "amplitude", "real_imag", "amp_phase", "phase"]
DynamicRsaMode = Literal["generalization", "aligned", "lagged"]
RsaInputKind = Literal[
    "epochs",
    "epochs_spectrum",
    "epochs_tfr",
    "evokeds",
    "spectrums",
    "average_tfrs",
    "source_estimates",
    "rsa_data",
]
RsaTemplate = (
    BaseEpochs | EpochsSpectrum | EpochsTFR | SourceEstimate | VolSourceEstimate | None
)
_COMPLEX_DERIVED_FEATURES = {"real_imag", "amp_phase", "phase"}
_ALL_TFR_FEATURES = {"power", "amplitude", *_COMPLEX_DERIVED_FEATURES}

_AGGREGATORS: Mapping[str, Callable[..., np.ndarray]] = {
    "mean": np.mean,
    "median": np.median,
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
}


def _as_real_float(data: Any) -> np.ndarray:
    """Coerce array-like to real float, rejecting complex spectral data."""
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        raise ValueError(
            "Complex spectral data is not supported. Pass power, magnitude, "
            "or another real-valued representation."
        )
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    return arr.astype(float, copy=False)


def _validate_axes(data: np.ndarray, axes: Sequence[str]) -> None:
    """Check that ``axes`` has one unique name per data dimension."""
    if data.ndim != len(axes):
        raise ValueError(
            f"data has {data.ndim} dimensions, but axes has {len(axes)} entries."
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes must be unique, got {list(axes)!r}.")


def _normalize_coords(
    data: np.ndarray,
    axes: Sequence[str],
    coords: Mapping[str, Sequence[Any]] | None,
) -> dict[str, np.ndarray]:
    """Fill missing coordinates with integer ranges and validate lengths."""
    coord_map: dict[str, np.ndarray] = {}
    coords = {} if coords is None else coords
    unknown = sorted(set(coords) - set(axes))
    if unknown:
        raise ValueError(
            f"Unknown coordinate keys: {unknown}. Coordinate keys must match axes."
        )
    for idx, axis_name in enumerate(axes):
        values = coords.get(axis_name, np.arange(data.shape[idx]))
        values_arr = np.asarray(values)
        if len(values_arr) != data.shape[idx]:
            raise ValueError(
                f"Coordinate length for axis {axis_name!r} is {len(values_arr)}, "
                f"but data axis has length {data.shape[idx]}."
            )
        coord_map[axis_name] = values_arr
    return coord_map


@dataclass(frozen=True)
class RsaPrepOptions:
    """Array / feature preparation options threaded through the input Adapters
    and the spectral feature pipeline.

    Built by ``prepare_rsa_data`` from its keyword arguments, forwarded to the
    input-kind extraction (``_extract_input``) as a single value, and read
    directly by the feature-pipeline cores (``_prepare_spectral_core`` /
    ``_prepare_morlet_rsa_core``), which consume this object whole rather than
    receiving the fields as separate keyword arguments. ``space``, ``src``, and
    ``picks`` are not included: ``space`` / ``src`` drive geometry construction,
    and ``picks`` is consumed once by the Adapter's ``working_object`` phase,
    not the array / feature preparation captured here.

    Parameters
    ----------
    axes, coords
        Required only for a raw NumPy array input (one axis name per dimension,
        plus optional coordinate values); unused for MNE / ``RsaData`` input.
    time
        Time handling, in one of three forms:

        - ``None`` keeps the existing ``time`` axis.
        - ``(start, stop)`` keeps native time points in the range controlled by
          ``range_closed`` (default closed on both ends).
        - ``{"name": (start, stop)}`` averages ranges and replaces ``time``
          with a named ``window`` axis.
    freq
        Frequency handling with the same three forms as ``time``; named ranges
        replace ``freq`` with a named ``band`` axis.
    tfr
        Morlet ``tfr_array_morlet`` config (``{"freqs": ..., "n_cycles": ...,
        ...}``) used to compute a TFR from time-domain data before feature
        projection. ``None`` means the input is already spectral or stays
        time-domain.
    tfr_feature
        Real-valued representation to use when the data contains complex TFR
        coefficients. ``"power"`` keeps real input unchanged and maps complex
        input to ``abs(z) ** 2``; ``"amplitude"`` maps complex input to
        ``abs(z)``; ``"real_imag"``, ``"amp_phase"``, and ``"phase"`` add a
        ``component`` axis with real-valued components.
    log
        Apply a safe ``log10`` transform before range averaging. With named
        ranges this yields mean log-power, not the log of mean raw power.
    log_floor
        Positive floor passed to ``safe_log10_power``. If ``None``, the floor
        is ``median(positive_values) * 1e-6``.
    standardize
        ``None`` leaves feature scales unchanged. ``"features"`` z-scores each
        non-item feature across the item axis after log / range preprocessing;
        NaNs are excluded from the per-feature mean / std, zero-variance
        features become zero, and standardization runs across all items before
        any ``mne_rsa`` cross-validation folds are created.
    range_closed
        Range endpoint rule: ``"both"``, ``"left"``, ``"right"``, or
        ``"neither"``. If ``None`` (default), tuple selections use ``"both"``
        and named-range aggregation uses ``"left"`` so adjacent bands / windows
        tile without sharing a boundary bin.
    agg
        Aggregation for named frequency bands, named time windows, and later
        axis averaging.
    """

    axes: Sequence[str] | None = None
    coords: Mapping[str, Sequence[Any]] | None = None
    time: AxisSpec = None
    freq: AxisSpec = None
    tfr: Mapping[str, Any] | None = None
    tfr_feature: TfrFeatureMode = "power"
    log: bool = False
    log_floor: float | None = None
    standardize: StandardizeMode = None
    range_closed: str | None = None
    agg: AggFunc = "mean"


@dataclass(frozen=True)
class RsaData:
    """Named-axis real-valued data used by the RSA helpers.

    Parameters
    ----------
    data
        Numeric data. The first axis does not have to be ``item`` at
        construction time, but RSA expects an ``item`` axis to be present.
    axes
        One name per data dimension, e.g. ``("item", "space", "freq", "time")``.
    coords
        Coordinate values for each axis. Missing coordinates default to
        integer indices.
    """

    data: np.ndarray
    axes: tuple[str, ...]
    coords: dict[str, np.ndarray]

    def __post_init__(self):
        data = _as_real_float(self.data)
        axes = tuple(self.axes)
        _validate_axes(data, axes)
        coords = _normalize_coords(data, axes, self.coords)

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "coords", coords)

    def axis_index(self, axis: str) -> int:
        """Return the integer index for an axis name."""

        if axis not in self.axes:
            raise ValueError(f"Axis {axis!r} is not present.")
        return self.axes.index(axis)


class SpaceDescription:
    """Abstract base for the typed spatial seam carried on `PreparedRsaData`.

    Concrete variants: `SensorSpace`, `SourceSpace`.
    """

    is_grouped: bool

    def searchlight_patches(
        self, radius: float | None = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        raise NotImplementedError(
            "SpaceDescription is abstract; use SensorSpace or SourceSpace."
        )

    def assembly_role(self) -> Literal["per_channel", "per_vertex", "per_group"]:
        raise NotImplementedError(
            "SpaceDescription is abstract; use SensorSpace or SourceSpace."
        )


@dataclass(frozen=True)
class SensorSpace(SpaceDescription):
    """Sensor-space variant of `SpaceDescription`.

    Stores post-pick channel names plus a picked `mne.Info`. Positions are
    read from `info["chs"][i]["loc"][:3]` at runtime when a non-zero radius
    search is requested. A non-zero radius is a Euclidean distance in meters;
    selection is delegated to mne-rsa (channels with ``dist < radius``), so the
    boundary matches both the source searchlight and mne-rsa's own sensor
    searchlight.
    """

    channel_names: tuple[str, ...]
    info: Any
    patches: dict[str, np.ndarray] | None = None
    is_grouped: bool = False

    def searchlight_patches(
        self, radius: float | None = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if self.patches is not None:
            if radius is not None and radius != 0:
                raise ValueError(
                    "Non-default radii['space'] is invalid for a grouped "
                    "SensorSpace; explicit groups already define the patches."
                )
            patches_list = [np.asarray(idx, dtype=int) for idx in self.patches.values()]
            centers = np.asarray(list(self.patches.keys()), dtype=object)
            return patches_list, centers

        n_channels = len(self.channel_names)
        if radius is None:
            return [np.arange(n_channels, dtype=int)], np.asarray(["all"], dtype=object)
        if radius == 0:
            patches_list = [np.asarray([idx], dtype=int) for idx in range(n_channels)]
            centers = np.asarray(self.channel_names, dtype=object)
            return patches_list, centers

        positions = np.vstack(
            [self.info["chs"][idx]["loc"][:3] for idx in range(n_channels)]
        )
        if not np.isfinite(positions).all():
            raise ValueError("Channel locations contain non-finite values.")
        # Delegate radius selection to mne-rsa so the boundary (strict
        # ``dist < radius``) matches the source searchlight and mne-rsa's own
        # sensor searchlight.
        from mne_rsa.searchlight import _get_in_radius
        from scipy.spatial import distance

        dist = distance.squareform(distance.pdist(positions))
        patches_list = [
            np.asarray(_get_in_radius(dist, idx, float(radius)), dtype=int)
            for idx in range(n_channels)
        ]
        centers = np.asarray(self.channel_names, dtype=object)
        return patches_list, centers

    def assembly_role(self) -> Literal["per_channel", "per_vertex", "per_group"]:
        return "per_group" if self.is_grouped else "per_channel"


@dataclass(frozen=True)
class SourceSpace(SpaceDescription):
    """Source-space variant of `SpaceDescription`.

    Stores per-hemi vertices and the `mne.SourceSpaces` geometry (when
    available) so a spatial radius search can measure true geodesic distance
    in meters, delegating to mne-rsa's distance machinery. `patches` carries
    grouped vertex selections (Labels, ROIs, named source patches); when
    `None`, search positions are individual vertices.
    """

    vertices: tuple[np.ndarray, ...]
    src: Any | None = None
    patches: dict[str, np.ndarray] | None = None
    is_grouped: bool = False
    estimate_kind: Literal["surface", "volume"] = "surface"
    subject: str | None = None
    subjects_dir: str | None = None

    def searchlight_patches(
        self, radius: float | None = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if self.patches is not None:
            if radius is not None and radius != 0:
                raise ValueError(
                    "Non-default radii['space'] is invalid for a grouped "
                    "SourceSpace; explicit groups already define the patches."
                )
            patches_list = [np.asarray(idx, dtype=int) for idx in self.patches.values()]
            centers = np.asarray(list(self.patches.keys()), dtype=object)
            return patches_list, centers

        n_total = sum(int(len(v)) for v in self.vertices)
        if radius is None:
            return [np.arange(n_total, dtype=int)], np.asarray(["all"], dtype=object)
        if radius == 0:
            patches_list = [np.asarray([idx], dtype=int) for idx in range(n_total)]
            centers = np.arange(n_total)
            return patches_list, centers

        # Geodesic ball in meters. Delegate distance + radius selection to
        # mne-rsa so the patches match ``mne_rsa.rsa_stcs(spatial_radius=...)``
        # exactly (surface: geodesic along the mesh; volume: 3-D distance).
        if self.src is None:
            raise ValueError(
                "Source spatial radius search requires the source-space "
                "geometry, but the SourceSpace was built without one. Pass "
                "src= to prepare_rsa_data."
            )
        radius_m = float(radius)
        if radius_m < 0:
            raise ValueError("Source spatial radius (meters) must be non-negative.")
        try:
            from mne_rsa.searchlight import _get_in_radius
            from mne_rsa.source_level import _get_distance_matrix
        except ImportError as exc:
            raise ImportError(
                "Source geodesic searchlight requires mne-rsa to be installed."
            ) from exc

        dist = _get_distance_matrix(self.src, dist_lim=radius_m)
        if dist.shape[0] != n_total:
            raise ValueError(
                "Source distance matrix does not match the vertex count "
                f"({dist.shape[0]} vs {n_total}); the src and source estimates "
                "must use the same in-use vertices."
            )
        patches_list = [
            np.asarray(_get_in_radius(dist, idx, radius_m), dtype=int)
            for idx in range(n_total)
        ]
        centers = np.arange(n_total)
        return patches_list, centers

    def assembly_role(self) -> Literal["per_channel", "per_vertex", "per_group"]:
        return "per_group" if self.is_grouped else "per_vertex"


@dataclass(frozen=True)
class AxisGroups:
    """Named grouping carried on one prepared axis.

    ``original_axis`` is the pre-grouping axis (``"time"`` / ``"freq"`` /
    ``"space"``). ``mode`` is the reduction. ``groups`` is the ordered
    ``(name, member_indices)``, where the indices address the original axis.
    ``original_coords`` carries the original numeric coordinates so aggregate
    results can be back-filled onto the native grid at assembly; it is ``None``
    for ``"space"`` (whose geometry is recovered from the template) and for
    ``"elements"`` entries (no back-fill needed).
    """

    original_axis: str
    mode: GroupReduceMode
    groups: tuple[tuple[str, np.ndarray], ...]
    original_coords: np.ndarray | None = None


def _validate_axis_group_bounds(
    axis_groups: Mapping[str, AxisGroups],
    rsa_data: RsaData,
) -> None:
    """Check that every group's member indices address a valid axis position.

    The normal path derives ``axis_groups`` from ``rsa_data`` and is correct by
    construction; this guard only bites when a caller hand-builds a
    ``PreparedRsaData`` with explicit groups, so a stale or partial mapping
    cannot silently steer the searchlight. Elements-mode indices address the
    native axis carried under the same key; aggregate-mode indices address the
    original (pre-collapse) grid recorded in ``original_coords``. Space
    aggregates recover their grid from the template at assembly, so there is no
    length to check here and they are skipped.
    """

    for axis, ag in axis_groups.items():
        if ag.mode == "elements":
            n_elements = rsa_data.data.shape[rsa_data.axis_index(axis)]
        elif ag.original_coords is not None:
            n_elements = len(ag.original_coords)
        else:
            continue
        for name, indices in ag.groups:
            members = np.atleast_1d(np.asarray(indices, dtype=int))
            if members.size and (members.min() < 0 or members.max() >= n_elements):
                raise ValueError(
                    f"axis_groups[{axis!r}] group {name!r} has member "
                    f"indices outside the axis length ({n_elements})."
                )


@dataclass(frozen=True)
class PreparedRsaData:
    """Typed Layer 1 to Layer 2 seam.

    Carries the named-axis ``RsaData`` plus the metadata downstream callers
    actually use: input provenance, an MNE template for output assembly,
    Item labels, optional typed sensor/source geometry, and explicit time/freq
    axis metadata. ``space`` is about MNE geometry, not just an axis named
    ``"space"``: abstract array / `RsaData` inputs may carry a generic
    coordinate ``"space"`` axis while leaving ``space=None``.

    Cross-field invariants enforced by ``__post_init__``:

    - ``time_is_grouped`` implies ``time_axis == "window"``
    - ``freq_is_grouped`` implies ``freq_axis == "band"``
    - ``space is not None`` implies a ``"space"`` axis is present
    - MNE-backed inputs with a ``"space"`` axis carry a `SpaceDescription`
    - ``item_labels`` length matches the item axis when present
    - every ``axis_groups`` key is an axis of ``rsa_data``

    ``axis_groups`` records named grouping, keyed by the axis name as it appears
    in ``rsa_data`` (the result axis). ``"elements"`` entries (numeric ``time`` /
    ``freq``) keep the native axis; the searchlight turns each group into one
    multi-index patch (one RSA per group). ``"aggregate"`` entries are the
    coarse axes (``"window"`` / ``"band"`` / ``"roi"``) produced by collapsing
    members; they carry the membership so the result can be back-filled onto the
    original grid at assembly. Spatial elements-mode grouping instead rides
    ``space`` / `SpaceDescription.patches` (geometry + ``per_group`` assembly).
    """

    rsa_data: RsaData
    input_kind: RsaInputKind
    template: RsaTemplate = None
    item_labels: np.ndarray | None = None
    space: SpaceDescription | None = None
    time_axis: str | None = None
    time_is_grouped: bool = False
    freq_axis: str | None = None
    freq_is_grouped: bool = False
    axis_groups: dict[str, AxisGroups] | None = None

    def __post_init__(self):
        if self.time_is_grouped and self.time_axis != "window":
            raise ValueError(
                "time_is_grouped=True requires time_axis='window', "
                f"got {self.time_axis!r}."
            )
        if self.freq_is_grouped and self.freq_axis != "band":
            raise ValueError(
                "freq_is_grouped=True requires freq_axis='band', "
                f"got {self.freq_axis!r}."
            )
        if self.space is not None and "space" not in self.rsa_data.axes:
            raise ValueError(
                "PreparedRsaData.space is set but rsa_data has no 'space' axis."
            )
        if (
            self.input_kind != "rsa_data"
            and "space" in self.rsa_data.axes
            and self.space is None
        ):
            raise ValueError(
                "MNE-backed PreparedRsaData with a 'space' axis requires "
                "PreparedRsaData.space to carry SensorSpace or SourceSpace "
                "geometry."
            )
        if self.axis_groups is not None:
            unknown = sorted(set(self.axis_groups) - set(self.rsa_data.axes))
            if unknown:
                raise ValueError(
                    f"axis_groups keys must be axes of rsa_data; got {unknown}."
                )
            _validate_axis_group_bounds(self.axis_groups, self.rsa_data)
        if self.item_labels is not None:
            item_idx = self.rsa_data.axes.index("item")
            if len(self.item_labels) != self.rsa_data.data.shape[item_idx]:
                raise ValueError(
                    "item_labels length does not match the item axis "
                    f"({len(self.item_labels)} vs "
                    f"{self.rsa_data.data.shape[item_idx]})."
                )


# Leading axes that name the model dimension rather than a searchlight axis.
_MODEL_AXES = frozenset({"model", "model_time"})


def _validate_result_axes(
    rsa: np.ndarray,
    axes: tuple[str, ...],
    coords: Mapping[str, Sequence[Any]] | None,
) -> dict[str, np.ndarray]:
    """Validate that ``axes`` / ``coords`` are the full contract for ``rsa``.

    One unique axis name per ``rsa`` dimension; ``coords`` keys exactly match
    ``axes``; each coordinate length matches its axis. Returns the normalized
    coordinate dict.
    """

    axes = tuple(axes)
    if rsa.ndim != len(axes):
        raise ValueError(
            f"rsa has {rsa.ndim} dimensions, but axes has {len(axes)} entries."
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes must be unique, got {list(axes)!r}.")
    coords = {} if coords is None else dict(coords)
    if set(coords) != set(axes):
        raise ValueError(
            f"coords keys {sorted(coords)} must exactly match axes {list(axes)}."
        )
    out: dict[str, np.ndarray] = {}
    for idx, axis in enumerate(axes):
        values = np.asarray(coords[axis])
        if len(values) != rsa.shape[idx]:
            raise ValueError(
                f"Coordinate length for axis {axis!r} is {len(values)}, but the "
                f"rsa axis has length {rsa.shape[idx]}."
            )
        out[axis] = values
    return out


@dataclass(frozen=True)
class DynamicInfo:
    """Typed secondary outputs and metadata for dynamic RSA.

    Carried on ``RsaResult.dynamic``; ``None`` for static RSA. ``full_rsa`` is
    the full model-time x search-grid map, self-described by ``full_rsa_axes`` /
    ``full_rsa_coords`` so it survives save/load; for generalization mode it
    equals the primary ``rsa``.
    """

    mode: str
    dynamic_axis: str
    full_rsa: np.ndarray
    full_rsa_axes: tuple[str, ...]
    full_rsa_coords: dict[str, np.ndarray]
    model_indices: np.ndarray | None = None
    lags: np.ndarray | None = None
    lag_tolerance: float | None = None
    model_times: np.ndarray | None = None

    def __post_init__(self):
        full_rsa = np.asarray(self.full_rsa)
        axes = tuple(self.full_rsa_axes)
        coords = _validate_result_axes(full_rsa, axes, self.full_rsa_coords)
        object.__setattr__(self, "full_rsa", full_rsa)
        object.__setattr__(self, "full_rsa_axes", axes)
        object.__setattr__(self, "full_rsa_coords", coords)


@dataclass(frozen=True)
class RsaResult:
    """Canonical RSA output.

    ``axes`` / ``coords`` are the complete dimension contract for ``rsa``: one
    ``axes`` entry per ``rsa`` dimension, with a matching coordinate in
    ``coords``. A leading ``"model"`` (static multi-model) or ``"model_time"``
    (dynamic) axis appears here when present. ``info`` is audit-only metadata
    (run / model / prepared summaries); ``space`` carries the typed geometry
    ``to_mne`` needs; ``dynamic`` carries dynamic-RSA secondary outputs and is
    ``None`` for static RSA.
    """

    rsa: np.ndarray
    axes: tuple[str, ...]
    coords: dict[str, np.ndarray]
    info: dict[str, Any] = field(default_factory=dict)
    space: SpaceDescription | None = None
    dynamic: DynamicInfo | None = None

    def __post_init__(self):
        rsa = np.asarray(self.rsa)
        axes = tuple(self.axes)
        coords = _validate_result_axes(rsa, axes, self.coords)
        object.__setattr__(self, "rsa", rsa)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "coords", coords)

    @property
    def search_axes(self) -> tuple[str, ...]:
        """The searchlight grid axes, excluding any leading model axis."""
        return tuple(axis for axis in self.axes if axis not in _MODEL_AXES)

    @property
    def search_coords(self) -> dict[str, np.ndarray]:
        """Coordinates for the searchlight grid axes."""
        return {axis: self.coords[axis] for axis in self.search_axes}

    def to_mne(self) -> Any:
        """Convert to an MNE container, or raise ``ValueError`` if unsupported."""
        from ._assemble import rsa_result_to_mne

        return rsa_result_to_mne(self)

    def save(self, fname: Any, *, overwrite: bool = False) -> None:
        """Save to an HDF5 file (see :mod:`._io`)."""
        from ._io import save_rsa_result

        save_rsa_result(self, fname, overwrite=overwrite)

    @classmethod
    def load(cls, fname: Any) -> RsaResult:
        """Load an `RsaResult` saved by :meth:`save`."""
        from ._io import load_rsa_result

        return load_rsa_result(fname)


__all__ = [
    "AggFunc",
    "RangeSpec",
    "AxisSpec",
    "StandardizeMode",
    "GroupReduceMode",
    "TfrFeatureMode",
    "DynamicRsaMode",
    "RsaInputKind",
    "RsaTemplate",
    "Label",
    "SourceEstimate",
    "VolSourceEstimate",
    "EpochsSpectrum",
    "EpochsTFR",
    "BaseEpochs",
    "RsaData",
    "RsaPrepOptions",
    "RsaResult",
    "DynamicInfo",
    "SpaceDescription",
    "SensorSpace",
    "SourceSpace",
    "PreparedRsaData",
    "AxisGroups",
]
