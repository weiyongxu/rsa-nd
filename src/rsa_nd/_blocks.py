"""Searchlight execution engine: plan -> layout -> backend scores.

This module owns the MNE-agnostic machinery that turns a ``PreparedRsaData``
plus a searchlight request into RSA scores:

- ``SearchlightPlan`` + ``build_searchlight_plan`` (the searchlight grammar),
- ``RsaArrayLayout`` + ``build_array_layout`` + ``FlatPatchGenerator`` (the
  ``item x feature x sample`` layout and its backend patch iterator),
- ``run_searchlight_rsa`` (one batched ``mne_rsa.rsa_array`` call, reshaped
  onto the searchlight grid).

``_core`` composes these behind ``rsa_nd``. This module is private.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from ._types import PreparedRsaData, RsaData, _as_real_float


def _has_duplicates(values: Sequence[Any]) -> bool:
    scalars = np.asarray(values).tolist()
    return len(set(scalars)) != len(scalars)


# --------------------------------------------------------------------------- #
# Searchlight plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SearchlightPlan:
    """Typed searchlight description.

    Parameters
    ----------
    search_axes
        Names of the axes the searchlight slides over. Order matches
        `axis_patches` and `shape`.
    axis_patches
        For each search axis, the per-position selectors into that axis.
        Each selector is either a `slice` (radius=0 / contiguous) or an
        `ndarray` of integer indices.
    search_coords
        For each search axis, the per-position center value (numeric for
        radius searches, name for explicit patches).
    shape
        Number of searchlight positions per axis, in `search_axes` order.
    """

    search_axes: tuple[str, ...]
    axis_patches: dict[str, tuple[slice | np.ndarray, ...]]
    search_coords: dict[str, np.ndarray]
    shape: tuple[int, ...]

    @property
    def n_patches(self) -> int:
        total = 1
        for length in self.shape:
            total *= int(length)
        return total


def build_searchlight_plan(
    prepared: PreparedRsaData,
    search_axes: tuple[str, ...],
    radii: Mapping[str, float | int | None],
) -> SearchlightPlan | None:
    """Resolve search axes, radii, and element groups into one ``SearchlightPlan``.

    Returns ``None`` when ``search_axes`` is empty (no searchlight; the RSA
    runs once over all features).

    Elements-mode grouping is resolved upstream by ``prepare_rsa_data`` and
    carried on ``prepared.axis_groups`` (numeric ``time`` / ``freq`` axes).
    When a search axis has element groups, each group becomes one explicit
    multi-index patch, so the searchlight yields one RSA value per group with
    every member point kept as a joint feature. Spatial grouping rides
    ``prepared.space``.

    Raises
    ------
    ValueError
        If a radius key is not a search axis, if an axis has both a radius and
        element groups, if an element-grouped axis is not searched, if a search
        axis is absent from ``prepared.rsa_data``, or if a search-axis
        coordinate set is invalid for the requested radius.
    """

    rsa_data = prepared.rsa_data
    # Elements-mode entries are the time/freq groups searched one-RSA-per-group;
    # aggregate entries (roi/window/band) collapse their axis instead, and
    # spatial element grouping rides ``prepared.space``, so neither appears here.
    element_groups = {
        axis: tuple((name, indices) for name, indices in ag.groups)
        for axis, ag in (prepared.axis_groups or {}).items()
        if ag.mode == "elements"
    }
    if "item" in search_axes:
        raise ValueError("'item' cannot be a search axis.")

    missing = [axis for axis in search_axes if axis not in rsa_data.axes]
    if missing:
        raise ValueError(f"Search axes not present in data: {missing}")

    unknown_radii = sorted(set(radii) - set(search_axes))
    if unknown_radii:
        raise ValueError(
            f"Unknown radii keys: {unknown_radii}. Radii keys must be search axes."
        )
    ungrouped = sorted(set(element_groups) - set(search_axes))
    if ungrouped:
        raise ValueError(
            f"Element-grouped axes {ungrouped} must be in search_axes; "
            "elements-mode grouping defines searchlight positions."
        )
    both = sorted(set(radii) & set(element_groups))
    if both:
        raise ValueError(
            f"Axes {both} have both a radius and element groups; choose one."
        )

    if not search_axes:
        return None

    axis_patches: dict[str, tuple[slice | np.ndarray, ...]] = {}
    search_coords: dict[str, np.ndarray] = {}

    for axis_name in search_axes:
        values = np.asarray(rsa_data.coords[axis_name])
        if axis_name == "space" and prepared.space is not None:
            patches, centers = prepared.space.searchlight_patches(
                radii.get(axis_name, 0)
            )
            axis_patches[axis_name] = tuple(
                _validate_patch_indices(patches, n_elements=len(values))
            )
            search_coords[axis_name] = centers
        elif axis_name in element_groups:
            patches = _validate_patch_indices(
                [idx for _, idx in element_groups[axis_name]],
                n_elements=len(values),
            )
            axis_patches[axis_name] = tuple(patches)
            search_coords[axis_name] = np.asarray(
                [name for name, _ in element_groups[axis_name]], dtype=object
            )
        else:
            radius = radii.get(axis_name, 0)
            patches, centers = _axis_radius_patches(values, radius)
            axis_patches[axis_name] = tuple(patches)
            search_coords[axis_name] = centers

    patch_shape = tuple(len(axis_patches[axis]) for axis in search_axes)
    return SearchlightPlan(
        search_axes=search_axes,
        axis_patches=axis_patches,
        search_coords=search_coords,
        shape=patch_shape,
    )


def _validate_patch_indices(
    patches: list[np.ndarray],
    *,
    n_elements: int,
) -> list[np.ndarray]:
    """Validate explicit index patches (space geometry or element groups)."""
    validated: list[np.ndarray] = []
    for patch in patches:
        indices = np.asarray(patch, dtype=int)
        if indices.size == 0:
            raise ValueError("Searchlight patches may not be empty.")
        if np.any(indices < 0) or np.any(indices >= n_elements):
            raise ValueError("Searchlight patch index is out of bounds.")
        validated.append(indices)
    return validated


def _axis_radius_patches(
    values: np.ndarray,
    radius: float | int | None,
) -> tuple[list[slice | np.ndarray], np.ndarray]:
    if radius is None:
        return [slice(None)], np.asarray(["all"], dtype=object)

    values = np.asarray(values)
    if _has_duplicates(values):
        raise ValueError(
            "Duplicate coordinate values are not supported on search axes."
        )
    if np.issubdtype(values.dtype, np.number):
        radius_float = float(radius)
        if radius_float < 0:
            raise ValueError("Searchlight radii must be non-negative.")
        numeric_values = values.astype(float)
        mask = np.abs(numeric_values[:, np.newaxis] - numeric_values[np.newaxis, :])
        mask = mask <= radius_float
        patches: list[slice | np.ndarray] = []
        for idx, row in enumerate(mask):
            selected = np.flatnonzero(row)
            if radius_float == 0 and selected.size == 1:
                patches.append(slice(idx, idx + 1))
            else:
                patches.append(selected)
        return patches, values

    if radius != 0:
        raise ValueError("Non-numeric search axes only support radius=0 or None.")
    return [slice(idx, idx + 1) for idx in range(len(values))], values


# --------------------------------------------------------------------------- #
# Array layout and backend patch iterator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RsaArrayLayout:
    """Typed array layout.

    Parameters
    ----------
    X
        Flattened array shaped `(n_items, n_features, n_samples)`. When
        `sample_axis is None`, `n_samples == 1`.
    feature_axes
        Names of the axes flattened into the feature dimension, in the
        order they appear in `X`'s flat index.
    feature_shape
        Per-axis lengths of the un-flattened feature dimensions. Used to
        recover flat feature indices from per-axis selectors at run time.
    sample_axis
        Name of the axis kept as the sample dimension, or `None` if no
        sample axis was requested.
    sample_values
        Coordinate values along `sample_axis` (or `[0]` if no sample axis).
    """

    X: np.ndarray
    feature_axes: tuple[str, ...]
    feature_shape: tuple[int, ...]
    sample_axis: str | None
    sample_values: np.ndarray


def build_array_layout(
    rsa_data: RsaData,
    *,
    feature_axes: Sequence[str] | None = None,
    sample_axis: str | None = None,
) -> RsaArrayLayout:
    """Flatten ``rsa_data`` into ``item x feature x sample`` shape.

    When ``feature_axes`` is None, every non-Item, non-sample axis becomes a
    feature axis (current default behavior; preserved for slice 1).
    """

    axes = tuple(rsa_data.axes)
    if "item" not in axes:
        raise ValueError("RsaData must have an 'item' axis.")

    if feature_axes is None:
        feature_axes_tuple = tuple(
            axis for axis in axes if axis != "item" and axis != sample_axis
        )
    else:
        feature_axes_tuple = tuple(feature_axes)

    X, feature_axes_tuple, feature_shape = _flatten_named_array(
        rsa_data.data,
        axes,
        feature_axes=feature_axes_tuple,
        sample_axis=sample_axis,
    )

    if sample_axis is None:
        sample_values = np.array([0])
    else:
        sample_values = np.asarray(rsa_data.coords[sample_axis])

    return RsaArrayLayout(
        X=X,
        feature_axes=feature_axes_tuple,
        feature_shape=feature_shape,
        sample_axis=sample_axis,
        sample_values=sample_values,
    )


def _flatten_named_array(
    data: np.ndarray,
    axes: Sequence[str],
    *,
    feature_axes: Sequence[str],
    sample_axis: str | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    """Low-level flatten primitive backing ``build_array_layout``.

    Returns the flattened ``X`` plus the resolved feature-axes tuple and the
    pre-flatten feature shape needed to recover indices later.
    """

    axes = tuple(axes)
    arr = _as_real_float(data)
    feature_axes_tuple = tuple(feature_axes)
    requested = {"item", *feature_axes_tuple}
    if sample_axis is not None:
        requested.add(sample_axis)

    missing = [axis for axis in requested if axis not in axes]
    if missing:
        raise ValueError(f"Requested axes not present: {missing}")

    item_idx = axes.index("item")
    feature_indices = [axes.index(axis) for axis in feature_axes_tuple]
    if sample_axis is None:
        transpose_order = [item_idx, *feature_indices]
    else:
        sample_idx = axes.index(sample_axis)
        transpose_order = [item_idx, *feature_indices, sample_idx]

    arr = np.transpose(arr, transpose_order)
    n_items = arr.shape[0]
    feature_shape = arr.shape[1 : 1 + len(feature_axes_tuple)]
    n_features = int(np.prod(feature_shape, dtype=int)) if feature_shape else 1

    if sample_axis is None:
        X = arr.reshape(n_items, n_features, 1)
    else:
        n_samples = arr.shape[-1]
        X = arr.reshape(n_items, n_features, n_samples)

    return X, feature_axes_tuple, tuple(int(length) for length in feature_shape)


def _flat_feature_indices_from_vectors(
    index_vectors: Sequence[np.ndarray],
    feature_shape: tuple[int, ...],
) -> np.ndarray:
    """Map per-axis index vectors to flat C-order feature indices.

    Used by `FlatPatchGenerator` (below) to turn each Searchlight position's
    per-axis selectors into the flat feature indices mne-rsa expects.
    """

    if len(index_vectors) != len(feature_shape):
        raise ValueError("index_vectors must have one entry per feature axis.")
    if not feature_shape:
        return np.array([0], dtype=int)

    strides = _c_order_strides(feature_shape)
    offsets = np.array([0], dtype=int)
    for values, length, stride in zip(index_vectors, feature_shape, strides):
        indices = np.atleast_1d(np.asarray(values, dtype=int))
        if __debug__ and np.any((indices < 0) | (indices >= length)):
            raise IndexError("Feature patch index is out of bounds.")
        offsets = (offsets[:, np.newaxis] + indices[np.newaxis, :] * stride).ravel()
    return offsets.astype(int, copy=False)


def _c_order_strides(shape: tuple[int, ...]) -> np.ndarray:
    strides = np.ones(len(shape), dtype=int)
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * int(shape[idx + 1])
    return strides


class FlatPatchGenerator:
    """Bridge `SearchlightPlan` and `RsaArrayLayout` to mne-rsa's iterator.

    Yields ``(item_slice, feature_indices, sample_slice)`` tuples, one per
    Searchlight position, in C order over ``plan.shape``.
    """

    def __init__(
        self,
        plan: SearchlightPlan,
        feature_axes: tuple[str, ...],
        feature_shape: tuple[int, ...],
    ):
        self.plan = plan
        self.feature_axes = feature_axes
        self.feature_shape = feature_shape
        # `shape` is read by mne_rsa.rsa_array to reshape its 1-D output to
        # the multi-axis Searchlight shape. Without it, callers see a flat
        # (n_models, n_patches) array instead of (n_models, *plan.shape).
        self.shape = plan.shape
        self._axis_positions = {axis: idx for idx, axis in enumerate(feature_axes)}
        self._default_vectors = [
            None if axis in plan.search_axes else np.arange(length, dtype=int)
            for axis, length in zip(feature_axes, feature_shape)
        ]

    def __iter__(self):
        for combo in self._iter_patch_combos():
            index_vectors = list(self._default_vectors)
            for axis_name, selector in zip(self.plan.search_axes, combo):
                axis_pos = self._axis_positions[axis_name]
                length = self.feature_shape[axis_pos]
                if isinstance(selector, slice):
                    index_vectors[axis_pos] = np.arange(
                        *selector.indices(length), dtype=int
                    )
                else:
                    index_vectors[axis_pos] = np.atleast_1d(
                        np.asarray(selector, dtype=int)
                    )
            feature_indices = _flat_feature_indices_from_vectors(
                index_vectors,
                self.feature_shape,
            )
            yield (slice(None), feature_indices, slice(None))

    def _iter_patch_combos(self):
        patch_lists = [self.plan.axis_patches[axis] for axis in self.plan.search_axes]
        yield from product(*patch_lists)

    def __len__(self):
        return int(np.prod(self.shape, dtype=int))


# --------------------------------------------------------------------------- #
# Searchlight scoring
# --------------------------------------------------------------------------- #
def run_searchlight_rsa(
    layout: RsaArrayLayout,
    plan: SearchlightPlan | None,
    rdm_model: np.ndarray | list[np.ndarray],
    *,
    data_rdm_metric: str,
    data_rdm_params: Mapping[str, Any],
    rsa_metric: str,
    ignore_nan: bool,
    labels_X: Sequence[Any] | None,
    labels_rdm_model: Sequence[Any] | None,
    n_folds: int | Any | None,
    n_jobs: int,
    verbose: bool,
) -> np.ndarray:
    """Score one prepared ``layout`` against ``rdm_model`` in a single backend call.

    ``plan is None`` runs RSA once over all features; otherwise
    ``FlatPatchGenerator`` drives the searchlight and the backend output is
    reshaped onto ``plan.shape`` (with any leading model dimension preserved).
    """

    try:
        import mne_rsa
    except ImportError as exc:
        raise ImportError("rsa_nd requires mne-rsa to be installed.") from exc

    patches = (
        None
        if plan is None
        else FlatPatchGenerator(plan, layout.feature_axes, layout.feature_shape)
    )
    values = np.asarray(
        mne_rsa.rsa_array(
            layout.X,
            rdm_model,
            patches=patches,
            data_rdm_metric=data_rdm_metric,
            data_rdm_params=data_rdm_params,
            rsa_metric=rsa_metric,
            ignore_nan=ignore_nan,
            labels_X=labels_X,
            labels_rdm_model=labels_rdm_model,
            n_folds=n_folds,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    )
    if plan is None:
        return values
    flat = _flatten_backend_values(values, plan.shape)
    prefix = flat.shape[:-1]
    return flat.reshape((*prefix, *plan.shape))


def _flatten_backend_values(
    values: np.ndarray,
    patch_shape: tuple[int, ...],
) -> np.ndarray:
    """Normalise ``mne_rsa.rsa_array`` output to ``(*model_prefix, n_patches)``."""
    arr = np.asarray(values)
    n_patches = int(np.prod(patch_shape, dtype=int))
    if arr.shape == patch_shape:
        return arr.reshape(n_patches)
    if patch_shape and arr.shape[-len(patch_shape) :] == patch_shape:
        prefix = arr.shape[: -len(patch_shape)]
        return arr.reshape((*prefix, n_patches))
    if arr.shape[-1:] == (n_patches,):
        return arr
    raise ValueError(
        "rsa_array returned an unexpected shape "
        f"{arr.shape}; expected patch dimensions {patch_shape}."
    )
