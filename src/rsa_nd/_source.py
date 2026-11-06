"""Source-space geometry helpers for RSA input preparation.

This private module owns the source-estimate compatibility checks and the
Label / ROI -> source-vertex patch resolution used by ``prepare_rsa_data``
(via ``build_source_space``) when an MNE source-estimate input carries a
grouped ``space=`` of Labels / ROIs. Source RSA always runs the dense path:
``rsa_nd`` materialises the full source-vertex tensor and uses these patches
as Searchlight neighbourhoods.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ._types import Label, SourceEstimate, SourceSpace, VolSourceEstimate


def check_source_estimates_compatible(
    stcs: Sequence[SourceEstimate | VolSourceEstimate],
) -> None:
    """Raise unless all source estimates share shape, vertices, and time."""

    if not stcs:
        raise ValueError("stcs must contain at least one SourceEstimate.")

    first = stcs[0]
    first_vertices = [np.asarray(vertices) for vertices in first.vertices]
    first_times = np.asarray(first.times, dtype=float)
    first_shape = np.asarray(first.data).shape
    for idx, stc in enumerate(stcs[1:], 1):
        if np.asarray(stc.data).shape != first_shape:
            raise ValueError(
                "All source estimates must have the same data shape; "
                f"stc 0 has {first_shape}, stc {idx} has {np.asarray(stc.data).shape}."
            )
        vertices = [np.asarray(values) for values in stc.vertices]
        if len(vertices) != len(first_vertices) or any(
            not np.array_equal(a, b) for a, b in zip(first_vertices, vertices)
        ):
            raise ValueError("All source estimates must use the same vertices.")
        if not np.allclose(np.asarray(stc.times, dtype=float), first_times):
            raise ValueError("All source estimates must use the same time points.")


def build_source_space(
    stcs: Sequence[SourceEstimate | VolSourceEstimate],
    *,
    space_arg: Any,
    src: Any,
) -> SourceSpace:
    """Build the typed source `SpaceDescription` for prepared source data.

    The ``mne.SourceSpaces`` ``src`` (when given) is carried through so a
    spatial radius search can measure geodesic distance in meters; it is not
    required for grouped (Label / ROI) searchlights.
    """

    if not stcs:
        raise ValueError("stcs must contain at least one SourceEstimate.")

    first = stcs[0]
    vertices = tuple(np.asarray(v, dtype=int) for v in first.vertices)
    estimate_kind = "volume" if len(vertices) == 1 else "surface"

    # resolve_source_space_patches never returns empty (it raises), so grouped
    # is exactly "a space= grouping was given".
    patches = (
        None if space_arg is None else resolve_source_space_patches(space_arg, first)
    )
    return SourceSpace(
        vertices=vertices,
        src=src,
        patches=patches,
        is_grouped=patches is not None,
        estimate_kind=estimate_kind,
        subject=getattr(first, "subject", None),
    )


def source_roi_patches(
    stc: SourceEstimate | VolSourceEstimate,
    rois: Sequence[Label],
) -> tuple[dict[str, np.ndarray], list[Label]]:
    """Convert source-space Labels / ROIs to source-vertex index patches.

    The returned indices address rows in ``stc.data``. Empty Labels after
    restricting to the source estimate vertices are skipped.

    Parameters
    ----------
    stc
        Source estimate whose vertices define the available source space.
    rois
        Labels or ROIs to restrict to the vertices present in ``stc``.

    Returns
    -------
    patches
        Mapping from ROI name to row indices in ``stc.data``.
    kept_rois
        ROIs containing at least one vertex present in ``stc``, in input order.

    Raises
    ------
    TypeError
        If ``stc`` is not SourceEstimate-like.
    ValueError
        If two retained ROIs have the same name.
    """

    _validate_source_estimate_like(stc)
    vertex_selection_to_indices = _load_vertex_selection_to_indices()

    patches: dict[str, np.ndarray] = {}
    kept_rois: list[Label] = []
    for idx, roi in enumerate(rois):
        name = str(getattr(roi, "name", None) or f"roi_{idx}")
        if name in patches:
            raise ValueError(f"Duplicate ROI name {name!r}.")
        indices = _indices_for_labels(
            stc,
            [roi],
            vertex_selection_to_indices=vertex_selection_to_indices,
        )
        if indices.size == 0:
            continue
        patches[name] = indices
        kept_rois.append(roi)

    return patches, kept_rois


def resolve_source_space_patches(
    space_arg: Any,
    stc: SourceEstimate | VolSourceEstimate,
) -> dict[str, np.ndarray]:
    """Convert ``space=`` Labels / ROIs / patch mapping to vertex indices."""

    _validate_source_estimate_like(stc)
    vertex_selection_to_indices = _load_vertex_selection_to_indices()

    patches: dict[str, np.ndarray] = {}
    for name, labels in _iter_source_patch_specs(space_arg):
        indices = _indices_for_labels(
            stc,
            labels,
            vertex_selection_to_indices=vertex_selection_to_indices,
        )
        if indices.size == 0:
            continue
        patches[str(name)] = indices

    if not patches:
        raise ValueError(
            "No source ROI / Label contained vertices present in the source estimates."
        )
    return patches


def _iter_source_patch_specs(
    space_arg: Any,
) -> list[tuple[str, list[Any]]]:
    if isinstance(space_arg, Mapping):
        items = list(space_arg.items())
    else:
        items = [
            (str(getattr(entry, "name", None) or f"roi_{idx}"), entry)
            for idx, entry in enumerate(space_arg)
        ]
    return [
        (str(name), list(entry) if isinstance(entry, (list, tuple)) else [entry])
        for name, entry in items
    ]


def _indices_for_labels(
    stc: SourceEstimate | VolSourceEstimate,
    labels: Sequence[Any],
    *,
    vertex_selection_to_indices: Callable[[Any, Any], Any],
) -> np.ndarray:
    all_indices: list[int] = []
    for label in labels:
        try:
            indices = np.asarray(
                vertex_selection_to_indices(stc.vertices, label),
                dtype=int,
            )
        except ValueError:
            if _label_has_no_stc_vertices(stc.vertices, label):
                continue
            raise
        if indices.size:
            all_indices.extend(indices.tolist())
    if not all_indices:
        return np.asarray([], dtype=int)
    return np.unique(np.asarray(all_indices, dtype=int))


def _load_vertex_selection_to_indices() -> Callable[[Any, Any], Any]:
    try:
        from mne_rsa.source_level import vertex_selection_to_indices
    except ImportError as exc:
        raise ImportError(
            "Source ROI patch resolution requires mne-rsa to be installed."
        ) from exc
    return vertex_selection_to_indices


def _validate_source_estimate_like(stc: Any) -> None:
    if not hasattr(stc, "vertices") or not hasattr(stc, "data"):
        raise TypeError("stc must be an MNE SourceEstimate-like object.")


def _label_has_no_stc_vertices(
    stc_vertices: Sequence[np.ndarray],
    label: Any,
) -> bool:
    if not hasattr(label, "get_vertices_used") or not hasattr(label, "hemi"):
        return False
    if label.hemi == "lh":
        return len(label.get_vertices_used(stc_vertices[0])) == 0
    if label.hemi == "rh":
        return (
            len(stc_vertices) < 2 or len(label.get_vertices_used(stc_vertices[1])) == 0
        )
    return False


__all__ = [
    "build_source_space",
    "check_source_estimates_compatible",
    "resolve_source_space_patches",
    "source_roi_patches",
]
