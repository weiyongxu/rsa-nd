"""RSA frontend: PreparedRsaData -> RSA scores.

The public surface (``rsa_nd``, ``compute_rsa``) stays MNE-agnostic.
``rsa_nd`` coerces its input to a typed ``PreparedRsaData``,
resolves Item labels, and delegates the searchlight execution to
:mod:`._blocks` (plan -> layout -> ``mne_rsa.rsa_array``). This module owns:

- model-RDM shape helpers (square vs condensed),
- ``rsa_nd`` (the runner) and ``compute_rsa`` (the one-call frontend).

The searchlight engine itself -- ``SearchlightPlan`` / ``build_searchlight_plan``,
``build_array_layout`` / ``FlatPatchGenerator`` / ``run_searchlight_rsa`` --
lives in :mod:`._blocks`. This module is private; the public surface is
``rsa_nd`` / ``compute_rsa``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._blocks import (
    build_array_layout,
    build_searchlight_plan,
    run_searchlight_rsa,
)
from ._types import (
    PreparedRsaData,
    RsaData,
    RsaResult,
    SensorSpace,
    SourceSpace,
    SpaceDescription,
)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


# --------------------------------------------------------------------------- #
# Model-RDM shape helpers
# --------------------------------------------------------------------------- #
def n_items_from_condensed(n_pairs: int) -> int | None:
    """Invert ``n_pairs == n_items * (n_items - 1) // 2``.

    Returns ``None`` when ``n_pairs`` is not a valid condensed-RDM length for
    some ``n_items >= 2``.
    """

    n_items = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    if n_items < 2 or n_items * (n_items - 1) // 2 != n_pairs:
        return None
    return n_items


def square_or_condensed_n_items(rdm: np.ndarray) -> int:
    """Return ``n_items`` for one square or condensed RDM.

    Raises ``ValueError`` for any other shape.
    """

    arr = np.asarray(rdm)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Square model RDMs must have shape (n_items, n_items).")
        return arr.shape[0]
    if arr.ndim == 1:
        n_items = n_items_from_condensed(arr.size)
        if n_items is None:
            raise ValueError("Condensed model RDM vectors have invalid length.")
        return n_items
    raise ValueError("Each model RDM must be square or condensed.")


# --------------------------------------------------------------------------- #
# Public frontend
# --------------------------------------------------------------------------- #
def rsa_nd(
    data: RsaData | np.ndarray | PreparedRsaData,
    rdm_model: np.ndarray | list[np.ndarray],
    *,
    axes: Sequence[str] | None = None,
    coords: Mapping[str, Sequence[Any]] | None = None,
    search_axes: Sequence[str] = (),
    feature_axes: Sequence[str] | None = None,
    sample_axis: str | None = None,
    radii: Mapping[str, float | int | None] | None = None,
    data_rdm_metric: str = "correlation",
    data_rdm_params: Mapping[str, Any] | None = None,
    rsa_metric: str = "spearman",
    ignore_nan: bool = False,
    labels: Sequence[Any] | None = None,
    labels_rdm_model: Sequence[Any] | None = None,
    model_names: Sequence[str] | None = None,
    n_folds: int | Any | None = 1,
    n_jobs: int = 1,
    verbose: bool = False,
) -> RsaResult:
    """Run RSA over arbitrary real-valued named dimensions.

    Accepts:

    - ``PreparedRsaData`` from ``prepare_rsa_data(...)`` (typed seam path).
    - ``RsaData``, or an ``np.ndarray`` plus ``axes``/``coords``: wrapped into a
      minimal ``PreparedRsaData(input_kind="rsa_data", space=None)`` (an ndarray
      is just an ``RsaData`` the caller did not wrap by hand).

    The ``feature_axes`` and ``sample_axis`` arguments are explicit overrides
    on the default "every non-Item, non-Search axis becomes a Feature" rule.
    Spatial Searchlight geometry for MNE inputs lives on ``PreparedRsaData.space``;
    build it with ``prepare_rsa_data(..., space=...)``.

    ``radii`` maps each search axis to its searchlight radius, in that axis's
    own units: a ``"space"`` radius is in **meters** (geodesic for source,
    Euclidean for sensor), while numeric axes such as ``"time"`` / ``"freq"``
    use their coordinate units (e.g. seconds / Hz). Note this differs from
    mne-rsa, whose ``temporal_radius`` is expressed in **samples**. ``radius=0``
    (the default for a search axis absent from ``radii``) makes each element its
    own patch; ``radius=None`` pools the whole axis into a single patch.

    Named grouping (aggregate into bands/windows/ROIs, or keep group members as
    elements) is configured on ``prepare_rsa_data`` via ``<axis>=`` plus
    ``<axis>_mode``; an elements-mode axis is then searched here like any other
    search axis (one RSA per group). ``radii`` is the continuous,
    non-grouped searchlight base case.

    Passing a list of model RDMs (or a 3-D ``(n_models, n_items, n_items)``
    stack) adds a leading ``"model"`` axis to the result; name its coordinates
    with ``model_names``. ``RsaResult.axes`` / ``coords`` are the full dimension
    contract for the returned ``rsa`` array.
    """

    prepared = _coerce_to_prepared(data, axes=axes, coords=coords)
    rsa_data = prepared.rsa_data
    data_rdm_params = {} if data_rdm_params is None else dict(data_rdm_params)
    radii_map = {} if radii is None else dict(radii)
    search_axes_tuple = tuple(search_axes)

    backend_rdms, n_models, has_model_axis, model_coord = _normalize_model_rdms(
        rdm_model, model_names
    )

    resolved_feature_axes = _resolve_backend_feature_axes(
        rsa_data.axes,
        search_axes=search_axes_tuple,
        feature_axes=feature_axes,
        sample_axis=sample_axis,
    )

    if sample_axis is not None and sample_axis not in rsa_data.axes:
        raise ValueError(
            f"sample_axis {sample_axis!r} is not present in rsa_data.axes."
        )

    plan = build_searchlight_plan(prepared, search_axes_tuple, radii_map)

    layout = build_array_layout(
        rsa_data,
        feature_axes=resolved_feature_axes,
        sample_axis=sample_axis,
    )
    labels_X = _resolve_item_labels(
        rsa_data,
        labels=labels,
        prepared_item_labels=prepared.item_labels,
        labels_rdm_model=labels_rdm_model,
    )

    rsa_values = run_searchlight_rsa(
        layout,
        plan,
        backend_rdms,
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

    rsa, axes_out, coords_out = _result_array_axes_coords(
        rsa_values,
        plan,
        search_axes_tuple,
        has_model_axis=has_model_axis,
        n_models=n_models,
        model_coord=model_coord,
    )
    info, space = _result_info_from_prepared(
        prepared,
        run={
            "search_axes": search_axes_tuple,
            "radii": radii_map,
            "feature_axes": None if feature_axes is None else tuple(feature_axes),
            "sample_axis": sample_axis,
            "data_rdm_metric": data_rdm_metric,
            "data_rdm_params": data_rdm_params,
            "rsa_metric": rsa_metric,
            "ignore_nan": ignore_nan,
            "n_folds": n_folds,
        },
        model={
            "n_models": n_models,
            "names": None if model_names is None else list(model_names),
        },
    )
    return RsaResult(rsa=rsa, axes=axes_out, coords=coords_out, info=info, space=space)


def _coerce_to_prepared(
    data: RsaData | np.ndarray | PreparedRsaData,
    *,
    axes: Sequence[str] | None,
    coords: Mapping[str, Sequence[Any]] | None,
) -> PreparedRsaData:
    if isinstance(data, PreparedRsaData):
        if axes is not None or coords is not None:
            raise ValueError("axes/coords are not used when passing PreparedRsaData.")
        return data
    if isinstance(data, RsaData):
        if axes is not None or coords is not None:
            raise ValueError("axes/coords are not used when passing RsaData.")
        rsa_data = data
    elif isinstance(data, np.ndarray):
        if np.iscomplexobj(data):
            raise ValueError(
                "rsa_nd accepts real-valued data only. Use prepare_rsa_data() "
                "with tfr_feature= to project complex TFR coefficients first."
            )
        if axes is None:
            raise ValueError("axes must be provided when data is a NumPy array.")
        rsa_data = RsaData(
            data=data,
            axes=tuple(axes),
            coords={} if coords is None else dict(coords),
        )
    else:
        raise TypeError(
            "rsa_nd data must be PreparedRsaData, RsaData, or a NumPy array."
        )
    # An ndarray is just an RsaData the caller did not wrap; both are the
    # abstract, MNE-agnostic input kind.
    return PreparedRsaData(rsa_data=rsa_data, input_kind="rsa_data", space=None)


def _resolve_backend_feature_axes(
    data_axes: tuple[str, ...],
    *,
    search_axes: tuple[str, ...],
    feature_axes: Sequence[str] | None,
    sample_axis: str | None,
) -> tuple[str, ...]:
    """Resolve caller feature axes into the backend feature layout.

    Search axes are local feature axes for ``mne_rsa.rsa_array``: the
    searchlight patch selects a subset of those features at each position.
    Keep that implementation detail behind the ``rsa_nd`` interface.
    """

    if sample_axis is not None and sample_axis in search_axes:
        raise ValueError(f"sample_axis {sample_axis!r} cannot also be a Search axis.")
    if feature_axes is None:
        return tuple(
            axis for axis in data_axes if axis != "item" and axis != sample_axis
        )

    resolved = list(dict.fromkeys(tuple(feature_axes)))
    if sample_axis is not None and sample_axis in resolved:
        raise ValueError(f"sample_axis {sample_axis!r} cannot also be a Feature axis.")
    for axis in data_axes:
        if axis in search_axes and axis not in resolved:
            resolved.append(axis)
    return tuple(resolved)


def _resolve_item_labels(
    rsa_data: RsaData,
    *,
    labels: Sequence[Any] | None,
    prepared_item_labels: np.ndarray | None,
    labels_rdm_model: Sequence[Any] | None,
) -> Sequence[Any] | None:
    """Resolve Item labels for one ``rsa_nd`` run.

    This enforces the run-time label precedence and **does not**
    inspect any MNE object -- events / Condition resolution belong to the
    Layer 1 input Adapters, which populate ``PreparedRsaData.item_labels`` at
    preparation time. Precedence:

    1. ``labels=`` explicitly passed to ``rsa_nd(...)`` wins.
    2. Else ``prepared_item_labels`` (from the Layer 1 adapter).
    3. Else ``rsa_data.coords["item"]``.

    Raises ``ValueError`` if ``labels_rdm_model`` is provided but the resolved
    labels do not match it as a set, and the caller did not pass ``labels=``.
    """

    if labels is not None:
        labels_X: Sequence[Any] | None = labels
    elif prepared_item_labels is not None:
        labels_X = list(prepared_item_labels)
    else:
        labels_X = rsa_data.coords.get("item")

    if labels_rdm_model is not None and labels is None:
        if labels_X is None or set(map(_python_scalar, labels_X)) != set(
            map(_python_scalar, labels_rdm_model)
        ):
            raise ValueError(
                "labels must be provided, or item coordinates must match "
                "labels_rdm_model, when labels_rdm_model is used. If your "
                "data labels and model labels use different types, pass "
                "labels= explicitly."
            )
    return labels_X


def _normalize_model_rdms(
    rdm_model: np.ndarray | list[np.ndarray],
    model_names: Sequence[str] | None,
) -> tuple[np.ndarray | list[np.ndarray], int, bool, np.ndarray]:
    """Normalize the model RDM input into a backend value plus a model axis.

    A 2-D square RDM (or a 1-D condensed vector) is one model with **no** model
    axis. A list/tuple of RDMs, or a 3-D ``(n_models, n_items, n_items)`` stack,
    is many models with an explicit ``"model"`` axis. A 2-D non-square array is
    ambiguous and rejected. Returns
    ``(backend_rdms, n_models, has_model_axis, model_coord)``.
    """

    if isinstance(rdm_model, (list, tuple)):
        backend: np.ndarray | list[np.ndarray] = [np.asarray(r) for r in rdm_model]
        if not backend:
            raise ValueError("rdm_model list must contain at least one RDM.")
        n_models = len(backend)
        has_model_axis = True
    else:
        arr = np.asarray(rdm_model)
        if arr.ndim == 3:
            if arr.shape[1] != arr.shape[2]:
                raise ValueError(
                    "A 3-D model RDM stack must be (n_models, n_items, n_items)."
                )
            backend = [arr[idx] for idx in range(arr.shape[0])]
            n_models = arr.shape[0]
            has_model_axis = True
        elif arr.ndim == 2 and arr.shape[0] != arr.shape[1]:
            raise ValueError(
                "A 2-D non-square model array is ambiguous; pass a list of RDMs "
                "or a 3-D (n_models, n_items, n_items) stack for multi-model RSA."
            )
        else:
            backend = arr
            n_models = 1
            has_model_axis = False

    if model_names is not None:
        if not has_model_axis:
            raise ValueError(
                "model_names is only valid for multi-model RSA (a list of RDMs "
                "or a 3-D stack), not a single square / condensed model."
            )
        names = list(model_names)
        if len(names) != n_models:
            raise ValueError(
                f"model_names has {len(names)} entries but there are {n_models} models."
            )
        model_coord = np.asarray(names, dtype=object)
    else:
        model_coord = np.arange(n_models)

    return backend, n_models, has_model_axis, model_coord


def _result_array_axes_coords(
    rsa_values: np.ndarray,
    plan: Any,
    search_axes: tuple[str, ...],
    *,
    has_model_axis: bool,
    n_models: int,
    model_coord: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    """Shape the backend scores and name every result axis.

    ``axes`` / ``coords`` become the full contract for the returned ``rsa``: a
    leading ``"model"`` axis when ``has_model_axis``, then the searchlight grid.
    """

    rsa = np.asarray(rsa_values)
    if plan is None:
        search_shape: tuple[int, ...] = ()
        search_coords: dict[str, np.ndarray] = {}
    else:
        search_shape = tuple(plan.shape)
        search_coords = dict(plan.search_coords)

    if has_model_axis:
        target = (n_models, *search_shape)
        axes = ("model", *search_axes)
        coords = {"model": model_coord, **search_coords}
    else:
        target = search_shape
        axes = tuple(search_axes)
        coords = dict(search_coords)

    if rsa.shape != target:
        rsa = rsa.reshape(target)
    return rsa, axes, coords


def _result_space_from_prepared(
    space: SpaceDescription | None,
) -> SpaceDescription | None:
    """Copy the geometry the result needs, dropping the heavy source ``src``.

    Sensor ``Info`` is copied defensively so later mutations of the source
    object do not leak into the saved result.
    """

    if space is None:
        return None
    if isinstance(space, SensorSpace):
        info = space.info.copy() if space.info is not None else None
        return SensorSpace(
            channel_names=space.channel_names,
            info=info,
            patches=space.patches,
            is_grouped=space.is_grouped,
        )
    if isinstance(space, SourceSpace):
        return SourceSpace(
            vertices=space.vertices,
            src=None,
            patches=space.patches,
            is_grouped=space.is_grouped,
            estimate_kind=space.estimate_kind,
            subject=space.subject,
            subjects_dir=space.subjects_dir,
        )
    return space


def _result_info_from_prepared(
    prepared: PreparedRsaData,
    *,
    run: dict[str, Any],
    model: dict[str, Any],
) -> tuple[dict[str, Any], SpaceDescription | None]:
    """Build audit ``info`` and the typed result geometry from ``prepared``.

    Keeps input-specific metadata assembly in one place (Layer 1 is already
    normalized into ``prepared``). ``info`` is audit-only; the geometry
    ``to_mne`` needs rides the returned typed ``space``.
    """

    template = prepared.template
    info: dict[str, Any] = {
        "run": run,
        "model": model,
        "prepared": {
            "input_kind": prepared.input_kind,
            "input_axes": tuple(prepared.rsa_data.axes),
            "time_axis": prepared.time_axis,
            "freq_axis": prepared.freq_axis,
            "time_is_grouped": prepared.time_is_grouped,
            "freq_is_grouped": prepared.freq_is_grouped,
        },
        "mne": {
            "input_kind": prepared.input_kind,
            "subject": getattr(template, "subject", None),
        },
    }
    return info, _result_space_from_prepared(prepared.space)


def compute_rsa(
    inst_or_data: PreparedRsaData | RsaData | np.ndarray | Any,
    rdm_model: np.ndarray | list[np.ndarray],
    **kwargs: Any,
) -> RsaResult:
    """One-call RSA workflow: prepare input, then run RSA.

    Keyword arguments accepted by :func:`prepare_rsa_data` are consumed during
    preparation; the rest are forwarded to :func:`rsa_nd`. Always returns an
    :class:`RsaResult`; call ``result.to_mne()`` for an MNE container.

    Source-estimate inputs (including grouped ``space=`` ROI / Label TFR
    searchlights) run the dense path: the full source TFR tensor is
    materialised, so very large source spaces may need cropping or decimation.
    """

    from ._prepare import prepare_rsa_data, prepare_rsa_data_kwarg_names

    prep_keys = prepare_rsa_data_kwarg_names()
    prep_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in prep_keys}

    if isinstance(inst_or_data, PreparedRsaData):
        if prep_kwargs:
            unexpected = sorted(prep_kwargs)
            raise ValueError(
                "Preparation kwargs are not allowed when passing "
                f"PreparedRsaData; got {unexpected}."
            )
        prepared = inst_or_data
    else:
        prepared = prepare_rsa_data(inst_or_data, **prep_kwargs)

    return rsa_nd(prepared, rdm_model, **kwargs)


__all__ = [
    "compute_rsa",
    "RsaData",
    "RsaResult",
    "rsa_nd",
]
