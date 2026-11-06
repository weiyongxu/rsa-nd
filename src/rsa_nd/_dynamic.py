"""Dynamic model-RDM RSA: time-varying Model RDMs against neural data.

Owns both the dynamic Model-RDM machinery (normalization, view extraction,
lag handling) and the public ``rsa_dynamic`` / ``rsa_dynamic_windows``
frontends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._core import rsa_nd, square_or_condensed_n_items
from ._prepare import prepare_rsa_data, prepare_rsa_data_kwarg_names
from ._types import (
    DynamicInfo,
    DynamicRsaMode,
    PreparedRsaData,
    RsaData,
    RsaResult,
)


@dataclass(frozen=True)
class DynamicModelRdms:
    """Normalized dynamic Model RDM stack plus its model-time axis."""

    rdms: list[np.ndarray]
    times: np.ndarray
    n_items: int

    @property
    def n_models(self) -> int:
        """Number of model-time RDMs."""

        return len(self.rdms)


def dynamic_rdms_from_signal(
    signal_window: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Build a ``(n_times, n_items, n_items)`` model-RDM stack."""

    from scipy.spatial.distance import pdist, squareform

    n_items, n_times = signal_window.shape
    rdms = np.empty((n_times, n_items, n_items), dtype=float)
    for time_idx in range(n_times):
        column = signal_window[:, time_idx][:, np.newaxis]
        rdms[time_idx] = squareform(pdist(column, metric=metric))
    return rdms


def normalize_dynamic_model_rdms(
    model_rdms: np.ndarray | Sequence[np.ndarray],
    model_times: Sequence[float],
    labels_rdm_model: Sequence[Any] | None,
) -> DynamicModelRdms:
    """Normalize dynamic Model RDMs and validate their time / Item axes."""

    times = np.asarray(model_times, dtype=float)
    if times.ndim != 1:
        raise ValueError("model_times must be a 1-D numeric sequence.")
    if times.size == 0:
        raise ValueError("model_times must not be empty.")
    if not np.isfinite(times).all():
        raise ValueError("model_times must contain only finite values.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("model_times must be strictly increasing.")

    if isinstance(model_rdms, np.ndarray):
        rdms = _dynamic_rdms_from_array(model_rdms, n_model_times=len(times))
    else:
        rdms = [np.asarray(rdm) for rdm in model_rdms]

    if len(rdms) != len(times):
        raise ValueError(
            "model_times must have one value per dynamic model RDM "
            f"({len(times)} times, {len(rdms)} RDMs)."
        )
    if not rdms:
        raise ValueError("model_rdms must contain at least one RDM.")

    n_items: int | None = None
    for idx, rdm in enumerate(rdms):
        this_n_items = square_or_condensed_n_items(rdm)
        if n_items is None:
            n_items = this_n_items
        elif this_n_items != n_items:
            raise ValueError("All dynamic model RDMs must describe the same items.")
        rdms[idx] = np.asarray(rdm, dtype=float)

    assert n_items is not None
    if labels_rdm_model is not None and len(labels_rdm_model) != n_items:
        raise ValueError(
            "labels_rdm_model must have one label per model-RDM item "
            f"({len(labels_rdm_model)} labels, {n_items} items)."
        )
    return DynamicModelRdms(rdms=rdms, times=times, n_items=n_items)


def normalize_lags(lags: Sequence[float] | float | None) -> np.ndarray | None:
    """Normalize lag values for Lagged RSA."""

    if lags is None:
        return None
    lag_values = np.atleast_1d(np.asarray(lags, dtype=float))
    if lag_values.ndim != 1 or lag_values.size == 0:
        raise ValueError("lags must be a non-empty 1-D numeric sequence.")
    if not np.isfinite(lag_values).all():
        raise ValueError("lags must contain only finite values.")
    return lag_values


def as_dynamic_full_rsa(
    rsa_values: np.ndarray,
    *,
    n_models: int,
    search_axes: tuple[str, ...],
    search_coords: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return RSA values as ``model_time x *search_shape``."""

    full = np.asarray(rsa_values, dtype=float)
    search_shape = tuple(len(search_coords[axis]) for axis in search_axes)
    expected_shape = (n_models, *search_shape)
    if full.shape == expected_shape:
        return full
    if n_models == 1 and full.shape == search_shape:
        return full.reshape(expected_shape)
    raise ValueError(
        "Dynamic RSA expected rsa_nd to return shape "
        f"{expected_shape}, got {full.shape}."
    )


def resolve_lag_tolerance(
    model_times: np.ndarray,
    lag_tolerance: float | Literal["auto"] | None,
) -> float | None:
    """Resolve the lag-matching tolerance in seconds."""

    if lag_tolerance is None:
        return None
    if lag_tolerance == "auto":
        if len(model_times) < 2:
            return 0.0
        return float(np.median(np.diff(model_times)) / 2)
    tolerance = float(lag_tolerance)
    if tolerance < 0:
        raise ValueError("lag_tolerance must be non-negative, 'auto', or None.")
    return tolerance


def extract_dynamic_view(
    full_rsa: np.ndarray,
    search_axes: tuple[str, ...],
    search_coords: Mapping[str, np.ndarray],
    dynamic_axis: str,
    model_times: np.ndarray,
    tolerance: float | None,
    *,
    lags: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Aligned RSA or Lagged RSA from a full generalization matrix."""

    try:
        dynamic_values = np.asarray(search_coords[dynamic_axis], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("dynamic_axis coordinates must be numeric.") from exc
    if dynamic_values.ndim != 1:
        raise ValueError("dynamic_axis coordinates must be 1-D.")
    if not np.isfinite(dynamic_values).all():
        raise ValueError("dynamic_axis coordinates must contain only finite values.")

    if lags is None:
        layered = _nearest_model_indices(model_times, dynamic_values, tolerance)[
            np.newaxis, :
        ]
        rsa, model_indices = gather_dynamic_along_axis(
            full_rsa,
            search_axes=search_axes,
            dynamic_axis=dynamic_axis,
            indices_for_axis_layers=layered,
        )
        return rsa[0], model_indices[0]

    layered = np.stack(
        [
            _nearest_model_indices(model_times, dynamic_values - lag, tolerance)
            for lag in lags
        ],
        axis=0,
    )
    return gather_dynamic_along_axis(
        full_rsa,
        search_axes=search_axes,
        dynamic_axis=dynamic_axis,
        indices_for_axis_layers=layered,
    )


def gather_dynamic_along_axis(
    full_rsa: np.ndarray,
    *,
    search_axes: tuple[str, ...],
    dynamic_axis: str,
    indices_for_axis_layers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather model-time-indexed scores along one Search axis."""

    search_shape = full_rsa.shape[1:]
    dynamic_pos = search_axes.index(dynamic_axis)
    indices = np.asarray(indices_for_axis_layers, dtype=int)
    if indices.ndim == 1:
        indices = indices[np.newaxis, :]
    if indices.ndim != 2 or indices.shape[1] != search_shape[dynamic_pos]:
        raise ValueError(
            "indices_for_axis_layers must have shape (n_layers, n_dynamic_axis_values)."
        )

    n_layers = indices.shape[0]
    layer_shape = (n_layers, *search_shape)
    broadcast_shape = (
        n_layers,
        *(
            length if axis_idx == dynamic_pos else 1
            for axis_idx, length in enumerate(search_shape)
        ),
    )
    model_indices = np.broadcast_to(
        indices.reshape(broadcast_shape),
        layer_shape,
    ).copy()
    valid = (model_indices >= 0) & (model_indices < full_rsa.shape[0])
    safe_model_indices = np.where(valid, model_indices, 0)

    search_grids = np.indices(search_shape, sparse=False)
    search_index_arrays = [
        np.broadcast_to(grid[np.newaxis, ...], layer_shape) for grid in search_grids
    ]
    rsa = np.asarray(
        full_rsa[(safe_model_indices, *search_index_arrays)],
        dtype=float,
    )
    rsa[~valid] = np.nan
    model_indices[~valid] = -1
    return rsa, model_indices


def _dynamic_rdms_from_array(
    model_rdms: np.ndarray,
    *,
    n_model_times: int,
) -> list[np.ndarray]:
    arr = np.asarray(model_rdms)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[1] != arr.shape[2]:
            raise ValueError(
                "A 3-D dynamic RDM stack must have shape "
                "(n_model_times, n_items, n_items)."
            )
        return [arr[idx] for idx in range(arr.shape[0])]
    if arr.ndim == 2:
        if arr.shape[0] == arr.shape[1]:
            if n_model_times == 1:
                return [arr]
            raise ValueError(
                "A 2-D square model_rdms array is ambiguous for dynamic RSA. "
                "Use a 3-D stack for square RDMs across time, a list of RDMs, "
                "or pass one model_time for a single square RDM."
            )
        return [arr[idx] for idx in range(arr.shape[0])]
    raise ValueError(
        "model_rdms must be a square stack, a condensed-vector stack, or a "
        "sequence of RDMs."
    )


def _nearest_model_indices(
    model_times: np.ndarray,
    target_times: np.ndarray,
    tolerance: float | None,
) -> np.ndarray:
    targets = np.asarray(target_times, dtype=float)
    flat_targets = targets.ravel()
    distances = np.abs(model_times[:, np.newaxis] - flat_targets[np.newaxis, :])
    indices = np.argmin(distances, axis=0).astype(int)
    min_distances = distances[indices, np.arange(flat_targets.size)]
    valid = np.isfinite(flat_targets)
    if tolerance is not None:
        valid &= min_distances <= tolerance + np.finfo(float).eps
    indices[~valid] = -1
    return indices.reshape(targets.shape)


_DYNAMIC_MODES = ("generalization", "aligned", "lagged")

# Shared prep / RSA kwarg grammar, derived once from prepare_rsa_data's
# signature (see ._prepare). Re-exported here for the dynamic frontends.
_PREP_KEYS_RSA = prepare_rsa_data_kwarg_names()


def rsa_dynamic(
    inst_or_data: PreparedRsaData | RsaData | Any,
    model_rdms: np.ndarray | Sequence[np.ndarray],
    model_times: Sequence[float],
    *,
    mode: DynamicRsaMode = "generalization",
    dynamic_axis: str = "time",
    lags: Sequence[float] | float | None = None,
    lag_tolerance: float | Literal["auto"] | None = "auto",
    search_axes: Sequence[str] = ("time",),
    radii: Mapping[str, float | int | None] | None = None,
    **rsa_kwargs: Any,
) -> RsaResult:
    """Run RSA against time-varying model RDMs.

    Accepts anything ``prepare_rsa_data`` understands (including a
    pre-built ``PreparedRsaData``), or an ``RsaData`` / array.
    """

    if "rdm_model" in rsa_kwargs:
        raise TypeError("Use 'model_rdms', not 'rdm_model', with rsa_dynamic.")
    if mode not in _DYNAMIC_MODES:
        raise ValueError(f"mode must be one of {list(_DYNAMIC_MODES)}.")

    search_axes_tuple = tuple(search_axes)
    if dynamic_axis not in search_axes_tuple:
        raise ValueError("dynamic_axis must be present in search_axes.")

    dynamic_model = normalize_dynamic_model_rdms(
        model_rdms, model_times, rsa_kwargs.get("labels_rdm_model")
    )
    lag_values = normalize_lags(lags) if mode == "lagged" else None
    if mode == "lagged" and lag_values is None:
        raise ValueError("mode='lagged' requires lags.")

    if isinstance(inst_or_data, PreparedRsaData):
        prepared = inst_or_data
        unexpected = sorted(k for k in rsa_kwargs if k in _PREP_KEYS_RSA)
        if unexpected:
            raise ValueError(
                "Preparation kwargs are not allowed when passing "
                f"PreparedRsaData; got {unexpected}."
            )
    else:
        prep_kwargs = {
            key: rsa_kwargs.pop(key)
            for key in list(rsa_kwargs)
            if key in _PREP_KEYS_RSA
        }
        prepared = prepare_rsa_data(inst_or_data, **prep_kwargs)

    static = rsa_nd(
        prepared,
        dynamic_model.rdms,
        search_axes=search_axes_tuple,
        radii=radii,
        **rsa_kwargs,
    )
    search_grid = static.search_axes
    search_coords = static.search_coords
    full_rsa = as_dynamic_full_rsa(
        static.rsa,
        n_models=dynamic_model.n_models,
        search_axes=search_grid,
        search_coords=search_coords,
    )
    resolved_tolerance = resolve_lag_tolerance(dynamic_model.times, lag_tolerance)

    full_rsa_axes = ("model_time", *search_grid)
    full_rsa_coords = {"model_time": dynamic_model.times, **search_coords}

    if mode == "generalization":
        rsa_values = full_rsa
        model_indices = None
        axes_out: tuple[str, ...] = full_rsa_axes
        coords_out = dict(full_rsa_coords)
    else:
        rsa_values, model_indices = extract_dynamic_view(
            full_rsa,
            search_grid,
            search_coords,
            dynamic_axis,
            dynamic_model.times,
            resolved_tolerance,
            lags=lag_values,
        )
        if lag_values is None:
            axes_out = tuple(search_grid)
            coords_out = dict(search_coords)
        else:
            axes_out = ("lag", *search_grid)
            coords_out = {"lag": lag_values, **search_coords}

    return RsaResult(
        rsa=np.asarray(rsa_values),
        axes=axes_out,
        coords=coords_out,
        info=static.info,
        space=static.space,
        dynamic=DynamicInfo(
            mode=mode,
            dynamic_axis=dynamic_axis,
            full_rsa=full_rsa,
            full_rsa_axes=full_rsa_axes,
            full_rsa_coords=full_rsa_coords,
            model_indices=model_indices,
            lags=lag_values,
            lag_tolerance=resolved_tolerance,
            model_times=dynamic_model.times,
        ),
    )


def rsa_dynamic_windows(
    inst_or_data: PreparedRsaData | RsaData | Any,
    model_signal: np.ndarray,
    model_times: Sequence[float],
    *,
    window_s: float,
    n_windows: int,
    mode: DynamicRsaMode = "generalization",
    model_metric: str = "cityblock",
    dynamic_axis: str = "time",
    lags: Sequence[float] | float | None = None,
    lag_tolerance: float | Literal["auto"] | None = "auto",
    search_axes: Sequence[str] = ("time",),
    radii: Mapping[str, float | int | None] | None = None,
    random_state: int | np.random.Generator | None = None,
    **rsa_kwargs: Any,
) -> RsaResult:
    """Random-window dynamic RSA averaged into a relative-time result.

    Draws ``n_windows`` random ``window_s``-second windows, aligned in
    wall-clock time between the neural data and ``model_signal``, builds
    per-window model RDMs from ``model_signal`` with ``model_metric``,
    runs :func:`rsa_dynamic` on each window, and averages the scores.

    The result is expressed in *relative* (within-window) time: the
    generalization axes run from 0 to ``window_s``, recovering lag
    structure rather than absolute clip time. This is the recommended
    workflow for long (e.g. minute-scale) epochs where a single dynamic
    RSA over the full epoch is intractable.

    Parameters
    ----------
    inst_or_data
        Anything :func:`rsa_dynamic` accepts (epochs, ``RsaData``, array,
        or a prepared object).
    model_signal
        ``(n_items, n_model_times)`` scalar model time series (e.g. EDA
        amplitude per clip). Pairwise distances at each model time form
        the dynamic model RDMs.
    model_times
        Strictly increasing times (seconds) for ``model_signal``'s second
        axis.
    window_s
        Window length in seconds.
    n_windows
        Number of random windows to draw and average.
    model_metric
        ``scipy.spatial.distance.pdist`` metric used to build each
        per-time model RDM from ``model_signal``.
    random_state
        Seed or ``Generator`` for window placement.
    """

    if n_windows < 1:
        raise ValueError("n_windows must be >= 1.")
    if window_s <= 0:
        raise ValueError("window_s must be positive.")

    model_signal = np.asarray(model_signal, dtype=float)
    if model_signal.ndim != 2:
        raise ValueError("model_signal must be 2-D (n_items, n_model_times).")
    model_times_arr = np.asarray(model_times, dtype=float)
    if model_times_arr.ndim != 1 or model_times_arr.size != model_signal.shape[1]:
        raise ValueError(
            "model_times must be 1-D with one value per model_signal column."
        )
    if np.any(np.diff(model_times_arr) <= 0):
        raise ValueError("model_times must be strictly increasing.")

    if isinstance(inst_or_data, PreparedRsaData):
        prepared = inst_or_data
        if any(k in _PREP_KEYS_RSA for k in rsa_kwargs):
            raise ValueError(
                "Preparation kwargs are not allowed when passing PreparedRsaData."
            )
    else:
        prep_kwargs = {
            key: rsa_kwargs.pop(key)
            for key in list(rsa_kwargs)
            if key in _PREP_KEYS_RSA
        }
        prepared = prepare_rsa_data(inst_or_data, **prep_kwargs)

    rsa_data = prepared.rsa_data
    if dynamic_axis not in rsa_data.axes:
        raise ValueError(f"dynamic_axis {dynamic_axis!r} not in neural axes.")
    dyn_idx = rsa_data.axis_index(dynamic_axis)
    neural_times = np.asarray(rsa_data.coords[dynamic_axis], dtype=float)

    win_neural = int(round(window_s * _sfreq_from_times(neural_times)))
    win_model = int(round(window_s * _sfreq_from_times(model_times_arr)))
    if win_neural < 2 or win_model < 2:
        raise ValueError("window_s is too short for the sampling rates.")
    if win_neural > neural_times.size or win_model > model_times_arr.size:
        raise ValueError("window_s exceeds the available data length.")

    start_t = max(neural_times[0], model_times_arr[0])
    end_t = min(neural_times[-1], model_times_arr[-1])
    max_start = end_t - start_t - window_s
    if max_start < 0:
        raise ValueError("window_s exceeds the overlapping time span.")

    rng = np.random.default_rng(random_state)
    accum_rsa: np.ndarray | None = None
    accum_full: np.ndarray | None = None
    last: RsaResult | None = None

    for _ in range(n_windows):
        ts = start_t + rng.uniform(0.0, max_start)
        s_n = int(np.searchsorted(neural_times, ts))
        s_m = int(np.searchsorted(model_times_arr, ts))
        s_n = min(s_n, neural_times.size - win_neural)
        s_m = min(s_m, model_times_arr.size - win_model)

        neural_sl = [slice(None)] * rsa_data.data.ndim
        neural_sl[dyn_idx] = slice(s_n, s_n + win_neural)
        win_data = rsa_data.data[tuple(neural_sl)]
        eeg_rel = neural_times[s_n : s_n + win_neural] - neural_times[s_n]

        win_coords = dict(rsa_data.coords)
        win_coords[dynamic_axis] = eeg_rel
        neural_window = RsaData(data=win_data, axes=rsa_data.axes, coords=win_coords)

        sig_win = model_signal[:, s_m : s_m + win_model]
        model_rel = model_times_arr[s_m : s_m + win_model] - model_times_arr[s_m]
        model_rdms = dynamic_rdms_from_signal(sig_win, model_metric)

        result = rsa_dynamic(
            neural_window,
            model_rdms=model_rdms,
            model_times=model_rel,
            mode=mode,
            dynamic_axis=dynamic_axis,
            lags=lags,
            lag_tolerance=lag_tolerance,
            search_axes=search_axes,
            radii=radii,
            **rsa_kwargs,
        )
        if accum_rsa is None:
            accum_rsa = np.asarray(result.rsa, dtype=float).copy()
            accum_full = np.asarray(result.dynamic.full_rsa, dtype=float).copy()
        else:
            accum_rsa += result.rsa
            accum_full += result.dynamic.full_rsa
        last = result

    assert last is not None and accum_rsa is not None and accum_full is not None
    dyn = last.dynamic
    return RsaResult(
        rsa=accum_rsa / n_windows,
        axes=last.axes,
        coords=last.coords,
        info=last.info,
        space=last.space,
        dynamic=DynamicInfo(
            mode=dyn.mode,
            dynamic_axis=dyn.dynamic_axis,
            full_rsa=accum_full / n_windows,
            full_rsa_axes=dyn.full_rsa_axes,
            full_rsa_coords=dyn.full_rsa_coords,
            model_indices=dyn.model_indices,
            lags=dyn.lags,
            lag_tolerance=dyn.lag_tolerance,
            model_times=dyn.model_times,
        ),
    )


def _sfreq_from_times(times: np.ndarray) -> float:
    if times.size < 2:
        raise ValueError("Need at least two time points to infer a rate.")
    step = float(np.median(np.diff(times)))
    if step <= 0:
        raise ValueError("Time coordinates must be strictly increasing.")
    return 1.0 / step


__all__ = [
    "rsa_dynamic",
    "rsa_dynamic_windows",
]
