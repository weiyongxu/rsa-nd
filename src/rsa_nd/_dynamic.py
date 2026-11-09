"""RSA against model RDMs that change over time."""

from dataclasses import replace

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._data import RsaData
from ._result import DynamicInfo, RsaResult
from ._rsa import _prepare_from_kwargs, rsa_nd

_MODES = ("generalization", "aligned", "lagged")


def rsa_dynamic(
    inst,
    model_rdms,
    model_times,
    *,
    mode="generalization",
    dynamic_axis="time",
    lags=None,
    lag_tolerance="auto",
    search_axes=("time",),
    radii=None,
    **kwargs,
):
    """Run RSA against time-varying model RDMs.

    Parameters
    ----------
    inst : instance of Epochs | EpochsTFR | EpochsSpectrum | RsaData | ndarray | list
        The data, as for :func:`compute_rsa`.
    model_rdms : ndarray | list of ndarray
        One model RDM per model time: a list, a 3-D stack of square RDMs, or a
        2-D stack of condensed RDMs.
    model_times : array-like of float
        The strictly increasing time of each model RDM, in seconds.
    mode : str
        ``"generalization"`` scores every model time at every search position,
        adding a leading ``"model_time"`` axis. ``"aligned"`` keeps the model
        time nearest to each data time. ``"lagged"`` keeps the model time
        ``lag`` seconds before each data time, adding a leading ``"lag"`` axis.
    dynamic_axis : str
        The search axis compared with model time.
    lags : float | array-like of float | None
        The lags for ``mode="lagged"``, in seconds.
    lag_tolerance : float | "auto" | None
        Largest allowed gap between a target time and the nearest model time.
        Positions with a larger gap are NaN. ``"auto"`` uses half the model time
        step and ``None`` means no limit.
    search_axes : sequence of str
        The axes to search over. Must include ``dynamic_axis``.
    radii : dict | None
        Searchlight radii, as for :func:`rsa_nd`.
    **kwargs : dict
        Arguments of :func:`prepare_rsa_data` and :func:`rsa_nd`.

    Returns
    -------
    result : RsaResult
        The RSA scores. ``result.dynamic`` holds the full model-time map.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
    search_axes = tuple(search_axes)
    if dynamic_axis not in search_axes:
        raise ValueError("dynamic_axis must be in search_axes.")
    if (radii or {}).get(dynamic_axis, 0) is None:
        raise ValueError("dynamic_axis cannot have a radius of None.")
    model_times = _check_times(model_times, "model_times")
    rdms = _check_dynamic_rdms(model_rdms, model_times)
    if mode != "lagged":
        lags = None
    elif lags is None:
        raise ValueError("mode='lagged' needs lags.")
    else:
        lags = np.atleast_1d(np.asarray(lags, dtype=float))
        if lags.ndim != 1 or lags.size == 0 or not np.isfinite(lags).all():
            raise ValueError("lags must be a non-empty 1-D sequence of finite values.")
    tolerance = _resolve_tolerance(model_times, lag_tolerance)

    prepared = _prepare_from_kwargs(inst, kwargs)
    static = rsa_nd(prepared, rdms, search_axes=search_axes, radii=radii, **kwargs)
    search_axes, search_coords = static.search_axes, static.search_coords
    full_axes = ("model_time", *search_axes)
    full_coords = {"model_time": model_times, **search_coords}

    if mode == "generalization":
        rsa, axes, coords, indices = static.rsa, full_axes, full_coords, None
    else:
        values = static.coords[dynamic_axis]
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("dynamic_axis needs numeric coordinates.")
        pos = search_axes.index(dynamic_axis)
        targets = [values] if lags is None else [values - lag for lag in lags]
        gathered = [
            _gather(static.rsa, pos, _nearest(model_times, target, tolerance))
            for target in targets
        ]
        rsa = np.stack([scores for scores, _ in gathered])
        indices = np.stack([idx for _, idx in gathered])
        if lags is None:
            rsa, indices = rsa[0], indices[0]
            axes, coords = search_axes, search_coords
        else:
            axes, coords = ("lag", *search_axes), {"lag": lags, **search_coords}

    dynamic = DynamicInfo(
        mode=mode,
        dynamic_axis=dynamic_axis,
        full_rsa=static.rsa,
        full_rsa_axes=full_axes,
        full_rsa_coords=full_coords,
        model_indices=indices,
        lags=lags,
        lag_tolerance=tolerance,
        model_times=model_times,
    )
    return RsaResult(rsa, axes, coords, static.info, static.space, dynamic)


def rsa_dynamic_windows(
    inst,
    model_signal,
    model_times,
    *,
    window_s,
    n_windows,
    mode="generalization",
    model_metric="cityblock",
    dynamic_axis="time",
    lags=None,
    lag_tolerance="auto",
    search_axes=("time",),
    radii=None,
    random_state=None,
    **kwargs,
):
    """Average dynamic RSA over random windows of long recordings.

    Draws random windows, aligned in time between the data and the model
    signal, runs :func:`rsa_dynamic` on each, and averages the scores. Results
    use time within the window, from 0 to ``window_s``, so they show lag
    structure rather than absolute time.

    Parameters
    ----------
    inst : instance of Epochs | EpochsTFR | EpochsSpectrum | RsaData | ndarray | list
        The data, as for :func:`compute_rsa`.
    model_signal : ndarray, shape (n_items, n_model_times)
        A model value per item over time. Pairwise distances at each time form
        the model RDMs.
    model_times : array-like of float
        The strictly increasing times of ``model_signal``, in seconds.
    window_s : float
        The window length, in seconds.
    n_windows : int
        The number of windows.
    mode, dynamic_axis, lags, lag_tolerance, search_axes, radii
        See :func:`rsa_dynamic`.
    model_metric : str
        The :func:`scipy.spatial.distance.pdist` metric for the model RDMs.
    random_state : int | Generator | None
        Seed or generator for placing the windows.
    **kwargs : dict
        Arguments of :func:`prepare_rsa_data` and :func:`rsa_nd`.

    Returns
    -------
    result : RsaResult
        The averaged RSA scores.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be at least 1.")
    if window_s <= 0:
        raise ValueError("window_s must be positive.")
    model_signal = np.asarray(model_signal, dtype=float)
    if model_signal.ndim != 2:
        raise ValueError("model_signal must have shape (n_items, n_model_times).")
    model_times = _check_times(model_times, "model_times")
    if len(model_times) != model_signal.shape[1]:
        raise ValueError("model_times needs one value per model_signal column.")

    prepared = _prepare_from_kwargs(inst, kwargs)
    rsa_data = prepared.rsa_data
    if dynamic_axis not in rsa_data.axes:
        raise ValueError(f"The data has no {dynamic_axis!r} axis.")
    pos = rsa_data.axes.index(dynamic_axis)
    times = _check_times(rsa_data.coords[dynamic_axis], dynamic_axis)

    n_data = _window_length(window_s, times)
    n_model = _window_length(window_s, model_times)
    start = max(times[0], model_times[0])
    span = min(times[-1], model_times[-1]) - start - window_s
    if span < 0 or n_data > len(times) or n_model > len(model_times):
        raise ValueError("window_s is longer than the overlapping data.")

    rng = np.random.default_rng(random_state)
    results = []
    for _ in range(n_windows):
        onset = start + rng.uniform(0, span)
        first = min(np.searchsorted(times, onset), len(times) - n_data)
        data = np.take(rsa_data.data, np.arange(first, first + n_data), axis=pos)
        window_times = times[first : first + n_data] - times[first]
        coords = {**rsa_data.coords, dynamic_axis: window_times}
        window = RsaData(data, rsa_data.axes, coords)

        first = min(np.searchsorted(model_times, onset), len(model_times) - n_model)
        signal = model_signal[:, first : first + n_model]
        rdms = [
            squareform(pdist(column[:, np.newaxis], model_metric))
            for column in signal.T
        ]
        results.append(
            rsa_dynamic(
                replace(prepared, rsa_data=window),
                rdms,
                model_times[first : first + n_model] - model_times[first],
                mode=mode,
                dynamic_axis=dynamic_axis,
                lags=lags,
                lag_tolerance=lag_tolerance,
                search_axes=search_axes,
                radii=radii,
                **kwargs,
            )
        )

    last = results[-1]
    dynamic = replace(
        last.dynamic,
        full_rsa=np.mean([result.dynamic.full_rsa for result in results], axis=0),
    )
    rsa = np.mean([result.rsa for result in results], axis=0)
    return replace(last, rsa=rsa, dynamic=dynamic)


def _check_times(times, name):
    """Return ``times`` as floats, or raise unless they strictly increase."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence.")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return times


def _check_dynamic_rdms(model_rdms, model_times):
    """Return one model RDM per model time, as a list of arrays."""
    if isinstance(model_rdms, np.ndarray):
        square = model_rdms.ndim == 2 and model_rdms.shape[0] == model_rdms.shape[1]
        if model_rdms.ndim == 1 or (square and len(model_times) == 1):
            model_rdms = [model_rdms]
        elif square:
            raise ValueError(
                "A 2-D square model_rdms is ambiguous. Use a 3-D stack or a list."
            )
    rdms = [np.asarray(rdm, dtype=float) for rdm in model_rdms]
    if len(rdms) != len(model_times):
        raise ValueError(f"Got {len(rdms)} model RDMs and {len(model_times)} times.")
    if len({rdm.shape for rdm in rdms}) > 1:
        raise ValueError("All dynamic model RDMs must have the same shape.")
    return rdms


def _resolve_tolerance(model_times, lag_tolerance):
    """Return the largest gap allowed to the nearest model time, or None."""
    if lag_tolerance is None:
        return None
    if lag_tolerance == "auto":
        return np.median(np.diff(model_times)) / 2 if len(model_times) > 1 else 0.0
    if lag_tolerance < 0:
        raise ValueError("lag_tolerance must be non-negative, 'auto' or None.")
    return float(lag_tolerance)


def _nearest(model_times, targets, tolerance):
    """Return the nearest model index per target, with -1 for no match."""
    dist = np.abs(np.subtract.outer(model_times, targets))
    indices = dist.argmin(axis=0)
    if tolerance is not None:
        indices[dist.min(axis=0) > tolerance + np.finfo(float).eps] = -1
    return indices


def _gather(full_rsa, pos, indices):
    """Pick one model time per position along search axis ``pos``."""
    shape = [1] * full_rsa.ndim
    shape[pos + 1] = -1
    indices = np.broadcast_to(indices.reshape(shape), (1, *full_rsa.shape[1:]))
    scores = np.take_along_axis(full_rsa, np.maximum(indices, 0), axis=0)[0]
    return np.where(indices[0] < 0, np.nan, scores), indices[0]


def _window_length(window_s, times):
    """Return the number of samples of ``times`` in ``window_s`` seconds."""
    if len(times) < 2:
        raise ValueError("Need at least two time points.")
    n_samples = round(window_s / np.median(np.diff(times)))
    if n_samples < 2:
        raise ValueError("window_s is too short for the sampling rate.")
    return n_samples
