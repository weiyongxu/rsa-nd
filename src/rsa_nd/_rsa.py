"""Run RSA over named axes, through ``mne_rsa.rsa_array``."""

import inspect
from dataclasses import replace
from itertools import product

import mne_rsa
import numpy as np

from ._data import PreparedRsaData, RsaData, SensorSpace
from ._prepare import prepare_rsa_data
from ._result import RsaResult

_PREPARE_KWARGS = set(inspect.signature(prepare_rsa_data).parameters) - {"inst"}


def compute_rsa(inst, rdm_model, **kwargs):
    """Prepare data and run RSA in one call.

    Parameters
    ----------
    inst : instance of Epochs | EpochsTFR | EpochsSpectrum | RsaData | ndarray | list
        The data, as for :func:`prepare_rsa_data`. May also be
        :class:`PreparedRsaData`.
    rdm_model : ndarray | list of ndarray
        The model RDM or RDMs, as for :func:`rsa_nd`.
    **kwargs : dict
        Arguments of :func:`prepare_rsa_data` and :func:`rsa_nd`.

    Returns
    -------
    result : RsaResult
        The RSA scores.
    """
    return rsa_nd(_prepare_from_kwargs(inst, kwargs), rdm_model, **kwargs)


def rsa_nd(
    data,
    rdm_model,
    *,
    axes=None,
    coords=None,
    search_axes=(),
    radii=None,
    data_rdm_metric="correlation",
    data_rdm_params=None,
    rsa_metric="spearman",
    ignore_nan=False,
    labels=None,
    labels_rdm_model=None,
    model_names=None,
    n_folds=1,
    n_jobs=1,
    verbose=False,
):
    """Run RSA over any named dimensions.

    Every axis other than ``"item"`` provides features. Search axes also define
    the searchlight: each position uses the features inside its patch.

    Parameters
    ----------
    data : PreparedRsaData | RsaData | ndarray
        The data.
    rdm_model : ndarray | list of ndarray
        A square or condensed model RDM. A list of RDMs, or a 3-D array of
        square RDMs, adds a leading ``"model"`` axis to the result.
    axes : sequence of str | None
        Axis names. Required when ``data`` is an array.
    coords : dict | None
        Axis coordinates for an array ``data``.
    search_axes : sequence of str
        The axes to search over. The result has these axes, in this order.
    radii : dict | None
        Searchlight radius per search axis, in the units of the axis: meters
        for ``"space"`` (Euclidean for sensors, geodesic for source space, and
        strictly less than the radius, as in mne-rsa), and coordinate units for
        other axes, such as seconds for ``"time"`` (note that mne-rsa uses
        samples). The default of 0 makes each element its own patch. ``None``
        pools the whole axis, which is the same as not searching it.
    data_rdm_metric : str
        Distance metric for the data RDMs.
    data_rdm_params : dict | None
        Extra arguments for the distance metric.
    rsa_metric : str
        How to compare data and model RDMs. See :func:`mne_rsa.rsa_array`.
    ignore_nan : bool
        Whether to ignore NaNs in the RDMs.
    labels : sequence | None
        One label per item. By default the labels stored in ``data``, such as
        event codes, or else the ``"item"`` coordinates. Items sharing a label
        are repeats of the same item, and are averaged or used as folds.
    labels_rdm_model : sequence | None
        One label per model RDM row. By default, the sorted unique ``labels``.
    model_names : sequence of str | None
        Coordinates for the ``"model"`` axis.
    n_folds : int | sklearn splitter | None
        Number of cross-validation folds.
    n_jobs : int
        Number of parallel jobs.
    verbose : bool
        Whether to show a progress bar.

    Returns
    -------
    result : RsaResult
        The RSA scores.
    """
    prepared = _as_prepared(data, axes, coords)
    rsa_data = prepared.rsa_data
    if "item" not in rsa_data.axes:
        raise ValueError("The data needs an 'item' axis.")
    radii = dict(radii or {})
    rdms, model_coord = _check_model_rdms(rdm_model, model_names)
    search = _resolve_search(prepared, tuple(search_axes), radii)

    if labels is None:
        labels = prepared.item_labels
    if labels is None:
        labels = rsa_data.coords["item"]

    X = np.moveaxis(rsa_data.data, rsa_data.axes.index("item"), 0)
    feature_axes = [axis for axis in rsa_data.axes if axis != "item"]
    patches = _Searchlight(
        X.shape[1:] or (1,),
        [feature_axes.index(axis) for axis in search],
        [patches for patches, _ in search.values()],
    )
    scores = mne_rsa.rsa_array(
        X.reshape(len(X), -1),
        rdms,
        patches=patches,
        data_rdm_metric=data_rdm_metric,
        data_rdm_params=data_rdm_params or {},
        rsa_metric=rsa_metric,
        ignore_nan=ignore_nan,
        labels_X=list(labels),
        labels_rdm_model=labels_rdm_model,
        n_folds=n_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    axes_out = tuple(search)
    coords_out = {axis: centers for axis, (_, centers) in search.items()}
    if model_coord is not None:
        # mne-rsa drops the model dimension for a list of one RDM.
        scores = np.reshape(scores, (len(model_coord), *patches.shape))
        axes_out = ("model", *axes_out)
        coords_out["model"] = model_coord
    info = dict(
        input_kind=prepared.input_kind,
        input_axes=rsa_data.axes,
        search_axes=tuple(search),
        radii=radii,
        data_rdm_metric=data_rdm_metric,
        data_rdm_params=data_rdm_params or {},
        rsa_metric=rsa_metric,
        ignore_nan=ignore_nan,
        n_folds=n_folds,
    )
    return RsaResult(scores, axes_out, coords_out, info, _result_space(prepared.space))


class _Searchlight:
    """Iterate over feature indices, one set per searchlight position.

    ``mne_rsa.rsa_array`` reads ``shape`` to reshape its output.
    """

    def __init__(self, feature_shape, positions, patches):
        self.feature_shape = feature_shape
        self.positions = positions
        self.patches = patches
        self.shape = tuple(len(axis_patches) for axis_patches in patches)

    def __len__(self):
        return int(np.prod(self.shape))

    def __iter__(self):
        selectors = [np.arange(length) for length in self.feature_shape]
        for combo in product(*self.patches):
            for pos, patch in zip(self.positions, combo):
                selectors[pos] = patch
            flat = np.ravel_multi_index(np.ix_(*selectors), self.feature_shape)
            yield slice(None), flat.ravel()


def _prepare_from_kwargs(inst, kwargs):
    """Prepare ``inst``, removing the arguments for preparation from ``kwargs``."""
    prepare_kwargs = {
        key: kwargs.pop(key) for key in list(kwargs) if key in _PREPARE_KWARGS
    }
    if not isinstance(inst, PreparedRsaData):
        return prepare_rsa_data(inst, **prepare_kwargs)
    if prepare_kwargs:
        raise ValueError(
            f"{sorted(prepare_kwargs)} cannot be used with PreparedRsaData."
        )
    return inst


def _as_prepared(data, axes, coords):
    """Wrap ``data`` in PreparedRsaData, checking the arguments that go with it."""
    if isinstance(data, PreparedRsaData):
        if axes is not None or coords is not None:
            raise ValueError("axes and coords are not used with PreparedRsaData.")
        return data
    if isinstance(data, RsaData):
        if axes is not None or coords is not None:
            raise ValueError("axes and coords are not used with RsaData.")
        return PreparedRsaData(data)
    if isinstance(data, np.ndarray):
        if axes is None:
            raise ValueError("axes is required when the data is an array.")
        return PreparedRsaData(RsaData(data, axes, coords))
    raise TypeError(
        f"data must be PreparedRsaData, RsaData or an array, got {type(data).__name__}."
    )


def _check_model_rdms(rdm_model, model_names):
    """Return the RDMs for mne-rsa and the ``"model"`` coordinates, if any."""
    if not isinstance(rdm_model, (list, tuple)):
        rdm_model = np.asarray(rdm_model)
        if rdm_model.ndim == 2 and rdm_model.shape[0] != rdm_model.shape[1]:
            raise ValueError(
                "A 2-D model RDM must be square. Pass a list of RDMs or a 3-D array "
                "for several models."
            )
        if rdm_model.ndim < 3:
            if model_names is not None:
                raise ValueError("model_names needs several model RDMs.")
            return rdm_model, None
    rdms = [np.asarray(rdm) for rdm in rdm_model]
    if not rdms:
        raise ValueError("rdm_model must contain at least one RDM.")
    if model_names is None:
        return rdms, np.arange(len(rdms))
    if len(model_names) != len(rdms):
        raise ValueError(f"Got {len(model_names)} model_names for {len(rdms)} models.")
    return rdms, np.array(model_names, dtype=object)


def _resolve_search(prepared, search_axes, radii):
    """Return ``{axis: (patches, centers)}`` for each search axis, in order."""
    rsa_data = prepared.rsa_data
    groups = dict(prepared.element_groups)
    if getattr(prepared.space, "patches", None) is not None:
        groups["space"] = prepared.space.patches

    if len(set(search_axes)) != len(search_axes):
        raise ValueError(f"search_axes must be unique, got {list(search_axes)}.")
    if "item" in search_axes:
        raise ValueError("'item' cannot be a search axis.")
    missing = sorted(set(search_axes) - set(rsa_data.axes))
    if missing:
        raise ValueError(f"Search axes not present in the data: {missing}.")
    unknown = sorted(set(radii) - set(search_axes))
    if unknown:
        raise ValueError(f"radii has axes that are not searched: {unknown}.")
    if any(radius is not None and radius < 0 for radius in radii.values()):
        raise ValueError("Radii must be non-negative.")
    both = sorted(axis for axis in groups if radii.get(axis, 0) != 0)
    if both:
        raise ValueError(f"Axes {both} have named groups, so they take no radius.")
    search_axes = [axis for axis in search_axes if radii.get(axis, 0) is not None]
    unsearched = sorted(set(groups) - set(search_axes))
    if unsearched:
        raise ValueError(
            f"Named groups on {unsearched} define searchlight positions, so these "
            "axes must be in search_axes. To average each group instead, use "
            "'aggregate' mode."
        )

    search = {}
    for axis in search_axes:
        if axis in groups:
            names = np.array(list(groups[axis]), dtype=object)
            search[axis] = (list(groups[axis].values()), names)
        elif axis == "space" and prepared.space is not None:
            search[axis] = _space_patches(prepared.space, radii.get(axis, 0))
        else:
            search[axis] = _radius_patches(rsa_data.coords[axis], radii.get(axis, 0))
    return search


def _space_patches(space, radius):
    """Return the spatial patches within ``radius`` of each element, and names."""
    names = space.names
    if radius == 0:
        return [np.array([idx]) for idx in range(len(names))], names
    dist = space._distances(radius)
    if len(dist) != len(names):
        raise ValueError(
            f"src has {len(dist)} vertices but the data has {len(names)}. They must "
            "use the same vertices."
        )
    return [np.flatnonzero(row < radius) for row in dist], names


def _radius_patches(values, radius):
    """Return the patches within ``radius`` of each coordinate, and the centres."""
    if len(np.unique(values)) != len(values):
        raise ValueError("Search axes cannot have duplicate coordinates.")
    if not np.issubdtype(values.dtype, np.number):
        if radius != 0:
            raise ValueError("Axes with named coordinates only support a radius of 0.")
        return [np.array([idx]) for idx in range(len(values))], values
    dist = np.abs(np.subtract.outer(values, values).astype(float))
    # Coordinates such as 0.06 - 0.02 can land just above an exact radius.
    within = (dist <= radius) | np.isclose(dist, radius, rtol=1e-9, atol=0)
    return [np.flatnonzero(row) for row in within], values


def _result_space(space):
    """Copy the geometry for a result, without the large source space."""
    if isinstance(space, SensorSpace):
        return replace(space, info=space.info.copy())
    return None if space is None else replace(space, src=None)
