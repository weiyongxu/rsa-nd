"""RSA results: conversion to MNE containers and HDF5 persistence."""

import os
import tempfile
from dataclasses import asdict, dataclass, field

import mne
import numpy as np
from h5io import read_hdf5, write_hdf5
from mne.time_frequency import AverageTFRArray, SpectrumArray

from ._data import SensorSpace, SourceSpace, _check_axes

_KIND = "rsa_nd.rsa_result"
_TITLE = "rsa_nd_result"
_SCHEMA_VERSION = 1
# Files written before this package was split out of mne-ez stay readable.
_LEGACY_KIND = "mne_ez.rsa_result"
_LEGACY_TITLE = "mne_ez_rsa_result"
_MODEL_AXES = ("model", "model_time")
_MNE_AXES = ("space", "roi", "freq", "time")


@dataclass
class DynamicInfo:
    """Extra outputs of dynamic RSA, stored on :attr:`RsaResult.dynamic`.

    Parameters
    ----------
    mode : str
        The dynamic RSA mode.
    dynamic_axis : str
        The search axis compared with model time.
    full_rsa : ndarray
        Scores for every model time and search position.
    full_rsa_axes : tuple of str
        The axes of ``full_rsa``.
    full_rsa_coords : dict
        The coordinates of ``full_rsa``.
    model_indices : ndarray | None
        The model time index behind each score, with ``-1`` for no match.
    lags : ndarray | None
        The lags, in seconds.
    lag_tolerance : float | None
        The largest allowed gap between a target time and a model time.
    model_times : ndarray | None
        The time of each model RDM.
    """

    mode: str
    dynamic_axis: str
    full_rsa: np.ndarray
    full_rsa_axes: tuple
    full_rsa_coords: dict
    model_indices: np.ndarray | None = None
    lags: np.ndarray | None = None
    lag_tolerance: float | None = None
    model_times: np.ndarray | None = None

    def __post_init__(self):
        self.full_rsa = np.asarray(self.full_rsa)
        self.full_rsa_axes = tuple(self.full_rsa_axes)
        self.full_rsa_coords = _check_axes(
            self.full_rsa.shape, self.full_rsa_axes, self.full_rsa_coords
        )


@dataclass(repr=False)
class RsaResult:
    """The scores of an RSA run, with their axes and coordinates.

    Parameters
    ----------
    rsa : ndarray
        The RSA scores.
    axes : tuple of str
        One name per dimension of ``rsa``. Several models add a leading
        ``"model"`` axis.
    coords : dict
        Coordinate values per axis.
    info : dict
        Settings of the run, for the record.
    space : SensorSpace | SourceSpace | None
        Geometry of the ``"space"`` axis.
    dynamic : DynamicInfo | None
        Extra outputs of dynamic RSA.
    """

    rsa: np.ndarray
    axes: tuple
    coords: dict
    info: dict = field(default_factory=dict)
    space: SensorSpace | SourceSpace | None = None
    dynamic: DynamicInfo | None = None

    def __post_init__(self):
        self.rsa = np.asarray(self.rsa)
        self.axes = tuple(self.axes)
        self.coords = _check_axes(self.rsa.shape, self.axes, self.coords)

    def __repr__(self):
        dims = ", ".join(f"{a}: {n}" for a, n in zip(self.axes, self.rsa.shape))
        return f"<RsaResult | {dims or 'scalar'}>"

    @property
    def search_axes(self):
        """The searchlight axes, without any leading model axis."""
        return tuple(axis for axis in self.axes if axis not in _MODEL_AXES)

    @property
    def search_coords(self):
        """The coordinates of the searchlight axes."""
        return {axis: self.coords[axis] for axis in self.search_axes}

    def to_mne(self):
        """Convert to the matching MNE container.

        Returns
        -------
        inst : Evoked | Spectrum | AverageTFR | SourceEstimate | VolSourceEstimate
            A source estimate for per-vertex source results. Otherwise a sensor
            container, chosen by the ``time`` and ``freq`` axes. Results without
            sensor geometry use one channel per group, or a single ``"rsa"``
            channel when there is no spatial axis.

        Raises
        ------
        ValueError
            If the axes have no matching MNE container, for example with a
            ``model`` axis or named time windows.
        """
        return _to_mne(self)

    def save(self, fname, *, overwrite=False):
        """Save to an HDF5 file.

        Parameters
        ----------
        fname : path-like
            The file to write.
        overwrite : bool
            Whether to replace an existing file.
        """
        payload = dict(
            kind=_KIND,
            schema_version=_SCHEMA_VERSION,
            rsa=self.rsa,
            axes=list(self.axes),
            coords=_encode(self.coords),
            info=_encode(self.info),
            space=_space_to_dict(self.space),
            dynamic=_dynamic_to_dict(self.dynamic),
        )
        write_hdf5(
            str(fname), payload, overwrite=overwrite, title=_TITLE, slash="replace"
        )

    @classmethod
    def load(cls, fname):
        """Load a result written by :meth:`save`.

        Parameters
        ----------
        fname : path-like
            The file to read.

        Returns
        -------
        result : RsaResult
            The loaded result.
        """
        try:
            payload = read_hdf5(str(fname), title=_TITLE, slash="replace")
        except ValueError:
            payload = read_hdf5(str(fname), title=_LEGACY_TITLE, slash="replace")
        kind = payload.get("kind")
        if kind not in (_KIND, _LEGACY_KIND):
            raise ValueError(f"{fname} is not an rsa-nd result file (kind={kind!r}).")
        return cls(
            rsa=payload["rsa"],
            axes=payload["axes"],
            coords=payload["coords"],
            info=payload.get("info") or {},
            space=_space_from_dict(payload.get("space")),
            dynamic=_dynamic_from_dict(payload.get("dynamic")),
        )


def _to_mne(result):
    axes, coords = result.axes, result.coords
    if not axes:
        raise ValueError("to_mne() needs at least one search axis.")
    for axis in axes:
        if axis not in _MNE_AXES:
            raise ValueError(f"to_mne() cannot represent a {axis!r} axis.")
    for axis in {"freq", "time"} & set(axes):
        if not np.issubdtype(coords[axis].dtype, np.number):
            raise ValueError(f"to_mne() cannot represent named {axis!r} groups.")
    if "space" in axes and result.space is None:
        raise ValueError("to_mne() needs geometry for the 'space' axis.")

    order = [axis for axis in _MNE_AXES if axis in axes]
    data = np.transpose(result.rsa, [axes.index(axis) for axis in order])
    freqs, times = coords.get("freq"), coords.get("time")
    grouped = getattr(result.space, "patches", None) is not None
    if "space" in axes and not grouped:
        if isinstance(result.space, SourceSpace):
            return _to_source_estimate(result, data, order)
        info = result.space.info
    else:
        if "space" in axes or "roi" in axes:
            names = [str(name) for name in coords[order[0]]]
        else:
            names, data = ["rsa"], data[np.newaxis]
        info = mne.create_info(names, _sfreq(times), "eeg")
    if freqs is not None and times is not None:
        return AverageTFRArray(info, data, times, freqs)
    if freqs is not None:
        return SpectrumArray(data, info, freqs)
    if times is None:
        data, times = data[:, np.newaxis], np.zeros(1)
    return mne.EvokedArray(data, info, tmin=times[0], verbose=False)


def _to_source_estimate(result, data, order):
    if "freq" in order:
        raise ValueError("to_mne() cannot represent source results with a 'freq' axis.")
    times = result.coords.get("time", np.zeros(1))
    data = data.reshape(len(data), len(times))
    is_volume = result.space.estimate_kind == "volume"
    cls = mne.VolSourceEstimate if is_volume else mne.SourceEstimate
    return cls(
        data,
        list(result.space.vertices),
        tmin=times[0],
        tstep=1 / _sfreq(times),
        subject=result.space.subject,
    )


def _sfreq(times):
    return 1 / np.median(np.diff(times)) if times is not None and len(times) > 1 else 1


def _encode(obj):
    """Convert to types that h5io can store."""
    if isinstance(obj, dict):
        return {str(key): _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj if np.issubdtype(obj.dtype, np.number) else _encode(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def _space_to_dict(space):
    if space is None:
        return {"kind": "none"}
    if isinstance(space, SensorSpace):
        return dict(
            kind="sensor",
            channel_names=list(space.channel_names),
            info_fif=_info_to_bytes(space.info),
            patches=space.patches,
        )
    return dict(
        kind="source",
        vertices={str(idx): verts for idx, verts in enumerate(space.vertices)},
        patches=space.patches,
        estimate_kind=space.estimate_kind,
        subject=space.subject,
    )


def _space_from_dict(stored):
    kind = (stored or {}).get("kind", "none")
    if kind == "none":
        return None
    patches = stored.get("patches") or None
    if kind == "sensor":
        info = _info_from_bytes(stored["info_fif"])
        return SensorSpace(tuple(stored["channel_names"]), info, patches)
    vertices = stored["vertices"]
    return SourceSpace(
        vertices=tuple(vertices[key] for key in sorted(vertices, key=int)),
        patches=patches,
        estimate_kind=stored["estimate_kind"],
        subject=stored.get("subject"),
    )


def _info_to_bytes(info):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rsa-info.fif")
        mne.io.write_info(path, info)
        return np.fromfile(path, dtype=np.uint8)


def _info_from_bytes(blob):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rsa-info.fif")
        np.asarray(blob, dtype=np.uint8).tofile(path)
        return mne.io.read_info(path, verbose=False)


def _dynamic_to_dict(dynamic):
    if dynamic is None:
        return {"present": False}
    return {"present": True, **_encode(asdict(dynamic))}


def _dynamic_from_dict(stored):
    if not stored or not stored.pop("present", False):
        return None
    return DynamicInfo(**stored)
