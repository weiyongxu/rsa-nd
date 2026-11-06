"""Containers for prepared RSA data."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist, squareform


def _check_axes(shape, axes, coords):
    """Check axis names and coordinates, filling missing ones with integers."""
    if len(shape) != len(axes):
        raise ValueError(f"data has {len(shape)} dimensions but {len(axes)} axes.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes must be unique, got {list(axes)}.")
    unknown = sorted(set(coords) - set(axes))
    if unknown:
        raise ValueError(f"Unknown coordinate keys: {unknown}.")
    checked = {}
    for axis, n_values in zip(axes, shape):
        values = np.asarray(coords.get(axis, np.arange(n_values)))
        if len(values) != n_values:
            raise ValueError(
                f"Axis {axis!r} has {n_values} values but {len(values)} coordinates."
            )
        checked[axis] = values
    return checked


@dataclass
class RsaData:
    """Real-valued data with named axes.

    Parameters
    ----------
    data : array-like
        The data. Complex values are not allowed.
    axes : sequence of str
        One name per dimension, for example ``("item", "space", "time")``. RSA
        needs an ``"item"`` axis.
    coords : dict | None
        Coordinate values per axis. Missing axes get integer coordinates.
    """

    data: np.ndarray
    axes: tuple
    coords: dict = field(default_factory=dict)

    def __post_init__(self):
        data = np.asarray(self.data)
        if np.iscomplexobj(data):
            raise ValueError(
                "RsaData must be real-valued. Use prepare_rsa_data() with "
                "tfr_feature= to project complex values first."
            )
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(float)
        self.data = data
        self.axes = tuple(self.axes)
        self.coords = _check_axes(data.shape, self.axes, self.coords or {})


@dataclass
class SensorSpace:
    """Sensor geometry for spatial searchlights.

    Parameters
    ----------
    channel_names : tuple of str
        The channel names, in data order.
    info : mne.Info
        Measurement info holding the channel locations.
    patches : dict | None
        Channel indices per named group, or ``None`` for single channels.
    """

    channel_names: tuple
    info: object
    patches: dict | None = None

    @property
    def names(self):
        """Name of each spatial element."""
        return np.asarray(self.channel_names, dtype=object)

    def _distances(self, radius):
        pos = np.array([ch["loc"][:3] for ch in self.info["chs"]])
        if not np.isfinite(pos).all() or not pos.any(axis=1).all():
            raise ValueError("A space radius needs a location for every channel.")
        return squareform(pdist(pos))


@dataclass
class SourceSpace:
    """Source geometry for spatial searchlights.

    Parameters
    ----------
    vertices : tuple of ndarray
        Vertex numbers per hemisphere (one array for volume source spaces).
    src : mne.SourceSpaces | None
        Source space, needed for a spatial radius search.
    patches : dict | None
        Row indices per named group, or ``None`` for single vertices.
    estimate_kind : str
        Either ``"surface"`` or ``"volume"``.
    subject : str | None
        The subject name.
    """

    vertices: tuple
    src: object = None
    patches: dict | None = None
    estimate_kind: str = "surface"
    subject: str | None = None

    @property
    def names(self):
        """Name of each spatial element."""
        return np.arange(sum(len(verts) for verts in self.vertices))

    def _distances(self, radius):
        from mne_rsa.source_level import _get_distance_matrix

        if self.src is None:
            raise ValueError("A source space radius needs src= in prepare_rsa_data().")
        return _get_distance_matrix(self.src, dist_lim=radius)


@dataclass
class PreparedRsaData:
    """Data ready for :func:`rsa_nd.rsa_nd`.

    Parameters
    ----------
    rsa_data : RsaData
        The named feature tensor.
    input_kind : str
        The kind of input it came from.
    item_labels : ndarray | None
        One label per item. Items sharing a label are repeats of one item.
    space : SensorSpace | SourceSpace | None
        The geometry of the ``"space"`` axis.
    element_groups : dict
        For each axis searched by named group, the member indices per group.
    """

    rsa_data: RsaData
    input_kind: str = "array"
    item_labels: np.ndarray | None = None
    space: SensorSpace | SourceSpace | None = None
    element_groups: dict = field(default_factory=dict)

    def __post_init__(self):
        axes = self.rsa_data.axes
        if self.space is not None and "space" not in axes:
            raise ValueError("space is set but the data has no 'space' axis.")
        if self.item_labels is not None:
            n_items = self.rsa_data.data.shape[axes.index("item")]
            if len(self.item_labels) != n_items:
                raise ValueError(
                    f"Got {len(self.item_labels)} item_labels for {n_items} items."
                )
