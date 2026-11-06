"""HDF5 persistence for `RsaResult`, via ``h5io``.

Backs :meth:`RsaResult.save` / :meth:`RsaResult.load`. The on-disk payload is a
plain nested dict (``kind`` + ``schema_version`` + ``rsa`` / ``axes`` /
``coords`` / ``info`` / ``space`` / ``dynamic``) so it round-trips through
``h5io`` -- the same library MNE uses. Two values need encoding for h5io:
non-numeric coordinates (stored as lists, since h5io rejects object arrays) and
the sensor ``mne.Info`` (stored as a FIF byte blob via ``mne.io.write_info`` so
channel types, bads, locations, and montage survive). The prepared input tensor
is never saved.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from ._types import (
    DynamicInfo,
    RsaResult,
    SensorSpace,
    SourceSpace,
    SpaceDescription,
)

_KIND = "rsa_nd.rsa_result"
_SCHEMA_VERSION = 1
_TITLE = "rsa_nd_result"
# Files written before this package was split out of mne-ez stay readable.
_LEGACY_KIND = "mne_ez.rsa_result"
_LEGACY_TITLE = "mne_ez_rsa_result"


def save_rsa_result(result: RsaResult, fname: Any, *, overwrite: bool = False) -> None:
    """Write ``result`` to an HDF5 file."""
    from h5io import write_hdf5

    payload = {
        "kind": _KIND,
        "schema_version": _SCHEMA_VERSION,
        "rsa": np.asarray(result.rsa),
        "axes": list(result.axes),
        "coords": _coords_to_storage(result.coords),
        "info": _sanitize(result.info),
        "space": _space_to_storage(result.space),
        "dynamic": _dynamic_to_storage(result.dynamic),
    }
    write_hdf5(str(fname), payload, overwrite=overwrite, title=_TITLE, slash="replace")


def load_rsa_result(fname: Any) -> RsaResult:
    """Read an `RsaResult` written by :func:`save_rsa_result`."""
    from h5io import read_hdf5

    try:
        payload = read_hdf5(str(fname), title=_TITLE, slash="replace")
    except ValueError:
        payload = read_hdf5(str(fname), title=_LEGACY_TITLE, slash="replace")
    kind = payload.get("kind")
    if kind not in (_KIND, _LEGACY_KIND):
        raise ValueError(f"{fname} is not an rsa-nd result file (kind={kind!r}).")
    return RsaResult(
        rsa=np.asarray(payload["rsa"]),
        axes=tuple(payload["axes"]),
        coords=_coords_from_storage(payload["coords"]),
        info=payload.get("info") or {},
        space=_space_from_storage(payload.get("space")),
        dynamic=_dynamic_from_storage(payload.get("dynamic")),
    )


# --------------------------------------------------------------------------- #
# Coordinate / metadata encoding
# --------------------------------------------------------------------------- #
def _coords_to_storage(coords: dict[str, np.ndarray]) -> dict[str, Any]:
    """Numeric coords stay arrays; non-numeric (names) become lists for h5io."""
    out: dict[str, Any] = {}
    for key, value in coords.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            out[str(key)] = arr
        else:
            out[str(key)] = arr.tolist()
    return out


def _coords_from_storage(stored: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in stored.items()}


def _sanitize(obj: Any) -> Any:
    """Coerce audit ``info`` into h5io-safe types (tuples / object arrays -> lists)."""
    if isinstance(obj, dict):
        return {str(key): _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            return obj
        return [_sanitize(value) for value in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# --------------------------------------------------------------------------- #
# Geometry (SpaceDescription) encoding
# --------------------------------------------------------------------------- #
def _space_to_storage(space: SpaceDescription | None) -> dict[str, Any]:
    if space is None:
        return {"kind": "none"}
    if isinstance(space, SensorSpace):
        return {
            "kind": "sensor",
            "channel_names": list(space.channel_names),
            "info_fif": _info_to_bytes(space.info),
            "patches": _patches_to_storage(space.patches),
            "is_grouped": bool(space.is_grouped),
        }
    if isinstance(space, SourceSpace):
        return {
            "kind": "source",
            "vertices": {
                str(idx): np.asarray(verts) for idx, verts in enumerate(space.vertices)
            },
            "patches": _patches_to_storage(space.patches),
            "is_grouped": bool(space.is_grouped),
            "estimate_kind": space.estimate_kind,
            "subject": space.subject,
            "subjects_dir": space.subjects_dir,
        }
    raise TypeError(f"Cannot serialize space of type {type(space).__name__}.")


def _space_from_storage(stored: dict[str, Any] | None) -> SpaceDescription | None:
    if not stored:
        return None
    kind = stored.get("kind", "none")
    if kind == "none":
        return None
    if kind == "sensor":
        return SensorSpace(
            channel_names=tuple(stored["channel_names"]),
            info=_info_from_bytes(stored.get("info_fif")),
            patches=_patches_from_storage(stored.get("patches")),
            is_grouped=bool(stored["is_grouped"]),
        )
    if kind == "source":
        verts = stored["vertices"]
        vertices = tuple(np.asarray(verts[key]) for key in sorted(verts, key=int))
        return SourceSpace(
            vertices=vertices,
            src=None,
            patches=_patches_from_storage(stored.get("patches")),
            is_grouped=bool(stored["is_grouped"]),
            estimate_kind=stored["estimate_kind"],
            subject=stored.get("subject"),
            subjects_dir=stored.get("subjects_dir"),
        )
    raise ValueError(f"Unknown stored space kind {kind!r}.")


def _patches_to_storage(
    patches: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    if patches is None:
        return None
    return {str(name): np.asarray(idx, dtype=int) for name, idx in patches.items()}


def _patches_from_storage(
    stored: dict[str, Any] | None,
) -> dict[str, np.ndarray] | None:
    if not stored:
        return None
    return {str(name): np.asarray(idx, dtype=int) for name, idx in stored.items()}


def _info_to_bytes(info: Any) -> np.ndarray | None:
    """Serialize an ``mne.Info`` to a FIF byte blob (full fidelity)."""
    if info is None:
        return None
    from mne.io import write_info

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rsa-info.fif")
        write_info(path, info)
        return np.fromfile(path, dtype=np.uint8)


def _info_from_bytes(blob: Any) -> Any:
    if blob is None:
        return None
    from mne.io import read_info

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rsa-info.fif")
        np.asarray(blob, dtype=np.uint8).tofile(path)
        return read_info(path, verbose=False)


# --------------------------------------------------------------------------- #
# Dynamic-RSA secondary outputs
# --------------------------------------------------------------------------- #
def _dynamic_to_storage(dynamic: DynamicInfo | None) -> dict[str, Any]:
    if dynamic is None:
        return {"present": False}
    return {
        "present": True,
        "mode": dynamic.mode,
        "dynamic_axis": dynamic.dynamic_axis,
        "full_rsa": np.asarray(dynamic.full_rsa),
        "full_rsa_axes": list(dynamic.full_rsa_axes),
        "full_rsa_coords": _coords_to_storage(dynamic.full_rsa_coords),
        "model_indices": _optional_array(dynamic.model_indices),
        "lags": _optional_array(dynamic.lags),
        "lag_tolerance": dynamic.lag_tolerance,
        "model_times": _optional_array(dynamic.model_times),
    }


def _dynamic_from_storage(stored: dict[str, Any] | None) -> DynamicInfo | None:
    if not stored or not stored.get("present"):
        return None
    return DynamicInfo(
        mode=stored["mode"],
        dynamic_axis=stored["dynamic_axis"],
        full_rsa=np.asarray(stored["full_rsa"]),
        full_rsa_axes=tuple(stored["full_rsa_axes"]),
        full_rsa_coords=_coords_from_storage(stored["full_rsa_coords"]),
        model_indices=_optional_array(stored.get("model_indices")),
        lags=_optional_array(stored.get("lags")),
        lag_tolerance=stored.get("lag_tolerance"),
        model_times=_optional_array(stored.get("model_times")),
    )


def _optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value)


__all__ = ["save_rsa_result", "load_rsa_result"]
