"""RSA over any EEG/MEG features and dimensions."""

from importlib.metadata import PackageNotFoundError, version

from ._data import PreparedRsaData, RsaData, SensorSpace, SourceSpace
from ._dynamic import rsa_dynamic, rsa_dynamic_windows
from ._prepare import prepare_rsa_data, source_roi_patches
from ._result import DynamicInfo, RsaResult
from ._rsa import compute_rsa, rsa_nd

try:
    __version__ = version("rsa-nd")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "DynamicInfo",
    "PreparedRsaData",
    "RsaData",
    "RsaResult",
    "SensorSpace",
    "SourceSpace",
    "compute_rsa",
    "prepare_rsa_data",
    "rsa_dynamic",
    "rsa_dynamic_windows",
    "rsa_nd",
    "source_roi_patches",
]
