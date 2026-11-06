"""RSA over any EEG/MEG features and dimensions.

The package exposes a generic real-valued N-D RSA frontend, typed data
preparation, and dynamic / source-space workflows, built on MNE-Python and
mne-rsa.
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import (
    compute_rsa,
    rsa_nd,
)
from ._dynamic import (
    rsa_dynamic,
    rsa_dynamic_windows,
)
from ._features import safe_log10_power as safe_log10_power
from ._prepare import prepare_rsa_data
from ._source import source_roi_patches
from ._types import (
    DynamicInfo,
    PreparedRsaData,
    RsaData,
    RsaResult,
)
from ._types import (
    RsaInputKind as RsaInputKind,
)
from ._types import (
    RsaTemplate as RsaTemplate,
)
from ._types import (
    SensorSpace as SensorSpace,
)
from ._types import (
    SourceSpace as SourceSpace,
)
from ._types import (
    SpaceDescription as SpaceDescription,
)

try:
    __version__ = version("rsa-nd")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "RsaData",
    "RsaResult",
    "DynamicInfo",
    "PreparedRsaData",
    "compute_rsa",
    "rsa_nd",
    "rsa_dynamic",
    "rsa_dynamic_windows",
    "prepare_rsa_data",
    "source_roi_patches",
]
