"""Guard the private mne-rsa helpers that ``rsa_nd`` depends on.

The RSA module reaches into a few leading-underscore mne-rsa internals to make
its sensor / source searchlight boundaries match mne-rsa's own searchlight
exactly (see ``_types.SensorSpace`` / ``SourceSpace`` and
``_source``). Those names carry no deprecation guarantee, and mne-rsa
is intentionally left unpinned, so this test fails loudly in CI if an upgrade
moves or renames them -- rather than surfacing as an ImportError inside a
user's geodesic / radius searchlight call.
"""

from __future__ import annotations

import importlib


def test_private_mne_rsa_helpers_are_importable_and_callable() -> None:
    expected = {
        "mne_rsa.searchlight": ("_get_in_radius",),
        "mne_rsa.source_level": ("_get_distance_matrix", "vertex_selection_to_indices"),
    }
    for module_name, symbols in expected.items():
        module = importlib.import_module(module_name)
        for symbol in symbols:
            assert hasattr(module, symbol), (
                f"{module_name}.{symbol} is gone -- the installed mne-rsa moved "
                "a private helper rsa_nd relies on; update the RSA module "
                "or pin a compatible mne-rsa."
            )
            assert callable(getattr(module, symbol)), (
                f"{module_name}.{symbol} is no longer callable."
            )
