"""Pytest configuration shared across the rsa-nd test suite.

The RSA demo-pointed tests (``test_rsa_*_demo.py``) import the matching demo
script from ``examples/`` as a Python module. ``examples/`` is a flat script
directory (no ``__init__.py``, intentional for sphinx-gallery), so we expose
it on ``sys.path`` here.

We also guard the real-3D-rendering tests: VTK/OpenGL ``Brain`` rendering
hard-aborts (native ``SIGABRT``, uncatchable from Python) on GPU-less hosts,
which kills the whole pytest session. ``render3d``-marked tests are skipped
when a one-off, subprocess-isolated render probe fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_examples_str = str(_EXAMPLES_DIR)
if _EXAMPLES_DIR.is_dir() and _examples_str not in sys.path:
    sys.path.insert(0, _examples_str)


# A minimal render through the pyvistaqt/Qt path that MNE's ``Brain`` uses
# (plain pyvista off-screen succeeds via software GL even where the Qt
# OpenGL widget aborts, so it is not a faithful probe). Run in a
# subprocess so the native abort cannot take down the test session.
_RENDER_PROBE = (
    "import pyvista as pv, pyvistaqt;"
    "pv.OFF_SCREEN=True;"
    "bp=pyvistaqt.BackgroundPlotter(show=False, off_screen=True);"
    "bp.add_mesh(pv.Sphere());"
    "bp.close()"
)


def _can_render_3d() -> bool:
    """True iff a minimal off-screen 3D render succeeds in a subprocess."""

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _RENDER_PROBE],
            capture_output=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


_HAS_3D: bool | None = None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "render3d: needs a working 3D/OpenGL backend (real VTK Brain "
        "render); auto-skipped on hosts where the render probe aborts.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    global _HAS_3D
    if not any(item.get_closest_marker("render3d") for item in items):
        return
    if _HAS_3D is None:
        _HAS_3D = _can_render_3d()
    if _HAS_3D:
        return
    skip = pytest.mark.skip(
        reason="no working 3D/OpenGL backend (render probe aborted)"
    )
    for item in items:
        if item.get_closest_marker("render3d"):
            item.add_marker(skip)
