"""Output-directory helpers for RSA example scripts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def examples_output_dir(example_file: str | Path) -> Path:
    """Return the shared ``examples/output`` directory for an example file.

    Parameters
    ----------
    example_file : str | Path
        Path to an example script, usually ``__file__``.

    Returns
    -------
    pathlib.Path
        Directory where generated example artifacts should be written.
    """

    return Path(example_file).resolve().parent / "output"


def clean_generated_outputs(
    output_dir: str | Path,
    patterns: Iterable[str],
) -> list[Path]:
    """Remove generated files in one output directory.

    Parameters
    ----------
    output_dir : str | Path
        Directory that stores generated artifacts.
    patterns : iterable of str
        Glob patterns for files that are safe to remove.

    Returns
    -------
    list of pathlib.Path
        Removed files, sorted by path.
    """

    root = Path(output_dir)
    root.mkdir(exist_ok=True)
    removed = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return sorted(removed)
