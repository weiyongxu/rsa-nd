# rsa-nd Agent Guidelines

RSA over any EEG/MEG features and dimensions, built on MNE-Python and mne-rsa.

## Commands
```bash
uv sync                               # Install dependencies
uv run pytest tests/ -m "not slowtest"  # Run fast tests
uv run ruff check . && uv run ruff format --check .
```

## Standards
- Type hints required (`from __future__ import annotations`)
- NumPy-style docstrings
- Mark tests that need MNE sample data with `@pytest.mark.slowtest`
- The version comes from git tags via `hatch-vcs`; never hardcode it
- `RsaResult.load` must keep reading files labelled `mne_ez.rsa_result`
