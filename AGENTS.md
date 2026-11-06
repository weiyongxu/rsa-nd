# rsa-nd Agent Guidelines

RSA over any EEG/MEG features and dimensions, built on MNE-Python and mne-rsa.

## Commands
```bash
uv sync                               # Install dependencies
uv run pytest tests/ -m "not slowtest"  # Run fast tests
uv run ruff check . && uv run ruff format --check .
```

## Standards
- Follow NumPy and MNE-Python: NumPy-style docstrings with types in the docstring,
  no type hints in signatures, short functions, comments only where needed
- Private modules (`_rsa.py`); the public API is listed in `__init__.py`
- Mark tests that need MNE sample data with `@pytest.mark.slowtest`
- `tests/test_combinations.py` checks results against a brute-force reference; set
  `RSA_ND_ALL_CASES=1` to run all of its roughly 35,000 cases
- The version comes from git tags via `hatch-vcs`; never hardcode it
- `RsaResult.load` must keep reading files labelled `mne_ez.rsa_result`
