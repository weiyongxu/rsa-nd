# rsa-nd Agent Guidelines

RSA over any EEG/MEG features and dimensions, built on MNE-Python and mne-rsa.

## Commands
```bash
uv sync                               # Install dependencies
uv run pytest tests/ -m "not slowtest"  # Run fast tests
uv run pytest tests/ -m slowtest      # Run tests that need MNE sample data
uv run ruff check . && uv run ruff format --check .
uv run sphinx-build -b html docs/source docs/build/html  # Runs the examples
```

The README figures are committed in `docs/source/_static/`. Regenerate them after
changing `docs/make_concept_figure.py` or `examples/plot_00_overview.py`:
```bash
uv run python docs/make_concept_figure.py
MPLBACKEND=Agg uv run python -c "import runpy; runpy.run_path('examples/plot_00_overview.py')['fig'].savefig('docs/source/_static/overview.png', dpi=200)"
```

## Standards
- Follow NumPy and MNE-Python: NumPy-style docstrings with types in the docstring,
  no type hints in signatures, short functions, comments only where needed
- Private modules (`_rsa.py`); the public API is listed in `__init__.py`
- Mark tests that need MNE sample data with `@pytest.mark.slowtest`; CI runs them
  in the `slow` job of `.github/workflows/test.yml`
- `tests/test_combinations.py` checks results against a brute-force reference; set
  `RSA_ND_ALL_CASES=1` to run all of its roughly 35,000 cases
- The version comes from git tags via `hatch-vcs`; never hardcode it
- `RsaResult.load` must keep reading files labelled `mne_ez.rsa_result`
