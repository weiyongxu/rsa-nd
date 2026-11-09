# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

A cleanup with no change to any result.

### Documentation
- The examples are numbered in reading order, from `plot_01_overview.py` to
  `plot_06_group_stats.py`, so the gallery shows them in the README's order.
- The API pages show the parameter types written in the docstrings, such as
  `mne.Info`, instead of the bare annotations of the data classes, such as
  `object`: the docs no longer use `sphinx-autodoc-typehints`.

### Internal
- Every private helper has a one-line docstring.
- The two checks of model RDMs have different names (`_check_model_rdms` and
  `_check_dynamic_rdms`), and the radii are checked once, not per axis type.
- Deprecation and future warnings fail the tests, as in mne-ez, and the
  pre-commit hooks use the ruff and codespell versions of the lock file.
- CI has three test jobs instead of six: `lint` runs the pre-commit hooks,
  `test` runs every test with Python 3.13, and `oldest` runs the fast tests
  with Python 3.10 and MNE 1.8. The GitHub actions are their current, Node 24
  versions, and the tests can also be started by hand from the Actions tab.

## [0.1.0] - 2026-11-08

First standalone version, split out of the `rsa` module of `mne-ez`. Code that
imported `mne_ez.rsa` should import `rsa_nd` instead. Result files written by
`mne-ez` still load.

### Documentation
- The README and docs explain the gap the package fills and how it fits a real
  MNE and mne-rsa analysis.
- A schematic shows how the package works and what the choice of axes changes.
- A new overview example searches brain parcels, frequency bands and time for two
  models in one call, and provides the headline figure. The docs gallery now runs
  the examples and shows their figures.
- Each gallery figure now makes one point and marks the known answer: the
  simulated effect, the true lag, or the hemisphere that the stimuli reach. Bands
  keep the same colours in every figure, and lines carry their own labels instead
  of a legend.
- A new group example runs RSA for 10 participants of the LIMO EEG dataset, with
  a face model whose differences grow with image clarity, controlled for phase
  coherence and clarity. It then tests the scores across participants with a
  cluster permutation test over channels and time. The docs build downloads and
  caches these participants.

### Changed
- Rewritten for simplicity, in the style of NumPy and MNE-Python: about a third of
  the previous code, in five modules.
- `RsaData` is public. `SpaceDescription`, `RsaInputKind`, `RsaTemplate` and
  `safe_log10_power` are gone.
- `rsa_nd` no longer takes `feature_axes` or `sample_axis`: every axis other than
  `item` provides features.
- A radius of `None` now pools the axis and leaves it out of the result, instead
  of keeping a one-element axis.
- `PreparedRsaData` no longer holds `template`, `time_axis`, `freq_axis`,
  `time_is_grouped`, `freq_is_grouped` or `axis_groups`.
- `tfr=` passes its arguments straight to `tfr_array_morlet`, so `n_cycles` now
  defaults to MNE's 7 cycles rather than 1.
- `mne-rsa` is limited to 1.x, because `rsa-nd` uses a private helper from it.

### Fixed
- Radii that fall exactly on a sampling step no longer drop neighbours at some
  positions because of rounding (0.06 - 0.02 is slightly above 0.04).
- `standardize="features"` now runs after ROI aggregation, so the final features
  really are standardized.
- `decim` in `tfr=` no longer breaks the time axis.
- `rsa_dynamic_windows` keeps channel or vertex geometry and item labels.
- Named space groups that are not searched now raise an error instead of being
  ignored.
- `to_mne()` gives a clear error whenever a result has no matching MNE container,
  and now also converts source results without a time axis.
- Lists of items must now share times and frequencies, not just channels.
- `n_folds` splitters and callables no longer break `save()`.
- Labels spanning both hemispheres (`BiHemiLabel`) now work with `space=`.

[Unreleased]: https://github.com/weiyongxu/rsa-nd/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/weiyongxu/rsa-nd/releases/tag/v0.1.0
