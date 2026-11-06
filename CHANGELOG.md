# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First standalone version, split out of the `rsa` module of `mne-ez`. Code that
imported `mne_ez.rsa` should import `rsa_nd` instead. Result files written by
`mne-ez` still load.

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
