# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- First standalone version, split out of the `rsa` module of `mne-ez`. Code that
  imported `mne_ez.rsa` should import `rsa_nd` instead.
- `RsaData` is now part of the public API.

### Changed
- `mne-rsa` is limited to 1.x because `rsa-nd` uses private helpers from it.
- New result files are labelled `rsa_nd`. `RsaResult.load()` still reads files
  written by `mne-ez`.
