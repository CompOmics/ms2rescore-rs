# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ms2dip_extract_intensity_grid`: extract fixed-size, NaN-masked intensity grids from
  annotated spectra for MS²DIP-style CNN training targets (caller-defined ion-type/charge
  row layout, max-of-collisions intensity fill)

## [0.5.0] - 2026-07-13

First stable 0.5.0 release. No functional changes since 0.5.0-beta.1.

## [0.5.0-beta.1] - 2026-06-18

### Added
- `annotate_ms2_spectra`: annotate MS2 spectra with theoretical fragment ions
  (tolerance in Da or ppm, configurable fragmentation model and mass mode)
- `score_ms2_spectra`: compute per-ion-series MS2 scoring features (matched
  ions, longest sequence, explained intensity) with optional hyperscore
- `ms2pip_compute_features`: compute 139 MS2PIP XGBoost feature vectors per
  cleavage site from ProForma peptide strings
- `ms2pip_compute_theoretical_mz`: compute theoretical fragment m/z arrays
  (numpy float32) for arbitrary ion types and charges
- `ms2pip_extract_targets`: extract per-ion-type observed intensity arrays
  from annotated spectra for model training
- `AnnotatedMS2Spectrum` and `FragmentAnnotation` types with pickle support
- Codebase restructured into `types/`, `io/`, `scoring/`, and `ms2pip/` modules
- ARM64 wheel builds for Windows and macOS

## [0.4.3] - 2025-09-30

### Fixed
- Crash on empty spectra (contributed by @di-hardt)

### Changed
- Updated mzdata to 0.59.2

## [0.4.2] - 2025-04-01

### Changed
- Updated mzdata to 0.48.3

## [0.4.1] - 2025-01-15

### Fixed
- Thermo RAW files without pre-centroided peaks now read correctly (vendor
  centroiding is now requested explicitly)

### Changed
- Updated timsrust to 0.4

## [0.4.0-1] - 2024-11-20

### Changed
- Added Python 3.13 wheel builds

## [0.4.0] - 2024-10-24

### Added
- Thermo RAW file support via mzdata
- `is_supported_file_type` utility function

### Changed
- Dropped x86 Linux wheel builds
- Updated mzdata

## [0.3.0] - 2024-07-17

### Changed
- Updated timsrust to 0.3.0

## [0.2.1] - 2024-07-01

### Added
- Tests for `get_ms2_spectra`

### Changed
- Updated mzdata to 0.20.0

## [0.2.0] - 2024-04-08

### Added
- `get_ms2_spectra`: read MS2 spectra from spectrum files

## [0.1.0] - 2024-04-05

### Added
- Initial release: `get_precursor_info` for mzML, MGF, and Bruker TDF files

[Unreleased]: https://github.com/compomics/ms2rescore-rs/compare/v0.5.0-beta.1...HEAD
[0.5.0-beta.1]: https://github.com/compomics/ms2rescore-rs/compare/v0.4.3...v0.5.0-beta.1
[0.4.3]: https://github.com/compomics/ms2rescore-rs/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/compomics/ms2rescore-rs/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/compomics/ms2rescore-rs/compare/v0.4.0-1...v0.4.1
[0.4.0-1]: https://github.com/compomics/ms2rescore-rs/compare/v0.4.0...v0.4.0-1
[0.4.0]: https://github.com/compomics/ms2rescore-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/compomics/ms2rescore-rs/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/compomics/ms2rescore-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/compomics/ms2rescore-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/compomics/ms2rescore-rs/releases/tag/v0.1.0
