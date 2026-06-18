# ms2rescore-rs

Rust core for [MS²Rescore](https://github.com/compomics/ms2rescore) and
[MS²PIP](https://github.com/compomics/ms2pip), exposed as a Python package via PyO3.

Provides fast, parallelized implementations of spectrum reading, fragment ion annotation, MS2PIP feature extraction, and MS2 scoring features used during PSM rescoring.

## Installation

```sh
pip install ms2rescore-rs
```

## Functions

| Function | Description |
|---|---|
| `get_precursor_info(path)` | Read precursor metadata from mzML, MGF, or Bruker raw files |
| `get_ms2_spectra(path)` | Read MS2 spectra from mzML, MGF, or Bruker raw files |
| `annotate_ms2_spectra(spectra, proformas, ...)` | Match observed peaks to theoretical fragment ions |
| `score_ms2_spectra(spectra, seq_lens, ...)` | Compute per-series matched-ion scoring features |
| `ms2pip_compute_features(proformas, ...)` | Compute 139 MS2PIP XGBoost feature vectors per cleavage site |
| `ms2pip_compute_theoretical_mz(proformas, ...)` | Compute theoretical fragment m/z arrays |
| `ms2pip_extract_targets(spectra, intensities, ...)` | Extract observed intensity targets for model training |

Supported input formats: mzML, mzMLb, MGF, Thermo RAW, Bruker TDF/TIMS.

ProForma 2.0 notation is used throughout for peptide sequences and modifications.

## Development

Requires Rust and [maturin](https://github.com/PyO3/maturin).

```sh
uv run maturin develop
```
