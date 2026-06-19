pub mod file_types;
mod mzdata;
mod timsrust;

use std::collections::HashMap;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

use crate::types::ms2_spectrum::MS2Spectrum;
use crate::types::precursor::Precursor;
use file_types::{match_file_type, SpectrumFileType};

/// Check if spectrum path matches a supported file type.
#[pyfunction]
pub fn is_supported_file_type(spectrum_path: String) -> bool {
    let file_type = match_file_type(&spectrum_path);

    !matches!(file_type, SpectrumFileType::Unknown)
}

/// Get mapping of spectrum identifiers to precursor information.
#[pyfunction]
pub fn get_precursor_info(
    py: Python<'_>,
    spectrum_path: String,
) -> PyResult<HashMap<String, Precursor>> {
    let file_type = match_file_type(&spectrum_path);

    // Can't raise PyValueError below in a detached thread, so check here first
    if matches!(file_type, SpectrumFileType::Unknown) {
        return Err(PyValueError::new_err("Unsupported file type"));
    }

    let precursors = py.detach(|| match file_type {
        SpectrumFileType::MascotGenericFormat
        | SpectrumFileType::MzML
        | SpectrumFileType::MzMLb
        | SpectrumFileType::ThermoRaw => mzdata::parse_precursor_info(&spectrum_path),
        SpectrumFileType::BrukerRaw => timsrust::parse_precursor_info(&spectrum_path),
        SpectrumFileType::Unknown => unreachable!("handled above"),
    });

    match precursors {
        Ok(precursors) => Ok(precursors),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}

/// Get MS2 spectra from a spectrum file.
#[pyfunction]
pub fn get_ms2_spectra(
    py: Python<'_>,
    spectrum_path: String,
) -> PyResult<Vec<MS2Spectrum>> {
    let file_type = match_file_type(&spectrum_path);

    // Can't raise PyValueError below in a detached thread, so check here first
    if matches!(file_type, SpectrumFileType::Unknown) {
        return Err(PyValueError::new_err("Unsupported file type"));
    }

    let spectra = py.detach(|| match file_type {
        SpectrumFileType::MascotGenericFormat
        | SpectrumFileType::MzML
        | SpectrumFileType::MzMLb
        | SpectrumFileType::ThermoRaw => mzdata::read_ms2_spectra(&spectrum_path),
        SpectrumFileType::BrukerRaw => timsrust::read_ms2_spectra(&spectrum_path),
        SpectrumFileType::Unknown => unreachable!("handled above"),
    });

    match spectra {
        Ok(spectra) => Ok(spectra),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}
