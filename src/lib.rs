mod file_types;
mod ms2_spectrum;
mod parse_mzdata;
mod parse_timsrust;
mod precursor;

use std::collections::HashMap;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use file_types::{match_file_type, SpectrumFileType};
use ms2_spectrum::MS2Spectrum;
use precursor::Precursor;

/// Check if spectrum path matches a supported file type.
#[pyfunction]
pub fn is_supported_file_type(spectrum_path: String) -> bool {
    let file_type = match_file_type(&spectrum_path);

    !matches!(file_type, SpectrumFileType::Unknown)
}

/// Get mapping of spectrum identifiers to precursor information.
#[pyfunction]
pub fn get_precursor_info(py: Python<'_>, spectrum_path: String) -> PyResult<HashMap<String, Precursor>> {
    let file_type = match_file_type(&spectrum_path);

    let precursors = py.allow_threads(|| match file_type {
        SpectrumFileType::MascotGenericFormat
        | SpectrumFileType::MzML
        | SpectrumFileType::MzMLb
        | SpectrumFileType::ThermoRaw => parse_mzdata::parse_precursor_info(&spectrum_path),
        SpectrumFileType::BrukerRaw => parse_timsrust::parse_precursor_info(&spectrum_path),
        SpectrumFileType::Unknown => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Unsupported file type",
        )),
    });

    match precursors {
        Ok(precursors) => Ok(precursors),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}

/// Get MS2 spectra from a spectrum file.
#[pyfunction]
pub fn get_ms2_spectra(py: Python<'_>, spectrum_path: String) -> PyResult<Vec<ms2_spectrum::MS2Spectrum>> {
    let file_type = match_file_type(&spectrum_path);

    let spectra = py.allow_threads(|| match file_type {
        SpectrumFileType::MascotGenericFormat
        | SpectrumFileType::MzML
        | SpectrumFileType::MzMLb
        | SpectrumFileType::ThermoRaw => parse_mzdata::read_ms2_spectra(&spectrum_path),
        SpectrumFileType::BrukerRaw => parse_timsrust::read_ms2_spectra(&spectrum_path),
        SpectrumFileType::Unknown => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Unsupported file type",
        )),
    });

    match spectra {
        Ok(spectra) => Ok(spectra),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ms2rescore_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Precursor>()?;
    m.add_class::<MS2Spectrum>()?;
    m.add_function(wrap_pyfunction!(is_supported_file_type, m)?)?;
    m.add_function(wrap_pyfunction!(get_precursor_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_ms2_spectra, m)?)?;
    Ok(())
}
