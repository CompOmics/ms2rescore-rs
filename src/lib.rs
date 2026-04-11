mod file_types;
mod isotope_envelope;
mod ms2_spectrum;
mod parse_mzdata;
mod parse_timsrust;
mod precursor;

use std::collections::HashMap;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

use file_types::{match_file_type, SpectrumFileType};
use isotope_envelope::IsotopeEnvelope;
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
pub fn get_precursor_info(spectrum_path: String) -> PyResult<HashMap<String, Precursor>> {
    let file_type = match_file_type(&spectrum_path);

    let precursors = match file_type {
        SpectrumFileType::MascotGenericFormat
        | SpectrumFileType::MzML
        | SpectrumFileType::MzMLb
        | SpectrumFileType::ThermoRaw => parse_mzdata::parse_precursor_info(&spectrum_path),
        SpectrumFileType::BrukerRaw => parse_timsrust::parse_precursor_info(&spectrum_path),
        SpectrumFileType::Unknown => return Err(PyValueError::new_err("Unsupported file type")),
    };

    match precursors {
        Ok(precursors) => Ok(precursors),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}

/// Get MS2 spectra from a spectrum file.
#[pyfunction]
pub fn get_ms2_spectra(spectrum_path: String) -> PyResult<Vec<ms2_spectrum::MS2Spectrum>> {
    let file_type = match_file_type(&spectrum_path);

    let spectra = match file_type {
        SpectrumFileType::MascotGenericFormat
        | SpectrumFileType::MzML
        | SpectrumFileType::MzMLb
        | SpectrumFileType::ThermoRaw => parse_mzdata::read_ms2_spectra(&spectrum_path),
        SpectrumFileType::BrukerRaw => parse_timsrust::read_ms2_spectra(&spectrum_path),
        SpectrumFileType::Unknown => return Err(PyValueError::new_err("Unsupported file type")),
    };

    match spectra {
        Ok(spectra) => Ok(spectra),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}

/// Extract isotope envelopes from MS1 scans for all MS2 precursors.
///
/// For each MS2 spectrum in the file, finds the preceding MS1 scan and
/// extracts the isotope envelope (M0, M1, M2, ...) around the precursor m/z.
///
/// Args:
///     spectrum_path: Path to spectrum file (.RAW, .mzML, etc.)
///     n_peaks: Number of isotope peaks to extract (default: 4)
///     ppm_tolerance: Mass tolerance for peak matching in ppm (default: 10.0)
///
/// Returns:
///     List of IsotopeEnvelope objects, one per MS2 spectrum.
#[pyfunction]
#[pyo3(signature = (spectrum_path, n_peaks=4, ppm_tolerance=20.0))]
pub fn get_isotope_envelopes(
    spectrum_path: String,
    n_peaks: usize,
    ppm_tolerance: f64,
) -> PyResult<Vec<IsotopeEnvelope>> {
    let file_type = match_file_type(&spectrum_path);

    let envelopes = match file_type {
        SpectrumFileType::MascotGenericFormat
        | SpectrumFileType::MzML
        | SpectrumFileType::MzMLb
        | SpectrumFileType::ThermoRaw => {
            isotope_envelope::extract_envelopes_mzdata(&spectrum_path, n_peaks, ppm_tolerance)
        }
        _ => return Err(PyValueError::new_err(
            "Isotope envelope extraction only supported for mzML, mzMLb, and Thermo .RAW files"
        )),
    };

    match envelopes {
        Ok(envelopes) => Ok(envelopes),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ms2rescore_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Precursor>()?;
    m.add_class::<MS2Spectrum>()?;
    m.add_class::<IsotopeEnvelope>()?;
    m.add_function(wrap_pyfunction!(is_supported_file_type, m)?)?;
    m.add_function(wrap_pyfunction!(get_precursor_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_ms2_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(get_isotope_envelopes, m)?)?;
    Ok(())
}
