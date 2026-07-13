mod annotation;
mod io;
mod ms2pip;
mod scoring;
mod types;
mod utils;

use pyo3::prelude::*;

use types::annotation::{AnnotatedMS2Spectrum, FragmentAnnotation};
use types::ms2_spectrum::MS2Spectrum;
use types::precursor::Precursor;

/// A Python module implemented in Rust.
#[pymodule]
fn ms2rescore_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Precursor>()?;
    m.add_class::<MS2Spectrum>()?;
    m.add_class::<FragmentAnnotation>()?;
    m.add_class::<AnnotatedMS2Spectrum>()?;
    m.add_function(wrap_pyfunction!(io::is_supported_file_type, m)?)?;
    m.add_function(wrap_pyfunction!(io::get_precursor_info, m)?)?;
    m.add_function(wrap_pyfunction!(io::get_ms2_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(annotation::annotate_ms2_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(scoring::ms2::score_ms2_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(
        scoring::spectrum_prediction::ms2pip_features_from_prediction_peak_arrays,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ms2pip::feature_vectors::ms2pip_compute_features,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ms2pip::theoretical_mz::ms2pip_compute_theoretical_mz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ms2pip::targets::ms2pip_extract_targets,
        m
    )?)?;
    Ok(())
}
