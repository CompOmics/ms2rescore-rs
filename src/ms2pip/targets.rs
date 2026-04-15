use std::collections::HashMap;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::types::annotation::AnnotatedMS2Spectrum;

/// Floor value for unmatched ions: log2(0.001)
const LOG2_FLOOR: f32 = -9.965_784;

/// Extract per-ion-type observed intensity arrays from annotated spectra.
///
/// For each spectrum, maps peak annotations + observed intensities onto
/// per-ion-type arrays of length `seq_len - 1`, taking the max intensity
/// when multiple peaks match the same theoretical ion position.
///
/// Sequence length is derived from the max annotation position in each spectrum.
/// Unmatched positions are filled with `log2(0.001)`.
#[pyfunction]
pub fn ms2pip_extract_targets(
    py: Python<'_>,
    annotated_spectra: Vec<Py<AnnotatedMS2Spectrum>>,
    intensities: Vec<Py<PyArray1<f32>>>,
    ion_types: Vec<String>,
) -> PyResult<Vec<HashMap<String, Py<PyArray1<f32>>>>> {
    let n = annotated_spectra.len();
    if intensities.len() != n {
        return Err(PyException::new_err(
            "Input arrays must have identical length: annotated_spectra, intensities",
        ));
    }

    // Extract owned data under the GIL
    struct OwnedData {
        peak_annotations: Vec<Vec<(String, usize)>>, // (ion_key, 0-indexed position)
        intensities: Vec<f32>,
        n_ions: usize,
    }

    let mut owned: Vec<OwnedData> = Vec::with_capacity(n);
    for i in 0..n {
        let spec_ref = annotated_spectra[i].bind(py);
        let spec = spec_ref.borrow();

        let intensities_arr = intensities[i].bind(py);
        let intensities_vec = unsafe { intensities_arr.as_slice()? }.to_vec();

        // Derive n_ions (seq_len - 1) from max annotation position
        let mut max_pos: usize = 0;
        let peak_annotations: Vec<Vec<(String, usize)>> = spec
            .peak_annotations
            .iter()
            .map(|annotations| {
                annotations
                    .iter()
                    .filter_map(|ann| {
                        if ann.position < 1 {
                            return None;
                        }
                        let ion_key = if ann.charge <= 1 {
                            ann.series.clone()
                        } else {
                            format!("{}{}", ann.series, ann.charge)
                        };
                        if !ion_types.contains(&ion_key) {
                            return None;
                        }
                        let idx = ann.position - 1;
                        if ann.position > max_pos {
                            max_pos = ann.position;
                        }
                        Some((ion_key, idx))
                    })
                    .collect()
            })
            .collect();

        owned.push(OwnedData {
            peak_annotations,
            intensities: intensities_vec,
            n_ions: max_pos,
        });
    }

    // Release GIL and process in parallel
    let results: Vec<HashMap<String, Vec<f32>>> = py.detach(|| {
        owned
            .into_par_iter()
            .map(|data| {
                let mut target_map: HashMap<String, Vec<f32>> = ion_types
                    .iter()
                    .map(|it| (it.clone(), vec![LOG2_FLOOR; data.n_ions]))
                    .collect();

                for (peak_idx, annotations) in data.peak_annotations.iter().enumerate() {
                    let intensity = data.intensities.get(peak_idx).copied().unwrap_or(0.0);
                    for (ion_key, idx) in annotations {
                        if let Some(arr) = target_map.get_mut(ion_key) {
                            if *idx < arr.len() {
                                arr[*idx] = arr[*idx].max(intensity);
                            }
                        }
                    }
                }

                target_map
            })
            .collect()
    });

    // Convert to numpy arrays
    Ok(results
        .into_iter()
        .map(|target_map| {
            target_map
                .into_iter()
                .map(|(key, vals)| (key, PyArray1::from_vec(py, vals).into()))
                .collect()
        })
        .collect())
}
