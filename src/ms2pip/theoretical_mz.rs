use std::collections::HashMap;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use rustyms::prelude::CompoundPeptidoformIon;

use crate::annotation::{parse_fragmentation_model, parse_mass_mode};
use crate::utils::{extract_charge, parse_fragment};

/// Compute theoretical m/z values for fragment ions.
///
/// Uses rustyms fragment generation, consistent with `annotate_ms2_spectra`.
/// Returns a dict mapping ion type (e.g. "b", "y", "b2") to a list of m/z
/// values ordered by ion position, length = seq_len - 1.
#[pyfunction]
pub fn ms2pip_compute_theoretical_mz(
    py: Python<'_>,
    proformas: Vec<String>,
    ion_types: Vec<String>,
    fragmentation_model: String,
    mass_mode: String,
) -> PyResult<Vec<HashMap<String, Py<PyArray1<f32>>>>> {
    let model = parse_fragmentation_model(&fragmentation_model)?;
    let mode = parse_mass_mode(&mass_mode)?;

    let results: Result<Vec<HashMap<String, Vec<f64>>>, String> = py.detach(|| {
        proformas
            .par_iter()
            .enumerate()
            .map(|(i, pf)| {
                let compound = CompoundPeptidoformIon::pro_forma(pf, None)
                    .map_err(|e| format!("ProForma at index {i}: {e}"))?;

                let charge = extract_charge(&compound).unwrap_or(1) as isize;

                let seq_len = compound
                    .peptidoforms()
                    .next()
                    .map(|pf| pf.sequence().len())
                    .unwrap_or(0);

                if seq_len < 2 {
                    return Ok(ion_types
                        .iter()
                        .map(|it| (it.clone(), Vec::new()))
                        .collect());
                }

                let n_ions = seq_len - 1;
                let frag_charge =
                    rustyms::system::isize::Charge::new::<rustyms::system::e>(charge);
                let frags = compound.generate_theoretical_fragments(frag_charge, &model);

                let mut mz_map: HashMap<String, Vec<f64>> = ion_types
                    .iter()
                    .map(|it| (it.clone(), vec![0.0; n_ions]))
                    .collect();

                for frag in &frags {
                    let (series, position, frag_charge) = match parse_fragment(frag) {
                        Some(info) => info,
                        None => continue,
                    };

                    if position < 1 || position > n_ions {
                        continue;
                    }

                    let ion_key = if frag_charge <= 1 {
                        series.to_string()
                    } else {
                        format!("{series}{frag_charge}")
                    };

                    if let Some(arr) = mz_map.get_mut(&ion_key) {
                        let idx = position - 1;
                        if let Some(mz) = frag.mz(mode) {
                            if arr[idx] == 0.0 {
                                arr[idx] = mz.value;
                            }
                        }
                    }
                }

                Ok(mz_map)
            })
            .collect()
    });

    let results = results.map_err(PyValueError::new_err)?;
    Ok(results
        .into_iter()
        .map(|mz_map| {
            mz_map
                .into_iter()
                .map(|(key, vals)| {
                    let f32_vals: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
                    (key, PyArray1::from_vec(py, f32_vals).into())
                })
                .collect()
        })
        .collect())
}
