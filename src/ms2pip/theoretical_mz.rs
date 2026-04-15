use std::collections::HashMap;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use rustyms::prelude::CompoundPeptidoformIon;

use crate::annotation::{parse_fragmentation_model, parse_mass_mode};
use crate::utils::{build_theoretical_fragments, extract_charge};

#[derive(Clone)]
struct IonTypeSpec {
    key: String,
    series: char,
    charge: usize,
}

fn parse_ion_type_spec(ion_type: &str) -> Option<IonTypeSpec> {
    let mut chars = ion_type.chars();
    let series = chars.next()?;
    if !matches!(series, 'a' | 'b' | 'c' | 'x' | 'y' | 'z') {
        return None;
    }

    let suffix: String = chars.collect();
    let charge = if suffix.is_empty() {
        1
    } else {
        suffix.parse().ok()?
    };

    Some(IonTypeSpec {
        key: ion_type.to_string(),
        series,
        charge,
    })
}

/// Compute theoretical m/z values for fragment ions.
///
/// Uses rustyms fragment generation, consistent with `annotate_ms2_spectra`.
/// Returns a dict mapping ion type (e.g. "b", "y", "b2") to a numpy float32
/// array of m/z values ordered by ion position, length = seq_len - 1.
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
    let ion_specs: Vec<IonTypeSpec> = ion_types
        .iter()
        .map(|ion_type| {
            parse_ion_type_spec(ion_type).ok_or_else(|| {
                PyValueError::new_err(format!("Unsupported ion type: {ion_type}"))
            })
        })
        .collect::<PyResult<_>>()?;

    let ion_index: HashMap<(char, usize), usize> = ion_specs
        .iter()
        .enumerate()
        .map(|(idx, spec)| ((spec.series, spec.charge), idx))
        .collect();

    let results: Result<Vec<Vec<Vec<f32>>>, String> = py.detach(|| {
        proformas
            .par_iter()
            .enumerate()
            .map(|(i, pf)| {
                let peptidoform = CompoundPeptidoformIon::pro_forma(pf, None)
                    .map_err(|e| format!("ProForma at index {i}: {e}"))?;

                let charge = extract_charge(&peptidoform).unwrap_or(1) as isize;

                let seq_len = peptidoform
                    .peptidoforms()
                    .next()
                    .map(|pf| pf.sequence().len())
                    .unwrap_or(0);

                if seq_len < 2 {
                    return Ok(vec![Vec::new(); ion_specs.len()]);
                }

                let n_ions = seq_len - 1;
                let frag_charge =
                    rustyms::system::isize::Charge::new::<rustyms::system::e>(charge);
                let fragments =
                    build_theoretical_fragments(&peptidoform, frag_charge, &model, mode);

                let mut mz_arrays = vec![vec![0.0_f32; n_ions]; ion_specs.len()];

                for frag in &fragments {
                    if frag.position < 1 || frag.position > n_ions {
                        continue;
                    }

                    if let Some(&array_idx) = ion_index.get(&(frag.series, frag.charge)) {
                        let pos_idx = frag.position - 1;
                        if mz_arrays[array_idx][pos_idx] == 0.0 {
                            mz_arrays[array_idx][pos_idx] = frag.mz as f32;
                        }
                    }
                }

                Ok(mz_arrays)
            })
            .collect()
    });

    let results = results.map_err(PyValueError::new_err)?;
    Ok(results
        .into_iter()
        .map(|mz_arrays| {
            ion_specs
                .iter()
                .cloned()
                .zip(mz_arrays)
                .map(|(spec, vals)| {
                    (spec.key, PyArray1::from_vec(py, vals).into())
                })
                .collect()
        })
        .collect())
}
