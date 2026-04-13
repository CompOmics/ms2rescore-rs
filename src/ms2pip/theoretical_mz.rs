use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use rustyms::prelude::CompoundPeptidoformIon;

use crate::annotation::{parse_fragmentation_model, parse_mass_mode};
use crate::utils::{extract_charge, parse_ion_series_and_index};

/// Map a rustyms Fragment to its ms2pip ion type string (e.g. "b", "y", "b2").
/// Returns None for non-backbone ions.
fn fragment_ion_type(frag: &rustyms::fragment::Fragment) -> Option<String> {
    let ion_str = frag.ion.to_string();
    let (series, _) = parse_ion_series_and_index(&ion_str)?;
    let charge = frag.charge.value.unsigned_abs();
    if charge <= 1 {
        Some(series.to_string())
    } else {
        Some(format!("{series}{charge}"))
    }
}

/// Extract the 1-indexed ion position from a fragment.
fn fragment_position(frag: &rustyms::fragment::Fragment) -> Option<usize> {
    let ion_str = frag.ion.to_string();
    let (_, position) = parse_ion_series_and_index(&ion_str)?;
    Some(position)
}

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
) -> PyResult<Vec<HashMap<String, Vec<f64>>>> {
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
                    let ion_type = match fragment_ion_type(frag) {
                        Some(it) if mz_map.contains_key(&it) => it,
                        _ => continue,
                    };
                    let position = match fragment_position(frag) {
                        Some(p) if p >= 1 && p <= n_ions => p,
                        _ => continue,
                    };

                    if let Some(mz) = frag.mz(mode) {
                        let idx = position - 1;
                        let arr = mz_map.get_mut(&ion_type).unwrap();
                        if arr[idx] == 0.0 {
                            arr[idx] = mz.value;
                        }
                    }
                }

                Ok(mz_map)
            })
            .collect()
    });

    results.map_err(PyValueError::new_err)
}
