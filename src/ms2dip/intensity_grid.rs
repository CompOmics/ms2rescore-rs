use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::types::annotation::AnnotatedMS2Spectrum;

/// Owned, PyO3-free per-spectrum input to `build_intensity_grid`.
struct OwnedSpec {
    peak_annotations: Vec<Vec<(String, usize, usize)>>, // (series, position, charge)
    intensity: Vec<f32>,
    seq_len: usize,
    precursor_charge: usize,
}

/// Build one flattened `(n_rows, max_peptide_length)` intensity grid (row-major) for a
/// single spectrum. Pure function, no PyO3 dependency, so it's directly unit-testable.
fn build_intensity_grid(
    data: &OwnedSpec,
    row_charges: &[usize],
    row_index: &HashMap<(String, usize), usize>,
    reverse_set: &HashSet<&str>,
    max_peptide_length: usize,
) -> Vec<f32> {
    let n_rows = row_charges.len();
    let mut grid = vec![f32::NAN; n_rows * max_peptide_length];
    let n_cols = data.seq_len.min(max_peptide_length).saturating_sub(1);

    // Unmask: 0.0 baseline wherever the (row charge, column) is chemically possible.
    for (row, charge) in row_charges.iter().enumerate() {
        if *charge > data.precursor_charge {
            continue;
        }
        for col in 0..n_cols {
            grid[row * max_peptide_length + col] = 0.0;
        }
    }

    // Fill observed intensities, taking the max when multiple peaks collide on a cell.
    for (peak_idx, annotations) in data.peak_annotations.iter().enumerate() {
        let intensity = data.intensity.get(peak_idx).copied().unwrap_or(0.0);
        for (series, position, charge) in annotations {
            let Some(&row) = row_index.get(&(series.clone(), *charge)) else {
                continue;
            };
            if *position < 1 {
                continue;
            }
            let col = if reverse_set.contains(series.as_str()) {
                data.seq_len.checked_sub(*position + 1)
            } else {
                Some(*position - 1)
            };
            let Some(col) = col else { continue };
            if col >= max_peptide_length {
                continue;
            }
            let cell = &mut grid[row * max_peptide_length + col];
            if cell.is_nan() {
                continue; // chemically impossible cell; leave masked
            }
            *cell = cell.max(intensity);
        }
    }

    grid
}

/// Extract fixed-size, NaN-masked intensity grids for MS²DIP-style training targets.
///
/// For each spectrum, builds a `(row_ion_types.len(), max_peptide_length)` array:
/// - `NaN` outside the chemically-possible region (fragment charge above the PSM's
///   precursor charge, or backbone position beyond `seq_len - 1`).
/// - `0.0` inside that region wherever no matching peak was observed.
/// - the observed peak intensity (max of all matches, if more than one peak maps to
///   the same cell) wherever a `FragmentAnnotation` matches a requested `(series, charge)`
///   row.
///
/// Row order is caller-defined via the parallel `row_ion_types`/`row_charges` arrays
/// (e.g. `["b", "y", "b", "y"]` / `[1, 1, 2, 2]`) rather than hardcoded, so callers are
/// free to include any ion series annotate_ms2_spectra produced (a/b/c/x/y/z), not just
/// b/y.
///
/// Series listed in `reverse_ions` (typically `["y"]`, sometimes also `["z"]` for ETD)
/// have their backbone position reversed onto the same column axis as the forward
/// series, so a b-ion and its complementary y-ion for the same cleavage site land in
/// the same column.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ms2dip_extract_intensity_grid(
    py: Python<'_>,
    annotated_spectra: Vec<Py<AnnotatedMS2Spectrum>>,
    seq_lens: Vec<usize>,
    precursor_charges: Vec<usize>,
    row_ion_types: Vec<String>,
    row_charges: Vec<usize>,
    reverse_ions: Vec<String>,
    max_peptide_length: usize,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let n = annotated_spectra.len();
    if seq_lens.len() != n || precursor_charges.len() != n {
        return Err(PyValueError::new_err(
            "Input arrays must have identical length: annotated_spectra, seq_lens, precursor_charges",
        ));
    }
    if row_ion_types.len() != row_charges.len() {
        return Err(PyValueError::new_err(
            "row_ion_types and row_charges must have identical length",
        ));
    }
    let n_rows = row_ion_types.len();
    let reverse_set: HashSet<&str> = reverse_ions.iter().map(|s| s.as_str()).collect();

    let mut row_index: HashMap<(String, usize), usize> = HashMap::with_capacity(n_rows);
    for (i, (series, charge)) in row_ion_types.iter().zip(row_charges.iter()).enumerate() {
        row_index.insert((series.clone(), *charge), i);
    }

    // Extract owned data under the GIL.
    let mut owned: Vec<OwnedSpec> = Vec::with_capacity(n);
    for i in 0..n {
        let spec_ref = annotated_spectra[i].bind(py);
        let spec = spec_ref.borrow();

        if spec.intensity.len() != spec.peak_annotations.len() {
            return Err(PyValueError::new_err(format!(
                "Spectrum {i}: intensity length {} != peak count {}",
                spec.intensity.len(),
                spec.peak_annotations.len()
            )));
        }

        let peak_annotations: Vec<Vec<(String, usize, usize)>> = spec
            .peak_annotations
            .iter()
            .map(|annotations| {
                annotations
                    .iter()
                    .map(|ann| (ann.series.clone(), ann.position, ann.charge))
                    .collect()
            })
            .collect();

        owned.push(OwnedSpec {
            peak_annotations,
            intensity: spec.intensity.clone(),
            seq_len: seq_lens[i],
            precursor_charge: precursor_charges[i],
        });
    }

    // Release the GIL and build grids in parallel.
    let grids: Vec<Vec<f32>> = py.detach(|| {
        owned
            .into_par_iter()
            .map(|data| build_intensity_grid(&data, &row_charges, &row_index, &reverse_set, max_peptide_length))
            .collect()
    });

    // Rebuild PyO3/numpy objects under the GIL.
    Ok(grids
        .into_iter()
        .map(|flat| {
            let arr = ndarray::Array2::from_shape_vec((n_rows, max_peptide_length), flat)
                .expect("grid size matches n_rows * max_peptide_length by construction");
            PyArray2::from_array(py, &arr).into()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row_index(pairs: &[(&str, usize)]) -> (Vec<usize>, HashMap<(String, usize), usize>) {
        let charges: Vec<usize> = pairs.iter().map(|(_, c)| *c).collect();
        let mut idx = HashMap::new();
        for (i, (series, charge)) in pairs.iter().enumerate() {
            idx.insert((series.to_string(), *charge), i);
        }
        (charges, idx)
    }

    #[test]
    fn test_basic_b_y_charge1() {
        // Peptide of length 4 (e.g. "PEPT"), so 3 backbone cleavage sites.
        // One b1 peak and one y1 peak, both charge 1, plus one unmatched peak.
        let data = OwnedSpec {
            peak_annotations: vec![
                vec![("b".to_string(), 1, 1)],
                vec![("y".to_string(), 1, 1)],
                Vec::new(),
            ],
            intensity: vec![10.0, 20.0, 5.0],
            seq_len: 4,
            precursor_charge: 2,
        };
        let (row_charges, row_index) = make_row_index(&[("b", 1), ("y", 1)]);
        let reverse_set: HashSet<&str> = ["y"].into_iter().collect();

        let grid = build_intensity_grid(&data, &row_charges, &row_index, &reverse_set, 5);
        // Row-major (2 rows, 5 cols)
        let at = |row: usize, col: usize| grid[row * 5 + col];

        // n_cols = min(4, 5) - 1 = 3 possible columns (0,1,2), rest NaN.
        // Row 0 = b1: b-ion at position 1 -> col 0 -> intensity 10.0
        assert_eq!(at(0, 0), 10.0);
        assert_eq!(at(0, 1), 0.0);
        assert_eq!(at(0, 2), 0.0);
        assert!(at(0, 3).is_nan());
        assert!(at(0, 4).is_nan());

        // Row 1 = y1: reversed, col = seq_len - position - 1 = 4 - 1 - 1 = 2 -> intensity 20.0
        assert_eq!(at(1, 0), 0.0);
        assert_eq!(at(1, 1), 0.0);
        assert_eq!(at(1, 2), 20.0);
        assert!(at(1, 3).is_nan());
        assert!(at(1, 4).is_nan());
    }

    #[test]
    fn test_masks_by_precursor_charge() {
        let data = OwnedSpec {
            peak_annotations: vec![],
            intensity: vec![],
            seq_len: 5,
            precursor_charge: 1, // charge-2 rows must stay fully NaN
        };
        let (row_charges, row_index) = make_row_index(&[("b", 1), ("b", 2)]);
        let reverse_set: HashSet<&str> = ["y"].into_iter().collect();

        let grid = build_intensity_grid(&data, &row_charges, &row_index, &reverse_set, 6);
        let at = |row: usize, col: usize| grid[row * 6 + col];

        // Row 0 (charge 1) unmasked for cols 0..3
        assert_eq!(at(0, 0), 0.0);
        assert_eq!(at(0, 3), 0.0);
        assert!(at(0, 4).is_nan());
        // Row 1 (charge 2, above precursor charge 1) fully NaN
        for col in 0..6 {
            assert!(at(1, col).is_nan());
        }
    }

    #[test]
    fn test_takes_max_on_collision() {
        let data = OwnedSpec {
            peak_annotations: vec![
                vec![("b".to_string(), 1, 1)],
                vec![("b".to_string(), 1, 1)],
            ],
            intensity: vec![5.0, 50.0],
            seq_len: 3,
            precursor_charge: 1,
        };
        let (row_charges, row_index) = make_row_index(&[("b", 1)]);
        let reverse_set: HashSet<&str> = HashSet::new();

        let grid = build_intensity_grid(&data, &row_charges, &row_index, &reverse_set, 4);
        assert_eq!(grid[0], 50.0); // max(5.0, 50.0)
    }

    #[test]
    fn test_unrequested_ion_type_ignored() {
        // An "a"-ion match should be silently dropped when only b/y rows are requested.
        let data = OwnedSpec {
            peak_annotations: vec![vec![("a".to_string(), 1, 1)]],
            intensity: vec![99.0],
            seq_len: 4,
            precursor_charge: 2,
        };
        let (row_charges, row_index) = make_row_index(&[("b", 1), ("y", 1)]);
        let reverse_set: HashSet<&str> = ["y"].into_iter().collect();

        let grid = build_intensity_grid(&data, &row_charges, &row_index, &reverse_set, 5);
        // No cell should carry 99.0 anywhere.
        assert!(!grid.contains(&99.0));
    }
}
