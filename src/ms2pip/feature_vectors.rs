use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustyms::prelude::CompoundPeptidoformIon;

use crate::utils::{aa_to_ms2pip_index, extract_charge};

/// Number of features per cleavage site.
const N_FEATURES: usize = 139;

/// Number of standard amino acids used in feature computation.
const N_AA: usize = 19;

/// 4 amino acid property tables, each indexed by AA index (0..18).
/// Order: basicity, helicity, hydrophobicity, pI.
const AA_PROPERTIES: [[u32; N_AA]; 4] = [
    // Basicity
    [37, 35, 59, 129, 94, 0, 210, 81, 191, 106, 101, 117, 115, 343, 49, 90, 60, 134, 104],
    // Helicity
    [68, 23, 33, 29, 70, 58, 41, 73, 32, 66, 38, 0, 40, 39, 44, 53, 71, 51, 55],
    // Hydrophobicity
    [51, 75, 25, 35, 100, 16, 3, 94, 0, 82, 12, 0, 22, 22, 21, 39, 80, 98, 70],
    // pI (isoelectric point)
    [32, 23, 0, 4, 27, 32, 48, 32, 69, 29, 26, 35, 28, 79, 29, 28, 31, 31, 28],
];

/// Compute floor-based quartiles matching the C code.
/// Input must be a sorted slice. `denom` is the denominator used in the
/// quantile index: `floor(q * denom)`. The C code uses different denominators
/// for different contexts: `peplen-1` for peptide-level, `i` for N-ion
/// (= n_elements - 1), and `peplen-i-1` for C-ion (= n_elements, NOT
/// n_elements - 1). This inconsistency is a bug in the original C code but
/// is intentionally replicated here for XGBoost model compatibility.
fn quartiles_with_denom(sorted: &[u32], denom: usize) -> [u32; 5] {
    let n = sorted.len();
    if n == 0 {
        return [0; 5];
    }
    let q0 = sorted[0];
    let q1 = sorted[((denom as f64 * 0.25).floor() as usize).min(n - 1)];
    let q2 = sorted[((denom as f64 * 0.5).floor() as usize).min(n - 1)];
    let q3 = sorted[((denom as f64 * 0.75).floor() as usize).min(n - 1)];
    let q4 = sorted[n - 1];
    [q0, q1, q2, q3, q4]
}

/// Extract amino acid indices and charge from a ProForma string.
/// Returns (aa_indices, charge). Modified residues are mapped to their base AA.
fn parse_proforma(proforma: &str) -> Result<(Vec<usize>, usize), String> {
    let compound = CompoundPeptidoformIon::pro_forma(proforma, None)
        .map_err(|e| format!("Failed to parse ProForma '{proforma}': {e}"))?;

    let charge = extract_charge(&compound)
        .ok_or_else(|| format!("No charge state found in '{proforma}'. MS2PIP requires a charge (e.g. 'PEPTIDE/2')."))?;

    // Extract amino acid sequence — must be exactly one peptidoform
    let peptidoform_ions = compound.peptidoform_ions();
    if peptidoform_ions.len() != 1 {
        return Err(format!(
            "Expected exactly 1 peptidoform ion in '{proforma}', found {}",
            peptidoform_ions.len()
        ));
    }

    let mut aa_indices = Vec::new();
    for peptidoform in compound.peptidoforms() {
        for pos in peptidoform.sequence() {
            let aa = pos.aminoacid.aminoacid();
            let idx = aa_to_ms2pip_index(aa).ok_or_else(|| {
                format!("Unsupported amino acid '{aa}' in '{proforma}'")
            })?;
            aa_indices.push(idx);
        }
    }

    if aa_indices.is_empty() {
        return Err(format!("No amino acids found in '{proforma}'"));
    }

    Ok((aa_indices, charge))
}

/// Compute 139 features for each cleavage site of a single peptide.
/// Returns a flat Vec<f32> of length (peplen-1) * 139.
fn compute_features_single(aa_indices: &[usize], charge: usize) -> Vec<f32> {
    let peplen = aa_indices.len();
    if peplen < 2 {
        return Vec::new();
    }

    let n_ions = peplen - 1;
    let mut features = vec![0.0_f32; n_ions * N_FEATURES];

    // --- Shared peptide-level features (27 total) ---

    // [0] p_length, [1] p_charge
    let p_length = peplen as f32;
    let p_charge = charge as f32;

    // [2..6] charge one-hot
    let charge_onehot: [f32; 5] = [
        if charge == 1 { 1.0 } else { 0.0 },
        if charge == 2 { 1.0 } else { 0.0 },
        if charge == 3 { 1.0 } else { 0.0 },
        if charge == 4 { 1.0 } else { 0.0 },
        if charge >= 5 { 1.0 } else { 0.0 },
    ];

    // [7..26] peptide property quartiles (4 props × 5 quartiles)
    let mut peptide_quartiles = [[0_u32; 5]; 4];
    for prop_idx in 0..4 {
        let mut prop_values: Vec<u32> = aa_indices
            .iter()
            .map(|&ai| AA_PROPERTIES[prop_idx][ai])
            .collect();
        prop_values.sort_unstable();
        peptide_quartiles[prop_idx] = quartiles_with_denom(&prop_values, peplen - 1);
    }

    // --- Precompute per-property values for each position ---
    let prop_at: Vec<[u32; 4]> = aa_indices
        .iter()
        .map(|&ai| {
            [
                AA_PROPERTIES[0][ai],
                AA_PROPERTIES[1][ai],
                AA_PROPERTIES[2][ai],
                AA_PROPERTIES[3][ai],
            ]
        })
        .collect();

    // --- Precompute cumulative AA counts and property sums ---
    // count_prefix[i][aa] = count of AA in positions 0..i (exclusive)
    // prop_prefix[i][p] = sum of property p in positions 0..i (exclusive)
    let mut count_prefix = vec![[0_u32; N_AA]; peplen + 1];
    let mut prop_prefix = vec![[0_u32; 4]; peplen + 1];

    for i in 0..peplen {
        count_prefix[i + 1] = count_prefix[i];
        count_prefix[i + 1][aa_indices[i]] += 1;

        prop_prefix[i + 1] = prop_prefix[i];
        for p in 0..4 {
            prop_prefix[i + 1][p] += prop_at[i][p];
        }
    }

    // --- Compute features per cleavage site ---
    // The C code uses 1-indexed peptide_buf where peptide_buf[0] = nterm_mod.
    // For unmodified peptides, peptide_buf[0] = 0, so props[j][0] = AA index 0's
    // property (Alanine). We replicate this with a "virtual" position -1 that maps
    // to AA index 0. The positional features pi-1, pi, pi+1, pi+2 in the C code
    // reference peptide_buf[i-1], peptide_buf[i], peptide_buf[i+1], peptide_buf[i+2]
    // where i is 0-indexed and peptide_buf is 1-indexed.
    //
    // The C code also uses an incremental count_c that starts as the full peptide
    // and removes from the RIGHT end (position peplen-i in 1-indexed = peplen-1-i
    // in 0-indexed) at each step.

    // Build prop_at_buf matching C's peptide_buf: [nterm_prop, aa0_prop, aa1_prop, ...]
    // C code bug: peptide_buf[0] = nterm_mod = 0 for unmodified peptides, so
    // props[j][peptide_buf[0]] = props[j][0] = Alanine's property, regardless of
    // the actual first residue. This is used for positional feature pi when i=0.
    // Intentionally replicated for XGBoost model compatibility.
    let prop_at_buf: Vec<[u32; 4]> = std::iter::once([
        AA_PROPERTIES[0][0],
        AA_PROPERTIES[1][0],
        AA_PROPERTIES[2][0],
        AA_PROPERTIES[3][0],
    ])
    .chain(prop_at.iter().copied())
    .collect();

    // Incremental c AA counts (matching C code: start full, remove from right)
    let mut inc_count_n = [0_u32; N_AA];
    let mut inc_count_c = [0_u32; N_AA];
    for &ai in aa_indices {
        inc_count_c[ai] += 1;
    }

    for ion_idx in 0..n_ions {
        let offset = ion_idx * N_FEATURES;
        let i = ion_idx;

        // -- Shared features [0..26] --
        features[offset] = p_length;
        features[offset + 1] = p_charge;
        for k in 0..5 {
            features[offset + 2 + k] = charge_onehot[k];
        }
        for prop_idx in 0..4 {
            for q in 0..5 {
                features[offset + 7 + prop_idx * 5 + q] =
                    peptide_quartiles[prop_idx][q] as f32;
            }
        }

        // -- Ion lengths [27..28] --
        // C code: v[fnum++] = i+1; v[fnum++] = peplen-i;
        features[offset + 27] = (i + 1) as f32;
        features[offset + 28] = (peplen - i) as f32;

        // -- AA counts [29..66] --
        // C code updates counts BEFORE writing them:
        // count_n[peptide_buf[i+1]]++ → add position i (0-indexed)
        // count_c[peptide_buf[peplen-i]]-- → remove position peplen-1-i (0-indexed)
        inc_count_n[aa_indices[i]] += 1;
        inc_count_c[aa_indices[peplen - 1 - i]] -= 1;

        for aa in 0..N_AA {
            features[offset + 29 + aa * 2] = inc_count_n[aa] as f32;
            features[offset + 29 + aa * 2 + 1] = inc_count_c[aa] as f32;
        }

        // -- Property features [67..138] --
        for prop_idx in 0..4 {
            let prop_offset = offset + 67 + prop_idx * 18;

            // Positional features (6) — using prop_at_buf (1-indexed like C)
            // C code: props[j][peptide_buf[1]], props[j][peptide_buf[peplen]],
            //         props[j][peptide_buf[i-1]] (or 0), props[j][peptide_buf[i]],
            //         props[j][peptide_buf[i+1]], props[j][peptide_buf[i+2]] (or 0)
            let p0 = prop_at_buf[1][prop_idx] as f32;
            let p_minus1 = prop_at_buf[peplen][prop_idx] as f32;
            let pi_minus1 = if i == 0 {
                0.0
            } else {
                prop_at_buf[i - 1][prop_idx] as f32 // peptide_buf[i-1]
            };
            let pi = prop_at_buf[i][prop_idx] as f32; // peptide_buf[i]
            let pi_plus1 = prop_at_buf[i + 1][prop_idx] as f32; // peptide_buf[i+1]
            let pi_plus2 = if i == peplen - 1 {
                0.0
            } else {
                prop_at_buf[i + 2][prop_idx] as f32 // peptide_buf[i+2]
            };

            features[prop_offset] = p0;
            features[prop_offset + 1] = p_minus1;
            features[prop_offset + 2] = pi_minus1;
            features[prop_offset + 3] = pi;
            features[prop_offset + 4] = pi_plus1;
            features[prop_offset + 5] = pi_plus2;

            // N-terminal ion: sum + quartiles (6)
            // C code: positions 0..=i (0-indexed), denom for quartiles = i
            let mut n_props: Vec<u32> = (0..=i).map(|j| prop_at[j][prop_idx]).collect();
            let n_sum: u32 = n_props.iter().sum();
            n_props.sort_unstable();
            let n_q = quartiles_with_denom(&n_props, i);

            features[prop_offset + 6] = n_sum as f32;
            for q in 0..5 {
                features[prop_offset + 7 + q] = n_q[q] as f32;
            }

            // C-terminal ion: sum + quartiles (6)
            // C code: positions i+1..peplen-1 (0-indexed).
            // C code bug: denom = peplen-i-1 = n_elements, while peptide-level
            // and N-ion use n_elements-1. Intentionally replicated for model compat.
            let c_n_elements = peplen - i - 1;
            let mut c_props: Vec<u32> =
                (i + 1..peplen).map(|j| prop_at[j][prop_idx]).collect();
            let c_sum: u32 = c_props.iter().sum();
            c_props.sort_unstable();
            let c_q = quartiles_with_denom(&c_props, c_n_elements);

            features[prop_offset + 12] = c_sum as f32;
            for q in 0..5 {
                features[prop_offset + 13 + q] = c_q[q] as f32;
            }
        }
    }

    features
}

/// Compute MS2PIP feature vectors for XGBoost prediction.
///
/// Takes ProForma strings with charge (e.g. "PEPTIDE/2") and returns
/// a flat numpy array per peptide of length (n_ions * 139).
/// Reshape to (n_ions, 139) on the Python side with n_ions = seq_len - 1.
#[pyfunction]
pub fn ms2pip_compute_features(
    py: Python<'_>,
    proformas: Vec<String>,
) -> PyResult<Vec<Py<PyArray1<f32>>>> {
    // Parse and compute in parallel (no GIL needed — pure Rust)
    let results: Result<Vec<Vec<f32>>, String> = py.detach(|| {
        proformas
            .par_iter()
            .enumerate()
            .map(|(i, pf)| {
                let (aa_indices, charge) = parse_proforma(pf)
                    .map_err(|e| format!("ProForma at index {i}: {e}"))?;
                Ok(compute_features_single(&aa_indices, charge))
            })
            .collect()
    });

    let results = results.map_err(PyValueError::new_err)?;

    // Convert to numpy arrays (needs GIL)
    let arrays: Vec<Py<PyArray1<f32>>> = results
        .into_iter()
        .map(|flat| PyArray1::from_vec(py, flat).into())
        .collect();

    Ok(arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quartiles_with_denom() {
        // peplen-1 formula: denom = n-1
        assert_eq!(
            quartiles_with_denom(&[1, 2, 3, 4, 5], 4),
            [1, 2, 3, 4, 5]
        );
        assert_eq!(quartiles_with_denom(&[10], 0), [10, 10, 10, 10, 10]);
        assert_eq!(quartiles_with_denom(&[], 0), [0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_compute_features_single_shape() {
        let aa = vec![0, 1, 2, 3]; // ACDE
        let features = compute_features_single(&aa, 2);
        assert_eq!(features.len(), 3 * N_FEATURES);
    }

    #[test]
    fn test_compute_features_single_peptide_level() {
        let aa = vec![0, 1, 2, 3]; // ACDE, charge=2
        let features = compute_features_single(&aa, 2);
        assert_eq!(features[0], 4.0); // p_length
        assert_eq!(features[1], 2.0); // p_charge
        assert_eq!(features[2], 0.0); // charge_1
        assert_eq!(features[3], 1.0); // charge_2
        // Same shared features for all ions
        assert_eq!(features[N_FEATURES], 4.0);
        assert_eq!(features[N_FEATURES + 1], 2.0);
    }

    #[test]
    fn test_compute_features_ion_lengths() {
        let aa = vec![0, 1, 2, 3, 4]; // ACDEF (5 AAs, 4 ions)
        let features = compute_features_single(&aa, 2);
        // C code: n_len = i+1, c_len = peplen-i
        assert_eq!(features[27], 1.0); // ion 0: n=1
        assert_eq!(features[28], 5.0); // ion 0: c=5
        assert_eq!(features[N_FEATURES + 27], 2.0); // ion 1: n=2
        assert_eq!(features[N_FEATURES + 28], 4.0); // ion 1: c=4
        assert_eq!(features[3 * N_FEATURES + 27], 4.0); // ion 3: n=4
        assert_eq!(features[3 * N_FEATURES + 28], 2.0); // ion 3: c=2
    }

    #[test]
    fn test_compute_features_c_reference_acde() {
        // Exact reference values from C code for ACDE/2
        let aa = vec![0, 1, 2, 3]; // A, C, D, E
        let features = compute_features_single(&aa, 2);

        let expected_ion0: Vec<f32> = vec![
            4, 2, 0, 1, 0, 0, 0, 35, 35, 37, 59, 129, 23, 23, 29, 33, 68, 25, 25,
            35, 51, 75, 0, 0, 4, 23, 32, 1, 4, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 37, 129, 0, 37, 37, 35, 37, 37, 37, 37, 37, 37, 223, 35, 35,
            59, 129, 129, 68, 29, 0, 68, 68, 23, 68, 68, 68, 68, 68, 68, 85, 23, 23,
            29, 33, 33, 51, 35, 0, 51, 51, 75, 51, 51, 51, 51, 51, 51, 135, 25, 25,
            35, 75, 75, 32, 4, 0, 32, 32, 23, 32, 32, 32, 32, 32, 32, 27, 0, 0, 4,
            23, 23,
        ]
        .into_iter()
        .map(|x| x as f32)
        .collect();

        let ion0 = &features[..N_FEATURES];
        assert_eq!(ion0.len(), expected_ion0.len());
        for (idx, (&got, &exp)) in ion0.iter().zip(expected_ion0.iter()).enumerate() {
            assert_eq!(got, exp, "Ion 0, feature {idx}: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_compute_features_short_peptide() {
        let aa = vec![0, 1]; // AC
        let features = compute_features_single(&aa, 1);
        assert_eq!(features.len(), N_FEATURES);
    }

    #[test]
    fn test_compute_features_single_aa() {
        let aa = vec![0];
        let features = compute_features_single(&aa, 1);
        assert!(features.is_empty());
    }
}
