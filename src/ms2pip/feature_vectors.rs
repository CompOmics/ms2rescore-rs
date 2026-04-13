use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustyms::prelude::CompoundPeptidoformIon;

use crate::utils::extract_charge;

/// Number of features per cleavage site.
const N_FEATURES: usize = 139;

/// Number of standard amino acids used in feature computation.
const N_AA: usize = 19;

/// AA single-letter codes in ms2pip index order.
/// L is mapped to I (index 7).
const AA_CHARS: [char; N_AA] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];

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

/// Map a single-letter AA code to its ms2pip index (0..18).
/// L is mapped to I (index 7). Returns None for unknown AAs.
fn aa_to_index(aa: char) -> Option<usize> {
    match aa {
        'A' => Some(0),
        'C' => Some(1),
        'D' => Some(2),
        'E' => Some(3),
        'F' => Some(4),
        'G' => Some(5),
        'H' => Some(6),
        'I' | 'L' => Some(7),
        'K' => Some(8),
        'M' => Some(9),
        'N' => Some(10),
        'P' => Some(11),
        'Q' => Some(12),
        'R' => Some(13),
        'S' => Some(14),
        'T' => Some(15),
        'V' => Some(16),
        'W' => Some(17),
        'Y' => Some(18),
        _ => None,
    }
}

/// Compute floor-based quartiles matching the C code.
/// Input must be a sorted slice.
fn quartiles(sorted: &[u32]) -> [u32; 5] {
    let n = sorted.len();
    if n == 0 {
        return [0; 5];
    }
    if n == 1 {
        return [sorted[0]; 5];
    }
    let q0 = sorted[0];
    let q1 = sorted[((n - 1) as f64 * 0.25).floor() as usize];
    let q2 = sorted[((n - 1) as f64 * 0.5).floor() as usize];
    let q3 = sorted[((n - 1) as f64 * 0.75).floor() as usize];
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
            let aa_str = aa.to_string();
            let aa_char = aa_str.chars().next().ok_or_else(|| {
                format!("Empty amino acid string in '{proforma}'")
            })?;
            let idx = aa_to_index(aa_char).ok_or_else(|| {
                format!("Unknown amino acid '{aa_char}' in '{proforma}'")
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
        peptide_quartiles[prop_idx] = quartiles(&prop_values);
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
    for ion_idx in 0..n_ions {
        let offset = ion_idx * N_FEATURES;
        let i = ion_idx; // Cleavage after position i (0-indexed)

        // N-terminal ion: positions 0..=i (length i+1)
        // C-terminal ion: positions i+1..peplen-1 (length peplen-1-i)
        let n_len = (i + 1) as f32;
        let c_len = (peplen - 1 - i) as f32;

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
        features[offset + 27] = n_len;
        features[offset + 28] = c_len;

        // -- AA counts [29..66] --
        // N-terminal: count in positions 0..=i → count_prefix[i+1]
        // C-terminal: count in positions i+1..peplen → count_prefix[peplen] - count_prefix[i+1]
        for aa in 0..N_AA {
            let n_count = count_prefix[i + 1][aa];
            let c_count = count_prefix[peplen][aa] - count_prefix[i + 1][aa];
            features[offset + 29 + aa * 2] = n_count as f32;
            features[offset + 29 + aa * 2 + 1] = c_count as f32;
        }

        // -- Property features [67..138] --
        // 4 properties × 18 features each = 72 features
        for prop_idx in 0..4 {
            let prop_offset = offset + 67 + prop_idx * 18;

            // Positional features (6):
            // p0: first position of peptide
            let p0 = prop_at[0][prop_idx] as f32;
            // p-1: last position of peptide
            let p_minus1 = prop_at[peplen - 1][prop_idx] as f32;
            // pi-1: position before cleavage (or 0 if i==0)
            let pi_minus1 = if i > 0 { prop_at[i - 1][prop_idx] as f32 } else { 0.0 };
            // pi: cleavage position
            let pi = prop_at[i][prop_idx] as f32;
            // pi+1: position after cleavage
            let pi_plus1 = prop_at[i + 1][prop_idx] as f32;
            // pi+2: two positions after cleavage (or 0 if at end)
            let pi_plus2 = if i + 2 < peplen {
                prop_at[i + 2][prop_idx] as f32
            } else {
                0.0
            };

            features[prop_offset] = p0;
            features[prop_offset + 1] = p_minus1;
            features[prop_offset + 2] = pi_minus1;
            features[prop_offset + 3] = pi;
            features[prop_offset + 4] = pi_plus1;
            features[prop_offset + 5] = pi_plus2;

            // N-terminal ion: sum + quartiles (6)
            let n_sum = prop_prefix[i + 1][prop_idx] as f32;
            let mut n_props: Vec<u32> = (0..=i).map(|j| prop_at[j][prop_idx]).collect();
            n_props.sort_unstable();
            let n_q = quartiles(&n_props);

            features[prop_offset + 6] = n_sum;
            for q in 0..5 {
                features[prop_offset + 7 + q] = n_q[q] as f32;
            }

            // C-terminal ion: sum + quartiles (6)
            let c_sum = (prop_prefix[peplen][prop_idx] - prop_prefix[i + 1][prop_idx]) as f32;
            let mut c_props: Vec<u32> = (i + 1..peplen).map(|j| prop_at[j][prop_idx]).collect();
            c_props.sort_unstable();
            let c_q = quartiles(&c_props);

            features[prop_offset + 12] = c_sum;
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
    fn test_aa_to_index() {
        assert_eq!(aa_to_index('A'), Some(0));
        assert_eq!(aa_to_index('L'), Some(7)); // L maps to I
        assert_eq!(aa_to_index('I'), Some(7));
        assert_eq!(aa_to_index('Y'), Some(18));
        assert_eq!(aa_to_index('X'), None);
    }

    #[test]
    fn test_quartiles() {
        assert_eq!(quartiles(&[1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]);
        assert_eq!(quartiles(&[10]), [10, 10, 10, 10, 10]);
        assert_eq!(quartiles(&[]), [0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_compute_features_single_shape() {
        // ACDE = 4 AA → 3 ions × 139 features = 417 values
        let aa = vec![0, 1, 2, 3]; // A, C, D, E
        let features = compute_features_single(&aa, 2);
        assert_eq!(features.len(), 3 * N_FEATURES);
    }

    #[test]
    fn test_compute_features_single_peptide_level() {
        let aa = vec![0, 1, 2, 3]; // A, C, D, E, charge=2
        let features = compute_features_single(&aa, 2);

        // First ion's features
        assert_eq!(features[0], 4.0); // p_length
        assert_eq!(features[1], 2.0); // p_charge
        assert_eq!(features[2], 0.0); // charge_1
        assert_eq!(features[3], 1.0); // charge_2
        assert_eq!(features[4], 0.0); // charge_3

        // Check same shared features repeat for second ion
        assert_eq!(features[N_FEATURES], 4.0); // p_length
        assert_eq!(features[N_FEATURES + 1], 2.0); // p_charge
    }

    #[test]
    fn test_compute_features_ion_lengths() {
        let aa = vec![0, 1, 2, 3, 4]; // ACDEF, 4 ions
        let features = compute_features_single(&aa, 2);

        // Ion 0: n_len=1, c_len=4
        assert_eq!(features[27], 1.0);
        assert_eq!(features[28], 4.0);

        // Ion 1: n_len=2, c_len=3
        assert_eq!(features[N_FEATURES + 27], 2.0);
        assert_eq!(features[N_FEATURES + 28], 3.0);

        // Ion 3: n_len=4, c_len=1
        assert_eq!(features[3 * N_FEATURES + 27], 4.0);
        assert_eq!(features[3 * N_FEATURES + 28], 1.0);
    }

    #[test]
    fn test_compute_features_short_peptide() {
        // Minimum length: 2 AA → 1 ion
        let aa = vec![0, 1]; // AC
        let features = compute_features_single(&aa, 1);
        assert_eq!(features.len(), N_FEATURES);
    }

    #[test]
    fn test_compute_features_single_aa() {
        // 1 AA → 0 ions → empty
        let aa = vec![0];
        let features = compute_features_single(&aa, 1);
        assert!(features.is_empty());
    }
}
