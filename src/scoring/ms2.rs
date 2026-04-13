use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::types::annotation::AnnotatedMS2Spectrum;
use crate::utils::{ln_factorial, longest_true_run};

const ION_SERIES: [&str; 6] = ["a", "b", "c", "x", "y", "z"];
const N_TERM: [usize; 3] = [0, 1, 2]; // a, b, c
const C_TERM: [usize; 3] = [3, 4, 5]; // x, y, z

fn series_index(s: &str) -> Option<usize> {
    ION_SERIES.iter().position(|&x| x == s)
}

fn hyperscore(n_nterm: usize, n_cterm: usize, sum_nterm: f64, sum_cterm: f64) -> f64 {
    let sum = if (sum_nterm + sum_cterm) > 0.0 {
        sum_nterm + sum_cterm
    } else {
        1.0
    };
    ln_factorial(n_nterm) + ln_factorial(n_cterm) + sum.ln()
}

/// Pre-computed feature name strings for each ion series.
struct FeatureNames {
    matched: [String; 6],
    matched_pct: [String; 6],
    longest: [String; 6],
    ratio: [String; 6],
}

impl FeatureNames {
    fn new() -> Self {
        let mut matched = std::array::from_fn(|_| String::new());
        let mut matched_pct = std::array::from_fn(|_| String::new());
        let mut longest = std::array::from_fn(|_| String::new());
        let mut ratio = std::array::from_fn(|_| String::new());
        for (i, &s) in ION_SERIES.iter().enumerate() {
            matched[i] = format!("matched_{s}_ions");
            matched_pct[i] = format!("matched_{s}_ions_pct");
            longest[i] = format!("longest_{s}_ion_sequence");
            ratio[i] = format!("ln_explained_{s}_ion_ratio");
        }
        FeatureNames {
            matched,
            matched_pct,
            longest,
            ratio,
        }
    }
}

/// Compute MS2 scoring features from annotated spectra.
///
/// Always emits features for all 6 primary ion series (a, b, c, x, y, z).
/// Series not in `active_ion_series` get NaN values.
/// Pass the series that your fragmentation model produces, e.g. `["a", "b", "y"]` for CID/HCD.
#[pyfunction]
pub fn score_ms2_spectra(
    py: Python<'_>,
    spectra: Vec<Py<AnnotatedMS2Spectrum>>,
    seq_lens: Vec<usize>,
    active_ion_series: Vec<String>,
    calculate_hyperscore: bool,
) -> PyResult<Vec<HashMap<String, f64>>> {
    let n = spectra.len();
    if seq_lens.len() != n {
        return Err(pyo3::exceptions::PyException::new_err(
            "spectra and seq_lens must have identical length",
        ));
    }

    // Determine active series from the fragmentation model (not from matched annotations)
    let mut active = [false; 6];
    for s in &active_ion_series {
        if let Some(idx) = series_index(s) {
            active[idx] = true;
        }
    }

    struct OwnedAnnotated {
        peak_intensities: Vec<f32>,
        peak_annotations: Vec<Vec<(usize, usize)>>, // (series_index, position)
        seq_len: usize,
    }

    let mut owned: Vec<OwnedAnnotated> = Vec::with_capacity(n);

    // Extract data (must hold GIL)
    for (i, spec_py) in spectra.iter().enumerate() {
        let spec_ref = spec_py.bind(py);
        let spec = spec_ref.borrow();

        let peak_intensities: Vec<f32> = spec.intensity.clone();
        let peak_annotations: Vec<Vec<(usize, usize)>> = spec
            .peak_annotations
            .iter()
            .map(|anns| {
                anns.iter()
                    .filter_map(|a| {
                        let idx = series_index(&a.series)?;
                        Some((idx, a.position))
                    })
                    .collect()
            })
            .collect();

        owned.push(OwnedAnnotated {
            peak_intensities,
            peak_annotations,
            seq_len: seq_lens[i],
        });
    }

    let active = Arc::new(active);
    let names = Arc::new(FeatureNames::new());

    // Release GIL and parallelize
    let results: Vec<HashMap<String, f64>> = py.detach(|| {
        owned
            .into_par_iter()
            .map(|item| {
                let seq_len = item.seq_len;
                if seq_len == 0 {
                    return HashMap::new();
                }

                let n_features = 3 + 4 * 6 + 1 + if calculate_hyperscore { 1 } else { 0 };
                let mut feats: HashMap<String, f64> = HashMap::with_capacity(n_features);

                // Per-series tracking with fixed-size arrays
                let mut flags: [Vec<bool>; 6] = std::array::from_fn(|_| vec![false; seq_len]);
                let mut intensity_sum = [0.0_f32; 6];
                let mut matched_ints: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());

                let pseudo = 1e-5_f64;
                let mut total_intensity = 0.0_f32;
                let mut matched_intensity = 0.0_f32;

                for (peak_idx, anns) in item.peak_annotations.iter().enumerate() {
                    let inten = item.peak_intensities[peak_idx];
                    total_intensity += inten;

                    if !anns.is_empty() {
                        matched_intensity += inten;

                        for &(si, position) in anns {
                            if position < seq_len {
                                flags[si][position] = true;
                                intensity_sum[si] += inten;
                                if calculate_hyperscore {
                                    matched_ints[si].push(inten);
                                }
                            }
                        }
                    }
                }

                // Cast to f64 at the output boundary (Python float precision)
                let total = total_intensity as f64;
                let matched = matched_intensity as f64;

                feats.insert(
                    "ln_explained_intensity".to_string(),
                    (matched + pseudo).ln(),
                );
                feats.insert(
                    "ln_total_intensity".to_string(),
                    (total + pseudo).ln(),
                );
                let explained_ratio = if total > 0.0 {
                    (matched / total + pseudo).ln()
                } else {
                    pseudo.ln()
                };
                feats.insert("ln_explained_intensity_ratio".to_string(), explained_ratio);

                let mut total_matched = 0usize;
                let mut total_possible = 0usize;

                for i in 0..6 {
                    if !active[i] {
                        feats.insert(names.matched[i].clone(), f64::NAN);
                        feats.insert(names.matched_pct[i].clone(), f64::NAN);
                        feats.insert(names.longest[i].clone(), f64::NAN);
                        feats.insert(names.ratio[i].clone(), f64::NAN);
                        continue;
                    }

                    let matched = flags[i].iter().filter(|&&x| x).count();
                    total_matched += matched;
                    total_possible += seq_len;

                    feats.insert(names.matched[i].clone(), matched as f64);
                    feats.insert(names.matched_pct[i].clone(), matched as f64 / seq_len as f64);
                    feats.insert(
                        names.longest[i].clone(),
                        longest_true_run(&flags[i]) as f64,
                    );

                    let ratio = if matched_intensity > 0.0 {
                        (intensity_sum[i] as f64 / matched_intensity as f64 + pseudo).ln()
                    } else {
                        pseudo.ln()
                    };
                    feats.insert(names.ratio[i].clone(), ratio);
                }

                feats.insert(
                    "matched_ions_pct".to_string(),
                    if total_possible > 0 {
                        total_matched as f64 / total_possible as f64
                    } else {
                        0.0
                    },
                );

                if calculate_hyperscore {
                    let (mut n_nterm, mut sum_nterm) = (0usize, 0.0_f64);
                    let (mut n_cterm, mut sum_cterm) = (0usize, 0.0_f64);

                    for &i in &N_TERM {
                        n_nterm += matched_ints[i].len();
                        sum_nterm += matched_ints[i].iter().map(|&v| v as f64).sum::<f64>();
                    }
                    for &i in &C_TERM {
                        n_cterm += matched_ints[i].len();
                        sum_cterm += matched_ints[i].iter().map(|&v| v as f64).sum::<f64>();
                    }

                    feats.insert(
                        "hyperscore".to_string(),
                        hyperscore(n_nterm, n_cterm, sum_nterm, sum_cterm),
                    );
                }

                feats
            })
            .collect()
    });

    Ok(results)
}
