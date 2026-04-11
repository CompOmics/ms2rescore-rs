use std::collections::HashMap;

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::types::annotation::AnnotatedMS2Spectrum;
use crate::utils::{ln_factorial, longest_true_run};

/// The 6 primary ion series, in order.
const ION_SERIES: [&str; 6] = ["a", "b", "c", "x", "y", "z"];

/// N-terminal ion series (used for hyperscore grouping).
const N_TERM_SERIES: [&str; 3] = ["a", "b", "c"];

/// C-terminal ion series (used for hyperscore grouping).
const C_TERM_SERIES: [&str; 3] = ["x", "y", "z"];

fn hyperscore(n_nterm: usize, n_cterm: usize, sum_nterm: f64, sum_cterm: f64) -> f64 {
    let sum = if (sum_nterm + sum_cterm) > 0.0 {
        sum_nterm + sum_cterm
    } else {
        1.0
    };
    ln_factorial(n_nterm) + ln_factorial(n_cterm) + sum.ln()
}

/// Determine which ion series are "active" — i.e., have at least one annotation
/// across all spectra. Series not present in any annotation get NaN features.
fn active_series(spectra: &[&AnnotatedMS2Spectrum]) -> HashMap<String, bool> {
    let mut active: HashMap<String, bool> = ION_SERIES
        .iter()
        .map(|s| (s.to_string(), false))
        .collect();

    for spec in spectra {
        for peak_anns in &spec.peak_annotations {
            for ann in peak_anns {
                if let Some(v) = active.get_mut(&ann.series) {
                    *v = true;
                }
            }
        }
    }

    active
}

/// Compute MS2 scoring features from annotated spectra.
///
/// Always emits features for all 6 primary ion series (a, b, c, x, y, z).
/// Series not produced by the fragmentation model get NaN values.
#[pyfunction]
pub fn score_ms2_spectra(
    py: Python<'_>,
    spectra: Vec<Py<AnnotatedMS2Spectrum>>,
    seq_lens: Vec<usize>,
    calculate_hyperscore: bool,
) -> PyResult<Vec<HashMap<String, f64>>> {
    let n = spectra.len();
    if seq_lens.len() != n {
        return Err(pyo3::exceptions::PyException::new_err(
            "spectra and seq_lens must have identical length",
        ));
    }

    // Extract owned data while holding the GIL
    struct OwnedAnnotated {
        peak_intensities: Vec<f64>,
        peak_annotations: Vec<Vec<(String, usize)>>, // (series, position)
        seq_len: usize,
    }

    let mut owned: Vec<OwnedAnnotated> = Vec::with_capacity(n);
    let mut all_spectra_ref: Vec<pyo3::PyRef<'_, AnnotatedMS2Spectrum>> = Vec::with_capacity(n);

    for spec_py in &spectra {
        let spec_ref = spec_py.bind(py);
        let spec = spec_ref.borrow();
        all_spectra_ref.push(spec);
    }

    // Determine active series from all spectra
    let borrowed_spectra: Vec<&AnnotatedMS2Spectrum> = all_spectra_ref.iter().map(|r| &**r).collect();
    let active = active_series(&borrowed_spectra);

    for (i, spec) in all_spectra_ref.iter().enumerate() {
        let peak_intensities: Vec<f64> = spec.intensity.iter().map(|&x| x as f64).collect();
        let peak_annotations: Vec<Vec<(String, usize)>> = spec
            .peak_annotations
            .iter()
            .map(|anns| {
                anns.iter()
                    .map(|a| (a.series.clone(), a.position))
                    .collect()
            })
            .collect();

        owned.push(OwnedAnnotated {
            peak_intensities,
            peak_annotations,
            seq_len: seq_lens[i],
        });
    }

    let active = std::sync::Arc::new(active);

    // Release GIL and parallelize
    let results: Vec<HashMap<String, f64>> = py.detach(|| {
        owned
            .into_par_iter()
            .map(|item| {
                let seq_len = item.seq_len;
                let mut feats: HashMap<String, f64> = HashMap::new();

                // If seq_len is 0, return empty features
                if seq_len == 0 {
                    return feats;
                }

                // Per-series tracking: flags (matched positions) and intensity sums
                let mut series_flags: HashMap<&str, Vec<bool>> = HashMap::new();
                let mut series_intensity_sum: HashMap<&str, f64> = HashMap::new();
                let mut series_matched_ints: HashMap<&str, Vec<f64>> = HashMap::new();

                for &s in &ION_SERIES {
                    series_flags.insert(s, vec![false; seq_len]);
                    series_intensity_sum.insert(s, 0.0);
                    series_matched_ints.insert(s, Vec::new());
                }

                let pseudo = 1e-5_f64;
                let mut total_intensity = 0.0_f64;
                let mut matched_intensity = 0.0_f64;

                // Walk peaks
                for (peak_idx, anns) in item.peak_annotations.iter().enumerate() {
                    let inten = item.peak_intensities[peak_idx];
                    total_intensity += inten;

                    if !anns.is_empty() {
                        matched_intensity += inten;

                        for (series, position) in anns {
                            if let Some(flags) = series_flags.get_mut(series.as_str()) {
                                if *position < seq_len {
                                    flags[*position] = true;
                                    *series_intensity_sum
                                        .get_mut(series.as_str())
                                        .unwrap() += inten;
                                    if calculate_hyperscore {
                                        series_matched_ints
                                            .get_mut(series.as_str())
                                            .unwrap()
                                            .push(inten);
                                    }
                                }
                            }
                        }
                    }
                }

                // Aggregated intensity features
                feats.insert(
                    "ln_explained_intensity".to_string(),
                    (matched_intensity + pseudo).ln(),
                );
                feats.insert(
                    "ln_total_intensity".to_string(),
                    (total_intensity + pseudo).ln(),
                );
                let explained_ratio = if total_intensity > 0.0 {
                    (matched_intensity / total_intensity + pseudo).ln()
                } else {
                    pseudo.ln()
                };
                feats.insert("ln_explained_intensity_ratio".to_string(), explained_ratio);

                // Per-series features
                let mut total_matched = 0usize;
                let mut total_possible = 0usize;

                for &s in &ION_SERIES {
                    let is_active = *active.get(s).unwrap_or(&false);

                    if !is_active {
                        // NaN for inactive series
                        feats.insert(format!("matched_{s}_ions"), f64::NAN);
                        feats.insert(format!("matched_{s}_ions_pct"), f64::NAN);
                        feats.insert(format!("longest_{s}_ion_sequence"), f64::NAN);
                        feats.insert(format!("ln_explained_{s}_ion_ratio"), f64::NAN);
                        continue;
                    }

                    let flags = &series_flags[s];
                    let matched = flags.iter().filter(|&&x| x).count();
                    let intensity_sum = series_intensity_sum[s];

                    total_matched += matched;
                    total_possible += seq_len;

                    feats.insert(format!("matched_{s}_ions"), matched as f64);
                    feats.insert(
                        format!("matched_{s}_ions_pct"),
                        matched as f64 / seq_len as f64,
                    );
                    feats.insert(
                        format!("longest_{s}_ion_sequence"),
                        longest_true_run(flags) as f64,
                    );

                    let ratio = if matched_intensity > 0.0 {
                        (intensity_sum / matched_intensity + pseudo).ln()
                    } else {
                        pseudo.ln()
                    };
                    feats.insert(format!("ln_explained_{s}_ion_ratio"), ratio);
                }

                // Aggregated matched ions pct
                feats.insert(
                    "matched_ions_pct".to_string(),
                    if total_possible > 0 {
                        total_matched as f64 / total_possible as f64
                    } else {
                        0.0
                    },
                );

                // Hyperscore
                if calculate_hyperscore {
                    let mut n_nterm = 0usize;
                    let mut n_cterm = 0usize;
                    let mut sum_nterm = 0.0_f64;
                    let mut sum_cterm = 0.0_f64;

                    for &s in &N_TERM_SERIES {
                        let ints = &series_matched_ints[s];
                        n_nterm += ints.len();
                        sum_nterm += ints.iter().sum::<f64>();
                    }
                    for &s in &C_TERM_SERIES {
                        let ints = &series_matched_ints[s];
                        n_cterm += ints.len();
                        sum_cterm += ints.iter().sum::<f64>();
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
