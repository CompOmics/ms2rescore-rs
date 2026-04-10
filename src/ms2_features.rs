use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ms2_spectrum::MS2Spectrum;

use ordered_float::OrderedFloat;
use rustyms::annotation::model::FragmentationModel;
use rustyms::annotation::AnnotatableSpectrum;
use rustyms::chemistry::MassMode;
use rustyms::prelude::CompoundPeptidoformIon;
use rustyms::spectrum::{PeakSpectrum, RawPeak, RawSpectrum};
use rustyms::system::f64::MassOverCharge;
use rustyms::system::mass_over_charge::thomson;

fn parse_fragmentation_model(s: &str) -> PyResult<FragmentationModel> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cidhcd" | "cid_hcd" | "cid-hcd" => Ok((*FragmentationModel::cid_hcd()).clone()),
        "etd" => Ok((*FragmentationModel::etd()).clone()),
        "ethcd" | "et+hcd" | "et_hcd" => Ok((*FragmentationModel::ethcd()).clone()),
        "all" => Ok((*FragmentationModel::all()).clone()),
        other => Err(PyValueError::new_err(format!(
            "Unsupported fragmentation_model: {other}. Expected one of: cidhcd, etd, ethcd, all."
        ))),
    }
}

fn parse_mass_mode(s: &str) -> PyResult<MassMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "monoisotopic" | "mono" => Ok(MassMode::Monoisotopic),
        "average" | "avg" => Ok(MassMode::Average),
        other => Err(PyValueError::new_err(format!(
            "Unsupported mass_mode: {other}. Expected: monoisotopic, average."
        ))),
    }
}
// ---- Hyperscore helpers (stable; avoids factorial overflow) ----

fn ln_factorial(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).ln()).sum()
}

fn hyperscore(ny: usize, nb: usize, sum_y: f64, sum_b: f64) -> f64 {
    let sum = if (sum_y + sum_b) > 0.0 {
        sum_y + sum_b
    } else {
        1.0
    };
    ln_factorial(ny) + ln_factorial(nb) + sum.ln()
}

// ---- Feature helpers ----

fn longest_true_run(flags: &[bool]) -> usize {
    let mut max_run = 0usize;
    let mut cur = 0usize;
    for &v in flags {
        if v {
            cur += 1;
            max_run = max_run.max(cur);
        } else {
            cur = 0;
        }
    }
    max_run
}

/// Parse an ion string like "b5", "y7", possibly with extra suffixes (e.g. charge notation).
/// We extract the leading series letter and the first contiguous digit run.
fn parse_ion_series_and_index(ion: &str) -> Option<(char, usize)> {
    let ion = ion.trim();
    let mut chars = ion.chars();
    let series = chars.next()?;
    if series != 'b' && series != 'y' {
        return None;
    }
    let rest: String = chars.collect();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let idx = digits.parse::<usize>().ok()?;
    Some((series, idx))
}

// ---- Core batch function ----

#[pyfunction]
pub fn ms2_features_from_ms2spectra(
    py: Python<'_>,
    spectra: Vec<Py<MS2Spectrum>>,
    proformas: Vec<String>,
    seq_lens: Vec<usize>,
    fragmentation_model: String,
    mass_mode: String,
    calculate_hyperscore: bool,
) -> PyResult<Vec<HashMap<String, f64>>> {
    let n = spectra.len();
    if proformas.len() != n || seq_lens.len() != n {
        return Err(PyException::new_err(
            "Input arrays must have identical length: spectra, proformas, seq_lens",
        ));
    }

    // ---- Copy spectrum data out of Python objects (must hold GIL) ----
    #[derive(Clone)]
    struct OwnedSpec {
        id: String,
        mz: Vec<f64>,
        intensity: Vec<f64>,
        seq_len: usize,
        precursor_charge: i32,
        proforma: String,
    }

    let mut owned: Vec<OwnedSpec> = Vec::with_capacity(n);
    for i in 0..n {
        let spec_ref = spectra[i].bind(py);
        let spec = spec_ref.borrow();

        owned.push(OwnedSpec {
            id: spec.identifier.clone(),
            mz: spec.mz.iter().map(|&x| x as f64).collect(),
            intensity: spec.intensity.iter().map(|&x| x as f64).collect(),
            seq_len: seq_lens[i],
            precursor_charge: spec
                .precursor
                .as_ref()
                .map(|p| p.charge as i32)
                .unwrap_or(0),
            // mimic your Python: psm.peptidoform.proforma.split("/")[0]
            proforma: proformas[i]
                .split('/')
                .next()
                .unwrap_or(&proformas[i])
                .to_string(),
        });
    }

    // ---- Configure rustyms model/mode ----
    let model = parse_fragmentation_model(&fragmentation_model)?;
    let mode = parse_mass_mode(&mass_mode)?;

    // Matching parameters (tolerance etc.). Start with default; expose knobs later.
    let params = rustyms::annotation::model::MatchingParameters::default();

    // ---- Precompute theoretical fragments per unique peptide+charge+model ----
    // Charge included because fragment charge handling depends on precursor charge.
    type FragList = Vec<rustyms::fragment::Fragment>;

    let mut frag_cache: HashMap<(String, i32), Arc<FragList>> = HashMap::new();
    for item in &owned {
        let key = (item.proforma.clone(), item.precursor_charge);
        if item.precursor_charge <= 0 {
            frag_cache.insert(
                (item.proforma.clone(), item.precursor_charge),
                Arc::new(Vec::new()),
            );
            continue;
        }
        if frag_cache.contains_key(&key) {
            continue;
        }

        // Parse peptide
        let peptide = match CompoundPeptidoformIon::pro_forma(&item.proforma, None) {
            Ok(p) => p,
            Err(_) => {
                // Store empty fragments to mimic your Python behavior: return [] on parse/annotate failure
                frag_cache.insert(key, Arc::new(Vec::new()));
                continue;
            }
        };

        // Generate theoretical fragments up to the precursor charge.
        // If you want to cap this (e.g., min(charge, 2)) for speed, do it here.
        let frag_charge = rustyms::system::isize::Charge::new::<rustyms::system::e>(
            item.precursor_charge as isize,
        );
        let frags = peptide.generate_theoretical_fragments(frag_charge, &model);
        frag_cache.insert(key, Arc::new(frags));
    }

    let frag_cache = Arc::new(frag_cache);
    let params = Arc::new(params);

    // ---- Heavy work: release GIL and parallelize ----
    let results: Result<Vec<HashMap<String, f64>>, String> = py.detach(|| {
        owned
            .into_par_iter()
            .map(|item| {
                if item.mz.len() != item.intensity.len() {
                    return Err(format!(
                        "Spectrum {}: mz/intensity length mismatch",
                        item.id
                    ));
                }
                if item.seq_len == 0 {
                    return Ok(HashMap::new());
                }
                if item.precursor_charge <= 0 {
                    return Ok(HashMap::new());
                }

                let key = (item.proforma.clone(), item.precursor_charge);
                let empty: FragList = Vec::new();
                let frags: &FragList = frag_cache.get(&key).map(|x| x.as_ref()).unwrap_or(&empty);

                if frags.is_empty() {
                    return Ok(HashMap::new());
                }

                // Parse peptide again for annotation call (cheap compared to fragment generation; you can cache peptide too later)
                let peptide = match CompoundPeptidoformIon::pro_forma(&item.proforma, None) {
                    Ok(p) => p,
                    Err(_) => return Ok(HashMap::new()),
                };

                // Build a RawSpectrum (rustyms)
                let mut spectrum = RawSpectrum::default();
                spectrum.title = item.id.clone();
                spectrum.num_scans = 1;

                // Build peaks and extend into spectrum
                let peaks: Vec<RawPeak> = item
                    .mz
                    .iter()
                    .zip(item.intensity.iter())
                    .map(|(&mz, &inten)| RawPeak {
                        mz: MassOverCharge::new::<thomson>(mz),
                        intensity: OrderedFloat(inten),
                    })
                    .collect();

                spectrum.extend(peaks);

                // Annotate against precomputed fragments
                let annotated = spectrum.annotate(peptide, frags.as_slice(), &params, mode);

                // ---- Compute features matching your Python behavior ----
                let seq_len = item.seq_len;
                let mut b_flags = vec![false; seq_len];
                let mut y_flags = vec![false; seq_len];

                let pseudo = 1e-5_f64;
                let mut total_intensity = 0.0_f64;
                let mut matched_intensity = 0.0_f64;

                let mut b_sum = 0.0_f64;
                let mut y_sum = 0.0_f64;

                // For hyperscore parity: count “matched b/y intensities” per fragment annotation, not per peak
                let mut b_ints: Vec<f64> = Vec::new();
                let mut y_ints: Vec<f64> = Vec::new();

                for peak in annotated.spectrum() {
                    let inten = peak.intensity.into_inner();
                    total_intensity += inten;

                    if !peak.annotation.is_empty() {
                        matched_intensity += inten;

                        for frag in peak.annotation.iter() {
                            // Convert ion id to string and parse leading b/y + index
                            let ion_str = frag.ion.to_string();
                            if let Some((series, idx)) = parse_ion_series_and_index(&ion_str) {
                                if idx < seq_len {
                                    match series {
                                        'b' => {
                                            b_sum += inten;
                                            b_flags[idx] = true;
                                            if calculate_hyperscore {
                                                b_ints.push(inten);
                                            }
                                        }
                                        'y' => {
                                            y_sum += inten;
                                            y_flags[idx] = true;
                                            if calculate_hyperscore {
                                                y_ints.push(inten);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }

                let matched_b = b_flags.iter().filter(|&&x| x).count();
                let matched_y = y_flags.iter().filter(|&&x| x).count();

                let mut feats: HashMap<String, f64> = HashMap::new();
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

                let b_ratio = if matched_intensity > 0.0 {
                    (b_sum / matched_intensity + pseudo).ln()
                } else {
                    pseudo.ln()
                };
                feats.insert("ln_explained_b_ion_ratio".to_string(), b_ratio);

                let y_ratio = if matched_intensity > 0.0 {
                    (y_sum / matched_intensity + pseudo).ln()
                } else {
                    pseudo.ln()
                };
                feats.insert("ln_explained_y_ion_ratio".to_string(), y_ratio);

                feats.insert(
                    "longest_b_ion_sequence".to_string(),
                    longest_true_run(&b_flags) as f64,
                );
                feats.insert(
                    "longest_y_ion_sequence".to_string(),
                    longest_true_run(&y_flags) as f64,
                );

                feats.insert("matched_b_ions".to_string(), matched_b as f64);
                feats.insert(
                    "matched_b_ions_pct".to_string(),
                    matched_b as f64 / seq_len as f64,
                );

                feats.insert("matched_y_ions".to_string(), matched_y as f64);
                feats.insert(
                    "matched_y_ions_pct".to_string(),
                    matched_y as f64 / seq_len as f64,
                );

                feats.insert(
                    "matched_ions_pct".to_string(),
                    (matched_b + matched_y) as f64 / (2.0 * seq_len as f64),
                );

                if calculate_hyperscore {
                    let ny = y_ints.len();
                    let nb = b_ints.len();
                    let sum_y: f64 = y_ints.iter().sum();
                    let sum_b: f64 = b_ints.iter().sum();
                    feats.insert("hyperscore".to_string(), hyperscore(ny, nb, sum_y, sum_b));
                }

                Ok(feats)
            })
            .collect()
    });

    match results {
        Ok(v) => Ok(v),
        Err(e) => Err(PyException::new_err(e)),
    }
}
