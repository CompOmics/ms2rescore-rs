use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::types::annotation::{AnnotatedMS2Spectrum, FragmentAnnotation};
use crate::types::ms2_spectrum::MS2Spectrum;

use ordered_float::OrderedFloat;
use rustyms::annotation::model::FragmentationModel;
use rustyms::annotation::AnnotatableSpectrum;
use rustyms::chemistry::MassMode;
use rustyms::prelude::CompoundPeptidoformIon;
use rustyms::spectrum::{PeakSpectrum, RawPeak, RawSpectrum};
use rustyms::system::f64::MassOverCharge;
use rustyms::system::mass_over_charge::thomson;

pub(crate) fn parse_fragmentation_model(s: &str) -> PyResult<FragmentationModel> {
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

pub(crate) fn parse_mass_mode(s: &str) -> PyResult<MassMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "monoisotopic" | "mono" => Ok(MassMode::Monoisotopic),
        "average" | "avg" => Ok(MassMode::Average),
        other => Err(PyValueError::new_err(format!(
            "Unsupported mass_mode: {other}. Expected: monoisotopic, average."
        ))),
    }
}

fn parse_tolerance(
    tolerance_value: f64,
    tolerance_mode: &str,
) -> PyResult<rustyms::quantities::Tolerance<MassOverCharge>> {
    match tolerance_mode.trim().to_ascii_lowercase().as_str() {
        "ppm" => Ok(rustyms::quantities::Tolerance::new_ppm(tolerance_value)),
        "da" => Ok(rustyms::quantities::Tolerance::new_absolute(
            MassOverCharge::new::<thomson>(tolerance_value),
        )),
        other => Err(PyValueError::new_err(format!(
            "Unsupported tolerance_mode: {other}. Expected: ppm, Da."
        ))),
    }
}

/// Parse an ion string like "b5", "y7", "c3", "z12", possibly with extra suffixes.
/// Extracts the leading series letter and the first contiguous digit run.
/// Supports all 6 primary ion series: a, b, c, x, y, z.
fn parse_ion_series_and_index(ion: &str) -> Option<(char, usize)> {
    let ion = ion.trim();
    let mut chars = ion.chars();
    let series = chars.next()?;
    if !matches!(series, 'a' | 'b' | 'c' | 'x' | 'y' | 'z') {
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

/// Parse charge from a rustyms Fragment's charge field.
fn extract_fragment_charge(frag: &rustyms::fragment::Fragment) -> usize {
    frag.charge
        .value
        .abs()
        .round() as usize
}

/// Annotate MS2 spectra with theoretical fragment ions.
#[pyfunction]
pub fn annotate_ms2_spectra(
    py: Python<'_>,
    spectra: Vec<Py<MS2Spectrum>>,
    proformas: Vec<String>,
    seq_lens: Vec<usize>,
    fragmentation_model: String,
    mass_mode: String,
    tolerance_value: f64,
    tolerance_mode: String,
) -> PyResult<Vec<AnnotatedMS2Spectrum>> {
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
        mz_f32: Vec<f32>,
        intensity_f32: Vec<f32>,
        mz: Vec<f64>,
        intensity: Vec<f64>,
        precursor: Option<crate::types::precursor::Precursor>,
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
            mz_f32: spec.mz.clone(),
            intensity_f32: spec.intensity.clone(),
            mz: spec.mz.iter().map(|&x| x as f64).collect(),
            intensity: spec.intensity.iter().map(|&x| x as f64).collect(),
            precursor: spec.precursor.clone(),
            seq_len: seq_lens[i],
            precursor_charge: spec
                .precursor
                .as_ref()
                .map(|p| p.charge as i32)
                .unwrap_or(0),
            proforma: proformas[i]
                .split('/')
                .next()
                .unwrap_or(&proformas[i])
                .to_string(),
        });
    }

    // ---- Configure rustyms model/mode/tolerance ----
    let model = parse_fragmentation_model(&fragmentation_model)?;
    let mode = parse_mass_mode(&mass_mode)?;
    let tolerance = parse_tolerance(tolerance_value, &tolerance_mode)?;
    let params = rustyms::annotation::model::MatchingParameters::default().tolerance(tolerance);

    // ---- Precompute theoretical fragments per unique peptide+charge+model ----
    type FragList = Vec<rustyms::fragment::Fragment>;

    let mut frag_cache: HashMap<(String, i32), Arc<FragList>> = HashMap::new();
    for item in &owned {
        let key = (item.proforma.clone(), item.precursor_charge);
        if item.precursor_charge <= 0 {
            frag_cache.insert(key, Arc::new(Vec::new()));
            continue;
        }
        if frag_cache.contains_key(&key) {
            continue;
        }

        let peptide = match CompoundPeptidoformIon::pro_forma(&item.proforma, None) {
            Ok(p) => p,
            Err(_) => {
                frag_cache.insert(key, Arc::new(Vec::new()));
                continue;
            }
        };

        let frag_charge = rustyms::system::isize::Charge::new::<rustyms::system::e>(
            item.precursor_charge as isize,
        );
        let frags = peptide.generate_theoretical_fragments(frag_charge, &model);
        frag_cache.insert(key, Arc::new(frags));
    }

    let frag_cache = Arc::new(frag_cache);
    let params = Arc::new(params);

    // ---- Heavy work: release GIL and parallelize ----
    let results: Result<Vec<AnnotatedMS2Spectrum>, String> = py.detach(|| {
        owned
            .into_par_iter()
            .map(|item| {
                if item.mz.len() != item.intensity.len() {
                    return Err(format!(
                        "Spectrum {}: mz/intensity length mismatch",
                        item.id
                    ));
                }

                let n_peaks = item.mz.len();

                // For spectra with no valid peptide/charge, return empty annotations
                if item.seq_len == 0 || item.precursor_charge <= 0 {
                    return Ok(AnnotatedMS2Spectrum {
                        identifier: item.id,
                        mz: item.mz_f32,
                        intensity: item.intensity_f32,
                        precursor: item.precursor,
                        peak_annotations: vec![Vec::new(); n_peaks],
                    });
                }

                let key = (item.proforma.clone(), item.precursor_charge);
                let empty: FragList = Vec::new();
                let frags: &FragList =
                    frag_cache.get(&key).map(|x| x.as_ref()).unwrap_or(&empty);

                if frags.is_empty() {
                    return Ok(AnnotatedMS2Spectrum {
                        identifier: item.id,
                        mz: item.mz_f32,
                        intensity: item.intensity_f32,
                        precursor: item.precursor,
                        peak_annotations: vec![Vec::new(); n_peaks],
                    });
                }

                let peptide = match CompoundPeptidoformIon::pro_forma(&item.proforma, None) {
                    Ok(p) => p,
                    Err(_) => {
                        return Ok(AnnotatedMS2Spectrum {
                            identifier: item.id,
                            mz: item.mz_f32,
                            intensity: item.intensity_f32,
                            precursor: item.precursor,
                            peak_annotations: vec![Vec::new(); n_peaks],
                        });
                    }
                };

                // Build a RawSpectrum (rustyms)
                let mut spectrum = RawSpectrum::default();
                spectrum.title = item.id.clone();
                spectrum.num_scans = 1;

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

                // Convert rustyms AnnotatedSpectrum to our peak-centric representation
                let peak_annotations: Vec<Vec<FragmentAnnotation>> = annotated
                    .spectrum()
                    .iter()
                    .map(|peak| {
                        peak.annotation
                            .iter()
                            .filter_map(|frag| {
                                let ion_str = frag.ion.to_string();
                                let (series, position) = parse_ion_series_and_index(&ion_str)?;
                                let charge = extract_fragment_charge(frag);
                                Some(FragmentAnnotation {
                                    series: series.to_string(),
                                    position,
                                    charge,
                                })
                            })
                            .collect()
                    })
                    .collect();

                Ok(AnnotatedMS2Spectrum {
                    identifier: item.id,
                    mz: item.mz_f32,
                    intensity: item.intensity_f32,
                    precursor: item.precursor,
                    peak_annotations,
                })
            })
            .collect()
    });

    match results {
        Ok(v) => Ok(v),
        Err(e) => Err(PyException::new_err(e)),
    }
}
