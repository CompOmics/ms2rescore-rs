use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::types::annotation::{AnnotatedMS2Spectrum, FragmentAnnotation};
use crate::types::ms2_spectrum::MS2Spectrum;
use crate::utils::parse_ion_series_and_index;

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


fn extract_fragment_charge(frag: &rustyms::fragment::Fragment) -> usize {
    frag.charge.value.unsigned_abs()
}

/// Annotate MS2 spectra with theoretical fragment ions.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
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

    // Copy spectrum data out of Python objects (must hold GIL)
    #[derive(Clone)]
    struct OwnedSpec {
        id: String,
        mz_f32: Vec<f32>,
        intensity_f32: Vec<f32>,
        precursor: Option<crate::types::precursor::Precursor>,
        seq_len: usize,
        precursor_charge: i32,
        proforma: String,
    }

    impl OwnedSpec {
        fn empty_annotated(self) -> AnnotatedMS2Spectrum {
            let n_peaks = self.mz_f32.len();
            AnnotatedMS2Spectrum {
                identifier: self.id,
                mz: self.mz_f32,
                intensity: self.intensity_f32,
                precursor: self.precursor,
                peak_annotations: vec![Vec::new(); n_peaks],
            }
        }
    }

    let mut owned: Vec<OwnedSpec> = Vec::with_capacity(n);
    for i in 0..n {
        let spec_ref = spectra[i].bind(py);
        let spec = spec_ref.borrow();

        owned.push(OwnedSpec {
            id: spec.identifier.clone(),
            mz_f32: spec.mz.clone(),
            intensity_f32: spec.intensity.clone(),
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

    let model = parse_fragmentation_model(&fragmentation_model)?;
    let mode = parse_mass_mode(&mass_mode)?;
    let tolerance = parse_tolerance(tolerance_value, &tolerance_mode)?;
    let params = rustyms::annotation::model::MatchingParameters::default().tolerance(tolerance);

    // Precompute theoretical fragments and parsed peptides per unique peptide+charge
    type CacheEntry = (CompoundPeptidoformIon, Vec<rustyms::fragment::Fragment>);

    let mut frag_cache: HashMap<(String, i32), Arc<Option<CacheEntry>>> = HashMap::new();
    for item in &owned {
        let key = (item.proforma.clone(), item.precursor_charge);
        if item.precursor_charge <= 0 || frag_cache.contains_key(&key) {
            continue;
        }

        let entry = CompoundPeptidoformIon::pro_forma(&item.proforma, None)
            .ok()
            .map(|peptide| {
                let frag_charge = rustyms::system::isize::Charge::new::<rustyms::system::e>(
                    item.precursor_charge as isize,
                );
                let frags = peptide.generate_theoretical_fragments(frag_charge, &model);
                (peptide, frags)
            });

        frag_cache.insert(key, Arc::new(entry));
    }

    let frag_cache = Arc::new(frag_cache);
    let params = Arc::new(params);

    // Release GIL and parallelize
    let results: Result<Vec<AnnotatedMS2Spectrum>, String> = py.detach(|| {
        owned
            .into_par_iter()
            .map(|item| {
                if item.mz_f32.len() != item.intensity_f32.len() {
                    return Err(format!(
                        "Spectrum {}: mz/intensity length mismatch",
                        item.id
                    ));
                }

                if item.mz_f32.is_empty() || item.seq_len == 0 || item.precursor_charge <= 0 {
                    return Ok(item.empty_annotated());
                }

                let key = (item.proforma.clone(), item.precursor_charge);
                let cache_entry = frag_cache.get(&key).and_then(|e| e.as_ref().as_ref());

                let (peptide, frags) = match cache_entry {
                    Some((p, f)) if !f.is_empty() => (p, f),
                    _ => return Ok(item.empty_annotated()),
                };

                // Build RawSpectrum from f32 data (convert to f64 in-place, no pre-stored copy)
                let mut spectrum = RawSpectrum::default();
                spectrum.title = item.id.clone();
                spectrum.num_scans = 1;

                let peaks: Vec<RawPeak> = item
                    .mz_f32
                    .iter()
                    .zip(item.intensity_f32.iter())
                    .map(|(&mz, &inten)| RawPeak {
                        mz: MassOverCharge::new::<thomson>(mz as f64),
                        intensity: OrderedFloat(inten as f64),
                    })
                    .collect();

                spectrum.extend(peaks);

                let annotated = spectrum.annotate(peptide.clone(), frags.as_slice(), &params, mode);

                let peak_annotations: Vec<Vec<FragmentAnnotation>> = annotated
                    .spectrum()
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
