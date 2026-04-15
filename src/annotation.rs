use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::types::annotation::{AnnotatedMS2Spectrum, FragmentAnnotation};
use crate::types::ms2_spectrum::MS2Spectrum;
use crate::utils::parse_fragment;

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


/// Annotate MS2 spectra with theoretical fragment ions.
#[pyfunction]
pub fn annotate_ms2_spectra(
    py: Python<'_>,
    spectra: Vec<Py<MS2Spectrum>>,
    proformas: Vec<String>,
    fragmentation_model: String,
    mass_mode: String,
    tolerance_value: f64,
    tolerance_mode: String,
) -> PyResult<Vec<AnnotatedMS2Spectrum>> {
    let n = spectra.len();
    if proformas.len() != n {
        return Err(PyException::new_err(
            "Input arrays must have identical length: spectra, proformas",
        ));
    }

    // Copy spectrum data out of Python objects (must hold GIL)
    #[derive(Clone)]
    struct OwnedSpec {
        id: String,
        mz_f32: Vec<f32>,
        intensity_f32: Vec<f32>,
        precursor: Option<crate::types::precursor::Precursor>,
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

        let proforma = proformas[i]
            .split('/')
            .next()
            .unwrap_or(&proformas[i])
            .to_string();

        owned.push(OwnedSpec {
            id: spec.identifier.clone(),
            mz_f32: spec.mz.clone(),
            intensity_f32: spec.intensity.clone(),
            precursor: spec.precursor.clone(),
            precursor_charge: spec
                .precursor
                .as_ref()
                .map(|p| p.charge as i32)
                .unwrap_or(0),
            proforma,
        });
    }

    let model = parse_fragmentation_model(&fragmentation_model)?;
    let mode = parse_mass_mode(&mass_mode)?;
    let tolerance = parse_tolerance(tolerance_value, &tolerance_mode)?;
    let params = rustyms::annotation::model::MatchingParameters::default().tolerance(tolerance);

    // Precompute theoretical fragments and parsed peptides per unique peptide+charge
    struct CacheEntry {
        peptide: CompoundPeptidoformIon,
        fragments: Vec<rustyms::fragment::Fragment>,
        seq_len: usize,
    }

    let unique_keys: Vec<(String, i32)> = {
        let mut seen = HashMap::new();
        for item in &owned {
            if item.precursor_charge > 0 {
                seen.entry((item.proforma.clone(), item.precursor_charge))
                    .or_insert(());
            }
        }
        seen.into_keys().collect()
    };

    let params = Arc::new(params);

    // Release GIL and parallelize both cache building and annotation
    let results: Result<Vec<AnnotatedMS2Spectrum>, String> = py.detach(|| {
        let frag_cache: Arc<HashMap<(String, i32), Arc<Option<CacheEntry>>>> = Arc::new(
            unique_keys
                .into_par_iter()
                .map(|(proforma, charge)| {
                    let entry = CompoundPeptidoformIon::pro_forma(&proforma, None)
                        .ok()
                        .map(|peptide| {
                            let seq_len = peptide
                                .peptidoforms()
                                .next()
                                .map(|pf| pf.sequence().len())
                                .unwrap_or(0);
                            let frag_charge =
                                rustyms::system::isize::Charge::new::<rustyms::system::e>(
                                    charge as isize,
                                );
                            let fragments =
                                peptide.generate_theoretical_fragments(frag_charge, &model);
                            CacheEntry {
                                peptide,
                                fragments,
                                seq_len,
                            }
                        });
                    ((proforma, charge), Arc::new(entry))
                })
                .collect(),
        );

        owned
            .into_par_iter()
            .map(|item| {
                if item.mz_f32.len() != item.intensity_f32.len() {
                    return Err(format!(
                        "Spectrum {}: mz/intensity length mismatch",
                        item.id
                    ));
                }

                if item.mz_f32.is_empty() || item.precursor_charge <= 0 {
                    return Ok(item.empty_annotated());
                }

                let key = (item.proforma.clone(), item.precursor_charge);
                let cache_entry = frag_cache.get(&key).and_then(|e| e.as_ref().as_ref());

                let entry = match cache_entry {
                    Some(e) if e.seq_len > 0 && !e.fragments.is_empty() => e,
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

                let annotated =
                    spectrum.annotate(entry.peptide.clone(), &entry.fragments, &params, mode);

                let peak_annotations: Vec<Vec<FragmentAnnotation>> = annotated
                    .spectrum()
                    .map(|peak| {
                        peak.annotation
                            .iter()
                            .filter_map(|frag| {
                                let (series, position, charge) = parse_fragment(frag)?;
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
