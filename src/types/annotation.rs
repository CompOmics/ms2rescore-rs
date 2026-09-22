use pyo3::prelude::*;

use crate::types::precursor::Precursor;

type AnnotatedMS2SpectrumReduceArgs = (
    String,
    Vec<f32>,
    Vec<f32>,
    Option<Precursor>,
    Vec<Vec<FragmentAnnotation>>,
    Vec<Vec<FragmentAnnotation>>,
);
type FragmentAnnotationReduceArgs = (String, usize, usize, String, String, f64, f64);

/// A single fragment annotation on a peak.
#[pyclass(module = "ms2rescore_rs", get_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct FragmentAnnotation {
    /// Ion series: "a", "b", "c", "x", "y", "z"
    pub series: String,
    /// 1-indexed ion position along the peptide backbone
    pub position: usize,
    /// Fragment charge state
    pub charge: usize,
    /// Fragment kind: "backbone", "satellite", "precursor", "diagnostic", "immonium"
    pub ion_type: String,
    /// Neutral loss label in Hill notation (e.g. "-H3PO4"), empty when none
    pub neutral_loss: String,
    /// Monoisotopic mass of the neutral loss (positive for a loss), 0.0 when none
    pub loss_mass: f64,
    /// Observed minus theoretical m/z of the matched peak (Th)
    pub mz_error: f64,
}

#[pymethods]
impl FragmentAnnotation {
    #[new]
    #[pyo3(signature = (series, position, charge, ion_type="backbone".to_string(), neutral_loss=String::new(), loss_mass=0.0, mz_error=0.0))]
    pub fn new(
        series: String,
        position: usize,
        charge: usize,
        ion_type: String,
        neutral_loss: String,
        loss_mass: f64,
        mz_error: f64,
    ) -> Self {
        FragmentAnnotation {
            series,
            position,
            charge,
            ion_type,
            neutral_loss,
            loss_mass,
            mz_error,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FragmentAnnotation(series='{}', position={}, charge={}, ion_type='{}', neutral_loss='{}', loss_mass={}, mz_error={:.5})",
            self.series, self.position, self.charge, self.ion_type, self.neutral_loss, self.loss_mass, self.mz_error
        )
    }

    pub fn __reduce__(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyAny>, FragmentAnnotationReduceArgs)> {
        let cls = py.import("ms2rescore_rs")?.getattr("FragmentAnnotation")?;
        Ok((
            cls.into(),
            (
                self.series.clone(),
                self.position,
                self.charge,
                self.ion_type.clone(),
                self.neutral_loss.clone(),
                self.loss_mass,
                self.mz_error,
            ),
        ))
    }
}

/// An MS2 spectrum annotated with fragment ion assignments.
///
/// Contains the original spectrum data alongside peak-centric annotations.
/// Each entry in `peak_annotations` corresponds to the peak at the same index
/// in `mz` / `intensity`.
#[pyclass(module = "ms2rescore_rs", from_py_object)]
#[derive(Debug, Clone)]
pub struct AnnotatedMS2Spectrum {
    /// Spectrum identifier
    #[pyo3(get)]
    pub identifier: String,
    /// Original m/z values
    #[pyo3(get)]
    pub mz: Vec<f32>,
    /// Original intensity values
    #[pyo3(get)]
    pub intensity: Vec<f32>,
    /// Original precursor information
    #[pyo3(get)]
    pub precursor: Option<Precursor>,
    /// Loss-free backbone ion (a, b, c, x, y, z) matches, stored sparsely as
    /// (peak index, annotation) sorted by peak index. Most peaks are unmatched, so one list
    /// per peak would cost several times more than the peak data itself. Exposed to Python
    /// as a per-peak list of lists via the `peak_annotations` getter, and as the raw pairs via
    /// `backbone`, which avoids rebuilding that view when a consumer only needs the matches.
    #[pyo3(get)]
    pub backbone: Vec<(u32, FragmentAnnotation)>,
    /// Extended annotations (neutral-loss variants, precursor, diagnostic, immonium and
    /// satellite ions) stored sparsely as (peak index, annotation), sorted by peak index.
    /// Exposed to Python as a per-peak list of lists via the `extended_annotations` getter, and
    /// as the raw pairs via `extended`.
    #[pyo3(get)]
    pub extended: Vec<(u32, FragmentAnnotation)>,
    /// Whether the spectrum was annotated with `extended=True`. When false the Python
    /// `extended_annotations` is an empty list instead of one empty list per peak.
    pub has_extended: bool,
}

#[pymethods]
impl AnnotatedMS2Spectrum {
    #[new]
    #[pyo3(signature = (identifier="".to_string(), mz=vec![], intensity=vec![], precursor=None, peak_annotations=vec![], extended_annotations=vec![]))]
    pub fn new(
        identifier: String,
        mz: Vec<f32>,
        intensity: Vec<f32>,
        precursor: Option<Precursor>,
        peak_annotations: Vec<Vec<FragmentAnnotation>>,
        extended_annotations: Vec<Vec<FragmentAnnotation>>,
    ) -> Self {
        let has_extended = !extended_annotations.is_empty();
        AnnotatedMS2Spectrum {
            identifier,
            mz,
            intensity,
            precursor,
            backbone: sparse(peak_annotations),
            extended: sparse(extended_annotations),
            has_extended,
        }
    }

    /// Per-peak fragment annotations. `peak_annotations[i]` lists the loss-free backbone ion
    /// matches for peak `i`; an empty list means the peak is unmatched.
    #[getter]
    pub fn peak_annotations(&self) -> Vec<Vec<FragmentAnnotation>> {
        per_peak(&self.backbone, self.mz.len())
    }

    /// Per-peak extended annotations: neutral-loss variants, precursor, diagnostic,
    /// immonium and satellite ions. Empty list unless annotated with `extended=True`.
    #[getter]
    pub fn extended_annotations(&self) -> Vec<Vec<FragmentAnnotation>> {
        if !self.has_extended {
            return Vec::new();
        }
        per_peak(&self.extended, self.mz.len())
    }

    fn __repr__(&self) -> String {
        let mut peaks: Vec<u32> = self.backbone.iter().map(|(idx, _)| *idx).collect();
        peaks.dedup();
        let n_annotated = peaks.len();
        format!(
            "AnnotatedMS2Spectrum(identifier='{}', peaks={}, annotated={})",
            self.identifier,
            self.mz.len(),
            n_annotated
        )
    }

    pub fn __reduce__(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyAny>, AnnotatedMS2SpectrumReduceArgs)> {
        let cls = py
            .import("ms2rescore_rs")?
            .getattr("AnnotatedMS2Spectrum")?;
        Ok((
            cls.into(),
            (
                self.identifier.clone(),
                self.mz.clone(),
                self.intensity.clone(),
                self.precursor.clone(),
                self.peak_annotations(),
                self.extended_annotations(),
            ),
        ))
    }
}

/// Flatten a per-peak list of lists into (peak index, annotation) pairs, sorted by peak index.
fn sparse(per_peak: Vec<Vec<FragmentAnnotation>>) -> Vec<(u32, FragmentAnnotation)> {
    per_peak
        .into_iter()
        .enumerate()
        .flat_map(|(idx, anns)| anns.into_iter().map(move |a| (idx as u32, a)))
        .collect()
}

/// Expand (peak index, annotation) pairs back into one list per peak.
fn per_peak(sparse: &[(u32, FragmentAnnotation)], n_peaks: usize) -> Vec<Vec<FragmentAnnotation>> {
    let mut out = vec![Vec::new(); n_peaks];
    for (idx, ann) in sparse {
        out[*idx as usize].push(ann.clone());
    }
    out
}
