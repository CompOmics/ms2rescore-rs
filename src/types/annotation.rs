use pyo3::prelude::*;

use crate::types::precursor::Precursor;

/// A single fragment annotation on a peak.
#[pyclass(module = "ms2rescore_rs", get_all)]
#[derive(Debug, Clone)]
pub struct FragmentAnnotation {
    /// Ion series: "a", "b", "c", "x", "y", "z"
    pub series: String,
    /// 1-indexed ion position along the peptide backbone
    pub position: usize,
    /// Fragment charge state
    pub charge: usize,
}

#[pymethods]
impl FragmentAnnotation {
    #[new]
    pub fn new(series: String, position: usize, charge: usize) -> Self {
        FragmentAnnotation {
            series,
            position,
            charge,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FragmentAnnotation(series='{}', position={}, charge={})",
            self.series, self.position, self.charge
        )
    }
}

/// An MS2 spectrum annotated with fragment ion assignments.
///
/// Contains the original spectrum data alongside peak-centric annotations.
/// Each entry in `peak_annotations` corresponds to the peak at the same index
/// in `mz` / `intensity`.
#[pyclass(module = "ms2rescore_rs", get_all)]
#[derive(Debug, Clone)]
pub struct AnnotatedMS2Spectrum {
    /// Spectrum identifier
    pub identifier: String,
    /// Original m/z values
    pub mz: Vec<f32>,
    /// Original intensity values
    pub intensity: Vec<f32>,
    /// Original precursor information
    pub precursor: Option<Precursor>,
    /// Per-peak fragment annotations. `peak_annotations[i]` lists the fragment
    /// matches for peak `i`. An empty vec means the peak is unmatched.
    pub peak_annotations: Vec<Vec<FragmentAnnotation>>,
}

#[pymethods]
impl AnnotatedMS2Spectrum {
    #[new]
    #[pyo3(signature = (identifier="".to_string(), mz=vec![], intensity=vec![], precursor=None, peak_annotations=vec![]))]
    pub fn new(
        identifier: String,
        mz: Vec<f32>,
        intensity: Vec<f32>,
        precursor: Option<Precursor>,
        peak_annotations: Vec<Vec<FragmentAnnotation>>,
    ) -> Self {
        AnnotatedMS2Spectrum {
            identifier,
            mz,
            intensity,
            precursor,
            peak_annotations,
        }
    }

    fn __repr__(&self) -> String {
        let n_annotated = self
            .peak_annotations
            .iter()
            .filter(|a| !a.is_empty())
            .count();
        format!(
            "AnnotatedMS2Spectrum(identifier='{}', peaks={}, annotated={})",
            self.identifier,
            self.mz.len(),
            n_annotated
        )
    }
}
