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
type FragmentAnnotationReduceArgs = (String, usize, usize, String, String, f64);

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
}

#[pymethods]
impl FragmentAnnotation {
    #[new]
    #[pyo3(signature = (series, position, charge, ion_type="backbone".to_string(), neutral_loss=String::new(), loss_mass=0.0))]
    pub fn new(
        series: String,
        position: usize,
        charge: usize,
        ion_type: String,
        neutral_loss: String,
        loss_mass: f64,
    ) -> Self {
        FragmentAnnotation {
            series,
            position,
            charge,
            ion_type,
            neutral_loss,
            loss_mass,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FragmentAnnotation(series='{}', position={}, charge={}, ion_type='{}', neutral_loss='{}', loss_mass={})",
            self.series, self.position, self.charge, self.ion_type, self.neutral_loss, self.loss_mass
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
            ),
        ))
    }
}

/// An MS2 spectrum annotated with fragment ion assignments.
///
/// Contains the original spectrum data alongside peak-centric annotations.
/// Each entry in `peak_annotations` corresponds to the peak at the same index
/// in `mz` / `intensity`.
#[pyclass(module = "ms2rescore_rs", get_all, from_py_object)]
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
    /// Only loss-free backbone ions (a, b, c, x, y, z) are listed here.
    pub peak_annotations: Vec<Vec<FragmentAnnotation>>,
    /// Per-peak extended annotations: neutral-loss variants, precursor, diagnostic,
    /// immonium and satellite ions. Empty unless annotated with `extended=True`.
    pub extended_annotations: Vec<Vec<FragmentAnnotation>>,
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
        AnnotatedMS2Spectrum {
            identifier,
            mz,
            intensity,
            precursor,
            peak_annotations,
            extended_annotations,
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
                self.peak_annotations.clone(),
                self.extended_annotations.clone(),
            ),
        ))
    }
}
