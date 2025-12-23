use pyo3::prelude::*;

use crate::precursor::Precursor;

#[pyclass(module = "ms2rescore_rs", get_all, set_all)]
#[derive(Debug, Clone)]
pub struct MS2Spectrum {
    pub identifier: String,
    pub mz: Vec<f32>,
    pub intensity: Vec<f32>,
    pub precursor: Option<Precursor>,
}

impl MS2Spectrum {
    pub fn new(
        identifier: String,
        mz: Vec<f32>,
        intensity: Vec<f32>,
        precursor: Option<Precursor>,
    ) -> Self {
        MS2Spectrum {
            identifier,
            mz,
            intensity,
            precursor,
        }
    }
}

#[pymethods]
impl MS2Spectrum {
    #[new]
    #[pyo3(signature = (identifier="".to_string(), mz=vec![], intensity=vec![], precursor=None))]
    pub fn py_new(
        identifier: String,
        mz: Vec<f32>,
        intensity: Vec<f32>,
        precursor: Option<Precursor>,
    ) -> Self {
        MS2Spectrum::new(identifier, mz, intensity, precursor)
    }

    fn __repr__(&self) -> String {
        format!(
            "MS2Spectrum(identifier='{}', mz=[..], intensity=[..], precursor={:?})",
            self.identifier, self.precursor
        )
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, (String, Vec<f32>, Vec<f32>, Option<Precursor>))> {
        let cls = py.import("ms2rescore_rs")?.getattr("MS2Spectrum")?;
        Ok((cls.into(), (self.identifier.clone(), self.mz.clone(), self.intensity.clone(), self.precursor.clone())))
    }
}
