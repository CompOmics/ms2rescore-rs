use pyo3::prelude::*;

/// Precursor information.
#[pyclass(module = "ms2rescore_rs", get_all, set_all)]
#[derive(Debug, Clone)]
pub struct Precursor {
    pub mz: f64,
    pub rt: f64,
    pub im: f64,
    pub charge: usize,
    pub intensity: f64,
}

#[pymethods]
impl Precursor {
    #[new]
    #[pyo3(signature = (mz=0.0, rt=0.0, im=0.0, charge=0, intensity=0.0))]
    pub fn new(mz: f64, rt: f64, im: f64, charge: usize, intensity: f64) -> Self {
        Precursor {
            mz,
            rt,
            im,
            charge,
            intensity,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Precursor(mz={}, rt={}, im={}, charge={}, intensity={})",
            self.mz, self.rt, self.im, self.charge, self.intensity
        )
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, (f64, f64, f64, usize, f64))> {
        let cls = py.import("ms2rescore_rs")?.getattr("Precursor")?;
        Ok((cls.into(), (self.mz, self.rt, self.im, self.charge, self.intensity)))
    }
}

impl Default for Precursor {
    fn default() -> Self {
        Precursor {
            mz: 0.0,
            rt: 0.0,
            im: 0.0,
            charge: 0,
            intensity: 0.0,
        }
    }
}
