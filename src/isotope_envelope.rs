use std::collections::HashMap;

use mzdata::prelude::*;
use mzdata::MZReader;
use pyo3::prelude::*;

const NEUTRON: f64 = 1.003355;

/// Isotope envelope extracted from an MS1 scan for a precursor.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct IsotopeEnvelope {
    /// Spectrum identifier (of the MS2 scan this precursor belongs to)
    pub spectrum_id: String,
    /// Peptide sequence (if known)
    pub peptide: String,
    /// Monoisotopic m/z
    pub mono_mz: f64,
    /// Charge state
    pub charge: usize,
    /// Retention time of the MS1 scan
    pub rt: f64,
    /// m/z values of isotope peaks [M0, M1, M2, ...]
    pub mz_values: Vec<f64>,
    /// Intensities of isotope peaks (raw, not normalized)
    pub intensities: Vec<f64>,
    /// Normalized intensities (sum to 1)
    pub normalized: Vec<f64>,
}

#[pymethods]
impl IsotopeEnvelope {
    pub fn __repr__(&self) -> String {
        format!(
            "IsotopeEnvelope(spectrum_id={}, mono_mz={:.4}, charge={}, peaks={})",
            self.spectrum_id,
            self.mono_mz,
            self.charge,
            self.intensities.len()
        )
    }
}

/// Extract isotope peaks from a centroided MS1 spectrum at a given precursor m/z.
///
/// Looks for peaks at mono_mz + k * NEUTRON / charge for k = 0..n_peaks.
fn extract_envelope_from_peaks(
    ms1_mzs: &[f64],
    ms1_intensities: &[f64],
    mono_mz: f64,
    charge: usize,
    n_peaks: usize,
    ppm_tolerance: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut mz_values = Vec::with_capacity(n_peaks);
    let mut intensities = Vec::with_capacity(n_peaks);
    let spacing = NEUTRON / charge as f64;

    for k in 0..n_peaks {
        let target_mz = mono_mz + k as f64 * spacing;
        let tol = target_mz * ppm_tolerance / 1e6;

        // Find the peak CLOSEST to expected m/z within tolerance
        let mut best_intensity = 0.0f64;
        let mut best_mz = target_mz;
        let mut best_dist = f64::MAX;

        let lo = ms1_mzs
            .partition_point(|&mz| mz < target_mz - tol);
        let hi = ms1_mzs
            .partition_point(|&mz| mz <= target_mz + tol);

        for i in lo..hi {
            let dist = (ms1_mzs[i] - target_mz).abs();
            if dist < best_dist {
                best_dist = dist;
                best_intensity = ms1_intensities[i] as f64;
                best_mz = ms1_mzs[i];
            }
        }

        mz_values.push(best_mz);
        intensities.push(best_intensity);
    }

    (mz_values, intensities)
}

/// Extract isotope envelopes from a raw spectrum file (Thermo .RAW, mzML, etc.)
///
/// For each MS2 spectrum, finds the preceding MS1 scan and extracts
/// the isotope envelope around the precursor m/z.
pub fn extract_envelopes_mzdata(
    spectrum_path: &str,
    n_peaks: usize,
    ppm_tolerance: f64,
) -> Result<Vec<IsotopeEnvelope>, std::io::Error> {
    let mut reader = MZReader::open_path(spectrum_path)?;

    // Enable centroiding for Thermo files
    if let MZReader::ThermoRaw(inner) = &mut reader {
        inner.set_centroiding(true);
    }

    let mut envelopes = Vec::new();
    let mut last_ms1_mzs: Vec<f64> = Vec::new();
    let mut last_ms1_intensities: Vec<f64> = Vec::new();
    let mut last_ms1_rt: f64 = 0.0;

    for spectrum in reader {
        let ms_level = spectrum.description.ms_level;

        if ms_level == 1 {
            // Store MS1 data for subsequent MS2 lookups
            let centroided = spectrum.into_centroid().unwrap();
            last_ms1_mzs = centroided.peaks.iter().map(|p| p.mz).collect();
            last_ms1_intensities = centroided.peaks.iter().map(|p| p.intensity as f64).collect();
            last_ms1_rt = centroided
                .description
                .acquisition
                .first_scan()
                .map(|s| s.start_time)
                .unwrap_or(0.0);
        } else if ms_level == 2 && !last_ms1_mzs.is_empty() {
            // Extract envelope from the last MS1 scan
            if let Some(precursor) = spectrum.precursor() {
                if let Some(ion) = precursor.ions.first() {
                    let mono_mz = ion.mz;
                    let charge = ion
                        .charge
                        .map(|c| c.unsigned_abs() as usize)
                        .unwrap_or(2); // default to charge 2

                    let (mz_values, intensities) = extract_envelope_from_peaks(
                        &last_ms1_mzs,
                        &last_ms1_intensities,
                        mono_mz,
                        charge,
                        n_peaks,
                        ppm_tolerance,
                    );

                    // Normalize
                    let total: f64 = intensities.iter().sum();
                    let normalized = if total > 0.0 {
                        intensities.iter().map(|i| i / total).collect()
                    } else {
                        vec![0.0; n_peaks]
                    };

                    envelopes.push(IsotopeEnvelope {
                        spectrum_id: spectrum.description.id.clone(),
                        peptide: String::new(),
                        mono_mz,
                        charge,
                        rt: last_ms1_rt,
                        mz_values,
                        intensities,
                        normalized,
                    });
                }
            }
        }
    }

    Ok(envelopes)
}
