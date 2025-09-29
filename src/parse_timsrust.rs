use std::collections::HashMap;

use crate::ms2_spectrum::MS2Spectrum;
use crate::precursor::Precursor;

impl From<timsrust::Precursor> for Precursor {
    fn from(precursor: timsrust::Precursor) -> Self {
        Precursor {
            mz: precursor.mz,
            rt: precursor.rt,
            im: precursor.im,
            charge: precursor.charge.unwrap_or(0),
            intensity: precursor.intensity.unwrap_or(0.0),
        }
    }
}

impl From<timsrust::Spectrum> for MS2Spectrum {
    fn from(spectrum: timsrust::Spectrum) -> Self {
        MS2Spectrum::new(
            spectrum.index.to_string(),
            spectrum.mz_values.iter().map(|mz| *mz as f32).collect(),
            spectrum
                .intensities
                .iter()
                .map(|intensity| *intensity as f32)
                .collect(),
            spectrum.precursor.map(Precursor::from),
        )
    }
}

/// Parse precursor info from spectrum files with timsrust
pub fn parse_precursor_info(
    spectrum_path: &str,
) -> Result<HashMap<String, Precursor>, std::io::Error> {
    let reader = timsrust::readers::SpectrumReader::new(spectrum_path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let spectra = reader
        .get_all()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let precursor_info = spectra
        .into_iter()
        .filter_map(|spectrum| match spectrum.precursor {
            Some(precursor) => Some((spectrum.index.to_string(), Precursor::from(precursor))),
            None => None,
        })
        .collect::<HashMap<_, _>>();

    Ok(precursor_info)
}

/// Read MS2 spectra from spectrum files with timsrust
pub fn read_ms2_spectra(spectrum_path: &str) -> Result<Vec<MS2Spectrum>, std::io::Error> {
    let reader = timsrust::readers::SpectrumReader::new(spectrum_path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let spectra = reader
        .get_all()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    Ok(spectra.into_iter().map(MS2Spectrum::from).collect())
}
