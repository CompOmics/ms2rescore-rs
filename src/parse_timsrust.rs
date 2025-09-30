use std::collections::HashMap;

use timsrust::readers::SpectrumReaderError;

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

/// Handles errors from the spectrum reader by converting each error into std::io::Error except for
/// the `Decompression`` fails error.
///
/// # Arguments
/// * `err` - The error to handle
///
fn handle_spectrum_reader_error(err: SpectrumReaderError) -> Result<(), std::io::Error> {
    // Unfortunately, the timsrust library does not make all errors publicly available yet,
    // so we have to convert them to string for comparison. As soon as the development branch of
    // timsrust is merged into the main branch, we can use the "No binary data"
    // (TdfBlobReaderError::EmptyData) error to be more specific.
    if err.to_string() == "Decompression fails" {
        return Ok(());
    }

    Err(std::io::Error::other(format!(
        "SpectrumReaderError: {}",
        err
    )))
}

/// Parse precursor info from spectrum files with timsrust
pub fn parse_precursor_info(
    spectrum_path: &str,
) -> Result<HashMap<String, Precursor>, std::io::Error> {
    let reader = timsrust::readers::SpectrumReader::new(spectrum_path)
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let spectra = (0..reader.len())
        .map(|index| match reader.get(index) {
            Ok(spectrum) => Ok(Some(spectrum)),
            Err(err) => handle_spectrum_reader_error(err).map(|_| None),
        })
        // resolve errors
        .collect::<Result<Vec<Option<timsrust::Spectrum>>, std::io::Error>>()
        .map_err(|e| std::io::Error::other(e.to_string()))?
        // remove None values
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

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
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let spectra = (0..reader.len())
        .map(|index| match reader.get(index) {
            Ok(spectrum) => Ok(Some(spectrum)),
            Err(err) => handle_spectrum_reader_error(err).map(|_| None),
        })
        // resolve errors
        .collect::<Result<Vec<Option<timsrust::Spectrum>>, std::io::Error>>()
        .map_err(|e| std::io::Error::other(e.to_string()))?
        // remove None values
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    Ok(spectra.into_iter().map(MS2Spectrum::from).collect())
}
