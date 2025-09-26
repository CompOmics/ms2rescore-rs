use mzdata::io::MassSpectrometryFormat;

pub enum SpectrumFileType {
    MascotGenericFormat,
    MzML,
    MzMLb,
    BrukerRaw,
    ThermoRaw,
    Unknown,
}

pub fn match_file_type(spectrum_path: &str) -> SpectrumFileType {
    match mzdata::io::infer_from_path(spectrum_path).0 {
        MassSpectrometryFormat::MGF => SpectrumFileType::MascotGenericFormat,
        MassSpectrometryFormat::MzML => SpectrumFileType::MzML,
        MassSpectrometryFormat::MzMLb => SpectrumFileType::MzMLb,
        MassSpectrometryFormat::ThermoRaw => SpectrumFileType::ThermoRaw,
        MassSpectrometryFormat::Unknown => {
            let extension = Path::new(&spectrum_path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
            match extension.as_str() {
                "d" | "ms2" => SpectrumFileType::BrukerRaw,
                _ => match (
                    folder_contains_extension(spectrum_path, "bin"),
                    folder_contains_extension(spectrum_path, "parquet"),
                ) {
                    (true, true) => SpectrumFileType::BrukerRaw,
                    _ => SpectrumFileType::Unknown,
                },
            }
        }
        _ => SpectrumFileType::Unknown
    }
}

fn folder_contains_extension(input: impl AsRef<std::path::Path>, extension: &str) -> bool {
    let folder_path: std::path::PathBuf = input.as_ref().to_path_buf();
    if !folder_path.is_dir() {
        return false;
    }
    if let Ok(entries) = std::fs::read_dir(folder_path) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == extension {
                    return true;
                }
            }
        }
    }
    false
}
