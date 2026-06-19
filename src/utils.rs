use rustyms::annotation::model::FragmentationModel;
use rustyms::chemistry::MassMode;
use rustyms::fragment::{Fragment, FragmentType};
use rustyms::prelude::CompoundPeptidoformIon;
use rustyms::quantities::{Tolerance, WithinTolerance};
use rustyms::sequence::AminoAcid;
use rustyms::system::f64::MassOverCharge;
use rustyms::system::isize::Charge;
use rustyms::system::mass_over_charge::thomson;

#[derive(Clone, Copy, Debug)]
pub struct CachedFragment {
    pub series: char,
    pub position: usize,
    pub charge: usize,
    pub mz: f64,
}

/// Floor value for log2-intensity clipping and unmatched-ion fill: log2(0.001).
/// Single source of truth; f32 consumers cast at use site.
pub const LOG2_FLOOR_F64: f64 = -9.965_784_284_662_087;

/// Compute ln(n!). For typical peptide-length inputs (n < 50) the
/// iterative sum is fast enough; a lookup table is not warranted.
pub fn ln_factorial(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).ln()).sum()
}

/// Parse a rustyms Fragment into (series_char, position, charge).
/// Returns None for non-backbone ions (glycan, immonium, precursor, etc.).
/// Position is 1-indexed (the series_number from rustyms).
pub fn parse_fragment(frag: &Fragment) -> Option<(char, usize, usize)> {
    let (series, pos) = match &frag.ion {
        FragmentType::a(pos, _) => ('a', pos),
        FragmentType::b(pos, _) => ('b', pos),
        FragmentType::c(pos, _) => ('c', pos),
        FragmentType::x(pos, _) => ('x', pos),
        FragmentType::y(pos, _) => ('y', pos),
        FragmentType::z(pos, _) => ('z', pos),
        _ => return None,
    };
    let charge = frag.charge.value.unsigned_abs();
    Some((series, pos.series_number, charge))
}

/// Extract the precursor charge from a parsed peptidoform.
/// Returns None if no charge carriers are present.
pub fn extract_charge(peptidoform: &CompoundPeptidoformIon) -> Option<usize> {
    peptidoform
        .peptidoforms()
        .next()
        .and_then(|pf| pf.get_charge_carriers())
        .map(|cc| cc.charge().value.unsigned_abs())
        .filter(|&c| c > 0)
}

pub fn build_theoretical_fragments(
    peptidoform: &CompoundPeptidoformIon,
    max_charge: Charge,
    model: &FragmentationModel,
    mode: MassMode,
) -> Vec<CachedFragment> {
    peptidoform
        .generate_theoretical_fragments(max_charge, model)
        .into_iter()
        .filter_map(|frag| {
            let (series, position, charge) = parse_fragment(&frag)?;
            let mz = frag.mz(mode)?.value;
            Some(CachedFragment {
                series,
                position,
                charge,
                mz,
            })
        })
        .collect()
}

pub fn search_sorted_mz(
    mz_values: &[f32],
    query_mz: f64,
    tolerance: &Tolerance<MassOverCharge>,
) -> Option<usize> {
    if mz_values.is_empty() {
        return None;
    }

    let index = mz_values
        .binary_search_by(|mz| (*mz as f64).total_cmp(&query_mz))
        .unwrap_or_else(|i| i);

    let start = index.saturating_sub(1);
    let end = (index + 1).min(mz_values.len() - 1);
    let query = MassOverCharge::new::<thomson>(query_mz);

    let mut closest: Option<(usize, MassOverCharge)> = None;
    let mut closest_ppm = f64::INFINITY;

    for (i, mz) in mz_values.iter().enumerate().take(end + 1).skip(start) {
        let observed = MassOverCharge::new::<thomson>(*mz as f64);
        let ppm = observed.ppm(query).value;
        if ppm < closest_ppm {
            closest_ppm = ppm;
            closest = Some((i, observed));
        }
    }

    closest.and_then(|(index, observed)| tolerance.within(&observed, &query).then_some(index))
}

/// Map a rustyms AminoAcid to its ms2pip index (0..18).
/// L is mapped to I (index 7). Returns None for non-standard AAs.
pub fn aa_to_ms2pip_index(aa: AminoAcid) -> Option<usize> {
    match aa {
        AminoAcid::Alanine => Some(0),
        AminoAcid::Cysteine => Some(1),
        AminoAcid::AsparticAcid => Some(2),
        AminoAcid::GlutamicAcid => Some(3),
        AminoAcid::Phenylalanine => Some(4),
        AminoAcid::Glycine => Some(5),
        AminoAcid::Histidine => Some(6),
        AminoAcid::Isoleucine | AminoAcid::Leucine | AminoAcid::AmbiguousLeucine => Some(7),
        AminoAcid::Lysine => Some(8),
        AminoAcid::Methionine => Some(9),
        AminoAcid::Asparagine => Some(10),
        AminoAcid::Proline => Some(11),
        AminoAcid::Glutamine => Some(12),
        AminoAcid::Arginine => Some(13),
        AminoAcid::Serine => Some(14),
        AminoAcid::Threonine => Some(15),
        AminoAcid::Valine => Some(16),
        AminoAcid::Tryptophan => Some(17),
        AminoAcid::Tyrosine => Some(18),
        _ => None,
    }
}

/// Compute the longest run of `true` values in a boolean slice.
pub fn longest_true_run(flags: &[bool]) -> usize {
    let mut max_run = 0usize;
    let mut cur = 0usize;
    for &v in flags {
        if v {
            cur += 1;
            max_run = max_run.max(cur);
        } else {
            cur = 0;
        }
    }
    max_run
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ln_factorial() {
        assert_eq!(ln_factorial(0), 0.0);
        assert_eq!(ln_factorial(1), 0.0);
        assert!((ln_factorial(5) - (120.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_longest_true_run() {
        assert_eq!(longest_true_run(&[]), 0);
        assert_eq!(longest_true_run(&[false, false]), 0);
        assert_eq!(longest_true_run(&[true, true, false, true]), 2);
        assert_eq!(longest_true_run(&[true, true, true]), 3);
        assert_eq!(longest_true_run(&[false, true, false, true, true, false]), 2);
    }

    #[test]
    fn test_aa_to_ms2pip_index() {
        assert_eq!(aa_to_ms2pip_index(AminoAcid::Alanine), Some(0));
        assert_eq!(aa_to_ms2pip_index(AminoAcid::Leucine), Some(7));
        assert_eq!(aa_to_ms2pip_index(AminoAcid::Isoleucine), Some(7));
        assert_eq!(aa_to_ms2pip_index(AminoAcid::Tyrosine), Some(18));
        assert_eq!(aa_to_ms2pip_index(AminoAcid::Unknown), None);
    }

    #[test]
    fn test_search_sorted_mz_absolute() {
        let mz = [100.0_f32, 200.0, 300.0];
        let tol = Tolerance::new_absolute(MassOverCharge::new::<thomson>(0.02));
        assert_eq!(search_sorted_mz(&mz, 200.01, &tol), Some(1));
        assert_eq!(search_sorted_mz(&mz, 200.05, &tol), None);
    }

    #[test]
    fn test_search_sorted_mz_ppm() {
        let mz = [100.0_f32, 200.0, 300.0];
        let tol = Tolerance::new_ppm(10.0);
        assert_eq!(search_sorted_mz(&mz, 200.001, &tol), Some(1));
        assert_eq!(search_sorted_mz(&mz, 200.01, &tol), None);
    }
}
