use rustyms::annotation::model::FragmentationModel;
use rustyms::chemistry::{MassMode, MolecularFormula};
use rustyms::fragment::{DiagnosticPosition, Fragment, FragmentType};
use rustyms::prelude::CompoundPeptidoformIon;
use rustyms::quantities::{Tolerance, WithinTolerance};
use rustyms::sequence::AminoAcid;
use rustyms::system::f64::MassOverCharge;
use rustyms::system::isize::Charge;
use rustyms::system::mass_over_charge::thomson;

#[derive(Clone, Debug)]
pub struct CachedFragment {
    pub series: char,
    pub position: usize,
    pub charge: usize,
    pub mz: f64,
    pub ion_type: &'static str,
    /// Hill-notation neutral loss label, empty when none
    pub neutral_loss: String,
    /// Monoisotopic mass of the neutral loss, positive for a loss
    pub loss_mass: f64,
}

impl CachedFragment {
    pub fn is_plain_backbone(&self) -> bool {
        self.ion_type == "backbone" && self.neutral_loss.is_empty()
    }
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

/// Parse any rustyms Fragment into (ion_type, series_char, position).
/// Returns None for fragment kinds that are not exposed (internal, glycan, unknown).
/// Position is 1-indexed; 0 when not applicable (precursor, reporter ions).
pub fn parse_fragment_extended(frag: &Fragment) -> Option<(&'static str, char, usize)> {
    let seq_pos = |p: &rustyms::fragment::PeptidePosition| match p.sequence_index {
        rustyms::sequence::SequencePosition::Index(i) => i + 1,
        _ => 0,
    };
    match &frag.ion {
        FragmentType::a(..)
        | FragmentType::b(..)
        | FragmentType::c(..)
        | FragmentType::x(..)
        | FragmentType::y(..)
        | FragmentType::z(..) => {
            let (series, pos, _) = parse_fragment(frag)?;
            Some(("backbone", series, pos))
        }
        FragmentType::d(pos, ..) => Some(("satellite", 'd', pos.series_number)),
        FragmentType::v(pos, ..) => Some(("satellite", 'v', pos.series_number)),
        FragmentType::w(pos, ..) => Some(("satellite", 'w', pos.series_number)),
        FragmentType::Precursor => Some(("precursor", 'M', 0)),
        FragmentType::Diagnostic(DiagnosticPosition::Peptide(pos, _)) => {
            Some(("diagnostic", 'D', seq_pos(pos)))
        }
        FragmentType::Diagnostic(_) => Some(("diagnostic", 'D', 0)),
        FragmentType::Immonium(pos, _) => {
            Some(("immonium", 'I', pos.as_ref().map(seq_pos).unwrap_or(0)))
        }
        _ => None,
    }
}

/// Neutral-loss label and mass for a fragment. ("", 0.0) when the fragment has no loss.
pub fn fragment_loss(frag: &Fragment) -> (String, f64) {
    if frag.neutral_loss.is_empty() {
        return (String::new(), 0.0);
    }
    let label: String = frag
        .neutral_loss
        .iter()
        .map(|l| l.hill_notation())
        .collect();
    let mut delta = MolecularFormula::default();
    for loss in &frag.neutral_loss {
        delta += loss;
    }
    (label, -delta.monoisotopic_mass().value)
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

/// Build theoretical fragments. With `extended == false` only loss-free backbone ions are kept.
pub fn build_theoretical_fragments(
    peptidoform: &CompoundPeptidoformIon,
    max_charge: Charge,
    model: &FragmentationModel,
    mode: MassMode,
    extended: bool,
) -> Vec<CachedFragment> {
    peptidoform
        .generate_theoretical_fragments(max_charge, model)
        .into_iter()
        .filter_map(|frag| {
            let (ion_type, series, position) = parse_fragment_extended(&frag)?;
            let (neutral_loss, loss_mass) = fragment_loss(&frag);
            let cached = CachedFragment {
                series,
                position,
                charge: frag.charge.value.unsigned_abs(),
                mz: frag.mz(mode)?.value,
                ion_type,
                neutral_loss,
                loss_mass,
            };
            (extended || cached.is_plain_backbone()).then_some(cached)
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
        assert_eq!(
            longest_true_run(&[false, true, false, true, true, false]),
            2
        );
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
