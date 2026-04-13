use rustyms::prelude::CompoundPeptidoformIon;

/// Compute ln(n!). For typical peptide-length inputs (n < 50) the
/// iterative sum is fast enough; a lookup table is not warranted.
pub fn ln_factorial(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).ln()).sum()
}

/// Parse an ion string like "b5", "y7", "c3", "z12" into (series, position).
/// Works on ASCII bytes to avoid heap allocations.
/// Supports all 6 primary ion series: a, b, c, x, y, z.
pub fn parse_ion_series_and_index(ion: &str) -> Option<(char, usize)> {
    let bytes = ion.trim().as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let series = bytes[0] as char;
    if !matches!(series, 'a' | 'b' | 'c' | 'x' | 'y' | 'z') {
        return None;
    }
    let digit_end = bytes[1..]
        .iter()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(bytes.len() - 1);
    if digit_end == 0 {
        return None;
    }
    let digits = std::str::from_utf8(&bytes[1..1 + digit_end]).ok()?;
    let idx = digits.parse::<usize>().ok()?;
    Some((series, idx))
}

/// Extract the precursor charge from a parsed CompoundPeptidoformIon.
/// Returns None if no charge carriers are present.
pub fn extract_charge(compound: &CompoundPeptidoformIon) -> Option<usize> {
    compound
        .peptidoforms()
        .next()
        .and_then(|pf| pf.get_charge_carriers())
        .map(|cc| cc.charge().value.unsigned_abs())
        .filter(|&c| c > 0)
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
}
