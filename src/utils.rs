/// Compute ln(n!) in a numerically stable way.
pub fn ln_factorial(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).ln()).sum()
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
