//! Arithmetic on non-negative quantities stored as natural logarithms, plus
//! the serde representation of such values (JSON has no infinities).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Returns `ln(exp(a) + exp(b))` without leaving log space.
pub(crate) fn log_add_exp(a: f64, b: f64) -> f64 {
    let (larger, smaller) = if a >= b { (a, b) } else { (b, a) };
    if larger == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    larger + (smaller - larger).exp().ln_1p()
}

/// Returns `ln(sum_i exp(v_i))` without leaving log space.
pub(crate) fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let largest = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if largest == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let scaled_sum: f64 = values.map(|value| (value - largest).exp()).sum();
    largest + scaled_sum.ln()
}

/// JSON form of a log-space value: a number, or the text `"-inf"` for the
/// logarithm of zero.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum LogValueRecord {
    Finite(f64),
    Text(String),
}

/// Serde adapter for `f64` fields holding logarithms; use it through
/// `#[serde(with = "crate::log_space::log_value")]`.
pub(crate) mod log_value {
    use super::{Deserialize, Deserializer, LogValueRecord, Serialize, Serializer};

    const NEGATIVE_INFINITY_TEXT: &str = "-inf";

    #[allow(clippy::trivially_copy_pass_by_ref)] // signature dictated by serde
    pub(crate) fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        if value.is_finite() {
            return LogValueRecord::Finite(*value).serialize(serializer);
        }
        if *value == f64::NEG_INFINITY {
            return LogValueRecord::Text(NEGATIVE_INFINITY_TEXT.to_owned()).serialize(serializer);
        }
        Err(serde::ser::Error::custom(format!(
            "log-space value {value} cannot be stored; only finite values and -inf are valid logarithms here"
        )))
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        match LogValueRecord::deserialize(deserializer)? {
            LogValueRecord::Finite(value) => Ok(value),
            LogValueRecord::Text(text) if text == NEGATIVE_INFINITY_TEXT => Ok(f64::NEG_INFINITY),
            LogValueRecord::Text(text) => Err(serde::de::Error::custom(format!(
                "unknown log-space value \"{text}\"; write a number or \"{NEGATIVE_INFINITY_TEXT}\""
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn log_add_exp_matches_direct_sum_far_below_underflow() {
        let offset = -5000.0;
        let result = log_add_exp(offset + 2.0_f64.ln(), offset + 3.0_f64.ln());
        assert!((result - (offset + 5.0_f64.ln())).abs() < TOLERANCE);
    }

    #[test]
    fn log_sum_exp_of_only_zeros_is_log_of_zero() {
        let values = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert_eq!(log_sum_exp(values.iter().copied()), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_ignores_zero_terms() {
        let values = [f64::NEG_INFINITY, 1.0_f64.ln(), 3.0_f64.ln()];
        assert!((log_sum_exp(values.iter().copied()) - 4.0_f64.ln()).abs() < TOLERANCE);
    }
}
