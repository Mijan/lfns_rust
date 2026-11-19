//! The standard normal distribution: density, distribution function, and
//! quantile function.

/// Functions of the standard normal distribution.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StandardNormal;

impl StandardNormal {
    /// Below this probability, and above one minus it, the quantile uses its
    /// tail approximation.
    const TAIL_PROBABILITY: f64 = 0.024_25;
    /// Smallest and (one minus) largest probability the quantile accepts;
    /// they map to about -8.2 and 8.2.
    const SMALLEST_PROBABILITY: f64 = f64::EPSILON / 2.0;

    const CENTRAL_NUMERATOR: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const CENTRAL_DENOMINATOR: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const TAIL_NUMERATOR: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const TAIL_DENOMINATOR: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    /// `ln` of the density at `z`.
    pub(crate) fn log_density(z: f64) -> f64 {
        -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln()
    }

    /// Probability of a value below `z`.
    pub(crate) fn cdf(z: f64) -> f64 {
        0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
    }

    /// The value below which a fraction `probability` of the distribution
    /// lies. Probabilities of 0 and 1 are moved inward by half a machine
    /// epsilon, so the result is always finite.
    ///
    /// Acklam's rational approximation, refined by one Halley step.
    pub(crate) fn quantile(probability: f64) -> f64 {
        let p = probability.clamp(Self::SMALLEST_PROBABILITY, 1.0 - Self::SMALLEST_PROBABILITY);
        let estimate = if p < Self::TAIL_PROBABILITY {
            Self::tail_quantile(p)
        } else if p > 1.0 - Self::TAIL_PROBABILITY {
            -Self::tail_quantile(1.0 - p)
        } else {
            let offset = p - 0.5;
            let squared = offset * offset;
            offset * Self::polynomial(&Self::CENTRAL_NUMERATOR, squared)
                / (Self::polynomial(&Self::CENTRAL_DENOMINATOR, squared) * squared + 1.0)
        };
        let error = (Self::cdf(estimate) - p) * (-Self::log_density(estimate)).exp();
        estimate - error / (1.0 + 0.5 * estimate * error)
    }

    /// The quantile of a probability in the lower tail.
    fn tail_quantile(p: f64) -> f64 {
        let q = (-2.0 * p.ln()).sqrt();
        Self::polynomial(&Self::TAIL_NUMERATOR, q) / (Self::polynomial(&Self::TAIL_DENOMINATOR, q) * q + 1.0)
    }

    /// Evaluates a polynomial with coefficients from the highest power down.
    fn polynomial(coefficients: &[f64], x: f64) -> f64 {
        coefficients
            .iter()
            .fold(0.0, |value, coefficient| value * x + coefficient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_matches_tabulated_values() {
        assert!((StandardNormal::quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-12);
        assert!(StandardNormal::quantile(0.5).abs() < 1e-15);
        assert!((StandardNormal::quantile(0.001) + 3.090_232_306_167_813).abs() < 1e-11);
    }

    #[test]
    fn quantile_inverts_the_distribution_function() {
        // Above z = 5 the distribution function is within 3e-7 of one and no
        // longer resolves z to the tolerance asked for here.
        for step in -70..=50 {
            let z = f64::from(step) * 0.1;
            let recovered = StandardNormal::quantile(StandardNormal::cdf(z));
            assert!((recovered - z).abs() < 1e-8 * (1.0 + z.abs()), "{z} -> {recovered}");
        }
    }

    #[test]
    fn extreme_probabilities_give_finite_quantiles() {
        assert!(StandardNormal::quantile(0.0).is_finite());
        assert!(StandardNormal::quantile(1.0).is_finite());
        assert!(StandardNormal::quantile(0.0) < -8.0 && StandardNormal::quantile(1.0) > 8.0);
    }
}
