//! The error type of the crate.

/// Everything that can go wrong while building observations, fitting a
/// mixture, or constructing one by hand.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DpgmmError {
    /// Observations were declared to have zero dimensions.
    #[error("observations need at least one dimension. Pass `n_dimensions >= 1`.")]
    NoDimensions,

    /// The flat buffer does not hold a whole number of points.
    #[error(
        "{n_values} values do not form points of {n_dimensions} dimensions. \
         Pass a buffer whose length is a multiple of `n_dimensions`, one point after the other."
    )]
    RaggedObservations {
        /// Length of the buffer.
        n_values: usize,
        /// Declared number of dimensions.
        n_dimensions: usize,
    },

    /// An observation holds `NaN` or an infinity.
    #[error("value {index} of the observations is not finite. Remove or repair that point before fitting.")]
    NonFiniteObservation {
        /// Position of the value in the flat buffer.
        index: usize,
    },

    /// There are too few points to estimate a covariance.
    #[error(
        "{n_observations} observations cannot determine a covariance in {n_dimensions} dimensions. \
         Provide at least {n_required} points."
    )]
    TooFewObservations {
        /// Number of points supplied.
        n_observations: usize,
        /// Number of dimensions.
        n_dimensions: usize,
        /// Smallest admissible number of points.
        n_required: usize,
    },

    /// The observations lie in a lower-dimensional subspace.
    #[error(
        "the sample covariance of the observations is singular: the points lie in a subspace \
         (a constant or linearly dependent coordinate). Drop the redundant coordinate or add \
         points that vary in it."
    )]
    DegenerateObservations,

    /// The number of Gibbs sweeps is zero.
    #[error("a fit needs at least one Gibbs sweep. Pass `n_sweeps >= 1`, e.g. `DpgmmConfig::new(50)`.")]
    NoSweeps,

    /// A matrix lost positive definiteness, or a draw became impossible.
    #[error(
        "numerical breakdown while {step}. The data are probably extremely ill-conditioned; \
         rescale the coordinates to comparable ranges and fit again."
    )]
    NumericalBreakdown {
        /// What was being computed.
        step: &'static str,
    },

    /// A hand-built mixture is not a probability distribution.
    #[error("invalid mixture: {reason}")]
    InvalidMixture {
        /// What is wrong and how to fix it.
        reason: String,
    },
}
