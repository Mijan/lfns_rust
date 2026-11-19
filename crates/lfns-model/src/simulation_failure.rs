//! Reasons a simulation or likelihood evaluation cannot be completed.

/// Why a model could not be simulated at some parameter values. The sampler
/// counts such parameter values as having zero likelihood.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SimulationFailure {
    /// A rate law evaluated to a negative number or to `NaN`.
    #[error(
        "the rate of reaction {reaction_index} is {value}; a propensity must be a non-negative \
         number. Check the kinetic law for subtractions or divisions that can change sign."
    )]
    InvalidRate {
        /// Position of the reaction in the model.
        reaction_index: usize,
        /// The offending value.
        value: f64,
    },

    /// More reaction events occurred between two time points than allowed.
    #[error(
        "more than {limit} reaction events between two time points. The parameters make the \
         model explode, or the limit is too low for this model; raise it with \
         `StochasticSimulator::with_max_events`."
    )]
    EventLimitExceeded {
        /// The limit that was hit.
        limit: u64,
    },

    /// A noise formula did not evaluate to a positive standard deviation.
    #[error(
        "the noise formula of observable {observable_index} evaluates to {value}; a standard \
         deviation must be positive. Check the noiseFormula in the PEtab observable table."
    )]
    InvalidNoise {
        /// Position of the observable in the model.
        observable_index: usize,
        /// The offending value.
        value: f64,
    },

    /// The differential equation solver could not complete the integration.
    #[error(
        "the ODE solver failed: {message}. The parameters probably make the model extremely \
         stiff or make it blow up; if this happens across the whole prior, loosen the tolerances \
         or raise the step limit in `OdeSettings`."
    )]
    SolverFailed {
        /// The solver's own description of the failure.
        message: String,
    },
}
