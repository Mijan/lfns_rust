//! Python bindings of the `lfns` crate, for notebooks and analysis scripts.
//!
//! The module exposes `run`, which performs a likelihood-free nested
//! sampling run with a likelihood written in Python, `PetabProblem`, which
//! runs a PEtab problem whose model is simulated in Rust, and the
//! `RunResult` both return. Parameters live in the unit cube (prior-CDF space).

// PyO3 hands extracted arguments to a function by value.
#![allow(clippy::needless_pass_by_value)]

mod petab_binding;
mod proposal_choice;
mod python_estimator;
mod report_recorder;
mod run_result;
mod run_settings;

use std::path::PathBuf;

use ::dpgmm::DpgmmConfig;
use ::lfns::{DpgmmProposalConfig, HeldOutFloor, LowAcceptanceRule, RunConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::petab_binding::PyPetabProblem;
use crate::proposal_choice::ProposalSettings;
use crate::python_estimator::PythonEstimator;
use crate::run_result::PyRunResult;
use crate::run_settings::{Checkpointing, RunSettings};

/// Turns a Rust error into a Python `ValueError` carrying its message.
pub(crate) fn to_value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Runs likelihood-free nested sampling.
///
/// `log_likelihood(unit_point, seed)` receives a point of the unit cube as a
/// `NumPy` array and a 64-bit seed for any randomness it needs. It returns the
/// log-likelihood (one realization of an unbiased estimate for a stochastic
/// model), `-inf` for zero likelihood, or `None` if the evaluation failed.
/// An exception raised by it stops the run and is re-raised.
///
/// `proposal` is `"dpgmm"` (uniform over the support of a DP-GMM fit to the
/// live points) or `"prior"` (the whole unit cube). The DP-GMM region keeps
/// all but the fraction `rejection_quantile` of the fitted mixture. An
/// iteration that follows one whose acceptance rate was below
/// `low_acceptance_threshold` (`None` switches this off) cuts off the fraction
/// `low_acceptance_quantile` instead, and with `recovery_threshold` keeps
/// doing so until an iteration's acceptance rate reaches that value (by
/// default it switches back as soon as the previous iteration was not
/// poor): a tighter region that accepts more
/// often but covers less of the mixture's tails. With `run_directory` the
/// run is checkpointed there after every iteration, and `resume=True`
/// continues the run stored there. `progress=True` prints one line per
/// iteration.
#[pyfunction]
#[pyo3(signature = (
    log_likelihood, n_parameters, *, n_live, n_replaced, seed = 0,
    termination_tolerance = RunConfig::DEFAULT_TERMINATION_TOLERANCE, max_iterations = None,
    proposal = "dpgmm", rejection_quantile = DpgmmProposalConfig::DEFAULT_REJECTION_QUANTILE,
    n_sweeps = DpgmmConfig::DEFAULT_N_SWEEPS, low_acceptance_threshold = Some(LowAcceptanceRule::DEFAULT_ACCEPTANCE_THRESHOLD),
    low_acceptance_quantile = LowAcceptanceRule::DEFAULT_REJECTION_QUANTILE,
    recovery_threshold = None,
    level_set_share_lost = Some(HeldOutFloor::DEFAULT_LEVEL_SET_SHARE),
    held_out_fraction = HeldOutFloor::DEFAULT_HELD_OUT_FRACTION,
    run_directory = None, resume = false, progress = false,
))]
// The many arguments are keyword-only in Python, where they form the public
// configuration surface of a run.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn run(
    py: Python<'_>,
    log_likelihood: Py<PyAny>,
    n_parameters: usize,
    n_live: u32,
    n_replaced: u32,
    seed: u64,
    termination_tolerance: f64,
    max_iterations: Option<u32>,
    proposal: &str,
    rejection_quantile: f64,
    n_sweeps: u32,
    low_acceptance_threshold: Option<f64>,
    low_acceptance_quantile: f64,
    recovery_threshold: Option<f64>,
    level_set_share_lost: Option<f64>,
    held_out_fraction: f64,
    run_directory: Option<PathBuf>,
    resume: bool,
    progress: bool,
) -> PyResult<PyRunResult> {
    let settings = RunSettings::new((n_live, n_replaced), seed, termination_tolerance, max_iterations)?
        .with_proposal(&ProposalSettings {
            name: proposal,
            rejection_quantile,
            n_sweeps,
            low_acceptance_threshold,
            low_acceptance_quantile,
            recovery_threshold,
            level_set_share_lost,
            held_out_fraction,
        })?
        .with_checkpointing(Checkpointing {
            run_directory,
            resume,
            progress,
        })?;
    let (estimator, first_exception) = PythonEstimator::new(log_likelihood, n_parameters);
    let outcome = settings.run_sampler(py, estimator);
    let raised = first_exception
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    match raised {
        Some(exception) => Err(exception),
        None => outcome,
    }
}

/// Likelihood-free nested sampling.
#[pymodule]
fn lfns(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRunResult>()?;
    module.add_class::<PyPetabProblem>()?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
