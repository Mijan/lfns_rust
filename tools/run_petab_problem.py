"""Run likelihood-free nested sampling on a PEtab problem.

    uv run python tools/run_petab_problem.py --problem <folder>/problem.yaml \
        --run-directory <folder for this run> --n-live 2000 --n-replaced 200

The model is simulated in Rust: as an ODE by default, or stochastically
with a particle filter (``--dynamics stochastic --n-particles H``). The run
is checkpointed in the run directory after every iteration; interrupt it at
any time and continue with the same command plus ``--resume``.

When the run stops, two files are added to the run directory:

- ``summary.json``: the evidence with its predicted error, the stop reason,
  the number of iterations and likelihood evaluations, and the effective
  sample size of the posterior;
- ``posterior.csv``: one row per posterior point with the values of the
  estimated parameters, its log-likelihood, and its normalized log-weight.

The number of threads is the number of cores; set ``RAYON_NUM_THREADS`` to
use fewer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import lfns

SUMMARY_FILE = "summary.json"
POSTERIOR_FILE = "posterior.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problem", type=Path, required=True, help="the PEtab YAML file")
    parser.add_argument("--run-directory", type=Path, required=True, help="where the run is checkpointed")
    parser.add_argument("--resume", action="store_true", help="continue the run stored in the run directory")
    parser.add_argument("--dynamics", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--n-particles", type=int, default=None, help="filter particles, for stochastic dynamics")
    parser.add_argument("--n-live", type=int, required=True, help="number of live points N")
    parser.add_argument("--n-replaced", type=int, required=True, help="points replaced per iteration r")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.01, help="termination tolerance")
    parser.add_argument("--max-iterations", type=int, default=None, help="stop after this many iterations")
    parser.add_argument("--n-sweeps", type=int, default=200, help="Gibbs sweeps of each DP-GMM fit")
    parser.add_argument("--rejection-quantile", type=float, default=0.001, help="quantile of the proposal floor")
    return parser.parse_args()


def write_posterior(problem: lfns.PetabProblem, result: lfns.RunResult, run_directory: Path) -> None:
    """Writes the weighted posterior sample in parameter values."""
    unit_points, log_weights = result.posterior()
    parameters = problem.to_parameters(unit_points)
    # The posterior lists the dead points in order of removal, then the live points.
    log_likelihoods = np.concatenate([result.dead_points()[1], result.live_points()[1]])
    table = np.column_stack([parameters, log_likelihoods, log_weights])
    header = ",".join([*problem.parameter_ids, "log_likelihood", "log_weight"])
    np.savetxt(run_directory / POSTERIOR_FILE, table, delimiter=",", header=header, comments="")


def write_summary(result: lfns.RunResult, seconds: float, run_directory: Path) -> dict[str, float | int | str]:
    """Writes the headline numbers of the run and returns them."""
    summary = {
        "stop_reason": result.stop_reason,
        "log_evidence": result.log_evidence,
        "relative_std": result.relative_std,
        "relative_floor_std": result.relative_floor_std,
        "n_iterations": result.n_iterations,
        "n_likelihood_evaluations": result.n_proposals,
        "n_failed_evaluations": int(result.reports()["n_failures"].sum()),
        "effective_sample_size": result.effective_sample_size(),
        "seconds_this_session": seconds,
    }
    (run_directory / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    problem = lfns.PetabProblem(str(args.problem))
    print(f"{len(problem.parameter_ids)} estimated parameters, {len(problem.condition_ids)} conditions, "
          f"{len(problem.observable_ids)} observables", flush=True)
    start = time.perf_counter()
    result = problem.run(
        dynamics=args.dynamics,
        n_particles=args.n_particles,
        n_live=args.n_live,
        n_replaced=args.n_replaced,
        seed=args.seed,
        termination_tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        n_sweeps=args.n_sweeps,
        rejection_quantile=args.rejection_quantile,
        run_directory=str(args.run_directory),
        resume=args.resume,
        progress=True,
    )
    summary = write_summary(result, time.perf_counter() - start, args.run_directory)
    write_posterior(problem, result, args.run_directory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
