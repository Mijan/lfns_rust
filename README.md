# lfns

Likelihood-free nested sampling, written in Rust for speed.

## Repository Structure

    .
    ├── Cargo.toml              # workspace: shared lints and release profile
    ├── Cargo.lock              # committed; CI builds with --locked
    ├── clippy.toml, rustfmt.toml
    ├── Makefile                # quality gates (same commands CI runs)
    ├── crates/
    │   ├── dpgmm/              # Dirichlet process Gaussian mixture density estimation
    │   ├── dpgmm-py/           # Python bindings of dpgmm (for the notebooks)
    │   ├── lfns/               # model-agnostic LF-NS algorithm
    │   ├── lfns-model/         # SBML and PEtab models, ODE and stochastic simulation
    │   └── lfns-py/            # Python bindings of lfns (for the notebooks)
    ├── examples/               # example problems (Antimony, SBML, PEtab)
    ├── tools/                  # scripts: run a PEtab problem, export, cross-check
    ├── notebooks/              # Jupyter notebooks illustrating the crates
    ├── pyproject.toml, uv.lock # Python environment of the notebooks
    ├── notes/                  # paper and supplementary material
    ├── CLAUDE.md               # code style and architecture rules
    └── README.md

- `dpgmm` fits a Dirichlet process Gaussian mixture to a point set by Gibbs
  sampling and returns a mixture that can be evaluated and sampled. It does
  not depend on `lfns` and can be used on its own.
- `lfns` contains the sampling loop, the evidence estimate with its variance
  and termination criterion, reproducible parallel proposal evaluation,
  checkpointing, and the DP-GMM proposal. A model plugs in through the
  `LikelihoodEstimator` trait.
- `lfns-model` reads a model from SBML and an estimation problem from PEtab
  (version 1) and provides its likelihood in two ways: exactly, by
  integrating the model as an ODE with a stiff solver, or as an unbiased
  estimate, by simulating it stochastically and running a bootstrap particle
  filter. See *Supported SBML and PEtab* below.

Each crate keeps its unit tests next to the code and its integration tests,
which run on problems with known answers, in `tests/`.

## Getting Started

Install a stable Rust toolchain (1.85 or newer) with the `rustfmt` and
`clippy` components, then:

``` bash
cargo build            # debug build
cargo build --release  # optimized build (fat LTO, one codegen unit)
cargo test
```

## Running a PEtab problem

``` bash
uv sync      # once: builds the Rust code and its Python bindings
uv run python tools/run_petab_problem.py --problem <folder>/problem.yaml \
    --run-directory <folder for this run> --n-live 2000 --n-replaced 200
```

The run prints one line per iteration and checkpoints itself in the run
directory after every iteration. Interrupt it at any time and continue with
the same command plus `--resume`; a resumed run reproduces an uninterrupted
one exactly. When it stops it writes `summary.json` (evidence, predicted
error, stop reason, cost) and `posterior.csv` (weighted posterior sample in
parameter values) into the run directory. `--help` lists all settings.

The same is available from Python as `lfns.PetabProblem(path).run(...)`; the
notebooks show it.

### Supported SBML and PEtab

- **SBML**: compartments with a constant size; species; parameters;
  reactions with kinetic laws; rate rules on species; assignment rules on
  parameters; events that assign parameters at a fixed time (trigger
  `time >= constant`), which is how piecewise-constant inputs such as feed
  schedules are expressed. Formulas may use arithmetic, powers, `exp`, `ln`,
  `log`, `sqrt`, `abs`, `max`, `min`, comparisons, and `piecewise`.
- **PEtab version 1**: one model; condition tables that set initial species
  values and parameters to numbers; normal measurement noise with a noise
  formula; parameters on a linear, `log`, or `log10` scale with the default
  prior (uniform on that scale between the bounds); trajectories identified
  by condition and `replicateId`.
- **Dynamics**: deterministic (ODE) for every supported model; stochastic
  only for models that consist of reactions alone.

Anything else (function definitions, initial assignments, state-triggered
events, other priors or noise distributions, pre-equilibration) is rejected
when the problem is loaded, with an error that names it. Antimony is not
read directly: convert it to SBML with the `antimony` package, as in
`examples/birth_death/README.md`.

### Tools

- `tools/run_petab_problem.py`: runs a problem, as above.
- `tools/simulate_with_roadrunner.py`: simulates a problem at its nominal
  parameters with RoadRunner, as an independent reference for the ODE
  simulation of this repository. Needs an environment with `roadrunner`.
- `tools/export_campaign_petab.py`: writes a culture-fit campaign of the
  Cultzyme model-training playground as a PEtab problem. It runs in the
  playground's environment and reads the playground's data; see its
  docstring.

## Notebooks

The notebooks are written in Python and call the Rust code through bindings
(`crates/dpgmm-py`, `crates/lfns-py`). They are illustrations, not tests, and are committed
with their outputs so they can be read without running them.

``` bash
uv sync                              # builds the bindings in release mode
uv run jupyter lab                   # opens the notebooks in the browser
uv sync --reinstall-package dpgmm --reinstall-package lfns   # rebuild after changing Rust code
```

- `notebooks/dpgmm_density_estimation.ipynb`: DP-GMM fits in one and two
  dimensions, their variability, runtimes, and the proposal region LF-NS
  derives from a fit.
- `notebooks/birth_death_petab.ipynb`: the birth-death example of the paper
  from Antimony and PEtab files to a posterior, with the model simulated in
  Rust and every step checked against the exact likelihood.
- `notebooks/birth_death_ode_petab.ipynb`: the same model and data treated
  as an ODE with three estimated parameters, checked against the exact
  evidence and posterior.
- `notebooks/lfns_toy_problem.ipynb`: one LF-NS run followed from start to
  finish on a problem with known evidence and posterior, with an exact and
  with a noisy likelihood.
- `notebooks/lfns_variance_validation.ipynb`: repeated runs that compare the
  error a run predicts for itself with the observed spread, for different
  numbers of live points and batch sizes.

## Quality Checks

``` bash
make check         # everything below, exactly as CI runs it
make format        # auto-format (cargo fmt)
make format_check  # formatting check
make lint          # cargo clippy, warnings are errors
make docs          # cargo doc, warnings are errors
make test          # cargo test
```

Lint levels are configured in `Cargo.toml` under `[lints]`: clippy
`pedantic`, `missing_docs`, `missing_debug_implementations`, and
`unsafe_code = "deny"`.

## Dependencies

``` bash
cargo add rand
cargo add --dev criterion
```

Commit `Cargo.lock` together with `Cargo.toml`. CI runs with `--locked`
and fails if the lock file is out of date.

## Performance

The release profile trades compile time for run time (`lto = "fat"`,
`codegen-units = 1`). To also target the CPU of the machine you build on:

``` bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Binaries built this way are not portable to older CPUs.

## Continuous Integration

GitHub Actions (`ci-quality.yaml`) runs on push, pull request, and manual
trigger: formatting check, clippy, rustdoc, and tests.

## License

MIT License © 2026 Jan Mikelson
