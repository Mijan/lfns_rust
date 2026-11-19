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
    │   ├── lfns-model/         # SBML and PEtab models, stochastic simulation, particle filter
    │   └── lfns-py/            # Python bindings of lfns (for the notebooks)
    ├── examples/               # example problems (Antimony, SBML, PEtab)
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
  (version 1), simulates the model stochastically, and estimates the
  likelihood with a bootstrap particle filter. Only a subset of SBML and
  PEtab is supported so far, and deterministic (ODE) simulation is not
  implemented yet; unsupported input is rejected with an error that names it.

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
