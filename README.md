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
    │   └── lfns/               # model-agnostic LF-NS algorithm
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
