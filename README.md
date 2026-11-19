# lfns

Likelihood-free nested sampling, written in Rust for speed.

## Repository Structure

    .
    ├── .github/workflows/ci-quality.yaml
    ├── Cargo.toml          # package metadata, lints, release profile
    ├── Cargo.lock          # committed; CI builds with --locked
    ├── clippy.toml         # clippy thresholds
    ├── Makefile            # quality gates (same commands CI runs)
    ├── src/lib.rs          # crate root
    ├── CLAUDE.md           # code style and architecture rules
    └── README.md

Unit tests live next to the code in `#[cfg(test)] mod tests`; integration
tests that exercise the public API go in `tests/`; benchmarks go in
`benches/`.

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
