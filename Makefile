# Quality gates. `make check` runs exactly what CI runs.
.PHONY: check format format_check lint docs test

check: format_check lint docs test

format:
	cargo fmt --all

format_check:
	cargo fmt --all -- --check

lint:
	cargo clippy --all-targets --all-features --locked -- -D warnings

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --locked

test:
	cargo test --all-features --locked
