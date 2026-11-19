# CLAUDE.md

## Code Style

### General Principles

- Write clean, readable code. Prioritize clarity over cleverness.
- Functions should be short (aim for <20 lines in the body). If a function
  does two things, split it into two functions.
- Names must be descriptive and intention-revealing. No abbreviations unless
  they are universally understood in the domain (see *Files and Naming* for
  how a domain abbreviation is adopted).
- Names must be readable at the CALL SITE, not only at the declaration.
  `self.iteration_factory.build(...)` narrates itself;
  `self.factory.build(...)` forces the reader to jump to the definition to
  learn "factory of what?". Qualify a field's role whenever the bare noun
  leaves that question open.
- One narrated step per line: when a chained call's intermediate value has a
  meaningful name, split the chain and name it
  (`let iteration = factory.build(...)?;` then
  `let report = iteration.run()?;`), so an error or backtrace points at the
  step that failed. Chains stay fine when the intermediate is nameless
  (`text.trim().to_lowercase()`, iterator adapter pipelines).
- No magic numbers or string literals. Extract them into named constants or
  config fields.
- Prefer explicit over implicit. Never rely on side effects for control flow.
- No commented-out code. Remove dead code; git history preserves it.
- Follow the Rust API Guidelines for naming: `snake_case` functions and
  modules, `UpperCamelCase` types, `SCREAMING_SNAKE_CASE` constants,
  `as_`/`to_`/`into_` conversion prefixes, no `get_` prefix on getters.

### Types

- Use `Option<T>` only when "absent" is a semantically meaningful value, not
  as a default-avoidance pattern.
- Wrap domain quantities that share a primitive type in newtypes
  (`struct Tolerance(f64)`, `struct ParticleCount(usize)`), so the compiler
  rejects a swapped argument.
- Never convert numbers with `as` when the cast can truncate, wrap, or lose
  sign. Use `From`/`TryFrom` and handle the failure.
- Use traits to define interfaces. Prefer generics or `impl Trait` (static
  dispatch) over `dyn Trait`; reach for `dyn Trait` only when the set of
  implementations must be chosen at run time and the call is not on a hot
  path.
- Generic parameters and associated types get meaningful names when a single
  letter does not say enough (`P: Prior`, `type Params`).

### Structs, Enums, and Traits

- Struct fields are **private** by default, `pub(crate)` or `pub(super)`
  only when a sibling module genuinely needs them, and exposed outside the
  module only through getter methods. Provide `&mut self` setters only when
  mutation is explicitly part of the design. Plain-data structs with no
  invariants are the exception: they may have `pub` fields.
- Getters describe what an object IS and are named as nouns
  (`n_features()`, `input_dim()`); they are cheap and take `&self`. Methods
  that perform an action or non-trivial computation are named as verbs.
- Value objects and configuration are immutable: no `&mut self` methods, no
  interior mutability. Mutable state needs strong justification.
- Use a trait when the set of implementations is open, an `enum` when it is
  closed. Never use raw strings or integers for categorical choices.
- Use associated types on a trait when implementors are parameterized by a
  type (e.g. a factory trait with `type Params`), so callers can name and
  construct the concrete type without knowing the implementor.
- Rust has no inheritance. When two types share behavior that varies in one
  dimension, put the shared logic in provided (default) trait methods and
  the varying part in a required method, or make one struct generic over a
  strategy trait. Use composition when the relationship is "has-a".
- Each type has a single responsibility. If it does two unrelated things,
  split it.
- Derive `Debug` on every public type. Derive `Clone`, `PartialEq`, and
  `Copy` only where the semantics are meaningful for the domain object.

### Functions

- Functions do one thing. If the name contains "and", it probably does two.
- Separate commands from queries, and let the signature and the VERB
  disclose the effect. A query takes `&self` and uses a computational verb
  (`estimate_`, `compare_`, `assemble_`, `derive_`); a method that persists
  or mutates takes `&mut self` (or does I/O) and says so (`record_`,
  `write_`, `save_`, `submit_`, `add_`, `narrow_to`). `with_*` is reserved
  for methods that consume or copy `self` and return the modified value;
  never use it for an in-place mutation.
- Never hide mutation behind `&self` with interior mutability (`Cell`,
  `RefCell`, `Mutex`) unless it is a cache whose effect is unobservable.
- Prefer pure functions (no side effects, deterministic) wherever possible.
  "Pure" is about effects, not placement: see *No free-floating functions*
  for where logic lives.
- Limit function arguments to 5 or fewer. If more are needed, group them into
  a config or params struct.
- Rust has no keyword arguments. When an argument is not self-evident from
  position, or two arguments share a type, pass newtypes or a params struct
  with named fields instead of bare positional primitives. Never pass a
  bare `bool` flag; use a two-variant enum.

### Error Handling

- Recoverable failures return `Result<T, E>` with a specific error type
  carrying an informative message. Define domain error enums (derive
  `std::error::Error`, e.g. with `thiserror`); do not return
  `Box<dyn Error>` or `String` from library code.
- Panics are for bugs only. No `unwrap()` in library code. `expect("...")`
  is allowed only when the invariant is provable at that line, and the
  message states the invariant.
- Never discard an error: no `let _ = fallible();`, no `.ok()` to silence a
  `Result`, no `unwrap_or_default()` to paper over a failure. Propagate with
  `?` or handle it explicitly.
- Validate inputs at public API boundaries. Internal functions can trust
  their callers; use `debug_assert!` to document what they rely on.
- Every validation message tells the caller how to fix the problem; see
  *Validation errors must contain the remediation*.

### Crate and Module Structure

- Organize code into clearly separated modules and sub-modules. Each module
  represents a coherent domain concept.
- Items are private by default. A parent module explicitly exports its
  public API with `pub use` re-exports; `lib.rs` re-exports the crate's
  public API. Do not make a module `pub` just to reach one item in it.
- The compiler allows cyclic `use` between modules of one crate; this
  project does not. If two modules need each other, extract the shared
  abstraction into a third module.
- Keep module files focused. A module with more than ~300 lines (excluding
  its unit tests) likely needs splitting.

### Files and Naming

- One module per substantial type, named after it: `IterationFactory` lives
  in `iteration_factory.rs`, with its unit tests in a `#[cfg(test)] mod
  tests` at the bottom of the same file. Small definitions that exist only
  in service of that type (its params struct, a local enum, its error type,
  a private helper) live in the same module.
- Module basenames are unique across the whole crate, so editor tabs,
  backtraces, and grep are unambiguous. Never use `mod.rs`: a module `foo`
  with children is `foo.rs` plus the directory `foo/`. (The one exception
  is `tests/common/mod.rs`, which Cargo requires for shared test helpers.)
- When repeating the domain word makes names long
  (`heart_rate/heart_rate_features.rs`), shorten it with an abbreviation
  that is standard in the domain and use that abbreviation consistently in
  module and type: `hr/hr_features.rs` holds `HrFeatures`,
  `br/br_features.rs` holds `BrFeatures`. Expand the abbreviation once in
  the parent module's `//!` doc comment. An abbreviation adopted this way is
  the "universally understood in the domain" exception to the no-abbreviation
  rule; it is never introduced ad hoc in a single identifier.
- Falling back to the module path (`heart_rate/features.rs`) is acceptable
  only when no standard abbreviation exists. Never mix both styles within one
  parent module.

---

## Architectural Design Principles

These are hard requirements, not style preferences. They prevent the design
drift that creates bugs and technical debt.

### A. Type shape and content

#### Types must mirror the domain, not the implementation

Ask: "what is this thing mathematically / conceptually?" The fields and
methods match that answer, nothing more. If a mathematical object is defined
by X and Y, the struct contains X and Y. Do not add Z because some algorithm
that consumes the object needs Z; that algorithm derives Z from X and Y, or
Z lives on the algorithm's side.

#### Design types so misuse is structurally impossible

The test of a type design: how hard is it to construct or use one WRONGLY,
with inconsistent fields, out of context, or bypassing the invariants its
consumers rely on? Prefer designs where the illegal state cannot be expressed
at all (an enum instead of two booleans, a newtype instead of a bare `f64`,
a non-empty collection type instead of a length check) over designs where it
is merely documented away.

Two kinds of data-bearing types get different constructor policies:

- A **value object**'s meaning is entirely in its fields: any combination
  that passes validation is legitimate, whoever built it. These have private
  fields and a PUBLIC validating constructor
  (`new(...) -> Result<Self, Error>`) that enforces every internal invariant
  (length agreement, ranges, cross-field consistency).
- A **result object** additionally CLAIMS provenance ("this is what the
  sampler/parser/resolver produced") and its fields can be individually
  valid yet jointly a lie. Close the front door: fields are private, the
  constructor is private or `pub(super)` to the producer's module, and
  outside code can only obtain one from the producer or a named constructor
  (`between(...)`, `of_result(...)`, `parse(...)`).

Rules that follow:

- **Invariants live on the TYPE, never only in one constructor.** Every way
  of creating a value funnels through the same validation: named
  constructors call `new`; `Deserialize` goes through
  `#[serde(try_from = "...")]`; `Default` is derived only if the default
  value satisfies the invariants. Constructors DERIVE; the type ENFORCES.
- A struct with invariants never has `pub` fields: a struct literal or a
  field assignment would skip the validation.
- If a doc comment needs to say "construct directly only in tests", the
  constructor wants to be closed: either enforce the invariants so direct
  construction is safe, or restrict its visibility.
- A constructor never takes two positional arguments of the same type;
  positionally swapped same-type arguments are exactly the silent misuse
  this section prevents. Use newtypes or a params struct with named fields.
- Do not reach for producer-only construction by default: it costs cheap
  test fixtures and struct-update syntax. Reserve it for objects whose claim
  cannot be validated from their own fields and whose misconstruction would
  flow into artifacts or decisions undetected. Where such a type must still
  evolve (a resumed run extending its history), give it a NAMED evolution
  method rather than exposing its fields.

#### Distinguish structural properties from behavioral properties

A domain object has two kinds of information: what it IS (structure) and
what it DOES (behavior). Represent them separately, and do not conflate
derived properties with defining ones.

- If two properties seem related but can diverge in edge cases, they are
  separate concepts and are stored or computed independently.
- Properties computable from defining properties are getter methods, not
  redundant stored fields. If computation is expensive, cache it
  (`std::cell::OnceCell` / `std::sync::OnceLock`).

#### Separate defining parameters from runtime inputs

The values that define what an object IS are distinguished from the inputs
it receives at execution time.

- A stochastic model is defined by its structure and parameters. Runtime
  inputs (initial states, time spans, solver settings, the random number
  generator) are passed at the call site, not stored in the same config or
  params struct.
- A sampler is defined by its configuration. The data it is run on and the
  RNG it draws from are arguments to `run()`, not constructor parameters.
- A factory's params struct contains only defining parameters.

This separation prevents conflation in sampling, serialization, and
composition.

#### No free-floating functions for logic that has state or identity

If a function constructs something, transforms something, or dispatches on
type, it belongs as an associated function, a method, or a method on an
enum. Free functions are acceptable only for pure stateless utilities (math
helpers, slice manipulation).

- A factory that selects an implementation based on an enum value is a
  method on the enum.
- A function that captures parameters and returns a closure becomes a struct
  that implements a trait, so the captured parameters are inspectable via
  getters. Closures are fine as local, throwaway arguments (iterator
  adapters, `sort_by`, tests); they are never stored in a struct field or
  returned from a production function.

#### Parameters belong inside callables, not alongside them

When a function is parameterized (e.g., a scoring function with
coefficients), the parameters are captured at construction time, not passed
at every call site.

- Bad: `score_fn` and `score_params` as parallel fields, params passed to fn
  at every evaluation.
- Good: the scoring object captures its params; the call site invokes
  `score.evaluate(&state)` with no extra arguments. The callable is a struct
  (not a closure) exposing a `params()` getter for serialization and
  debugging.

### B. Typed interfaces over untyped data

#### String-keyed maps are not a typed interface

`HashMap<String, f64>` or `serde_json::Value` configuration gives the caller
no type checking, no autocompletion, and no safe refactoring. A misspelled
key (`k_prodd` instead of `k_prod`) is a runtime failure, not a compile
error.

- More than two related parameters that could be misspelled are grouped into
  a struct with named, typed fields.
- A factory that constructs objects from a parameter set takes a typed
  struct, not a map, and is generic over (or has an associated type for) the
  params type.
- The single place where untyped input becomes typed params is the params
  type's `Deserialize` / `TryFrom` implementation, with
  `#[serde(deny_unknown_fields)]`. No other code indexes into a map by
  string key.
- Modules sharing parameter values need no special "sharing" machinery;
  passing the typed struct (by reference or `Copy`) is sufficient.

#### Positional indexing into flat arrays is not an API

If a slice or vector has semantic slots (`params[0]` = rate_constant,
`params[1]` = k_m), the mapping is defined in exactly one place: a struct
with `to_array()` / `from_array()` (or `to_vec()` / `try_from_slice()`). No
code outside that struct indexes into the array by position.

### C. Constants and single source of truth

#### Structural constants belong inside the type

A constant intrinsic to a specific type is an associated constant in that
type's `impl` block, not a module-level `const`. Module-level constants are
only for genuinely module-wide values. When several types in one module each
have structural constants, module-level placement leaves it ambiguous which
constant belongs to which type. Constants callers might customize are
exposed through the constructor or config, with the associated constant as
the default.

#### Structural constants have a single source of truth

When a dimension, count, or schema is determined by one module and consumed
by another, define it once and have every consumer derive from it. Never
duplicate a constant across files, even if both agree today.

- If a feature vector has N channels, define an enum listing the channels in
  the module that constructs the vector, give it an `ALL` array, and export
  `N_CHANNELS = Channel::ALL.len()`. Consumers import it. Where the
  dimension is known at compile time, carry it in the type
  (`[f64; N_CHANNELS]`, const generics) so a mismatch does not compile.
- Never store structural constants (dimensions dictated by the data format)
  in config. Config is for hyperparameters the user chooses; a dimension
  dictated by the code is a derived property.
- If adding a channel requires changing more than two locations (the enum +
  the builder), the abstraction is leaking.
- Co-locate field metadata (valid range, unit, description) with the field's
  type: a newtype per quantity carries its bounds as associated constants
  and enforces them in its constructor. A field name that appears in the
  struct and again in a separate metadata map keyed by string is a
  duplication bug waiting to happen.

### D. Module boundaries and layering

#### Orthogonality: consumers must not know about producers

Modules depend on interfaces, not on each other.

- A downstream consumer (solver, integrator, writer) accepts generic inputs
  (slices, matrices, a trait it defines itself), not a domain object. The
  domain object provides what the consumer needs via a method or by
  implementing the consumer's trait; the consumer never imports the domain
  type.
- The conversion between a symbolic domain object and its numeric
  representation happens at an explicit boundary (a dedicated converter
  module), not inside the numeric consumer.

#### Translate at boundaries; do not modify endpoints to fit each other

When two modules disagree on a convention (frame, units, schema), the fix is
a translator type at the boundary, not a patch to either endpoint. The
endpoints stay correct in their own domain; the translator is the only place
that knows both. Even when an endpoint *could* internalize the conversion
via a flag, prefer the external translator so the boundary is named.

- Translators carry the assumption in their name (`MetricToImperialAdapter`
  wrapping a metric sensor for an imperial consumer), so the cost of the
  translation is visible at the call site.
- A configurable convention on one endpoint is a smell: it becomes a global
  setting propagating through every call. Multiple thin translators are
  healthier.
- The endpoint to leave alone is the one correct in its own domain: a
  third-party crate staying bit-exact with upstream, a data source
  faithfully representing its format.

#### Shared utilities live in neutral modules; layering must not invert

When two sibling modules need the same helper, move it to a shared, neutral
module rather than reaching across module boundaries or duplicating it.

- A library module never imports from an application module (a binary, an
  example, a benchmark). If a library needs something an application
  provides, that something is mis-located.
- Inlining a small helper into a second consumer to avoid the refactor is
  technical debt: the day the two copies diverge silently, the bug ships.
- "Neutral" means the module sits at or below every consumer in the
  dependency graph, not inside one of the siblings.

#### Organize by dependency layer, not by feature slice, when components are coupled

Feature slices (one module per feature, each owning its types, logic, and
API) work only when features are independent. When components exchange
data, form a pipeline, or feed back into each other, feature-slicing forces
them to import one another and the graph tangles. Decompose by layer:

- Shared **data types and contracts** go *down* into a core layer everything
  depends on.
- Shared **algorithms** go *down* into a layer written against the core
  types only.
- The **wiring** that composes components into a pipeline goes *up* into an
  orchestration layer.
- Feature/domain modules sit in between, each depending only on layers
  below it, never on a sibling.

**A feedback loop in data flow must never become a cycle in the code
dependency graph.** If A produces what B consumes and B produces what A
consumes, define the exchanged objects in the core layer; both depend on
core; an orchestrator on top closes the loop. The whole graph is a DAG.
Decomposition test: can two components be built and tested without
importing each other? If not, the shared part has not been pushed far
enough down. Once the layers are stable, split them into separate crates of
a Cargo workspace: crates cannot depend on each other cyclically, so a
layer violation fails the build.

#### Never widen visibility to reach across a module boundary

Private means internal to its module. Other modules never depend on it.

- Do not raise an item to `pub(crate)` or `pub` just because another module
  wants to peek at it. Either commit to it as part of the module's interface
  or redesign so the access is unnecessary.
- To dispatch on the kind of an object from another module, use a trait
  method or an enum the object exposes. Never downcast (`Any`) to a concrete
  type owned by another module.

#### Binaries, examples, and benchmarks must use the library, not reimplement it

More than 10 lines of logic in `src/bin/`, `examples/`, or `benches/` that
duplicates a library type (e.g., a sampling loop that mirrors the sampler)
is a bug. Call the public API; if the API does not support what the caller
needs, extend the API.

- The sampler returns a result object that callers can inspect and
  serialize directly.
- A custom variant implements the relevant trait or passes configuration;
  it does not copy-paste the loop.

### E. Validation and errors

#### No silent fallbacks for missing data

If an estimator requires M >= 2 samples to compute variance and receives
M = 1, it returns an `Err`; it does not return zero. Silent fallbacks hide
bugs.

- Never return `0.0`, `NaN`, or an empty collection as a "safe default" for
  a quantity that cannot be computed. Never use `unwrap_or(0.0)` or
  `unwrap_or_default()` to that effect.
- Never let `NaN` or infinity cross a public boundary unannounced. Check
  where they can first arise and return an error naming the input that
  caused them.
- Never silently truncate, pad, or broadcast to make lengths match. `zip`
  stops at the shorter iterator without complaint: check lengths first and
  return an error stating expected and actual lengths.
- Integer arithmetic that can overflow uses `checked_*` and reports the
  failure; `wrapping_*` and `saturating_*` are used only where that
  arithmetic is the intended semantics.
- When dispatching on an optional capability (a trait method returning
  `Option`, a feature flag), the branch for an object lacking the capability
  returns an error unless the fallback behavior is provably correct. A
  plausible but incorrect result is worse than a crash.

#### Validation errors must contain the remediation

When a type rejects an input, the error message tells the caller exactly how
to fix it. "X is not Y" is insufficient; "X is not Y. Wrap it in `Z::new(...)`."
is the minimum bar. This matters most at numerically critical boundaries
(frame conventions, units, schemas), where the alternative to a clear error
is a silent wrong output.

- The reader of the error never needs to open the source. Include the
  remediation snippet, the suggested adapter, or the exact config change.
- Pair with type-tagged contracts (enums, newtypes) so mismatches are caught
  at compile time or at construction, not after data has flowed.
- Never auto-correct silently. The user should see the wrap / convert / cast
  in their own code; auto-correction is exactly the silent-bug pattern this
  rule prevents.

### F. Performance

Speed is the reason this project is written in Rust. These rules apply to
hot paths; everywhere else, clarity wins.

#### Measure before optimizing

- A performance claim needs a number. Benchmark hot paths in `benches/`
  (criterion) and profile release builds (`perf`, flamegraph) before and
  after a change. An optimization without a measured gain is reverted.
- Always measure with `--release`. Debug-build timings are meaningless.

#### Hot-path conventions

- No heap allocation inside hot loops. Allocate buffers once and reuse them;
  pass scratch space as `&mut [f64]` rather than returning a fresh `Vec`
  per call.
- Borrow instead of cloning. A `.clone()` in a hot path needs a reason.
- Use static dispatch (generics, `impl Trait`) so calls can be inlined.
- Prefer iterators over index loops: they avoid bounds checks and make
  length mismatches explicit.
- Keep data contiguous: a flat `Vec<f64>` with a typed accessor, not
  `Vec<Vec<f64>>`.
- `unsafe` is denied crate-wide. An exception requires a measured gain, an
  `#[allow(unsafe_code)]` on the smallest possible item, a `// SAFETY:`
  comment stating why the invariants hold, a safe public wrapper, and a
  test that exercises it.

#### Parallelism and reproducibility

- Parallelize at the outermost level of independent work (e.g. over
  particles), not inside the inner numeric kernels.
- Library code never uses a global or thread-local RNG. Every stochastic
  function takes its RNG as an argument (`&mut impl Rng`), and a run is
  fully determined by its seed.
- Parallel work derives one independent RNG stream per unit of work from
  the run seed, so results do not depend on thread count or scheduling.

---

## Documentation

- **Doc comments**: rustdoc `///` on all public types, traits, methods, and
  functions, with `# Errors`, `# Panics`, `# Safety`, and `# Examples`
  sections where applicable. `missing_docs` is enforced by the lint
  configuration.
- Private items get a one-line doc comment explaining intent, unless the
  name is fully self-documenting.
- A module-level `//!` comment at the top of each `.rs` file explains the
  module's purpose in one to three sentences.
- No doc comments that restate the name (`/// Returns the name.` on
  `name()`).
- Examples in doc comments are compiled and run by `cargo test`; keep them
  correct and minimal.

---

## Testing

### Structure

- **Unit tests** live in a `#[cfg(test)] mod tests` at the bottom of the
  module they test, so tests mirror the source tree by construction.
- **Integration tests** live in `tests/`, one file per area of the public
  API, and can only see the crate's public interface.
- Shared integration-test helpers live in `tests/common/mod.rs`.

### Philosophy

- **Test external interfaces, not internal behavior.** Test a private
  function through the public method that calls it, even though a unit-test
  module is able to reach it.
- **Tests explain why they exist.** Names state the contract verified:
  `exponential_decay_matches_analytical`.
- **Use analytical references for numerical tests**, not
  cross-implementation comparisons.
- **Statistical tests use a fixed seed, wide tolerances**, and enough
  samples to keep the failure rate below 1% if the seed changes.
- **Do not test trivial behavior.** Getters, derived `Debug`/`Clone`, and
  enum variants need no tests. Test validation logic (constructors that
  return `Err` on invalid inputs).

### Fixtures and Stubs

- Test doubles are small structs implementing the trait under test's
  dependency; they do not pull in heavyweight modules.
- Downstream component tests do NOT depend on upstream components being
  correct; use fakes or stubs to isolate.
- Prefer fixture-building functions over shared mutable state.

### Conventions

- Use the built-in test harness (`#[test]`, `cargo test`).
- For the same logic across multiple inputs, iterate over a table of named
  cases and include the case name in the assertion message.
- Never compare floats with `==`. Compare with an explicit absolute or
  relative tolerance.
- Tests that return `Result` use `?`; `unwrap()` is acceptable in tests.
- Gate tests that depend on an optional feature with
  `#[cfg(feature = "...")]`.
- All tests are fast (<5 seconds each) and use tiny configs and minimal
  data. Mark anything slower `#[ignore]` with a reason.

### What to prioritize when writing new tests

1. **Data pipeline correctness** (corrupted inputs silently poison
   everything downstream).
2. **Estimator and likelihood numerics** (sign/normalization bugs are silent
   and devastating).
3. **Interface contracts** (trait implementations, single-item vs. batched
   consistency).
4. **Factory/builder correctness** (config-to-object mapping dispatches to
   the correct implementations).

### What to skip

- Derived trait implementations and trivial getters (unless validation
  logic exists).
- Full-scale sampling runs (too slow for unit tests).
- External service integrations (require live credentials).

---

## Git Practices

- Commit messages are imperative and concise: "Add data loader", not "Added
  the data loader".
- One logical change per commit. Do not mix refactoring with feature
  additions.
- Run `make check` (format check, clippy, rustdoc, tests) before every
  commit. Commit `Cargo.lock` whenever dependencies change.
