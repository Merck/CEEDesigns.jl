# Changelog

All notable changes to CEEDesigns.jl are documented in this file.

## [Unreleased]

This update resolves a set of correctness, reproducibility, and API/usability
issues identified during a pre-JOSS code review. Each item lists the previous
behavior and the change. Issue tags (S#, C#, M#, N#, D#, P#) are internal
review references.

### Fixed — correctness & crashes

- **Categorical distance direction (S1).** `DiscreteDistance` returned `λ` for a
  *match* and `0` for a *mismatch* — the opposite of a distance. After collation
  through the `Exponential` similarity, historical rows that *matched* the
  evidence received the *lowest* weight. It now returns `0` on a match and `λ` on
  a mismatch, consistent with `QuadraticDistance`, so matching rows receive the
  highest similarity weight.
  *Note:* this changes the numerical output of any model that uses categorical
  features.

- **Symbol-keyed constraints crashed at run time (C1).**
  `ConditionalUncertaintyReductionMDP` validated `target_condition` keys by
  stringifying them but stored the original keys, so a `Dict(:Col => …)`
  constructed successfully and then threw "Column … not found" during
  `isterminal`. Keys are now normalized to `String` (and ranges validated) at
  construction, and `conditional_likelihood` accepts both `String` and `Symbol`
  keys.

- **Columns with missing values were rejected (C2).** The numeric check
  `eltype(col) <: Real` excluded `Union{Missing, Float64}` columns. It now uses
  `nonmissingtype(...) <: Real`, and the membership mask uses
  `coalesce(…, false)` so rows with `missing` in a constraint column are excluded
  rather than erroring.

- **`max_experiments` counted evidence entries (M1).** The budget check counted
  evidence dictionary entries (including prior evidence and each feature of a
  multi-feature experiment) instead of completed experiments. It now counts
  experiments whose features are all present.

- **Constant feature column produced an opaque error (M4).** A zero-variance
  column made `QuadraticDistance` divide by zero, surfacing as the unrelated
  "weights cannot contain Inf or NaN values". It now raises a clear
  `ArgumentError`, and `σ` is recomputed per call (removing a latent
  cross-column memoization bug).

- **`importance_weights` length was unchecked (N3).** A wrong-length vector failed
  later with an opaque broadcasting error. The length is now validated against
  `nrow(data)` with a clear `ArgumentError`.

### Fixed — reproducibility & concurrency

- **Shared RNG across parallel simulations (S2).** All `Sim` runoffs shared a
  single RNG object (a data race under `run_parallel`), and one solver instance
  was reused across thresholds and ensemble runs. Each `Sim` now gets an
  independent RNG derived from the caller's RNG; `default_solver(rng)` is
  seedable; `efficient_designs` / `conditional_efficient_designs` use a fresh
  solver and RNG per threshold; and `perform_ensemble_designs` uses an
  independent RNG per ensemble run. Results are reproducible from a single seed.

- **Static arrangement search used the global RNG (S3).** `optimal_arrangement`
  built its `DPWSolver` with the implicit `Random.GLOBAL_RNG`, which the threaded
  `StaticDesigns.efficient_designs` accessed concurrently. `optimal_arrangement`
  now takes an `rng`, and the threaded driver pre-seeds one independent RNG per
  task.

### Changed — API, documentation, robustness

- **Input validation uses `ArgumentError` (C3).** User-facing `@assert`s in the
  MDP constructors, `conditional_likelihood`, and `Entropy` now throw
  `ArgumentError` with stable messages.
- **`EfficientValueMDP` value signature (D1).** Docstring corrected from
  `value(evidence)` to `value(evidence, costs)`.
- **Documentation / defaults / naming (D2).** `Exponential` docstring corrected to
  `λ = 1/2` and `exp(-λ x)`; fixed typos; unified the `terminal_condition`
  default; added `tau_set` as the keyword for `perform_ensemble_designs` (the
  previous `thred_set` is still accepted).
- **Static designs & plotting (D3).** Experiment feature lists are normalized to
  `String` so `Symbol` features work; `plot_front` guards against empty input; a
  docstring referenced a nonexistent `get_labels` (now `make_labels`); and the
  ensemble output column `Average_Utility` was renamed to `Average_Cost` to
  reflect that it holds cost.
- **Ensemble frequency counting (M2).** `ensemble_to_dataframe` now de-duplicates
  identical points within a single run, so collapsed points (under
  `realized_uncertainty = true`) no longer inflate frequencies / MLASP votes.
- **Documentation note (P1/P2).** Documented that the uncertainty measure is
  normalized (thresholds on `[0, 1]`) and that the terminal information term is
  realized as a hard terminal condition rather than an additive penalty.

### Added

- Regression tests covering the above: `test/GenerativeDesigns/test_review_fixes.jl`
  (S1/C1/C2/M1/M4/N3), ensemble within-run de-duplication, and static arrangement
  reproducibility.

### Testing

- Full test suite passes with `JULIA_NUM_THREADS=4`.
