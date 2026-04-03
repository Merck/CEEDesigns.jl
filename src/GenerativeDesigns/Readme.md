## Enhanced dynamic CEED integration

This update integrates the core constraint-aware IBMDP functionality into `CEEDesigns` for the dynamic generative design workflow.

### What was added
- Added a new `ConditionalUncertaintyReductionMDP` type as a constraint-aware extension of the baseline `UncertaintyReductionMDP`
- Added support for:
  - posterior row weights via `weights(evidence)`
  - historical data via `data::DataFrame`
  - terminal constraint logic via `terminal_condition = (target_condition, tau)`
- Added helper `conditional_likelihood(...)` to compute posterior belief mass over target ranges
- Added public helpers:
  - `conditional_efficient_design`
  - `conditional_efficient_designs`
  - `perform_ensemble_designs`

### Behavioral updates
- Terminal states now require:
  - uncertainty below threshold, and
  - conditional likelihood above the specified belief threshold `tau`
- Transitions always incorporate sampled evidence
- Reward was updated to be **incremental per step** rather than based on full cumulative cost

### Tests
- Added basic tests for:
  - conditional likelihood
  - terminal-condition behavior
  - evidence-updating transitions
  - incremental reward behavior

### Scope
- Only the necessary `src/` logic for the enhanced dynamic algorithm was integrated
- `ValueIteration/`-related code was intentionally excluded