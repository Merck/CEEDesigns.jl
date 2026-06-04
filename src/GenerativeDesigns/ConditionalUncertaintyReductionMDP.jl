"""
Conditional (constraint-aware) uncertainty-reduction MDP.

This is an enhanced version of the baseline UncertaintyReductionMDP.
It adds:

  - weights(evidence)::Vector: posterior weights over historical rows
  - data::DataFrame: historical data (rows correspond to weights)
  - terminal_condition = (target_condition::Dict, tau::Float64): constraint ranges + belief threshold

`target_condition` is a `Dict{String, <:AbstractVector}` (or any pair-style
mapping) whose keys are column names in `data` and whose values are
two-element ranges `[rmin, rmax]`. The membership test is the inclusive
range `rmin <= x <= rmax`, so the keyed columns **must be numeric** (i.e.
`eltype(data[!, colname]) <: Real`). A non-numeric column (e.g. a
`Multiclass` categorical) raises `ArgumentError` at construction. Rows
containing `NaN` in any constraint column are excluded from the conditional
likelihood (because `NaN >= rmin` is `false`); this is documented behavior,
not silent dropping.

Behavior:

  - The MDP is terminal only when uncertainty <= threshold AND
    conditional likelihood >= tau (if constraints are provided).
  - Transitions always incorporate sampled evidence; feasibility is enforced
    through the terminal condition and reward-driven solver behavior.
"""
struct ConditionalUncertaintyReductionMDP{S, U, W} <: POMDPs.MDP{State, Vector{String}}
    # initial state
    initial_state::State
    # uncertainty threshold
    threshold::Float64

    # actions and costs
    costs::Dict{String, ActionCost}
    # monetary cost v. time tradeoff
    costs_tradeoff::NTuple{2, Float64}
    # maximum number of assays that can be run in parallel
    max_parallel::Int
    # discount
    discount::Float64
    # max experiments
    max_experiments::Int64
    # penalty if max number of experiments exceeded
    bigM::Int64

    # sample readouts from the posterior
    sampler::S
    # measure of uncertainty about the ground truth
    uncertainty::U

    # NEW: compute current state weights
    weights::W
    # NEW: historical data
    data::DataFrame
    # NEW: (target constraints, belief threshold tau)
    terminal_condition::Tuple{Dict, Float64}

    function ConditionalUncertaintyReductionMDP(
            costs;
            sampler::S,
            uncertainty::U,
            threshold,
            evidence = Evidence(),
            costs_tradeoff = (1.0, 0.0),
            max_parallel::Int = 1,
            discount = 1.0,
            bigM = const_bigM,
            max_experiments = bigM,
            weights::W,
            data,
            terminal_condition = (Dict(), 0.0),
        ) where {S, U, W}
        state = State((evidence, Tuple(zeros(2))))

        hasmethod(sampler, Tuple{Evidence, Vector{String}, AbstractRNG}) || throw(
            ArgumentError("`sampler` must implement a method accepting (evidence, features, rng)."),
        )
        hasmethod(uncertainty, Tuple{Evidence}) || throw(
            ArgumentError("`uncertainty` must implement a method accepting (evidence)."),
        )

        # Validate and normalize `target_condition`. Keys are stringified so Symbol
        # column names round-trip to the String column names in `data` (otherwise the
        # likelihood lookup fails at runtime). Columns must be numeric (allowing
        # `Union{Missing,<:Real}`) because the membership mask uses inclusive
        # `rmin <= x <= rmax`.
        raw_target_cond = first(terminal_condition)
        tau_value = last(terminal_condition)
        normalized_target_cond = Dict{String, Tuple{Float64, Float64}}()
        for (colname, range) in raw_target_cond
            scol = string(colname)
            if !(scol in names(data))
                throw(
                    ArgumentError(
                        "target_condition references column `$(colname)` which is not present in `data`.",
                    ),
                )
            end
            col_eltype = eltype(data[!, scol])
            if !(nonmissingtype(col_eltype) <: Real)
                throw(
                    ArgumentError(
                        "target_condition column `$(colname)` has eltype `$(col_eltype)`, " *
                            "but conditional ranges (rmin <= x <= rmax) are only supported on numeric columns " *
                            "(`Union{Missing,<:Real}` is allowed; missing rows are excluded). " *
                            "Coerce the column to a numeric scitype (e.g. Continuous) before constructing the MDP.",
                    ),
                )
            end
            if length(range) != 2
                throw(
                    ArgumentError(
                        "target_condition for column `$(colname)` must be a 2-element `[rmin, rmax]`; got `$(range)`.",
                    ),
                )
            end
            rmin, rmax = Float64(range[1]), Float64(range[2])
            rmin <= rmax || throw(
                ArgumentError(
                    "target_condition for column `$(colname)` has rmin=$(rmin) > rmax=$(rmax).",
                ),
            )
            normalized_target_cond[scol] = (rmin, rmax)
        end
        terminal_condition = (normalized_target_cond, Float64(tau_value))

        # Parse costs dict into CEED ActionCost format
        parsed_costs = Dict{String, ActionCost}(
            try
                    if action isa Pair && action[2] isa Pair
                        string(action[1]) => (;
                            costs = Tuple(Float64[action[2][1]..., 0][1:2]),
                            features = convert(Vector{String}, action[2][2]),
                        )
                elseif action isa Pair
                        string(action[1]) => (;
                            costs = Tuple(Float64[action[2]..., 0][1:2]),
                            features = String[action[1]],
                        )
                else
                        error()
                end
            catch
                    error("could not parse $action as an action")
            end for action in costs
        )

        return new{S, U, W}(
            state,
            threshold,
            parsed_costs,
            costs_tradeoff,
            max_parallel,
            discount,
            max_experiments,
            bigM,
            sampler,
            uncertainty,
            weights,
            data,
            terminal_condition,
        )
    end
end

# --- helper: conditional likelihood for multiple constraints ---
function conditional_likelihood(
        evidence;
        compute_weights,
        hist_data::DataFrame,
        target_condition::Dict,
    )
    w = compute_weights(evidence)
    length(w) == nrow(hist_data) || throw(
        ArgumentError(
            "weights length ($(length(w))) must match number of rows in hist_data ($(nrow(hist_data))).",
        ),
    )

    valid = trues(nrow(hist_data))
    for (colname, range) in target_condition
        scol = string(colname)
        scol in names(hist_data) ||
            throw(ArgumentError("target_condition column `$(colname)` not found in data."))
        rmin, rmax = range
        col = hist_data[!, scol]
        # `coalesce(..., false)` excludes rows whose constraint column is `missing`.
        valid .&= coalesce.((col .>= rmin) .& (col .<= rmax), false)
    end

    return sum(w[valid])
end

# --- POMDPs interface ---
function POMDPs.actions(m::ConditionalUncertaintyReductionMDP, state)
    all_actions = filter!(collect(keys(m.costs))) do a
        feats = m.costs[a].features
        !isempty(feats) && !any(f -> haskey(state.evidence, f), feats)
    end

    # Count *completed experiments* (all of an experiment's features present), not
    # raw evidence entries, so prior evidence / multi-feature experiments don't
    # mis-count against `max_experiments`.
    n_completed = count(keys(m.costs)) do a
        feats = m.costs[a].features
        !isempty(feats) && all(f -> haskey(state.evidence, f), feats)
    end

    return if !isempty(all_actions) && (n_completed < m.max_experiments)
        collect(powerset(all_actions, 1, m.max_parallel))
    else
        [[eox]]
    end
end

function POMDPs.isterminal(m::ConditionalUncertaintyReductionMDP, state)
    if haskey(state.evidence, eox)
        return true
    end

    unc_ok = (m.uncertainty(state.evidence) <= m.threshold)

    target_condition, tau = m.terminal_condition
    cond_ok = true
    if !isempty(target_condition)
        cond_ok =
            conditional_likelihood(
            state.evidence;
            compute_weights = m.weights,
            hist_data = m.data,
            target_condition = target_condition,
        ) >= tau
    end

    return unc_ok && cond_ok
end

POMDPs.discount(m::ConditionalUncertaintyReductionMDP) = m.discount
POMDPs.initialstate(m::ConditionalUncertaintyReductionMDP) = Deterministic(m.initial_state)

function POMDPs.transition(m::ConditionalUncertaintyReductionMDP, state, action_set)
    if action_set == [eox]
        return Deterministic(merge(state, Dict(eox => -1), (0.0, 0.0)))
    end

    # costs
    cost_m, cost_t = 0.0, 0.0
    for experiment in action_set
        cost_m += m.costs[experiment].costs[1]
        cost_t = max(cost_t, m.costs[experiment].costs[2])
    end

    # features
    features = vcat(map(a -> m.costs[a].features, action_set)...)

    return ImplicitDistribution() do rng
        observation = m.sampler(state.evidence, features, rng)
        return merge(state, observation, (cost_m, cost_t))
    end
end

function POMDPs.reward(m::ConditionalUncertaintyReductionMDP, previous_state, action, state)
    if action == [eox]
        return -m.bigM
    else
        return -sum((state.costs .- previous_state.costs) .* m.costs_tradeoff)
    end
end

# --- public helpers (parallel to baseline efficient_design/efficient_designs) ---

function conditional_efficient_design(
        costs;
        sampler,
        uncertainty,
        threshold,
        evidence = Evidence(),
        weights,
        data,
        terminal_condition = (Dict(), 0.0),
        solver = nothing,
        repetitions = 0,
        realized_uncertainty = false,
        mdp_options = (;),
        rng::AbstractRNG = default_rng(),
    )
    mdp = ConditionalUncertaintyReductionMDP(
        costs;
        sampler,
        uncertainty,
        threshold,
        evidence,
        weights,
        data,
        terminal_condition = terminal_condition,
        mdp_options...,
    )

    if isterminal(mdp, mdp.initial_state)
        return (
            (
                0.0, if realized_uncertainty
                    mdp.uncertainty(mdp.initial_state.evidence)
                else
                    threshold
                end,
            ),
            (; monetary_cost = 0.0, time = 0.0),
        )
    end

    planner = solve(isnothing(solver) ? default_solver(rng) : solver, mdp)
    action, info = action_info(planner, mdp.initial_state)

    return if repetitions > 0
        queue = [Sim(mdp, planner; rng = Xoshiro(rand(rng, UInt64))) for _ in 1:repetitions]
        stats = run_parallel(queue) do _, hist
            monetary_cost, time = hist[end][:s].costs
            return (;
                monetary_cost,
                time,
                combined_cost = -discounted_reward(hist),
                actions = hist[:a],
            )
        end

        if haskey(info, :tree)
            return (
                (-info[:best_Q], threshold),
                (;
                    planner,
                    arrangement = [action],
                    monetary_cost = mean(stats.monetary_cost),
                    time = mean(stats.time),
                    tree = info[:tree],
                    stats,
                ),
            )
        else
            return (
                (-info[:best_Q], threshold),
                (;
                    planner,
                    arrangement = [action],
                    monetary_cost = mean(stats.monetary_cost),
                    time = mean(stats.time),
                    stats,
                ),
            )
        end
    else
        if haskey(info, :tree)
            return (
                (-info[:best_Q], threshold),
                (; planner, arrangement = [action], tree = info[:tree]),
            )
        else
            return ((-info[:best_Q], threshold), (; planner, arrangement = [action]))
        end
    end
end

function conditional_efficient_designs(
        costs;
        sampler,
        uncertainty,
        thresholds,
        evidence = Evidence(),
        weights,
        data,
        terminal_condition = (Dict(), 0.0),
        solver = nothing,
        repetitions = 0,
        realized_uncertainty = false,
        mdp_options = (;),
        rng::AbstractRNG = default_rng(),
    )
    thresholds < 2 && throw(ArgumentError("`thresholds` must be at least 2 (got $thresholds); use `conditional_efficient_design` for a single threshold."))
    designs = []
    for threshold in range(0.0, 1.0, thresholds)
        @info "Current threshold level : $threshold"
        inner_rng = Xoshiro(rand(rng, UInt64))
        inner_solver = isnothing(solver) ? default_solver(inner_rng) : solver
        push!(
            designs,
            conditional_efficient_design(
                costs;
                sampler,
                uncertainty,
                threshold,
                evidence,
                weights,
                data,
                terminal_condition = terminal_condition,
                solver = inner_solver,
                repetitions,
                realized_uncertainty,
                mdp_options,
                rng = inner_rng,
            ),
        )
    end
    return front(x -> x[1], designs)
end

"""
    perform_ensemble_designs(costs; ..., tau_set = [0.9], N = 30)

Run an ensemble of `N` independent `conditional_efficient_designs` calls per
belief threshold `tau` in `tau_set` (the legacy name `thred_set` is still
accepted as an alias), returning a
`Dict{Float64, Vector}` keyed by `tau::Float64`. Each value is the vector of
`N` ensemble outputs (each output is itself the Pareto front returned by
`conditional_efficient_designs`).

Example access pattern:

```julia
results = perform_ensemble_designs(experiments; ..., tau_set = [0.6, 0.9])
runs_at_06 = results[0.6]   # Vector of length N
```
"""
function perform_ensemble_designs(
        costs;
        sampler,
        uncertainty,
        thresholds,
        evidence = Evidence(),
        weights,
        data,
        terminal_condition = (Dict(), 0.0),
        realized_uncertainty = false,
        solver = nothing,
        repetitions = 0,
        mdp_options = (;),
        tau_set = [0.9],
        thred_set = nothing,
        N = 30,
        rng::AbstractRNG = default_rng(),
    )
    # `thred_set` is the legacy spelling; prefer `tau_set`.
    tau_values = isnothing(thred_set) ? tau_set : thred_set
    results = Dict{Float64, Vector}()

    for tau in tau_values
        ensemble_results = []
        for i in 1:N
            @info "Running ensemble $i for belief threshold τ=$tau"
            # Independent, reproducible rng per ensemble run (derived from `rng`).
            run_rng = Xoshiro(rand(rng, UInt64))
            design = conditional_efficient_designs(
                costs;
                sampler = sampler,
                uncertainty = uncertainty,
                thresholds = thresholds,
                evidence = evidence,
                weights = weights,
                data = data,
                terminal_condition = (terminal_condition[1], tau),
                realized_uncertainty = realized_uncertainty,
                solver = solver,
                repetitions = repetitions,
                mdp_options = mdp_options,
                rng = run_rng,
            )
            push!(ensemble_results, design)
        end
        results[Float64(tau)] = ensemble_results
    end

    return results
end
