

"""
Conditional (constraint-aware) uncertainty-reduction MDP.

This is an enhanced version of the baseline UncertaintyReductionMDP.
It adds:
- weights(evidence)::Vector: posterior weights over historical rows
- data::DataFrame: historical data (rows correspond to weights)
- terminal_condition = (target_condition::Dict, tau::Float64): constraint ranges + belief threshold

Behavior:
- The MDP is terminal only when uncertainty <= threshold AND conditional likelihood >= tau (if constraints provided).
- Transition can optionally "block" evidence updates if conditional likelihood is below tau.
"""
struct ConditionalUncertaintyReductionMDP <: POMDPs.MDP{State,Vector{String}}
    # initial state
    initial_state::State
    # uncertainty threshold
    threshold::Float64

    # actions and costs
    costs::Dict{String,ActionCost}
    # monetary cost v. time tradeoff
    costs_tradeoff::NTuple{2,Float64}
    # maximum number of assays that can be run in parallel
    max_parallel::Int
    # discount
    discount::Float64
    # max experiments
    max_experiments::Int64
    # penalty if max number of experiments exceeded
    bigM::Int64

    # sample readouts from the posterior
    sampler::Function
    # measure of uncertainty about the ground truth
    uncertainty::Function

    # NEW: compute current state weights
    weights::Function
    # NEW: historical data
    data::DataFrame
    # NEW: (target constraints, belief threshold tau)
    terminal_condition::Tuple{Dict,Float64}

    function ConditionalUncertaintyReductionMDP(
        costs;
        sampler,
        uncertainty,
        threshold,
        evidence = Evidence(),
        costs_tradeoff = (1.0, 0.0),
        max_parallel::Int = 1,
        discount = 1.0,
        bigM = const_bigM,
        max_experiments = bigM,
        weights,
        data,
        terminal_condition = (Dict(), 0.0),
    )
        state = State((evidence, Tuple(zeros(2))))

        @assert hasmethod(sampler, Tuple{Evidence,Vector{String},AbstractRNG}) """
            `sampler` must implement a method accepting (evidence, features, rng).
        """
        @assert hasmethod(uncertainty, Tuple{Evidence}) """
            `uncertainty` must implement a method accepting (evidence).
        """

        # Parse costs dict into CEED ActionCost format
        parsed_costs = Dict{String,ActionCost}(
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

        return new(
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
    @assert length(w) == nrow(hist_data) "weights length must match number of rows in hist_data"

    valid = trues(nrow(hist_data))
    for (colname, range) in target_condition
        @assert colname in names(hist_data) "Column $colname not found in data."
        rmin, rmax = range
        valid .&= (hist_data[!, colname] .>= rmin) .& (hist_data[!, colname] .<= rmax)
    end

    return sum(w[valid])
end

# mark that an experiment was attempted so it won't be re-selected
function mark_attempts(evidence::Evidence, action_set)
    d = Dict{String,Any}()
    for a in action_set
        d["__attempted__" * a] = true
    end
    return merge(evidence, d)
end

# --- POMDPs interface ---
function POMDPs.actions(m::ConditionalUncertaintyReductionMDP, state)
    all_actions = filter!(collect(keys(m.costs))) do a
        attempted_key = "__attempted__" * a
        return !isempty(m.costs[a].features) &&
               !haskey(state.evidence, attempted_key) &&     # do not retry blocked action forever
               !in(first(m.costs[a].features), keys(state.evidence))
    end

    if !isempty(all_actions) && (length(state.evidence) < m.max_experiments)
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
        cond_ok = conditional_likelihood(
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
        # sampler may return Dict OR (Dict, ...)
        obs = m.sampler(state.evidence, features, rng)
        observation = obs isa Dict ? obs : first(obs)

        # state where we ALWAYS pay cost (even if we later block evidence update)
        base_costed_state = State((state.evidence, state.costs .+ (cost_m, cost_t)))

        # proposed next state (with evidence update)
        updated_state = merge(state, observation, (cost_m, cost_t))

        # if constraints exist, block evidence updates when likelihood < tau
        target_condition, tau = m.terminal_condition
        if !isempty(target_condition)
            p = conditional_likelihood(
                updated_state.evidence;
                compute_weights = m.weights,
                hist_data = m.data,
                target_condition = target_condition,
            )

            if p < tau
                # BLOCK evidence update, but:
                # 1) keep the paid costs
                # 2) mark the attempted action(s) so they won't be selected again
                attempted_evidence = mark_attempts(base_costed_state.evidence, action_set)
                return State((attempted_evidence, base_costed_state.costs))
            end
        end

        return updated_state
    end
end

function POMDPs.reward(m::ConditionalUncertaintyReductionMDP, _, action, state)
    if action == [eox]
        -m.bigM
    else
        -sum(state.costs .* m.costs_tradeoff)
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
    terminal_condition = (Dict(), 0.8),
    solver = default_solver,
    repetitions = 0,
    realized_uncertainty = false,
    mdp_options = (;),
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
            (0.0, if realized_uncertainty
                mdp.uncertainty(mdp.initial_state.evidence)
            else
                threshold
            end),
            (; monetary_cost = 0.0, time = 0.0),
        )
    end

    planner = solve(solver, mdp)
    action, info = action_info(planner, mdp.initial_state)

    if repetitions > 0
        queue = [Sim(mdp, planner) for _ = 1:repetitions]
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
    terminal_condition = (Dict(), 0.8),
    solver = default_solver,
    repetitions = 0,
    realized_uncertainty = false,
    mdp_options = (;),
)
    designs = []
    for threshold in range(0.0, 1.0, thresholds)
        @info "Current threshold level : $threshold"
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
                solver,
                repetitions,
                realized_uncertainty,
                mdp_options,
            ),
        )
    end
    return front(x -> x[1], designs)
end

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
    solver = default_solver,
    repetitions = 0,
    mdp_options = (;),
    thred_set = [0.9],
    N = 30,
)
    results = Dict()

    for tau in thred_set
        ensemble_results = []
        for i in 1:N
            @info "Running ensemble $i for belief threshold τ=$tau"
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
            )
            push!(ensemble_results, design)
        end
        results[:belief => tau] = ensemble_results
    end

    return results
end

