"""
    UncertaintyReductionMDP(costs; sampler, uncertainty, threshold, evidence=Evidence(), <keyword arguments>)

Structure that parametrizes the experimental decision-making process. It is used in the object interface of POMDPs.

In this experimental setup, our objective is to minimize the expected experimental cost while ensuring the uncertainty remains below a specified threshold.

Internally, a state of the decision process is modeled as a tuple `(evidence::Evidence, [total accumulated monetary cost, total accumulated execution time])`.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.

# Keyword Arguments

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `threshold`: a number representing the acceptable level of uncertainty about the target variable.
  - `evidence=Evidence()`: initial experimental evidence.
  - `costs_tradeoff`: tradeoff between monetary cost and execution time of an experimental designs, given as a tuple of floats.
  - `max_parallel`: maximum number of parallel experiments.
  - `discount`: this is the discounting factor utilized in reward computation.
  - `bigM`: it refers to the penalty that arises in a scenario where further experimental action is not an option, yet the uncertainty exceeds the allowable limit.
  - `max_experiments`: this denotes the maximum number of experiments that are permissible to be conducted.
"""
struct UncertaintyReductionMDP <: POMDPs.MDP{State,Vector{String}}
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

    ## sample readouts from the posterior
    sampler::Function
    ## measure of uncertainty about the ground truth
    uncertainty::Function

    function UncertaintyReductionMDP(
        costs;
        sampler,
        uncertainty,
        threshold,
        evidence = Evidence(),
        costs_tradeoff = (1, 0),
        max_parallel::Int = 1,
        discount = 1.0,
        bigM = const_bigM,
        max_experiments = bigM,
    )
        state = State((evidence, Tuple(zeros(2))))

        # check if `sampler`, `uncertainty` are compatible
        @assert hasmethod(sampler, Tuple{Evidence,Vector{String},AbstractRNG}) """`sampler` must implement a method accepting `(evidence, readout features, rng)` as its arguments."""
        @assert hasmethod(uncertainty, Tuple{Evidence}) """`uncertainty` must implement a method accepting `evidence` as its argument."""

        # actions and their costs
        costs = Dict{String,ActionCost}(
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
            costs,
            costs_tradeoff,
            max_parallel,
            discount,
            max_experiments,
            bigM,
            sampler,
            uncertainty,
        )
    end
end

"""
A penalized action that results in a terminal state, e.g., in situations where conducting additional experiments is not possible, but the level of uncertainty remains above an acceptable threshold.
"""
const eox = "EOX"

function POMDPs.actions(m::UncertaintyReductionMDP, state)
    all_actions = filter!(collect(keys(m.costs))) do a
        return !isempty(m.costs[a].features) &&
               !in(first(m.costs[a].features), keys(state.evidence))
    end

    if !isempty(all_actions) && (length(state.evidence) < m.max_experiments)
        collect(powerset(all_actions, 1, m.max_parallel))
    else
        [[eox]]
    end
end

function POMDPs.isterminal(m::UncertaintyReductionMDP, state)
    return haskey(state.evidence, eox) || (m.uncertainty(state.evidence) <= m.threshold)
end

POMDPs.discount(m::UncertaintyReductionMDP) = m.discount

POMDPs.initialstate(m::UncertaintyReductionMDP) = Deterministic(m.initial_state)

function POMDPs.transition(m::UncertaintyReductionMDP, state, action_set)
    if action_set == [eox]
        Deterministic(merge(state, Dict(eox => -1), (0.0, 0.0)))
    else
        # costs
        cost_m, cost_t = 0.0, 0.0
        for experiment in action_set
            cost_m += m.costs[experiment].costs[1] # monetary cost
            cost_t = max(cost_t, m.costs[experiment].costs[2]) # time
        end

        # readout features 
        features = vcat(map(action -> m.costs[action].features, action_set)...)
        ImplicitDistribution() do rng
            # sample readouts from history
            observation = m.sampler(state.evidence, features, rng)

            # create new evidence, add new information
            return merge(state, observation, (cost_m, cost_t))
        end
    end
end

function POMDPs.reward(m::UncertaintyReductionMDP, _, action, state)
    if action == [eox]
        -m.bigM
    else
        -sum(state.costs .* m.costs_tradeoff)
    end
end

"""
    efficient_design(costs; sampler, uncertainty, threshold, evidence=Evidence(), <keyword arguments>)

In the uncertainty reduction setup, minimize the expected experimental cost while ensuring the uncertainty remains below a specified threshold.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.

# Keyword Arguments

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `threshold`: uncertainty threshold.
  - `evidence=Evidence()`: initial experimental evidence.
  - `solver=default_solver`: a POMDPs.jl compatible solver used to solve the decision process. The default solver is [`DPWSolver`](https://juliapomdp.github.io/MCTS.jl/dev/dpw/).
  - `repetitions=0`: number of runoffs used to estimate the expected experimental cost.
  - `mdp_options`: a `NamedTuple` of additional keyword arguments that will be passed to the constructor of [`UncertaintyReductionMDP`](@ref).
  - `realized_uncertainty=false`: whenever the initial state uncertainty is below the selected threshold, return the actual uncertainty of this state.

# Example

```julia
(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy,
    similarity = Exponential(; λ = 5),
);
# initialize evidence
evidence = Evidence("Age" => 35, "Sex" => "M")
# set up solver (or use default)
solver = GenerativeDesigns.DPWSolver(; n_iterations = 60_000, tree_in_info = true)
designs = efficient_design(
    costs;
    experiments,
    sampler,
    uncertainty,
    threshold = 0.6,
    evidence,
    solver,            # planner
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
)
```
"""
function efficient_design(
    costs;
    sampler,
    uncertainty,
    threshold,
    evidence = Evidence(),
    solver = default_solver,
    repetitions = 0,
    realized_uncertainty = false,
    mdp_options = (;),
)
    mdp = UncertaintyReductionMDP(
        costs;
        sampler,
        uncertainty,
        threshold,
        evidence,
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
    else
        # planner
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
end

"""
    efficient_designs(costs; sampler, uncertainty, thresholds, evidence=Evidence(), <keyword arguments>)

In the uncertainty reduction setup, minimize the expected experimental resource spend over a range of uncertainty thresholds, and return the set of Pareto-efficient designs in the dimension of cost and uncertainty threshold.

Internally, an instance of the `UncertaintyReductionMDP` structure is created for every selected uncertainty threshold and the corresponding runoffs are simulated.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.

# Keyword Arguments

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `thresholds`: number of thresholds to consider uniformly in the range between 0 and 1, inclusive.
  - `evidence=Evidence()`: initial experimental evidence.
  - `solver=default_solver`: a POMDPs.jl compatible solver used to solve the decision process. The default solver is [`DPWSolver`](https://juliapomdp.github.io/MCTS.jl/dev/dpw/).
  - `repetitions=0`: number of runoffs used to estimate the expected experimental cost.
  - `mdp_options`: a `NamedTuple` of additional keyword arguments that will be passed to the constructor of [`UncertaintyReductionMDP`](@ref).
  - `realized_uncertainty=false`: whenever the initial state uncertainty is below the selected threshold, return the actual uncertainty of this state.

# Example

```julia
(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy,
    similarity = Exponential(; λ = 5),
);
# initialize evidence
evidence = Evidence("Age" => 35, "Sex" => "M")
# set up solver (or use default)
solver = GenerativeDesigns.DPWSolver(; n_iterations = 60_000, tree_in_info = true)
designs = efficient_designs(
    costs;
    experiments,
    sampler,
    uncertainty,
    thresholds = 6,
    evidence,
    solver,            # planner
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
)
```
"""
function efficient_designs(
    costs;
    sampler,
    uncertainty,
    thresholds,
    evidence = Evidence(),
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
            efficient_design(
                costs;
                sampler,
                uncertainty,
                threshold,
                evidence,
                solver,
                repetitions,
                realized_uncertainty,
                mdp_options,
            ),
        )
    end
    ## rewrite 
    return front(x -> x[1], designs)
end
