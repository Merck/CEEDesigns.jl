"""
    EfficientValueMDP(costs, sampler, value, evidence=Evidence(); <keyword arguments>)

Structure that parametrizes the experimental decision-making process. It is used in the object interface of POMDPs.

In this experimental setup, our objective is to maximize the value of the experimental evidence (such as clinical utility), adjusted for experimental costs.

Internally, the reward associated with a particular experimental `evidence` and with total accumulated `monetary_cost` and (optionally) `execution_time` is computed as `value(evidence) - costs_tradeoff' * [monetary_cost, execution_time]`.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.
  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `value`: a function of `(evidence)`; it quantifies the utility of experimental evidence.
  - `evidence=Evidence()`: initial experimental evidence.

# Keyword Arguments

  - 'costs_tradeoff': a vector of weights that trade off monetary cost and execution time. Defaults to `[1, 0]`.
  - `max_parallel`: maximum number of parallel experiments.
  - `discount`: this is the discounting factor utilized in reward computation.
  - `bigM`: it refers to the penalty that arises in a scenario where further experimental action is not an option, yet the uncertainty exceeds the allowable limit.
"""
struct EfficientValueMDP <: POMDPs.MDP{State,Vector{String}}
    # initial state
    initial_state::State

    # actions and costs
    costs::Dict{String,ActionCost}
    # maximum number of assays that can be run in parallel
    max_parallel::Int
    # discount
    discount::Float64

    ## sample readouts from the posterior
    sampler::Function
    ## measure of utility
    value::Function

    function EfficientValueMDP(
        costs,
        sampler,
        value,
        evidence = Evidence();
        max_parallel::Int = 1,
        discount = 1.0,
    )
        state = State((evidence, zeros(2)))

        # check if `sampler`, `uncertainty` are compatible
        @assert hasmethod(sampler, Tuple{Evidence,Vector{String},AbstractRNG}) """`sampler` must implement a method accepting `(evidence, readout features, rng)` as its arguments."""
        @assert hasmethod(value, Tuple{Evidence,Vector{Float64}}) """`value` must implement a method accepting `(evidence, costs)` as its argument."""

        # actions and their costs
        costs = Dict{String,ActionCost}(
            try
                if action isa Pair && action[2] isa Pair
                    string(action[1]) => (;
                        costs = Float64[action[2][1]..., 0][1:2],
                        features = convert(Vector{String}, action[2][2]),
                    )
                elseif action isa Pair
                    string(action[1]) => (;
                        costs = Float64[action[2]..., 0][1:2],
                        features = String[action[1]],
                    )
                else
                    error()
                end
            catch
                error("could not parse $action as an action")
            end for action in costs
        )

        return new(state, costs, max_parallel, discount, sampler, value)
    end
end

function POMDPs.actions(m::EfficientValueMDP, state)
    all_actions = filter!(collect(keys(m.costs))) do a
        return !isempty(m.costs[a].features) &&
               !in(first(m.costs[a].features), keys(state.evidence))
    end

    return collect(powerset(all_actions, 1, m.max_parallel))
end

POMDPs.isterminal(m::EfficientValueMDP, state) = isempty(actions(m, state))

POMDPs.discount(m::EfficientValueMDP) = m.discount

POMDPs.initialstate(m::EfficientValueMDP) = Deterministic(m.initial_state)

function POMDPs.transition(m::EfficientValueMDP, state, action_set)
    # costs
    costs = zeros(2)
    for experiment in action_set
        costs[1] += m.costs[experiment].costs[1] # monetary cost
        costs[2] = max(costs[2], m.costs[experiment].costs[2]) # time
    end

    # readout features
    features = vcat(map(action -> m.costs[action].features, action_set)...)
    ImplicitDistribution() do rng
        # sample readouts from history
        observation = m.sampler(state.evidence, features, rng)

        # create new evidence, add new information
        return merge(state, observation, costs)
    end
end

function POMDPs.reward(m::EfficientValueMDP, previous_state::State, _, state::State)
    return m.value(state.evidence, state.costs) -
           m.value(previous_state.evidence, previous_state.costs)
end

"""
    efficient_value(costs, sampler, value, evidence=Evidence(); <keyword arguments>)

Estimate the maximum value of experimental evidence (such as clinical utility), adjusted for experimental costs.

Internally, an instance of the `EfficientValueMDP` structure is created and a summary over `repetitions` runoffs is returned.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.
  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `value`: a function of `(evidence, (monetary costs, execution time))`; it quantifies the utility of experimental evidence.
  - `evidence=Evidence()`: initial experimental evidence.

# Keyword Arguments

  - `solver=default_solver`: a POMDPs.jl compatible solver used to solve the decision process. The default solver is [`DPWSolver`](https://juliapomdp.github.io/MCTS.jl/dev/dpw/).
  - `repetitions=0`: number of runoffs used to estimate the expected experimental cost.
  - `mdp_options`: a `NamedTuple` of additional keyword arguments that will be passed to the constructor of [`EfficientValueMDP`](@ref).

# Example

```julia
(; sampler, uncertainty, weights) =
    DistanceBased(data, "HeartDisease", Entropy, Exponential(; λ = 5));
value = (evidence, costs) -> (1 - uncertainty(evidence) + 0.005 * sum(costs));
# initialize evidence
evidence = Evidence("Age" => 35, "Sex" => "M")
# set up solver (or use default)
solver =
    GenerativeDesigns.DPWSolver(; n_iterations = 10_000, depth = 3, tree_in_info = true)
design = efficient_value(
    experiments,
    sampler,
    value,
    evidence;
    solver,            # planner
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
)
```
"""
function efficient_value(
    costs,
    sampler,
    value,
    evidence = Evidence();
    solver = default_solver,
    repetitions = 0,
    mdp_options = (;),
)
    mdp = EfficientValueMDP(costs, sampler, value, evidence; mdp_options...)

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
                adjusted_value = value(
                    mdp.initial_state.evidence,
                    mdp.initial_state.costs,
                ) + discounted_reward(hist),
                terminal_value = value(hist[end][:s].evidence, hist[end][:s].costs),
            )
        end

        if haskey(info, :tree)
            return (
                value(mdp.initial_state.evidence, mdp.initial_state.costs) + info[:best_Q],
                (;
                    planner,
                    arrangement = [action],
                    monetary_cost = mean(stats.monetary_cost),
                    time = mean(stats.time),
                    terminal_value = mean(stats.terminal_value),
                    tree = info[:tree],
                    stats,
                ),
            )
        else
            (
                value(mdp.initial_state.evidence, mdp.initial_state.costs) + info[:best_Q],
                (;
                    planner,
                    arrangement = [action],
                    monetary_cost = mean(stats.monetary_cost),
                    time = mean(stats.time),
                    terminal_value = mean(stats.terminal_value),
                    stats,
                ),
            )
        end
    else
        if haskey(info, :tree)
            return (
                value(mdp.initial_state.evidence, mdp.initial_state.costs) + info[:best_Q],
                (; planner, arrangement = [action], tree = info[:tree]),
            )
        else
            (
                value(mdp.initial_state.evidence, mdp.initial_state.costs) + info[:best_Q],
                (; planner, arrangement = [action]),
            )
        end
    end
end
