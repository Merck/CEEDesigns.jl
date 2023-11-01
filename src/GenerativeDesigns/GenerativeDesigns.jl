module GenerativeDesigns

using POMDPs
using POMDPTools: ImplicitDistribution, Deterministic
using POMDPSimulators

using Combinatorics
using DataFrames, ScientificTypes
using Statistics
using StatsBase: Weights, countmap, entropy, sample
using Random: default_rng, AbstractRNG
using MCTS

using ..CEED: front

export ResearchMDP, DistanceBased
export QuadraticStandardizedDistance, DiscreteMetric, Exponential
export Variance, Entropy
export State
export efficient_designs

"""
Represent state (knowledge) as an immutable dictionary.
"""
const State = Base.ImmutableDict{String,Any}

function Base.merge(d::State, dsrc::Dict)
    for (k, v) in dsrc
        d = State(d, k, v)
    end

    return d
end

State(p1::Pair, pairs::Pair...) = merge(State(), Dict(p1, pairs...))

include("distancebased.jl")

"""
Represent action as a named tuple `(; costs=[monetary cost, time], features)`.
"""
const ActionCost = NamedTuple{(:costs, :features),<:Tuple{Vector{Float64},Vector{String}}}

const const_bigM = 1_000_000

## define the MDP
"""
    ResearchMDP(costs, sampler, uncertainty, threshold, state=State(); <keyword arguments>)

Structure that parametrizes the experimental decision-making process. It is used in the object interface of POMDPs.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.
  - `sampler`: a function of `(state, features, rng)`, in which `state` denotes the current experimental state, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `state`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `threshold`: a number representing the acceptable level of uncertainty about the target variable.
  - `state=State()`: initial experimental evidence.

# Keyword Arguments

  - `costs_tradeoff`: tradeoff between monetary cost and execution time of an experimental designs, modeled as a vector of length two.
  - `max_parallel`: maximum number of parallel experiments.
  - `discount`: this is the discounting factor utilized in reward computation.
  - `bigM`: it refers to the penalty that arises in a scenario where further experimental action is not an option, yet the uncertainty exceeds the allowable limit.
  - `max_experiments`: this denotes the maximum number of experiments that are permissible to be conducted.    # initial state
"""
struct ResearchMDP <: POMDPs.MDP{State,Vector{String}}
    # initial state
    initial_state::State
    # uncertainty threshold
    threshold::Float64

    # actions and costs
    costs::Dict{String,ActionCost}
    # monetary cost v. time tradeoff
    costs_tradeoff::Vector{Float64}
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

    function ResearchMDP(
        costs,
        sampler,
        uncertainty,
        threshold,
        state = State();
        costs_tradeoff = [1, 0],
        max_parallel::Int = 1,
        discount = 1.0,
        bigM = const_bigM,
        max_experiments = bigM,
    )

        # check if `sampler`, `uncertainty` are compatible
        @assert hasmethod(sampler, Tuple{State,Vector{String},AbstractRNG}) """`sampler` must implement a method accepting `(state, readout features, rng)` as its arguments."""
        @assert hasmethod(uncertainty, Tuple{State}) """`uncertainty` must implement a method accepting `state` as its argument."""

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

        new(
            State(state...),
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

function POMDPs.actions(m::ResearchMDP, state)
    all_actions = filter!(collect(keys(m.costs))) do a
        !isempty(m.costs[a].features) && !in(first(m.costs[a].features), keys(state))
    end

    if !isempty(all_actions) && (length(state) < m.max_experiments)
        collect(powerset(all_actions, 1, m.max_parallel))
    else
        [[eox]]
    end
end

function POMDPs.isterminal(m::ResearchMDP, state)
    #haskey(state, eox) || (haskey(state, "kpuu") && println(m.uncertainty(state)))
    return haskey(state, eox) || (m.uncertainty(state) <= m.threshold)
end

POMDPs.discount(m::ResearchMDP) = m.discount

POMDPs.initialstate(m::ResearchMDP) = Deterministic(m.initial_state)

function POMDPs.transition(m::ResearchMDP, state, action_set)
    if action_set == [eox]
        Deterministic(merge(state, Dict(eox => -1)))
    else
        # readout features 
        features = vcat(map(action -> m.costs[action].features, action_set)...)

        ImplicitDistribution() do rng
            # sample readouts from history
            observation = m.sampler(state, features, rng)

            # create new state, add new information
            merge(state, observation)
        end
    end
end

function POMDPs.reward(m::ResearchMDP, _, action)
    if action == [eox]
        -m.bigM
    else
        costs = zeros(2)
        for experiment in action
            costs[1] += m.costs[experiment].costs[1] # monetary cost
            costs[2] = max(costs[2], m.costs[experiment].costs[2]) # time
        end

        -costs' * m.costs_tradeoff
    end
end

"""
    compute_execution_cost(costs, actions; discount=1.)

Compute monetary cost and execution time for a sequence of actions. Returns a named tuple `(; monetary_cost, time)`.
"""
function compute_execution_cost(costs, actions; discount = 1.0)
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

    monetary_cost = time = 0

    for action in actions
        (action == [eox]) && break

        time_group = 0 # total duration of parallel assays
        for experiment in action
            monetary_cost += discount * costs[experiment].costs[1] # monetary cost
            time_group = max(time_group, costs[experiment].costs[2]) # time
        end
        time += discount * time_group
    end

    (; monetary_cost, time)
end

const default_solver = DPWSolver(; n_iterations = 100_000, tree_in_info = true)
const default_repetitions = 20

"""
    efficient_designs(costs, sampler, uncertainty, n_thresholds, state=State(); <keyword arguments>)

Estimate the combined experimental costs of 'generative' experimental designs over a range of uncertainty thresholds, and return the set of Pareto-efficient designs in the dimension of cost and uncertainty threshold.

Internally, an instance of the `ResearchMDP` reference is created for every selected uncertainty threshold and the corresponding runoffs are simulated.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.
  - `sampler`: a function of `(state, features, rng)`, in which `state` denotes the current experimental state, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `state`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `n_thresholds`: number of thresholds to consider uniformly in the range between 0 and 1, inclusive.
  - `state=State()`: initial experimental evidence.

# Keyword Arguments

  - `solver=default_solver`: a POMDPs.jl compatible solver used to solve the decision process. The default solver is [`DPWSolver`](https://juliapomdp.github.io/MCTS.jl/dev/dpw/).
  - `repetitions=default_repetitions`: number of runoffs used to estimate the expected experimental cost.
  - `mdp_options`: a `NamedTuple` of additional keyword arguments that will be passed to the constructor of [`ResearchMDP`](@ref).

# Example

```julia
(; sampler, uncertainty, weights) =
    DistanceBased(data, "HeartDisease", Entropy, Exponential(; λ = 5));
# initialize state
state = State("Age" => 35, "Sex" => "M")
# set up solver (or use default)
solver = GenerativeDesigns.DPWSolver(; n_iterations = 60_000, tree_in_info = true)
designs = efficient_designs(
    experiments,
    sampler,
    uncertainty,
    6,
    state;
    solver,            # planner
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
)
```
"""
function efficient_designs(
    costs,
    sampler,
    uncertainty,
    n_thresholds,
    state = State();
    solver = default_solver,
    repetitions = default_repetitions,
    mdp_options = (;),
)
    designs = []
    for threshold in range(0.0, 1.0, n_thresholds)
        @info "Current threshold level : $threshold"
        mdp = ResearchMDP(costs, sampler, uncertainty, threshold, state; mdp_options...)
        if isterminal(mdp, state)
            push!(designs, ((0.0, threshold), (; monetary_cost = 0.0, time = 0.0)))
        else
            # planner
            planner = solve(solver, mdp)
            queue = [Sim(mdp, planner) for _ = 1:repetitions]

            stats = run_parallel(queue) do _, hist
                (; monetary_cost, time) = compute_execution_cost(costs, hist[:a])
                return (; monetary_cost, time, combined_cost = -discounted_reward(hist))
            end
            show(stats)
            action, info = action_info(planner, state)

            if haskey(info, :tree)
                push!(
                    designs,
                    (
                        (mean(stats.combined_cost), threshold),
                        (;
                            planner,
                            arrangement = [action],
                            monetary_cost = mean(stats.monetary_cost),
                            time = mean(stats.time),
                            tree = info[:tree],
                        ),
                    ),
                )
            else
                push!(
                    designs,
                    (
                        (mean(stats.combined_cost), threshold),
                        (;
                            planner,
                            arrangement = [action],
                            monetary_cost = mean(stats.monetary_cost),
                            time = mean(stats.time),
                        ),
                    ),
                )
            end
        end
    end
    ## rewrite 
    front(x -> x[1], designs)
end

end
