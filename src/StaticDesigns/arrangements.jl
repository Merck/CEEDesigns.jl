## optimal arrangement as a MDP policy
"""
ArrangementMDP(; experiments, experimental_costs, evals, max_parallel=1, tradeoff=(1, 0))
Structure to parametrize a MDP that is used to approximate the optimal experimental arrangement.
"""
Base.@kwdef struct ArrangementMDP{T1<:Number,T2<:Number} <:
                   POMDPs.MDP{Set{String},Set{String}}
    # experiments
    experiments::Set{String}
    # experimental costs
    experimental_costs::Dict{String,NTuple{2,Float64}}
    # loss, filtration
    evals::Dict{Set{String},ExperimentalEval}
    # maximum number of parallel experiments
    max_parallel::Int = 1
    # monetary cost v. time tradeoff
    tradeoff::Tuple{T1,T2} = (1, 0)
end

function POMDPs.actions(m::ArrangementMDP, state)
    return Set.(
        collect(powerset(collect(setdiff(m.experiments, state)), 1, m.max_parallel))
    )
end

POMDPs.isterminal(m::ArrangementMDP, state) = m.experiments == state

POMDPs.initialstate(::ArrangementMDP) = Deterministic(Set{String}())

function POMDPs.transition(::ArrangementMDP, state, action)
    # readout features 
    return Deterministic(state ∪ action)
end

function POMDPs.reward(m::ArrangementMDP, state, action)
    monetary_cost = m.evals[state].filtration * sum(a -> m.experimental_costs[a][1], action)
    time = maximum(a -> m.experimental_costs[a][2], action)

    return -sum(m.tradeoff .* (monetary_cost, time))
end

POMDPs.discount(::ArrangementMDP) = 1.0

const default_mdp_kwargs = (; n_iterations = 10_000, depth = 10, exploration_constant = 3.0)

"""
    optimal_arrangement(costs, evals, experiments; max_parallel=1, tradeoff=(1, 0), pomdp_kwargs=default_mdp_kwargs)
Find the optimal arrangement of a set of `experiments` given their costs and associated filtration rates.

Otherwise, the function returns a named tuple `(; arrangement, combined cost, monetary cost, time, planner)`, where `planner` is the MDP planner.

# Arguments
- `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.
- `evals`: a dictionary containing pairs `experimental subset => (; filtration rate, loss)`.
- `experiments`: a set of experiments to be executed.

# Keyword arguments
- `parallel`: to estimate the execution time of the design, define the number of experiments that can run concurrently.
 The experiments will subsequently be arranged in descending order based on their individual durations,
 and they will be then iteratively allocated into consecutive groups that represent parallel experiments.
- `tradeoff`: determines how to project the monetary cost and execution time of an experimental design onto a single combined cost.
- `mdp_kwargs`: arguments to a MDP solver [`DPWSolver`](https://juliapomdp.github.io/MCTS.jl/dev/dpw/#Double-Progressive-Widening).
"""
# if the execution times for the experimental setup are specified
function optimal_arrangement(
    costs::Dict{String,T},
    evals::Dict{Set{String},ExperimentalEval},
    experiments::Set{String};
    max_parallel = 1,
    tradeoff = (1, 0),
    mdp_kwargs = default_mdp_kwargs,
) where {T}
    experimental_costs = Dict{String,NTuple{2,Float64}}(
        k => if v isa Number
            (convert(Float64, v), v)
        else
            (convert(Float64, v[1]), convert(Float64, v[2]))
        end for (k, v) in costs
    )

    mdp = ArrangementMDP(; experiments, experimental_costs, evals, max_parallel, tradeoff)

    solver = DPWSolver(; mdp_kwargs...)
    planner = solve(solver, mdp)

    monetary_cost = time = 0.0
    state = Set{String}()
    arrangement = Set{String}[]
    while state != experiments
        next_action = action(planner, state)

        push!(arrangement, next_action)
        monetary_cost +=
            evals[state].filtration * sum(a -> experimental_costs[a][1], next_action)
        time += maximum(a -> experimental_costs[a][2], next_action)

        state = state ∪ next_action
    end

    return (;
        arrangement,
        monetary_cost,
        time,
        combined_cost = sum(tradeoff .* (monetary_cost, time)),
        planner,
    )
end
