module StaticDesigns

using DataFrames
using MLJ: evaluate, PerformanceEvaluation
using Combinatorics: powerset
using POMDPs
using POMDPTools: Deterministic
using MCTS

using ..CEED: front

export evaluate_experiments, efficient_designs

# performance evaluation: predictive loss, fraction of population not filtered out by the experiment 
const ExperimentalEval = NamedTuple{(:loss, :filtration),NTuple{2,Float64}}

# optimal arrangement as a MDP task
include("arrangements.jl")

# there are two methods of evaluate_experiments function. These two methods differ in their use cases and the type of input data they take:

# The first version of the evaluate_experiments function takes a predictive model as an input along with data (X and y) and evaluates the predictive accuracy over subsets of experiments using MLJ.evaluate function. This version is used when the target is a prediction task. The scores (predictive accuracy) for each subset of experiments are returned.

# The second version of the evaluate_experiments function takes dichotomous data (X) and evaluates the discriminative power of each subset of the experiments. This version is used for classifying binary labels indicating whether an entity was filtered out by the experiment or not. The result is a dictionary in which each key is a set of experiments and the value is the fraction of surviving population taken as metrics.

# The need for several methods arises from the flexibility provided by Julia's multiple dispatch system, which allows developers to define different behaviors of a function based on the type and number of its inputs, making the code more readable and optimizing the function's performance for different types of input. This feature is useful in dealing with different data types and tasks.

"""
    evaluate_experiments(experiments, model, X, y; zero_cost_features=[], evaluate_empty_subset=true, return_full_metrics=false, kwargs...)

Evaluate predictive accuracy over subsets of experiments, and return the metrics. The evaluation is facilitated by `MLJ.evaluate`; additional keyword arguments to this function will be passed to `evaluate`.

Evaluations are run in parallel.

# Arguments

  - `experiments`: a dictionary containing pairs `experiment => (cost =>) features`, where `features` is a subset of column names in `data`.
  - `model`: a predictive model whose accuracy will be evaluated.
  - `X`: a dataframe with features used for prediction.
  - `y`: the variable that we aim to predict.

# Keyword arguments

  - `max_cardinality`: maximum cardinality of experimental subsets (defaults to the number of experiments).
  - `zero_cost_features`: additional zero-cost features available for each experimental subset (defaults to an empty list).
  - `evaluate_empty_subset`: flag indicating whether to evaluate empty experimental subset. A constant column will be added if `zero_cost_features` is empty (defaults to true).
  - `return_full_metrics`: flag indicating whether to return full `MLJ.PerformanceEvaluation` metrics. Otherwise return an aggregate "measurement" for the first measure (defaults to false).

# Example

```julia
evaluate_experiments(
    experiments,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    zero_cost_features,
    measure = LogLoss(),
)
```
"""
function evaluate_experiments(
    experiments::Dict{String,T},
    model,
    X,
    y;
    max_cardinality = length(experiments),
    zero_cost_features = [],
    evaluate_empty_subset::Bool = true,
    return_full_metrics::Bool = false,
    kwargs...,
) where {T}
    # predictive accuracy scores over subsets of experiments
    scores = Dict{Set{String},return_full_metrics ? PerformanceEvaluation : Float64}()
    # generate all the possible subsets from the set of experiments, with a minimum size of 1 and maximum size of 'max_cardinality'
    experimental_subsets = collect(powerset(collect(keys(experiments)), 1, max_cardinality))
    # lock
    lk = ReentrantLock()

    Threads.@threads for exp_set in collect(experimental_subsets)
        features = eltype(names(X))[zero_cost_features...]
        foreach(
            x -> append!(
                features,
                experiments[x] isa Pair ? experiments[x][2] : experiments[x],
            ),
            exp_set,
        )
        perf_eval = evaluate(model, X[:, features], y; kwargs...)

        # acquire the lock to prevent race conditions
        lock(lk) do
            return push!(
                scores,
                Set(exp_set) =>
                    return_full_metrics ? perf_eval : first(perf_eval.measurement),
            )
        end
    end

    if evaluate_empty_subset
        X_ = if !isempty(zero_cost_features)
            X[!, zero_cost_features]
        else
            DataFrame(; dummy = fill(0.0, nrow(data)))
        end
        perf_eval = evaluate(model, X_, y; kwargs...)

        push!(
            scores,
            Set{String}() => return_full_metrics ? perf_eval : first(perf_eval.measurement),
        )
    end

    return scores
end

"""
    evaluate_experiments(experiments, X; zero_cost_features=[], evaluate_empty_subset=true)

Evaluate discriminative power for subsets of experiments, and return the metrics.

Evaluations are run in parallel.

# Arguments

  - `experiments`: a dictionary containing pairs `experiment => (cost =>) features`, where `features` is a subset of column names in `X`.
  - `X`: a dataframe containing binary labels, where `false` indicated that an entity was filtered out by the experiment (and should be removed from the triage).

# Keyword arguments

  - `zero_cost_features`: additional zero-cost features available for each experimental subset (defaults to an empty list).
  - `evaluate_empty_subset`: flag indicating whether to evaluate empty experimental subset.

# Example

```julia
evaluate_experiments(experiments, data_binary; zero_cost_features)
```
"""
function evaluate_experiments(
    experiments::Dict{String,T},
    X::DataFrame;
    zero_cost_features = [],
    evaluate_empty_subset::Bool = true,
) where {T}
    scores = Dict{Set{String},ExperimentalEval}()

    for exp_set in powerset(collect(keys(experiments)), 1)
        features = eltype(names(X))[zero_cost_features...]
        foreach(
            x -> append!(
                features,
                experiments[x] isa Pair ? experiments[x][2] : experiments[x],
            ),
            exp_set,
        )

        # calculate fraction of surviving population
        perf_eval = count(all, eachrow(X[!, features])) / nrow(X)

        push!(scores, Set(exp_set) => (; loss = perf_eval, filtration = perf_eval))
    end

    if evaluate_empty_subset
        if !isempty(zero_cost_features)
            perf_eval = count(all, eachrow(X[!, zero_cost_features])) / nrow(X)
        else
            perf_eval = 1.0
        end

        push!(scores, Set{String}() => (; loss = perf_eval, filtration = perf_eval))
    end

    return scores
end

"""
    efficient_designs(experiments, evals; max_parallel=1, tradeoff=(1, 0), mdp_kwargs=default_mdp_kwargs)

Return the set of Pareto-efficient experimental designs, given experimental costs, predictive accuracy (loss), and estimated filtration rates for experimental subsets.

# Arguments

  - `experiments`: a dictionary containing pairs `experiment => cost (=> features)`, where `cost` can either be scalar cost or a tuple `(monetary cost, execution time)`.
  - `evals`: a dictionary containing pairs `experimental subset => (; predictive loss, filtration)`.

# Keyword arguments

  - `parallel`: to estimate the execution time of the design, define the number of experiments that can run concurrently.
    The experiments will subsequently be arranged in descending order based on their individual durations,
    and they will be then iteratively allocated into consecutive groups that represent parallel experiments.
  - `tradeoff`: determines how to project the monetary cost and execution time of an experimental design onto a single combined cost.

# Example

```julia
efficient_designs(
    experiments_costs,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    eval_options = (; zero_cost_features, measure = LogLoss()),
    arrangement_options = (; max_parallel = 2, tradeoff = (0.0, 1)),
)
```
"""
function efficient_designs(
    experiments::Dict{String,T},
    evals::Dict{Set{String},S};
    max_parallel::Int = 1,
    tradeoff = (1, 0),
    mdp_kwargs = default_mdp_kwargs,
) where {T,S}
    experimental_costs = Dict(e => v isa Pair ? v[1] : v for (e, v) in experiments)

    evals = Dict{Set{String},ExperimentalEval}(
        if e isa Number
            s => (; loss = convert(Float64, e), filtration = 1.0)
        else
            s => (;
                loss = convert(Float64, e.loss),
                filtration = convert(Float64, e.filtration),
            )
        end for (s, e) in evals
    )

    # find the optimal arrangement for each experimental subset
    designs = []
    # lock to prevent race condition
    lk = ReentrantLock()

    Threads.@threads for design in collect(evals)
        arrangement = optimal_arrangement(
            experimental_costs,
            evals,
            design[1];
            max_parallel,
            tradeoff,
            mdp_kwargs,
        )

        lock(lk) do
            return push!(
                designs,
                (
                    (arrangement.combined_cost, design[2].loss),
                    (;
                        arrangement = arrangement.arrangement,
                        monetary_cost = arrangement.monetary_cost,
                        time = arrangement.time,
                    ),
                ),
            )
        end
    end

    return front(x -> x[1], designs)
end

"""
    efficient_designs(experiments, args...; eval_options, arrangement_options)

Evaluate predictive power for subsets of experiments, and return the set of Pareto-efficient experimental designs.

Internally, [`evaluate_experiments`](@ref) is called first, followed by [`efficient_designs`](@ref StaticDesigns.efficient_designs(experiments, ::Dict{Set{String}, Float64})).

# Keyword arguments

  - `eval_options`: keyword arguments to [`evaluate_experiments`](@ref).
  - `arrangement_options`: keyword arguments to [`efficient_designs`](@ref StaticDesigns.efficient_designs(experiments, ::Dict{Set{String}, Float64})).

# Example

```julia
efficient_designs(
    experiments_costs,
    data_binary;
    eval_options = (; zero_cost_features),
    arrangement_options = (; max_parallel = 2, tradeoff = (0.0, 1)),
)
```
"""
function efficient_designs(
    experiments::Dict{String,T},
    args...;
    eval_options = (;),
    arrangement_options = (;),
) where {T}
    evals = evaluate_experiments(experiments, args...; eval_options...)

    return efficient_designs(experiments, evals; arrangement_options...)
end

end
