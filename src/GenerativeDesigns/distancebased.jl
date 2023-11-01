# default readout-column distance functionals
"""
    QuadraticStandardizedDistance(; λ=1)

Return an anonymous function `(x, col; prior) -> λ * (x .- col).^2 / (2*σ2)`, where `σ2` is the variance of `col` calculated with respect to `prior`.
"""
QuadraticStandardizedDistance(; λ = 1) = function (x, col; prior = ones(length(col)))
    σ2 = var(col, Weights(prior); corrected=false)

    λ * (x .- col) .^ 2 / (2 * σ2)
end

"""
    DiscreteMetric(; λ=1)

Return an anonymous function `(x, col) -> λ * (x .== col)`.
"""
DiscreteMetric(; λ = 1) = function (x, col; _...)
    map(y -> y == x ? λ : 0.0, col)
end

# default similarity functional
"""
    Exponential(; λ=1)

Return an anonymous function `x -> exp(-λ * sum(x; init=0))`.
"""
Exponential(; λ = 1) = x -> exp(-λ * sum(x; init = 0))

# default uncertainty functionals
compute_variance(data::AbstractVector; weights) = var(data, Weights(weights))

compute_variance(data; weights) = var(Matrix(data), Weights(repeat(weights, size(data, 2))))

"""
    Variance(data; prior)

Return a function of `weights` that computes the percentage of variance in the data, compared to the variance calculated with respect to a specified `prior`.
"""
function Variance(data; prior)
    initial = compute_variance(data; weights = prior)
    return weights -> (compute_variance(data; weights) / initial)
end

function compute_entropy(labels; weights)
    aggregate_weights = collect(values(countmap(labels, Weights(weights))))
    return entropy(aggregate_weights ./ sum(aggregate_weights))
end

"""
    Entropy(labels; prior)

Return a function of `weights` that computes the percentage of information entropy, compared to the entropy calculated with respect to a specified `prior`.
"""
function Entropy(labels; prior)
    @assert elscitype(labels) <: Multiclass "labels must be of `Multiclass` scitype, but `elscitype(labels)=$(elscitype(labels))`"
    initial = compute_entropy(labels; weights = prior)
    return (weights -> compute_entropy(labels; weights) / initial)
end

"""
    DistanceBased(data, target, uncertainty, similarity=Exponential(), distances=Dict(); prior=ones(nrow(data)))

Compute distances between experimental evidence and historical readouts, and apply a 'similarity' functional to obtain probability mass for each row.

# Return Value

A named tuple with the following fields:

  - `sampler`: a function of `(state, features, rng)`, in which `state` denotes the current experimental state, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `state`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `weights`: a function of `state`; it returns probabilities (posterior) acrss the rows in `data`.

# Arguments

  - `data`: a dataframe with historical data.
  - `target`: target column name or a vector of target columns names.
  - `uncertainty`: a function that takes the subdataframe containing columns in targets along with prior, and returns an anonymous function taking a single argument (a probability vector over observations) and returns an uncertainty measure over targets.
  - `similarity`: a function that, for each row, takes distances between `row[col]` and `readout[col]`, and returns a non-negative probability mass for the row.
  - `distances`: a dictionary of pairs `colname => similarity functional`, where a similarity functional must implement the signature `(readout, col; prior)`. Defaults to [`QuadraticStandardizedDistance`](@ref) and [`DiscreteMetric`](@ref) for `Continuous` and `Multiclass` scitypes, respectively.

# Keyword Argumets

  - `prior`: prior across rows, uniform by default.

# Example

```julia
(; sampler, uncertainty, weights) =
    DistanceBased(data, "HeartDisease", Entropy, Exponential(; λ = 5));
```
"""
function DistanceBased(
    data::DataFrame,
    target,
    uncertainty,
    similarity = Exponential(),
    distances = Dict();
    prior = ones(nrow(data)),
)
    distances = Dict(
        try
            if haskey(distances, colname)
                string(colname) => distances[colname]
            elseif elscitype(data[!, colname]) <: Continuous
                string(colname) => QuadraticStandardizedDistance()
            elseif elscitype(data[!, colname]) <: Multiclass
                string(colname) => DiscreteMetric()
            else
                error()
            end
        catch
            error(
                """column $colname has scitype $(elscitype(data[!, colname])), which is not supported by default.
            Please provide a custom readout-column distances functional of the signature `(x, col; prior)`.""",
            )
        end for colname in names(data[!, Not(target)])
    )

    prior = Weights(prior)
    targets = target isa AbstractVector ? target : [target]
    compute_weights = function (state::State)
        if isempty(state)
            return prior
        else
            array_distances = zeros((nrow(data), length(state)))
            for (i, colname) in enumerate(keys(state))
                if colname ∈ targets
                    continue
                else
                    array_distances[:, i] .=
                        distances[colname](state[colname], data[!, colname]; prior)
                end
            end

            similarities =
                prior .*
                map(i -> similarity(array_distances[i, :]), 1:size(array_distances, 1))

            # hard match on target columns
            for colname in collect(keys(state)) ∩ targets
                similarities .*= data[!, colname] .== state[colname]
            end

            return Weights(similarities ./ sum(similarities))
        end
    end

    sampler = function (state::State, columns, rng = default_rng())
        observed = data[sample(rng, compute_weights(state)), :]

        Dict(c => observed[c] for c in columns)
    end

    f_uncertainty = uncertainty(data[!, target]; prior)
    compute_uncertainty = function (state::State)
        f_uncertainty(compute_weights(state))
    end

    (; sampler, uncertainty = compute_uncertainty, weights = compute_weights)
end
