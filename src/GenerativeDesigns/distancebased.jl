# function to compute the distance between a readout and column entries
"""
    QuadraticDistance(; λ=1, standardize=true)

This returns an anonymous function `(x, col; prior) -> λ * (x .- col).^2 / σ`.
If `standardize` is set to `true`, `σ` represents `col`'s variance calculated in relation to `prior`, otherwise `σ` equals one.
"""
QuadraticDistance(; λ = 1, standardize = true) =
    function (x, col; prior = ones(length(col)))
        σ = standardize ? var(col, Weights(prior); corrected = false) : 1

        return λ * (x .- col) .^ 2 / σ
    end

"""
    DiscreteDistance(; λ=1)

Return an anonymous function `(x, col) -> λ * (x .== col)`.
"""
DiscreteDistance(; λ = 1) = function (x, col; _...)
    return map(y -> y == x ? λ : 0.0, col)
end

# default similarity functional
"""
    Exponential(; λ=1)

Return an anonymous function `x -> exp(-λ * sum(x; init=0))`.
"""
Exponential(; λ = 1 / 2) = x -> exp(-λ * x)

# default uncertainty functionals
compute_variance(data::AbstractVector; weights) = var(data, Weights(weights))

compute_variance(data; weights) = sum(var(Matrix(data), Weights(weights), 1))

"""
    Variance(data; prior)

Return a function of `weights` that computes the fraction of variance in the data, relative to the variance calculated with respect to a specified `prior`.
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

Return a function of `weights` that computes the fraction of information entropy, relative to the entropy calculated with respect to a specified `prior`.
"""
function Entropy(labels; prior)
    @assert elscitype(labels) <: Multiclass "labels must be of `Multiclass` scitype, but `elscitype(labels)=$(elscitype(labels))`"
    initial = compute_entropy(labels; weights = prior)
    return (weights -> compute_entropy(labels; weights) / initial)
end

# Return a function that calculates the sum of distances in each row, column-wise, and applies weights based on the prior.
function sum_of_distances(data::DataFrame, targets::Vector, distances; prior::Weights)
    function (evidence::Evidence)
        if isempty(evidence)
            return zeros(nrow(data))
        else
            array_distances = zeros((nrow(data), length(evidence)))
            for (i, colname) in enumerate(keys(evidence))
                if colname ∈ targets
                    continue
                else
                    array_distances[:, i] .=
                        distances[colname](evidence[colname], data[!, colname]; prior)
                end
            end

            distances_sum = vec(sum(array_distances; init = 0.0, dims = 2))

            return distances_sum
        end
    end
end

"""
    MahalanobisDistance(; diagonal=0)

Returns a function that computes [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) between each row of `data` and the evidence.
For a singular covariance matrix, consider adding entries to the matrix's diagonal via the `diagonal` keyword.

# Arguments

  - `diagonal`: A scalar or vector to be added to the diagonal entries of the covariance matrix.

# Returns

It returns a high-level function of `(data, targets, prior)`.
When called, that function will return an internal function `compute_distances` that takes an `Evidence` and computes the Mahalanobis distance based on the input data and the evidence.
"""
function MahalanobisDistance(; diagonal = 0)
    function (data, targets, prior)
        non_targets = setdiff(names(data), targets)
        Σ = cov(Matrix(data[!, non_targets]), Weights(prior))
        println(non_targets)
        # add diagonal entries
        diagonal = diagonal isa Number ? fill(diagonal, size(Σ, 1)) : diagonal
        foreach(i -> Σ[i, i] += diagonal[i], axes(Σ, 1))

        if rank(Σ) < length(non_targets)
            #error("singular covariance matrix Σ (rank $(rank(Σ))). Consider adding entries to diagonal via `diagonal` keyword.")
        end

        # get inverse of Σ
        Λ = inv(Σ)

        compute_distances = function (evidence::Evidence)
            if isempty(evidence)
                return zeros(nrow(data))
            else
                vec_evidence = map(colname -> get(evidence, colname, 0), non_targets)
                distances = map(eachrow(data)) do row
                    vec_row = map(
                        colname -> haskey(evidence, colname) ? row[colname] : 0,
                        non_targets,
                    )
                    (vec_evidence - vec_row)' * Λ * (vec_evidence - vec_row)
                end

                return distances
            end
        end

        return compute_distances
    end
end

"""
    DistanceBased(data, target, uncertainty, similarity=Exponential(), distance=Dict(); prior=ones(nrow(data)))

Compute distances between experimental evidence and historical readouts, and apply a 'similarity' functional to obtain probability mass for each row.

Consider using [`QuadraticDistance`](@ref), [`DiscreteDistance`](@ref), and [`MahalanobisDistance`](@ref).

# Return value

A named tuple with the following fields:

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `weights`: a function of `evidence`; it returns probabilities (posterior) acrss the rows in `data`.

# Arguments

  - `data`: a dataframe with historical data.
  - `target`: target column name or a vector of target columns names.
  - `uncertainty`: a function that takes the subdataframe containing columns in targets along with prior, and returns an anonymous function taking a single argument (a probability vector over observations) and returns an uncertainty measure over targets.
  - `similarity`: a function that, for each row, takes distances between `row[col]` and `readout[col]`, and returns a non-negative probability mass for the row.
  - `distance`: a dictionary of pairs `colname => similarity functional`, where a similarity functional must implement the signature `(readout, col; prior)`. Defaults to [`QuadraticDistance`](@ref) and [`DiscreteDistance`](@ref) for `Continuous` and `Multiclass` scitypes, respectively.

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
    distance = Dict();
    prior = ones(nrow(data)),
)
    prior = Weights(prior)
    targets = target isa AbstractVector ? target : [target]

    if distance isa Dict
        distances = Dict(
            try
                if haskey(distance, colname)
                    string(colname) => distance[colname]
                elseif elscitype(data[!, colname]) <: Continuous
                    string(colname) => QuadraticDistance()
                elseif elscitype(data[!, colname]) <: Multiclass
                    string(colname) => DiscreteDistance()
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

        compute_distances = sum_of_distances(data, targets, distances; prior)
    elseif applicable(distance, data, targets, prior)
        compute_distances = distance(data, targets, prior)
    else
        error("distance $distance does not accept `(data, targets, prior)`")
    end

    # convert distances into probabilistic weights
    compute_weights = function (evidence::Evidence)
        similarities = prior .* map(x -> similarity(x), compute_distances(evidence))

        # hard match on target columns
        for colname in collect(keys(evidence)) ∩ targets
            similarities .*= data[!, colname] .== evidence[colname]
        end

        return Weights(similarities ./ sum(similarities))
    end

    sampler = function (evidence::Evidence, columns, rng = default_rng())
        observed = data[sample(rng, compute_weights(evidence)), :]

        return Dict(c => observed[c] for c in columns)
    end

    f_uncertainty = uncertainty(data[!, target]; prior)
    compute_uncertainty = function (evidence::Evidence)
        return f_uncertainty(compute_weights(evidence))
    end

    return (; sampler, uncertainty = compute_uncertainty, weights = compute_weights)
end
