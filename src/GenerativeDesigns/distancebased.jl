# function to compute the distance between a readout and column entries
"""
    QuadraticDistance(; λ=1, standardize=true)

This returns an anonymous function `(x, col; prior) -> λ * (x .- col).^2 / σ`.
If `standardize` is set to `true`, `σ` represents `col`'s variance calculated in relation to `prior`, otherwise `σ` equals one.
"""
function QuadraticDistance(; λ = 1, standardize = true)
    σ = nothing

    function (x, col; prior = ones(length(col)))
        if isnothing(σ)
            σ = standardize ? var(col, Weights(prior); corrected = false) : 1
        end

        return λ * (x .- col) .^ 2 / σ
    end
end

"""
    DiscreteDistance(; λ=1)

Return an anonymous function `(x, col) -> λ * (x .== col)`.
"""
DiscreteDistance(; λ = 1) = function (x, col; _...)
    return map(y -> y == x ? λ : 0.0, col)
end

# Default similarity functional
"""
    Exponential(; λ=1)

Return an anonymous function `x -> exp(-λ * sum(x; init=0))`.
"""
Exponential(; λ = 1 / 2) = x -> exp(-λ * x)

# default uncertainty functionals
function compute_variance(data::AbstractVector; weights)
    var(data, Weights(weights); corrected = false)
end

function compute_variance(data; weights)
    sum(var(Matrix(data), Weights(weights), 1; corrected = false))
end

"""
    Variance()

Return a function of `(data; prior)`. When this function is called as part of an instantiation procedure in [`DistanceBased`](@ref),
it returns an internal function of `weights` that computes the fraction of variance in the data, relative to the variance calculated with respect to a specified `prior`.
"""
Variance() = function (data; prior)
    initial = compute_variance(data; weights = prior)
    return weights -> (compute_variance(data; weights) / initial)
end

function compute_entropy(labels; weights)
    aggregate_weights = collect(values(countmap(labels, Weights(weights))))
    return entropy(aggregate_weights ./ sum(aggregate_weights))
end

"""
    Entropy()

Return a function of `(labels; prior)`.  When this function is called as part of an instantiation procedure in [`DistanceBased`](@ref),
it returns an internal function of `weights` that computes the fraction of information entropy, relative to the entropy calculated with respect to a specified `prior`.
"""
function Entropy()
    function (labels; prior)
        @assert elscitype(labels) <: Multiclass "labels must be of `Multiclass` scitype, but `elscitype(labels)=$(elscitype(labels))`"
        initial = compute_entropy(labels; weights = prior)
        return weights -> (compute_entropy(labels; weights) / initial)
    end
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
    SquaredMahalanobisDistance(; diagonal=0)

Returns a function that computes [squared Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) between each row of `data` and the evidence.
For a singular covariance matrix, consider adding entries to the matrix's diagonal via the `diagonal` keyword.

To accommodate missing values, we have implemented an approach described in https://www.jstor.org/stable/3559861, on page 285.

# Arguments

  - `diagonal`: A scalar to be added to the diagonal entries of the covariance matrix.

# Returns

It returns a high-level function of `(data, targets, prior)`.
When called, that function will return an internal function `compute_distances` that takes an `Evidence` and computes the squared Mahalanobis distance based on the input data and the evidence.
"""
function SquaredMahalanobisDistance(; diagonal = 0)
    function (data, targets, prior)
        non_targets = setdiff(names(data), targets)
        if !all(t -> t <: Real, eltype.(eachcol(data[!, non_targets])))
            @warn "Not all column types in the predictor matrix are numeric ($(eltype.(eachcol(data)))). This may cause errors."
        end

        Λ = Dict(
            Set(features) => begin
                Σ = cov(Matrix(data[!, features]), Weights(prior); corrected = false)
                # Add diagonal entries.
                foreach(i -> Σ[i, i] += diagonal, axes(Σ, 1))

                inv(Σ)
            end for features in powerset(non_targets, 1, length(non_targets))
        )

        compute_distances = function (evidence::Evidence)
            evidence_keys = collect(keys(evidence) ∩ non_targets)

            if length(evidence_keys) == 0
                return zeros(nrow(data))
            end

            vec_evidence = map(k -> evidence[k], evidence_keys)

            factor_p_q = length(non_targets) / length(evidence_keys)
            distances = map(eachrow(data)) do row
                # We retrieve the precomputed inverse of the covariance matrix coresponding to the "observed part"
                # See #12 and, in particular, https://www.jstor.org/stable/3559861
                Λ_marginal = Λ[Set(evidence_keys)]

                factor_p_q * dot(
                    (vec_evidence - Vector(row[evidence_keys])),
                    Λ_marginal * (vec_evidence - Vector(row[evidence_keys])),
                )
            end

            return distances
        end

        return compute_distances
    end
end

"""
    DistanceBased(data; target, uncertainty=Entropy(), similarity=Exponential(), distance=Dict(); prior=ones(nrow(data)))

Compute distances between experimental evidence and historical readouts, and apply a 'similarity' functional to obtain probability mass for each row.

Consider using [`QuadraticDistance`](@ref), [`DiscreteDistance`](@ref), and [`SquaredMahalanobisDistance`](@ref).

# Return value

A named tuple with the following fields:

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `weights`: a function of `evidence`; it returns probabilities (posterior) acrss the rows in `data`.

# Arguments

  - `data`: a dataframe with historical data.
  - `target`: target column name or a vector of target columns names.

# Keyword Argumets

  - `uncertainty`: a function that takes the subdataframe containing columns in targets along with prior, and returns an anonymous function taking a single argument (a probability vector over observations) and returns an uncertainty measure over targets.
  - `similarity`: a function that, for each row, takes distances between `row[col]` and `readout[col]`, and returns a non-negative probability mass for the row.
  - `distance`: a dictionary of pairs `colname => similarity functional`, where a similarity functional must implement the signature `(readout, col; prior)`. Defaults to [`QuadraticDistance`](@ref) and [`DiscreteDistance`](@ref) for `Continuous` and `Multiclass` scitypes, respectively.
  - `prior`: prior across rows, uniform by default.
  - `filter_range`: a dictionary of pairs `colname => (lower bound, upper bound)`. If there's data in the current state for a specific column specified in this list, only historical observations within the defined range for that column are considered.
  - `importance_weights`: a dictionary of pairs `colname` with either `weights` or a function `x -> weight`, which will be applied to each element of the column to obtain the vector of weights. If data for a given column is available in the current state, the product of the corresponding weights is used to adjust the similarity vector.

# Example

```julia
(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy(),
    similarity = Exponential(; λ = 5),
);
```
"""
function DistanceBased(
    data::DataFrame;
    target,
    uncertainty = Variance(),
    similarity = Exponential(),
    distance = Dict(),
    prior = ones(nrow(data)),
    filter_range = Dict(),
    importance_weights = Dict(),
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

    # Compute "column-wise" priors.
    weights = Dict{String,Vector{Float64}}()

    # If "importance weight" is a function, apply it to the column to get a numeric vector.
    for (colname, val) in importance_weights
        push!(
            weights,
            string(colname) =>
                (val isa Function ? map(x -> val(x), data[!, colname]) : val),
        )
    end

    # Convert "desirable ranges" into importance weights.
    for (colname, range) in filter_range
        colname = string(colname)
        within_range = (data[!, colname] .>= range[1]) .&& (data[!, colname] .<= range[2])

        weights[colname] = within_range .* get(weights, colname, ones(nrow(data)))
    end

    # Convert distances into probabilistic weights.
    compute_weights = function (evidence::Evidence)
        similarities = prior .* map(x -> similarity(x), compute_distances(evidence))

        # Perform hard match on target columns.
        for colname in collect(keys(evidence)) ∩ targets
            similarities .*= data[!, colname] .== evidence[colname]
        end

        # Adjustment based on "column-wise" priors.
        for colname in keys(evidence)
            if haskey(weights, colname)
                similarities .*= weights[colname]
            end
        end

        # If all similarities were zero, the `Weights` constructor would error.
        if sum(similarities) ≈ 0
            similarities .= 1
            for colname in collect(keys(evidence)) ∩ targets
                similarities .*= data[!, colname] .== evidence[colname]
            end
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
