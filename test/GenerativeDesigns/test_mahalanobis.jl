using CSV, DataFrames
data = CSV.File("GenerativeDesigns/data/heart_disease.csv") |> DataFrame

using ScientificTypes

types = Dict(
    :MaxHR => Continuous,
    :Cholesterol => Continuous,
    :Oldpeak => Continuous,
    :Age => Continuous,
    :RestingBP => Continuous,
    :Sex => Multiclass,
    :FastingBS => Continuous,
)
data = coerce(data, types);

continuous_cols = filter(colname -> eltype(data[!, colname]) == Float64, names(data))
data = data[!, continuous_cols ∪ ["HeartDisease"]]

using CEEDesigns, CEEDesigns.GenerativeDesigns

evidence = Evidence()

# test `DistanceBased` sampler
r = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Variance(),
    similarity = Exponential(; λ = 5),
    distance = SquaredMahalanobisDistance(; diagonal = 1),
);
@test all(x -> hasproperty(r, x), [:sampler, :uncertainty, :weights])
(; sampler, uncertainty, weights) = r

# test signatures
using Random: default_rng
@test applicable(sampler, evidence, ["HeartDisease"], default_rng())

@test applicable(uncertainty, evidence)
@test applicable(weights, evidence)

experiments = Dict(
    ## experiment => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["Oldpeak", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20.0 => ["FastingBS"],
    "HeartDisease" => 100.0,
)

# test `UncertaintyReductionMDP`

solver = GenerativeDesigns.DPWSolver(; n_iterations = 100, tree_in_info = true)

design = efficient_design(
    experiments;
    sampler,
    uncertainty,
    threshold = 0.0,
    evidence,
    solver,
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
);

@test design isa Tuple

designs = efficient_designs(
    experiments;
    sampler,
    uncertainty,
    thresholds = 4,
    evidence,
    solver,
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
);

@test designs isa Vector
@test all(design -> (design[1][1] ≈ 0) || hasproperty(design[2], :stats), designs)

designs = efficient_designs(
    experiments;
    sampler,
    uncertainty,
    thresholds = 4,
    evidence,
    solver,
    mdp_options = (; max_parallel = 1),
);

@test !hasproperty(designs[1][2], :stats)

designs = efficient_designs(
    experiments;
    sampler,
    uncertainty,
    thresholds = 4,
    evidence,
    solver,
    realized_uncertainty = true,
    mdp_options = (; max_parallel = 1),
);

@test designs[begin][1][2] ≈ uncertainty(evidence)

# test `EfficientValueMDP``

value = function (evidence, (monetary_cost, execution_time))
    return (1 - uncertainty(evidence)) - (0.005 * sum(monetary_cost))
end

## use less number of iterations to speed up build process
solver = GenerativeDesigns.DPWSolver(; n_iterations = 100, depth = 2, tree_in_info = true)

design = efficient_value(experiments; sampler, value, evidence, solver, repetitions = 5);
@test design isa Tuple
@test hasproperty(design[2], :stats)

design = efficient_value(experiments; sampler, value, evidence, solver);
@test design isa Tuple
@test !hasproperty(design[2], :stats)

# Regression: the distance must be invariant to evidence-insertion order.
# Prior to canonicalizing `evidence_keys` against `non_targets`, two
# evidences with the same content but different insertion orders produced
# different Mahalanobis distances when Σ had non-uniform off-diagonals.
let
    e1 = Evidence("RestingBP" => 130.0, "Cholesterol" => 250.0, "MaxHR" => 150.0)
    e2 = Evidence("MaxHR" => 150.0, "Cholesterol" => 250.0, "RestingBP" => 130.0)
    @test weights(e1) ≈ weights(e2)
    @test uncertainty(e1) ≈ uncertainty(e2)
end

# --- Severe #12: custom `distance` Dict closure must propagate its own errors ---

struct CustomDistanceError <: Exception end

@testset "Custom distance closure errors propagate" begin
    # User-supplied closure that throws a distinctive error. The previous implementation
    # wrapped this branch in a generic `try/catch` that masked the error as "unsupported
    # scitype". The error must now surface with its original message.
    bad_distance = (x, col; prior = ones(length(col))) -> throw(CustomDistanceError())

    # The custom-distance branch must NOT be wrapped in a try/catch that re-maps the
    # error to the generic "unsupported scitype" message. The original exception type
    # has to surface unchanged.
    r_bad = DistanceBased(
        data;
        target = "HeartDisease",
        uncertainty = Variance(),
        similarity = Exponential(; λ = 5),
        distance = Dict("MaxHR" => bad_distance),
    )
    @test_throws CustomDistanceError r_bad.weights(Evidence("MaxHR" => 150.0))
end

# --- Severe #15: collinear column => singular covariance ---

@testset "SquaredMahalanobisDistance: collinear column handling" begin
    # Synthetic collinear column: exact duplicate of MaxHR.
    data_collinear = copy(data)
    data_collinear[!, "MaxHR_dup"] = data_collinear[!, "MaxHR"]

    # diagonal == 0 (default) => singular subset Σ; the failure must be explicit.
    (; weights) = DistanceBased(
        data_collinear;
        target = "HeartDisease",
        uncertainty = Variance(),
        similarity = Exponential(; λ = 5),
        distance = SquaredMahalanobisDistance(; diagonal = 0),
    )
    # Evidence covering the collinear pair triggers the singular submatrix.
    bad_evidence = Evidence("MaxHR" => 150.0, "MaxHR_dup" => 150.0)
    @test_throws ArgumentError weights(bad_evidence)

    # diagonal > 0 => regularization kicks in, distances/weights are finite.
    (; weights) = DistanceBased(
        data_collinear;
        target = "HeartDisease",
        uncertainty = Variance(),
        similarity = Exponential(; λ = 5),
        distance = SquaredMahalanobisDistance(; diagonal = 1.0),
    )
    ok_weights = weights(bad_evidence)
    @test all(isfinite, ok_weights)
    @test sum(ok_weights) ≈ 1.0
end

# --- Severe #15: lazy cache returns the same numbers as a fresh instance ---

@testset "SquaredMahalanobisDistance: lazy-cache equivalence" begin
    # Build one shared instance whose internal cache will be populated by repeated calls,
    # and a second "control" instance for each evidence so we can compare against a freshly
    # computed reference (independent cache).
    shared = DistanceBased(
        data;
        target = "HeartDisease",
        uncertainty = Variance(),
        similarity = Exponential(; λ = 5),
        distance = SquaredMahalanobisDistance(; diagonal = 1.0),
    )

    fresh_weights(ev) = DistanceBased(
        data;
        target = "HeartDisease",
        uncertainty = Variance(),
        similarity = Exponential(; λ = 5),
        distance = SquaredMahalanobisDistance(; diagonal = 1.0),
    ).weights(ev)

    ev_a = Evidence("MaxHR" => 150.0)
    ev_b = Evidence("MaxHR" => 150.0, "Cholesterol" => 200.0)

    # First pass: populate the cache.
    w_a1 = shared.weights(ev_a)
    w_b1 = shared.weights(ev_b)
    # Second pass on `ev_a`: must hit the cache and return numerically identical weights.
    w_a2 = shared.weights(ev_a)
    @test w_a1 == w_a2

    # The cached weights must match a freshly built instance (no shared cache state).
    @test w_a1 ≈ fresh_weights(ev_a)
    @test w_b1 ≈ fresh_weights(ev_b)
end
