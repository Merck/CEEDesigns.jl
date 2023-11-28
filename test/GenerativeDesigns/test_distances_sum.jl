using CSV, DataFrames
data = CSV.File("GenerativeDesigns/data/heart_disease.csv") |> DataFrame

using ScientificTypes

types = Dict(
    :MaxHR => Continuous,
    :Cholesterol => Continuous,
    :ChestPainType => Multiclass,
    :Oldpeak => Continuous,
    :HeartDisease => Multiclass,
    :Age => Continuous,
    :ST_Slope => Multiclass,
    :RestingECG => Multiclass,
    :RestingBP => Continuous,
    :Sex => Multiclass,
    :FastingBS => Continuous,
    :ExerciseAngina => Multiclass,
)
data = coerce(data, types);

using CEED, CEED.GenerativeDesigns

evidence = Evidence("Age" => 35, "Sex" => "M")

# test `DistanceBased` sampler
r = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy,
    similarity = Exponential(; λ = 5),
);
@test all(x -> hasproperty(r, x), [:sampler, :uncertainty, :weights])
(; sampler, uncertainty, weights) = r

# test signatures
using Random: default_rng
@test applicable(sampler, evidence, ["HeartDisease"], default_rng)

@test applicable(uncertainty, evidence)
@test applicable(weights, evidence)

experiments = Dict(
    ## experiment => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
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
