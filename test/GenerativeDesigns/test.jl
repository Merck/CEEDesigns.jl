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

state = State("Age" => 35, "Sex" => "M")

# test `DistanceBased` sampler
r = DistanceBased(data, "HeartDisease", Entropy, Exponential(; λ = 5));
@test all(x -> hasproperty(r, x), [:sampler, :uncertainty, :weights])
(; sampler, uncertainty, weights) = r

# test signatures
using Random: default_rng
@test applicable(sampler, state, ["HeartDisease"], default_rng)

@test applicable(uncertainty, state)
@test applicable(weights, state)

experiments = Dict(
    ## experiment => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20.0 => ["FastingBS"],
    "HeartDisease" => 100.0,
)

solver = GenerativeDesigns.DPWSolver(; n_iterations = 10_000, tree_in_info = true)
designs = efficient_designs(
    experiments,
    sampler,
    uncertainty,
    4,
    state;
    solver,
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
);

@test designs isa Vector
