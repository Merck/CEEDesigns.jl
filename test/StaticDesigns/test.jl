using Random: seed!
using Random: Xoshiro
using CEEDesigns, CEEDesigns.StaticDesigns
using CSV, DataFrames

## predictive model from `MLJ`
# https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
data = CSV.File("StaticDesigns/data/heart_disease.csv") |> DataFrame

# available predictions
zero_cost_features = ["Age", "Sex", "ChestPainType", "ExerciseAngina"]

# experiment => cost => features
experiments = Dict(
    # experiment => cost => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20 => ["FastingBS"],
)

# prediction target
target = "HeartDisease"

using MLJ
import BetaML, MLJModels

# fix scitypes
types = Dict(
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

data = coerce(data, types)

classifier = @load RandomForestClassifier pkg = BetaML verbosity = 0
model = classifier(; n_trees = 20, max_depth = 10)

perf_eval = evaluate_experiments(
    experiments,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    zero_cost_features,
    measure = LogLoss(),
)
@test perf_eval isa Dict{Set{String}, Float64}

## binary dataset, use filtration
data = CSV.File("StaticDesigns/data/heart_binary.csv") |> DataFrame

# available predictions
zero_cost_features = ["Age", "Sex", "ChestPainType", "ExerciseAngina"]

# experiment => cost => features
experiments = Dict(
    # experiment => cost => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20 => ["FastingBS"],
)

seed!(1)
perf_eval = evaluate_experiments(experiments, data; zero_cost_features)
@test perf_eval isa
    Dict{Set{String}, NamedTuple{(:loss, :filtration), Tuple{Float64, Float64}}}

seed!(1)
designs = efficient_designs(experiments, perf_eval)
@test designs isa Vector

# test with (monetary cost, execution time)
# experiment => cost => features
experiments = Dict(
    # experiment => cost => features
    "BloodPressure" => (1.0, 1.0) => ["RestingBP"],
    "ECG" => (1.0, 5.0) => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => (1.0, 20.0) => ["Cholesterol"],
    "BloodSugar" => (1.0, 20.0) => ["FastingBS"],
)

## Regression test for #11: zero_cost_features as Vector{Symbol} should work.
# Reuse the binary dataset / experiments defined above.
data = CSV.File("StaticDesigns/data/heart_binary.csv") |> DataFrame
experiments_simple = Dict(
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20 => ["FastingBS"],
)

# (1) Symbol vector input — regression test for the MethodError documented in finding #11.
perf_eval_sym = evaluate_experiments(
    experiments_simple,
    data;
    zero_cost_features = [:Age, :Sex, :ChestPainType, :ExerciseAngina],
)
@test perf_eval_sym isa
    Dict{Set{String}, NamedTuple{(:loss, :filtration), Tuple{Float64, Float64}}}

# (2) String vector input — no regression.
perf_eval_str = evaluate_experiments(
    experiments_simple,
    data;
    zero_cost_features = ["Age", "Sex", "ChestPainType", "ExerciseAngina"],
)
@test perf_eval_str isa
    Dict{Set{String}, NamedTuple{(:loss, :filtration), Tuple{Float64, Float64}}}
# Both inputs should yield the same evaluation keys/values.
@test keys(perf_eval_sym) == keys(perf_eval_str)
@test all(perf_eval_sym[k] == perf_eval_str[k] for k in keys(perf_eval_sym))

## optimal_arrangement golden path with full evals.
seed!(1)
costs_only = Dict{String, Float64}(k => v[1] for (k, v) in experiments_simple)
arr = CEEDesigns.StaticDesigns.optimal_arrangement(
    costs_only,
    perf_eval_str,
    Set(keys(experiments_simple));
    mdp_kwargs = (; n_iterations = 200, depth = 5, exploration_constant = 3.0),
)
@test arr.arrangement isa Vector{Set{String}}
@test !isempty(arr.arrangement)
@test reduce(union, arr.arrangement) == Set(keys(experiments_simple))
@test arr.monetary_cost > 0

## M3 regression: optimal_arrangement with `max_cardinality`-truncated evals
## must raise a clear ArgumentError (not KeyError) when called over the full set.
# Simulate the truncation by dropping evals for subsets of size > 2 from a full eval dict.
truncated_evals = Dict(
    k => v for (k, v) in perf_eval_str if length(k) <= 2
)
@test_throws ArgumentError CEEDesigns.StaticDesigns.optimal_arrangement(
    costs_only,
    truncated_evals,
    Set(keys(experiments_simple));
    mdp_kwargs = (; n_iterations = 50, depth = 5, exploration_constant = 3.0),
)

## S3 regression: seeded rng => reproducible arrangement (no GLOBAL_RNG contention).
let
    repro_costs = Dict("A" => 1.0, "B" => 2.0, "C" => 3.0)
    repro_evals =
        Dict{Set{String}, NamedTuple{(:loss, :filtration), Tuple{Float64, Float64}}}(
        Set{String}() => (; loss = 1.0, filtration = 1.0),
        Set(["A"]) => (; loss = 0.7, filtration = 0.9),
        Set(["B"]) => (; loss = 0.6, filtration = 0.8),
        Set(["C"]) => (; loss = 0.5, filtration = 0.7),
        Set(["A", "B"]) => (; loss = 0.4, filtration = 0.6),
        Set(["A", "C"]) => (; loss = 0.4, filtration = 0.6),
        Set(["B", "C"]) => (; loss = 0.4, filtration = 0.6),
    )
    full = Set(["A", "B", "C"])
    mdp_kwargs = (; n_iterations = 500, depth = 5, exploration_constant = 3.0)
    a1 = CEEDesigns.StaticDesigns.optimal_arrangement(
        repro_costs, repro_evals, full; mdp_kwargs, rng = Xoshiro(123),
    )
    a2 = CEEDesigns.StaticDesigns.optimal_arrangement(
        repro_costs, repro_evals, full; mdp_kwargs, rng = Xoshiro(123),
    )
    @test a1.arrangement == a2.arrangement
    @test a1.combined_cost == a2.combined_cost
end
