using Random: seed!
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
@test perf_eval isa Dict{Set{String},Float64}

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
      Dict{Set{String},NamedTuple{(:loss, :filtration),Tuple{Float64,Float64}}}

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
