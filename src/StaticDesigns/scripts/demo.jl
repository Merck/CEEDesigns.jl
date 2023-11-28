using CEEDesigns, CEEDesigns.StaticDesigns
using CSV, DataFrames

## heart failure prediction dataset
# https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
data = CSV.File("data/heart.csv") |> DataFrame

# experiment => cost => features
experiments = Dict(
    # experiment => cost => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    #"ECG" => 5. => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    #"BloodCholesterol" => 20. => ["Cholesterol"],
    "BloodSugar" => 20 => ["FastingBS"],
)

# features that are always available
zero_cost_features = ["Age", "Sex", "ChestPainType", "ExerciseAngina"]

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

# models(matching(data, data[:, "HeartDisease"]))
classifier = @load RandomForestClassifier pkg = BetaML verbosity = 3
model = classifier(; n_trees = 20, max_depth = 10)

perf_eval = evaluate_experiments(
    experiments,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    zero_cost_features,
    measure = LogLoss(),
)

# explicit, use accuracies computed above
designs1 = efficient_designs(experiments, perf_eval)
# implicit, calculate accuracies automatically
designs2 = efficient_designs(
    experiments,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    eval_options = (; zero_cost_features, measure = LogLoss()),
)

# switch to plotly backend
CEEDesigns.plotly()

designs = designs2
plot_front(designs; labels = make_labels(designs), ylabel = "logloss")

# test with (monetary cost, execution time)
# experiment => cost => features
experiments = Dict(
    # experiment => cost => features
    "BloodPressure" => (1.0, 1.0) => ["RestingBP"],
    "ECG" => (5.0, 5.0) => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => (20.0, 20.0) => ["Cholesterol"],
    "BloodSugar" => (20.0, 20.0) => ["FastingBS"],
)

perf_eval = evaluate_experiments(
    experiments,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    zero_cost_features,
    measure = LogLoss(),
)

# implicit, calculate accuracies automatically
designs = efficient_designs(experiments, perf_eval; max_parallel = 2, tradeoff = (0.0, 1))

plot_front(designs; labels = make_labels(designs), ylabel = "logloss")
