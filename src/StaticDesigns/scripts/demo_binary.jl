
using CEED, CEED.StaticDesigns
using CSV, DataFrames

## synthetic heart disease dataset with binary labels
data = CSV.File("data/heart_binary.csv") |> DataFrame

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

perf_eval = evaluate_experiments(experiments, data; zero_cost_features)

# explicit, use accuracies computed above
designs = efficient_designs(experiments, perf_eval)

# switch to plotly backend
CEED.plotly()

plot_front(designs; labels = make_labels(designs), ylabel = "discriminative pwr")

# test with (monetary cost, execution time)
# experiment => cost => features
experiments = Dict(
    # experiment => cost => features
    "BloodPressure" => (1.0, 1.0) => ["RestingBP"],
    "ECG" => (1.0, 5.0) => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => (1.0, 20.0) => ["Cholesterol"],
    "BloodSugar" => (1.0, 20.0) => ["FastingBS"],
)

designs = efficient_designs(experiments, data; eval_options = (; zero_cost_features))

plot_front(designs; labels = make_labels(designs), ylabel = "discriminative pwr")
