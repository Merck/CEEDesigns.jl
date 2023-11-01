using CSV, DataFrames
using Distributions

features_discriminative_power = Dict(
    f => rand(0.7:0.05:0.95) for f in [
        "RestingBP",
        "RestingECG",
        "Oldpeak",
        "ST_Slope",
        "MaxHR",
        "Cholesterol",
        "FastingBS",
    ]
)

merge!(
    features_discriminative_power,
    Dict(
        f => rand(0.9:0.05:0.95) for f in ["Age", "Sex", "ChestPainType", "ExerciseAngina"]
    ),
)

data = DataFrame()
n_rows = 10_000

for (feat, p) in features_discriminative_power
    data[!, feat] = rand(Bernoulli(p), n_rows)
end

CSV.write("data/heart_binary.csv", data)
