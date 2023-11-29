
### ======================== comparison for active/passive learning ========================
##
using Plots
using CSV, DataFrames
cd(@__DIR__)
data = CSV.File("data/heart_disease.csv") |> DataFrame
data[1:10, :]

# We provide appropriate scientific types of the features.

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

# We specify the experiments along with the associated features:
experiments = Dict(
    ## experiment => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20.0 => ["FastingBS"],
    "HeartDisease" => 100.0,
)

##
# ## Generative Model for Outcomes Sampling
using CEED, CEED.GenerativeDesigns

function compare_efficient_designs(
    data,
    experiments,
    evidence;
    desirable_range = Dict(),
    target_constraints = Dict(),
    solver_options = Dict(),
)
    (; sampler, uncertainty, weights) =
        DistanceBased(data, "HeartDisease", Entropy, Exponential(; λ = 5))
    # Without desirable_range and target_constraints
    designs_without_constraints =
        efficient_designs(experiments, sampler, uncertainty, 6, evidence; solver_options...)

    # With desirable_range and target_constraints
    (; sampler, uncertainty, weights) = DistanceBased(
        data,
        "HeartDisease",
        Entropy,
        Exponential(; λ = 5);
        desirable_range = desirable_range,
        importance_sampling = true,
        target_constraints = target_constraints,
    )
    designs_with_constraints =
        efficient_designs(experiments, sampler, uncertainty, 6, evidence; solver_options...)

    # Plot results without constraints
    p1 = plot_front(
        designs_without_constraints;
        labels = make_labels(designs_without_constraints),
        ylabel = "% uncertainty",
    )
    title!(p1, "Without Constraints")

    # Plot results with constraints
    p2 = plot_front(
        designs_with_constraints;
        labels = make_labels(designs_with_constraints),
        ylabel = "% uncertainty",
    )
    title!(p2, "With Constraints")

    plt = plot(p1, p2; layout = (2, 1), size = (800, 600))
end
##



# Set seed for reproducibility
using Random: seed!
seed!(1)

# Define your evidence, sampler, and uncertainty function here
evidence = Evidence("Age" => 35, "Sex" => "M")
# ... initialize sampler and uncertainty ...

# Define solver options
solver_options = Dict(
    :solver => GenerativeDesigns.DPWSolver(;
        n_iterations = 100,
        exploration_constant = 5.0,
        tree_in_info = true,
    ),
    :mdp_options => (; max_parallel = 1),
    :repetitions => 5,
)

# Define desirable range and target constraints
mean_age = 53.5
std_age = 9
desirable_range = Dict("Age" => (mean_age - std_age, mean_age + std_age))
target_constraints = Dict("HeartDisease" => x -> any(x .== 1) ? 1.5 : 1.0)

# Call the function to compare and plot the results
active_passive_plt =compare_efficient_designs(
    data,
    experiments,
    evidence;
    desirable_range = desirable_range,
    target_constraints = target_constraints,
    solver_options = solver_options,
)
savefig(active_passive_plt, "active_passive_plt.png")