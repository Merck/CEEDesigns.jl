# # Heart Disease Triage Meets Generative Modeling

# Consider again a situation where a group of patients is tested for a specific disease. It may be costly to conduct an experiment yielding the definitive answer; instead, we want to utilize various proxy experiments that provide partial information about the presence of the disease.
# For details on the theoretical background and notation, please see our tutorial on [generative experimental designs](SimpleGenerative.md). This tutorial
# is a concrete application of the tools described in that document.

# Importantly, we aim to design personalized adaptive policies for each patient. At the beginning of the triage process, we use a patient's prior data, such as sex, age, or type of chest pain, to project a range of cost-efficient experimental designs. Internally, while constructing these designs, we incorporate multiple-step-ahead lookups to model probable experimental outcomes and consider the subsequent decisions for each outcome. Then after choosing a specific decision policy from this set and acquiring additional experimental readouts, we adjust the continuation based on this evidence.

# ## Heart Disease Dataset

# In this tutorial, we consider a dataset that includes 11 clinical features along with a binary variable indicating the presence of heart disease in patients. The dataset can be found at [Kaggle: Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It utilizes heart disease datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

using CSV, DataFrames
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

# ## Generative Model for Outcomes Sampling

using CEEDesigns, CEEDesigns.GenerativeDesigns

# As previously discussed, we provide a dataset of historical records, the target variable, along with an information-theoretic measure to quantify the uncertainty about the target variable.

# In what follows, we obtain three functions:
# - `sampler`: this is a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator,
# - `uncertainty`: this is a function of `evidence`,
# - `weights`: this represents a function of `evidence` that distributes probabilistic weights across the rows in the dataset.

# Note that internally, a state of the decision process is represented as a tuple `(evidence, costs)`.

# You can specify the method for computing the distance using the `distance` keyword. By default, the Kronecker delta and quadratic distance will be utilised for categorical and continuous features, respectively. 

(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy(),
    similarity = Exponential(; λ = 5),
);

# Alternatively, you can provide a dictionary of `feature => distance` pairs. The implemented distance functionals are `DiscreteDistance(; λ)` and `QuadraticDistance(; λ, standardize=true)`. In that case, the specified distance will be applied to the respective feature, after which the distances will be collated across the range of features.

# The above call is therefore equivalent to:

numeric_feats = filter(c -> c <: Real, eltype.(eachcol(data)))
categorical_feats = setdiff(names(data), numeric_feats)

DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy(),
    similarity = Exponential(; λ = 5),
    distance = merge(
        Dict(c => DiscreteDistance() for c in categorical_feats),
        Dict(c => QuadraticDistance() for c in numeric_feats),
    ),
);

# You can also use the squared Mahalanobis distance (`SquaredMahalanobisDistance(; diagonal)`). As the squared Mahalanobis distance only works with numeric features, we have to select a few, along with the target variable. For example, we could write:

DistanceBased(
    data[!, ["RestingBP", "MaxHR", "Cholesterol", "FastingBS", "HeartDisease"]];
    target = "HeartDisease",
    uncertainty = Entropy(),
    similarity = Exponential(; λ = 5),
    distance = SquaredMahalanobisDistance(; diagonal = 1),
);

# The package offers an additional flexibility by allowing an experiment to yield readouts over multiple features at the same time. In our scenario, we can consider the features `RestingECG`, `Oldpeak`, `ST_Slope`, and `MaxHR` to be obtained from a single experiment `ECG`.

# We specify the experiments along with the associated features:

experiments = Dict(
    ## experiment => features
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20.0 => ["FastingBS"],
    "HeartDisease" => 100.0,
)

# Let us inspect the distribution of belief for the following experimental evidence:

evidence = Evidence("Age" => 55, "Sex" => "M")
#
using StatsBase: countmap
using Plots
#
target_belief = countmap(data[!, "HeartDisease"], weights(evidence))
p = bar(
    0:1,
    [target_belief[0], target_belief[1]];
    xrot = 40,
    ylabel = "probability",
    color = :teal,
    title = "unc: $(round(uncertainty(evidence), digits=1))",
    kind = :bar,
    legend = false,
);
xticks!(p, 0:1, ["no disease", "disease"]);
p

# Let us next add an outcome of blood pressure measurement:

evidence_with_bp = merge(evidence, Dict("RestingBP" => 190))

target_belief = countmap(data[!, "HeartDisease"], weights(evidence_with_bp))
p = bar(
    0:1,
    [target_belief[0], target_belief[1]];
    xrot = 40,
    ylabel = "probability",
    color = :teal,
    title = "unc: $(round(uncertainty(evidence_with_bp), digits=2))",
    kind = :bar,
    legend = false,
);
xticks!(p, 0:1, ["no disease", "disease"]);
p

# ## Cost-Efficient Experimental Designs for Uncertainty Reduction

# In this experimental setup, our objective is to minimize the expected experimental cost while ensuring the uncertainty remains below a specified threshold.

# We use the provided function `efficient_designs` to construct the set of cost-efficient experimental designs for various levels of uncertainty threshold. In the following example, we generate 6 thresholds spaces evenly between 0 and 1, inclusive.

# Internally, for each choice of the uncertainty threshold, an instance of a Markov decision problem in [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) is created, and the `POMDPs.solve` is called on the problem. Afterwards, a number of simulations of the decision-making problem is run, all starting with the experimental `state`.

## set seed for reproducibility
using Random: seed!
seed!(1)
#
evidence = Evidence("Age" => 35, "Sex" => "M")
#
## use less number of iterations to speed up build process
solver = GenerativeDesigns.DPWSolver(;
    n_iterations = 20_000,
    exploration_constant = 5.0,
    tree_in_info = true,
)
designs = efficient_designs(
    experiments;
    sampler,
    uncertainty,
    thresholds = 6,
    evidence,
    solver,
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
);

# We plot the Pareto-efficient actions:

plot_front(designs; labels = make_labels(designs), ylabel = "% uncertainty")

# We render the search tree for the second design, sorted in descending order based on the uncertainty threshold:

using D3Trees
d3tree = D3Tree(designs[2][2].tree; init_expand = 2)

# ### Parallel Experiments

# We may exploit parallelism in the experimental arrangement. To that end, we first specify the monetary cost and execution time for each experiment, respectively.

experiments = Dict(
    ## experiment => (monetary cost, execution time) => features
    "BloodPressure" => (1.0, 1.0) => ["RestingBP"],
    "ECG" => (5.0, 5.0) => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => (20.0, 20.0) => ["Cholesterol"],
    "BloodSugar" => (20.0, 20.0) => ["FastingBS"],
    "HeartDisease" => (100.0, 100.0),
)

# We have to provide the maximum number of concurrent experiments. Additionally, we can specify the tradeoff between monetary cost and execution time. In our case, we aim to minimize the execution time.

## minimize time, two concurrent experiments at maximum
seed!(1)
## use less number of iterations to speed up build process
solver = GenerativeDesigns.DPWSolver(;
    n_iterations = 2_000,
    exploration_constant = 5.0,
    tree_in_info = true,
)
designs = efficient_designs(
    experiments;
    sampler,
    uncertainty,
    thresholds = 6,
    evidence,
    solver,
    mdp_options = (; max_parallel = 2, costs_tradeoff = (0, 1.0)),
    repetitions = 5,
);

# We plot the Pareto-efficient actions:

plot_front(designs; labels = make_labels(designs), ylabel = "% uncertainty")

# ## Efficient Value Experimental Designs

# In this experimental setup, we aim to maximize the value of experimental evidence, adjusted for the incurred experimental costs.

# For this purpose, we need to specify a function that quantifies the "value" of decision-process making state, modeled as a tuple of experimental evidence and costs.

value = function (evidence, (monetary_cost, execution_time))
    return (1 - uncertainty(evidence)) - (0.005 * sum(monetary_cost))
end

# Considering a discount factor $\lambda$, the total reward associated with the experimental state in an $n$-step decision process is given by $r = r_1 + \sum_{i=2}^n \lambda^{i-1} (r_i - r_{i-1})$, where $r_i$ is the value associated with the $i$-th state.

# In the following example, we also limit the maximum rollout horizon to 4.
#
seed!(1)
## use less number of iterations to speed up build process
solver = GenerativeDesigns.DPWSolver(; n_iterations = 2_000, depth = 4, tree_in_info = true)
design = efficient_value(
    experiments;
    sampler,
    value,
    evidence,
    solver,
    repetitions = 5,
    mdp_options = (; discount = 0.8),
);
#
design[1] # optimized cost-adjusted value
#
d3tree = D3Tree(design[2].tree; init_expand = 2)
