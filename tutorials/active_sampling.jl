# # Remarks for active sampling:
# The desirable_range and target_constraints in the distancebased_active.jl code serve different purposes, although they both influence the sampling process:
# Desirable Range Constraints
# The desirable_range is used to apply a filter to the data based on specified ranges for certain columns. The code snippet you provided checks if the values in a column fall within a specified range and creates a boolean array (within_range) that is true for rows that meet the criteria and false for those that don't. This boolean array is then used to element-wise multiply the similarities array, effectively setting the similarity to zero for rows that don't fall within the range. This is a form of hard constraint that excludes certain data points from being sampled.
# Target Constraints
# The target_constraints, on the other hand, apply a softer form of constraint by adjusting the weights of the data points based on certain conditions. Instead of excluding data points that don't meet the criteria, target_constraints increase the weight of those that do, making them more likely to be sampled, but still allowing for the possibility of sampling those that don't meet the criteria.
# Comparison and Necessity
# The key difference is that desirable_range acts as a binary filter (include or exclude), while target_constraints adjusts the probability of inclusion without necessarily excluding any data points.

# Whether to increase the weight for elements within the range or simply filter them out depends on the specific requirements of the problem you're trying to solve:

# - Filtering (Hard Constraint): If you are certain that data points outside the desirable range cannot possibly be relevant to your analysis or model, you might choose to filter them out completely.

# - Weighting (Soft Constraint): If data points outside the desirable range could still be relevant, but you want to focus more on those within the range, then applying a weight is more appropriate. This allows for a more nuanced sampling that can account for uncertainty or the potential relevance of all data points.

# In summary, whether to use filtering or weighting depends on the context and the desired strictness of the constraints. If strict exclusion is not necessary and you want to retain all data points with adjusted importance, then weighting is the appropriate choice. If certain data points should not be considered at all, filtering is the way to go.




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

# ## Generative Model for Outcomes Sampling

using CEED, CEED.GenerativeDesigns

# As previously discussed, we provide a dataset of historical records, the target variable, along with an information-theoretic measure to quantify the uncertainty about the target variable.

# In what follows, we obtain three functions:
# - `sampler`: this is a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator;
# - `uncertainty`: this is a function of `evidence`,
# - `weights`: this represents a function of `evidence` that distributes probabilistic weights across the rows in the dataset.

# Note that internally, a state of the decision process is represented as a tuple `(evidence, costs)`.



# ## Active Sampling
# Desirable Range
# We might want to focus on data points that fall within one standard deviation from the mean for continuous variables. For example, for Age with a mean of approximately 53.5, we could define a range around this mean.
mean_age = 53.5
std_age = 9 # Hypothetical standard deviation
desirable_range = Dict("Age" => (mean_age - std_age, mean_age + std_age))

# Target Constraints
# For binary variables like HeartDisease, we might want to increase the weight of samples with the disease present to balance the dataset if it's imbalanced.
target_constraints = Dict("HeartDisease" => x -> any(x .== 1) ? 1.5 : 1.0)


(; sampler, uncertainty, weights) =
    DistanceBased_active(
        data,
        "HeartDisease",
        Entropy,
        Exponential(; λ = 5),
        desirable_range = desirable_range,
        importance_sampling = true,
        target_constraints = target_constraints,
    );
# The CEED package offers an additional flexibility by allowing an experiment to yield readouts over multiple features at the same time. In our scenario, we can consider the features `RestingECG`, `Oldpeak`, `ST_Slope`, and `MaxHR` to be obtained from a single experiment `ECG`.

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
    n_iterations = 100,#20_000
    exploration_constant = 5.0,
    tree_in_info = true,
)
designs = efficient_designs(
    experiments,
    sampler,
    uncertainty,
    6,
    evidence;
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

# We have to provide the maximum number of concurrent experiments. Additionally, we can specify the tradeoff between monetary cost and execution time - in our case, we aim to minimize the execution time.

## minimize time, two concurrent experiments at maximum
seed!(1)
## use less number of iterations to speed up build process
solver = GenerativeDesigns.DPWSolver(;
    n_iterations = 100,
    exploration_constant = 5.0,
    tree_in_info = true,
)
designs = efficient_designs(
    experiments,
    sampler,
    uncertainty,
    6,
    evidence;
    solver,
    mdp_options = (; max_parallel = 2, costs_tradeoff = (0, 1.0)),
    repetitions = 5,
);

# We plot the Pareto-efficient actions:

plot_front(designs; labels = make_labels(designs), ylabel = "% uncertainty")

# ## Efficient Value Experimental Designs

# In this experimental setup, we aim to maximize the value of experimental evidence, adjusted for the incurred experimental costs.

# For this purpose, we need to specify a function that quantifies the 'value' of decision-process making state, modeled as a tuple of experimental evidence and costs.

value = function (evidence, (monetary_cost, execution_time))
    return (1 - uncertainty(evidence)) - (0.005 * sum(monetary_cost))
end

# Considering a discount factor $\lambda$, the total reward associated with the experimental state in an $n$-step decision process is given by $r = r_1 + \sum_{i=2}^n \lambda^{i-1} (r_i - r_{i-1})$, where $r_i$ is the value associated with the $i$-th state.

# In the following example, we also limit the maximum rollout horizon to 4.
#
seed!(1)
## use less number of iterations to speed up build process
solver =
    GenerativeDesigns.DPWSolver(; n_iterations = 20_000, depth = 4, tree_in_info = true)
design = efficient_value(
    experiments,
    sampler,
    value,
    evidence;
    solver,
    repetitions = 5,
    mdp_options = (; discount = 0.8),
);
#
design[1] # optimized cost-adjusted value
#
d3tree = D3Tree(design[2].tree; init_expand = 2)
