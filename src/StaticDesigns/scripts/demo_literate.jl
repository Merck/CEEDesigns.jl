# # Cost-Efficient Experimental Designs: Heart Disease Prediction

# Consider a situation where a group of patients has to be tested for a specific disease. It may be costly to conduct an experiment yielding the definitive answer; instead, we want to utilize various proxy experiments that yield partial information about the presence of the disease.

# ## Theoretical Framework

# In terms of machine learning, let us assume that a dataset of historical readouts over $n$ features $x_1, \ldots, x_n$ is available, and let $y$ denote the target variable that we want to predict.

# For each subset $S \subseteq {x_1, ..., x_n}$ of features, we evaluate the accuracy $a_S$ of a predictive model that predicts the value of $y$ based on readouts over features in $S$. Assuming the patient population follows the same distribution as the historical observations, the predictive accuracy serves as a proxy for the information gained from observing the features in $S$.

# In the cost-sensitive setting of CEEDesigns, observing the features $S$ incurs a cost $c_S$. Generally, this cost is specified in terms of monetary cost and execution time of an experiment. Considering the constraint of a maximum number of parallel experiments, the algorithm recommends an arrangement of experiments that minimizes the total running time. Eventually, for a fixed tradeoff between monetary cost and execution time, a combined cost $c_S$ is obtained.

# Assuming we know the accuracies $a_S$ and experimental costs $c_S$ for each subset $S \subseteq {x_1, ..., x_n}$, we can generate a set of Pareto-efficient experimental designs considering both predictive accuracy and cost.

# ## Heart Disease Dataset

# In this tutorial, we consider a dataset that includes 11 clinical features along with a binary variable indicating the presence of heart disease in patients. The dataset can be found at [Kaggle: Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It utilizes heart disease datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

using CSV, DataFrames
data = CSV.File("data/heart_disease.csv") |> DataFrame

# ## Predictive Accuracy

# The CEEDesigns package offers an additional flexibility by allowing an experiment to yield readouts over multiple features at the same time. In our scenario, we can consider the features `RestingECG`, `Oldpeak`, `ST_Slope`, and `MaxHR` to be obtained from a single experiment `ECG`.

# We specify the experiments along with the associated features:

experiments = Dict(
    ## experiment => features
    "BloodPressure" => ["RestingBP"],
    "ECG" => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => ["Cholesterol"],
    "BloodSugar" => ["FastingBS"],
)

# We may also provide additional zero-cost features, which are always available.
zero_cost_features = ["Age", "Sex", "ChestPainType", "ExerciseAngina"]

# And we specify the target for prediction.
target = "HeartDisease"

# ### Classifier

# We use [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) to evaluate the predictive accuracy over subsets of experimental features.

using MLJ
import BetaML, MLJModels

# We have to provide appropriate scientific types of the features.

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

# Next, we choose a particular predictive model that will evaluated in the sequel. We can list all models that are compatible with our dataset:

models(matching(data, data[:, "HeartDisease"]))

# Eventually, we fix `RandomForestClassifier` from [BetaML](https://github.com/sylvaticus/BetaML.jl)
classifier = @load RandomForestClassifier pkg = BetaML verbosity = 3
model = classifier(; n_trees = 20, max_depth = 10)

# ### Performance Evaluation

# We use `evaluate_experiments` from `CEEDesigns.StaticDesigns` to evaluate the predictive accuracy over subsets of experiments. We use `LogLoss` as the measure of accuracy. It is possible to pass additional keyword arguments that will propagate `MLJ.evaluate` (such as `measure`).

using CEEDesigns, CEEDesigns.StaticDesigns
perf_eval = evaluate_experiments(
    experiments,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    zero_cost_features,
    measure = LogLoss(),
)

# We plot performance measures evaluated for subsets of experiments.
plot_evals(perf_eval; ylabel = "logloss")

# ## Cost-Efficient Designs

# We specify the cost associated with the execution of each experiment. When it comes to groups of experiments, the costs simply sum up.

costs = Dict(
    ## experiment => cost
    "BloodPressure" => 1,
    "ECG" => 5,
    "BloodCholesterol" => 20,
    "BloodSugar" => 20,
)

# We use the provided function `efficient_designs` to construct the set of cost-efficient experimental designs.
designs = efficient_designs(costs, perf_eval)

## Switch to plotly backend for plotting
CEEDesigns.plotly()

plot_front(designs; labels = make_labels(designs), ylabel = "logloss")

# ### Duration of Experiment

# We may additionally specify the duration of an experiment. Furthermore, we  demonstrating the flexibility of `efficient_designs` as it can both evaluate the performance of experiments and generates efficient designs.

# test with (monetary cost, execution time)
# experiment => cost => features
experiments_costs = Dict(
    # experiment => (monetary cost, execution time) => features
    "BloodPressure" => (1.0, 1.0) => ["RestingBP"],
    "ECG" => (5.0, 5.0) => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => (20.0, 20.0) => ["Cholesterol"],
    "BloodSugar" => (20.0, 20.0) => ["FastingBS"],
)

# We have to provide the maximum number of concurrent experiments. Additionally, we can specify the tradeoff between monetary cost and execution time - in our case, we aim to minimize the execution time.

# Internally, `evaluate_experiments`is called first, followed by `efficient_designs`. Keyword arguments to the respective functions has to wrapped in `eval_options` and `arrangement_options` named tuples, respectively.

## Implicit, calculates accuracies automatically
designs = efficient_designs(
    experiments_costs,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    eval_options = (; zero_cost_features, measure = LogLoss()),
    arrangement_options = (; max_parallel = 2, tradeoff = (0.0, 1)),
)

# As we can see, the algorithm correctly suggests running experiments in parallel.
plot_front(designs; labels = make_labels(designs), ylabel = "logloss")
