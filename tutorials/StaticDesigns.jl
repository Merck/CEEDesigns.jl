# # Heart Disease Triage

# Consider a situation where a group of patients is tested for a specific disease. It may be costly to conduct an experiment yielding the definitive answer. Instead, we want to utilize various proxy experiments that provide partial information about the presence of the disease.

# ## Theoretical Framework

# Let us consider a set of $n$ experiments $E = \{ e_1, \ldots, e_n\}$. For each subset $S \subseteq E$ of experiments, we denote by $v_S$ the value of information acquired from conducting experiments in $S$.

# In the cost-sensitive setting of CEEDesigns, conducting an experiment $e$ incurs a cost $(m_e, t_e)$. Generally, this cost is specified in terms of monetary cost and execution time of the experiment.

# To compute the cost associated with carrying out a set of experiments $S$, we first need to introduce the notion of an arrangement $o$ of the experiments $S$. An arrangement is a partition of $S$. In other words, $o = (o_1, \ldots, o_l)$ for $l \leq |S|$, where $\bigcup_{i=1}^l o_i = S$ and $o_i \cap o_j = \emptyset$ for each $1\leq i < j \leq l$.

# Given a subset $S$ of experiments and their arrangement $o$, the total monetary cost and total execution time of the experimental design is given as $m_o = \sum_{e\in S} m_e$ and $t_o = \sum_{i=1}^l \max \{ t_e : e\in o_i\}$, respectively.

# For instance, consider the experiments $e_1,\, e_2,\, e_3$, and $e_4$ with associated costs $(1, 1)$, $(1, 3)$, $(1, 2)$, and $(1, 4)$. If we conduct experiments $e_1$ through $e_4$ in sequence, this would correspond to an arrangement $o = (\{ e_1 \}, \{ e_2 \}, \{ e_3 \}, \{ e_4 \})$ with a total cost of $m_o = 4$ and $t_o = 10$.

# However, if we decide to conduct $e_1$ in parallel with $e_3$, and $e_2$ with $e_4$, we would obtain an arrangement $o = (\{ e_1, e_3 \}, \{ e_2, e_4 \})$ with a total cost of $m_o = 4$, and $t_o = 3 + 4 = 7$.

# Given the constraint on the maximum number of parallel experiments, we devise an arrangement $o$ of experiments $S$ such that, for a fixed tradeoff between monetary cost and execution time, the expected combined cost $c_{s} = \lambda m_o + (1-\lambda) t_o$ is minimized (i.e., the execution time is minimized).

# Because $t_{o}$ is the sum of the maximum times of each partition of $S$, to minimize $t_{o}$, experiments with long execution times should be grouped together. Therefore, the optimal arrangement is one such that each partition's size equals the maximum number of parallel experiments, except possibly for the final partition, and that assignment into partitions is done in order of execution time.

# Continuing our example and assuming a maximum of two parallel experiments, the optimal arrangement is to conduct $e_1$ in parallel with $e_2$, and $e_3$ with $e_4$. This results in an arrangement $o = (\{ e_1, e_2 \}, \{ e_3, e_4 \})$ with a total cost of $m_o = 4$ and $t_o = 2 + 4 = 6$.

# Assuming the information values $v_S$ and optimized experimental costs $c_S$ for each subset $S \subseteq E$ of experiments, we then generate a set of cost-efficient experimental designs.

# ### Application to Predictive Modeling 

# Consider a dataset of historical readouts over $m$ features $X = \{x_1, \ldots, x_m\}$, and let $y$ denote the target variable that we want to predict.

# We assume that each experiment $e \in E$ yields readouts over a subset $X_e \subseteq X$ of features.

# Then, for each subset $S \subseteq E$ of experiments, we may model the value of information acquired by conducting the experiments in $S$ as the accuracy of a predictive model that predicts the value of $y$ based on readouts over features in $X_S = \bigcup_{e\in S} X_e$.

# ## Heart Disease Dataset

# In this tutorial, we consider a dataset that includes 11 clinical features along with a binary variable indicating the presence of heart disease in patients. The dataset can be found at [Kaggle: Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It utilizes heart disease datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

using CSV, DataFrames
data = CSV.File("data/heart_disease.csv") |> DataFrame
data[1:10, :]

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
using Random: seed!

# We provide appropriate scientific types of the features.

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
data = coerce(data, types);

# Next, we choose a particular predictive model that will evaluated in the sequel. We can list all models that are compatible with our dataset:

models(matching(data, data[:, "HeartDisease"]))

# Eventually, we fix `RandomForestClassifier` from [BetaML](https://github.com/sylvaticus/BetaML.jl)
classifier = @load RandomForestClassifier pkg = BetaML verbosity = 3
model = classifier(; n_trees = 20, max_depth = 10)

# ### Performance Evaluation

# We use `evaluate_experiments` from `CEEDesigns.StaticDesigns` to evaluate the predictive accuracy over subsets of experiments. We use `LogLoss` as a measure of accuracy. It is possible to pass additional keyword arguments, which will be passed to `MLJ.evaluate` (such as `measure`, shown below).

using CEEDesigns, CEEDesigns.StaticDesigns

#

seed!(1) # evaluation process generally is not deterministic
perf_eval = evaluate_experiments(
    experiments,
    model,
    data[!, Not("HeartDisease")],
    data[!, "HeartDisease"];
    zero_cost_features,
    measure = LogLoss(),
)

# We plot performance measures evaluated for subsets of experiments, sorted by performance measure.
plot_evals(
    perf_eval;
    f = x -> sort(collect(keys(x)); by = k -> x[k], rev = true),
    ylabel = "logloss",
)

# ## Cost-Efficient Designs

# We specify the cost associated with the execution of each experiment.

costs = Dict(
    ## experiment => cost
    "BloodPressure" => 1,
    "ECG" => 5,
    "BloodCholesterol" => 20,
    "BloodSugar" => 20,
)

# We use the provided function `efficient_designs` to construct the set of cost-efficient experimental designs.
designs = efficient_designs(costs, perf_eval)
#
plot_front(designs; labels = make_labels(designs), ylabel = "logloss")

# ### Parallel Experiments

# The previous example assumed that experiments had to be run sequentially. We can see how the optimal arrangement changes if we assume multiple experiments can be run in parallel. To that end, we first specify the monetary cost and execution time for each experiment, respectively.

experiments_costs = Dict(
    ## experiment => (monetary cost, execution time) => features
    "BloodPressure" => (1.0, 1.0) => ["RestingBP"],
    "ECG" => (5.0, 5.0) => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => (20.0, 20.0) => ["Cholesterol"],
    "BloodSugar" => (20.0, 20.0) => ["FastingBS"],
)

# We set the maximum number of concurrent experiments. Additionally, we can specify the tradeoff between monetary cost and execution time - in our case, we aim to minimize the execution time.

# Below, we demonstrate the flexibility of `efficient_designs` as it can both evaluate the performance of experiments and generate efficient designs. Internally, `evaluate_experiments` is called first, followed by `efficient_designs`. Keyword arguments to the respective functions has to wrapped in `eval_options` and `arrangement_options` named tuples, respectively.

## Implicit, calculates accuracies automatically.
seed!(1) # evaluation process generally is not deterministic
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
