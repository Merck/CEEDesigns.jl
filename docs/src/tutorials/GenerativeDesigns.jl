# # Heart Disease Triage Meets Generative Modeling

# Consider again a situation where a group of patients is tested for a specific disease. It may be costly to conduct an experiment yielding the definitive answer; instead, we want to utilize various proxy experiments that provide partial information about the presence of the disease.

# Importantly, we aim to design personalized adaptive policies for each patient. At the beginning of the triage process, we use a patient's prior data, such as sex, age, or type of chest pain, to project a range of cost-efficient experimental designs. Internally, while constructing these designs, we incorporate multiple-step-ahead lookups to model probable experimental outcomes and consider the subsequent decisions for each outcome. Then after choosing a specific decision policy from this set and acquiring additional experimental readouts, we adjust the continuation based on this evidence.

# The personalized experimental designs are motivated by the fact that the value of information collected from an experiment often differs across subsets of the entities involved in the triage process.

# ![information value matrix](assets/information_value_matrix.png)

# In the context of static designs, where each individual from the 'Population' undergoes the same triage process, 'Experiment C' would contribute the maximum information value. On the other hand, if we have the capability to adapt the triage process specifically for 'Population 1' and 'Population 2', we would choose 'Experiment 1' and 'Experiment 2' respectively, thereby improving the value of information acquired.

# ## Theoretical Framework

# Let us consider a set of $n$ experiments $E = \{ e_1, \ldots, e_n\}$, and let $y$ denote the target variable that we want to predict.

# We conceptualize the triage as a Markov decision process, in which we iteratively choose to conduct a subset of experiments $S \subseteq E$ and then, based on the experimental evidence, update our belief about the distribution of outcomes for the experiments that have not yet been conducted.

# Within the framework,
# - _state_ is modeled as the set of experiments conducted so far along with the acquired experimental evidence and accumulated costs;
# - _actions_ are subsets of experiments that have not yet been conducted; the size of these subsets is restricted by the maximum number of parallel experiments.

# Importantly, the outcome of a set $S$ of experiments is modeled as a random variable $e_S$, conditioned on the current state, i.e., combined evidence. This means that if in a given state outcomes from experiments in $S \subseteq E$ are available, the outcome of experiments in $S' \subseteq E \setminus S$ is drawn from a posterior $r \sim q(e_{S'} | e_S)$.

# We do not claim to know the 'best' way to define the posterior $q$. Instead, our approach is generic and allows us to consider any generative function that takes the current state and the set of experiments to be conducted, and returns a sample drawn from the implicit, theoretical posterior of the selected experiments.

# The information value associated with the state, derived from experimental evidence, can be modeled through any statistical or information-theoretic measure such as the variance or uncertainty associated with the target variable posterior $q(y|e_S)$.

# For example, consider [EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE](https://arxiv.org/pdf/1809.11142.pdf), where the authors use a VAE approach to model the posterior distribution and utilize Kullback-Leibler divergence to model the information gain.

# #### Distance-Based Sampling from Historical Data

# We implemented an approach that iteratively distributes a belief over historical entities likely to be similar to the entity currently being tested. As a result, we sample from the outputs of these historical entities, weighted by the probabilistic weight assigned by our system of belief. In doing so, we produce a specific model for $q(e_{S^{'}} | e_S)$ and $q(y | e_S)$.

# More specifically, consider a dataset of historical outputs across $m$ features $X = \{x_1, \ldots, x_m\}$ for $l$ entities (with entities and features representing rows and columns, respectively). Let's denote the target variable we want to predict as $y$.

# We assume that each experiment $e \in E$ yields readouts over a subset $X_e \subseteq X$ of features.

# In what follows, we discuss the assignment of probabilistic weights $w_j$ to each entity (or row in the dataset).

# In the initial state, when there is no experimental evidence gathered yet, we assign the weights according to a predetermined prior.

# If this is not the case, we have to compute the weights using particular distance and similarity functionals.

# For each feature $x\in X$, we consider a function $\rho_x$, which measures the distance between two outputs. By default, we consider:
# - Rescaled Kronecker delta (i.e., $\rho(x, y)=0$ only when $x=y$, and $\rho(x, y)= \lambda$ under any other circumstances, with $\lambda > 0$) for discrete features (i.e., features whose types are modeled as `MultiClass` type in [ScientificTypes.jl](https://github.com/JuliaAI/ScientificTypes.jl));
# - Rescaled squared distance $\rho(x, y) = λ  \frac{(x - y)^2}{2\sigma_2}$, where $\sigma_2$ is the variance of the feature column, estimated with respect to the prior for continuous features.

# Therefore, given an experimental state with readouts over the feature set $F \subseteq X$, we can calculate the total distance from the entity recorded in the $j$-th row as $d_j = \sum_{x\in F} \rho_x (\hat x, x_j)$, where $\hat x$ and $x_j$ denote the readout for feature $x$ for the entity being tested and the entity recorded in $j$-th column, respectively. 

# Alternatively, we could use the [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance#Definition).

# Next, we convert distances $d_j$ into probabilistic weights $w_j$. By default, we use a rescaled exponential function, i.e., we put $w_j = \exp(-\lambda d_j)$ for some $\lambda>0$. Notably, $\lambda$'s value determines how belief is distributed across the historical entities. Larger values of $\lambda$ concentrate the belief tightly around the 'closest' historical entities, while smaller values distribute more belief to more distant entities.

# Importantly, the proper choice of the distance functionals and the 'similarity' functional discussed above is a question of hyper-optimization. 

# Assuming the weights $w_j$ have been assigned, we can sample an index $\hat j \in \{ 1, \ldots, l\}$ according to these weights. To draw a sample $\hat x$ from feature $x\in X$, we let $\hat x$ be equal to the value in the column associated with feature $x$ for the $\hat j$-th row (entity). We can also take a sample from the target variable in the same way.

# ### Objective Sense
# The reward and stopping condition of the triage process can be interpreted in various ways. 
# - The triage may continue until the uncertainty about the posterior distribution of the target variable falls below a certain level. Our aim is to minimize the anticipated combined monetary cost and execution time of the triage (considered as a 'negative' reward). If all experiments are conducted without reaching below the required uncertainty level, or if the maximum number of experiments is exceeded, we penalize this scenario with a 'minus infinite' reward.
# - We may aim to minimize the expected uncertainty while being constrained by the costs of the experiment.
# - Alternatively, we could maximize the value of experimental evidence, adjusted for the incurred experimental costs.

# ### Policy Search
# Standard MDP algorithms can be used to solve this problem (offline learning) or construct the policy (online learning) for the sequential decision-making.

# Our MDP's state space is finite-dimensional but generally continuous due to the allowance of continuous features, which complicates the problem and few algorithms specialize in this area.

# We used the Double Progressive Widening Algorithm for this task as detailed in [A Comparison of Monte Carlo Tree Search and Mathematical Optimization for Large Scale Dynamic Resource Allocation](https://arxiv.org/abs/1405.5498).

# In a nutshell, the Double Progressive Widening (DPW) algorithm is designed for online learning in complex environments, particularly those known as Continuous Finite-dimensional Markov Decision Processes where the state space is continuous. The key idea behind DPW is to progressively expand the search tree during the Monte Carlo Tree Search (MCTS) process. The algorithm does so by dynamically and selectively adding states and actions to the tree based on defined heuristics.

# In the context of online learning, this algorithm addresses the complexity and challenges of real-time decision-making in domains with a large or infinite number of potential actions. As information is gathered in actual runtime, the algorithm explores and exploits this information to make optimal or near-optimal decisions. In other words, DPW permits the learning process to adapt on-the-fly as more data is made available, making it an effective tool for the dynamic and uncertain nature of online environments.

# In particular, at the core of the Double Progressive Widening (DPW) algorithm are several key components, including expansion, search, and rollout. 

# The search component is where the algorithm sifts through the search tree to determine the most promising actions or states to explore next. By using exploration-exploitation strategies, it can effectively balance its efforts between investigating previously successful actions and venturing into unexplored territories.

# The expansion phase is where the algorithm grows the search tree by adding new nodes, representing new state-action pairs, to the tree. This is done based on a predefined rule that dictates when and how much the tree should be expanded. This allows the algorithm to methodically discover and consider new potential actions without becoming overwhelmed with choices.

# Lastly, the rollout stage, also known as a simulation stage, is where the algorithm plays out a series of actions to the end of a game or scenario using a specific policy (like a random policy). The results of these rollouts are then used to update the estimates of the values of state-action pairs, increasing the accuracy of future decisions.

# ![One iteration of the MCTS algorithm, taken from https://ieeexplore.ieee.org/document/6145622](assets/mcts.png)

# In the above figure, nodes represent states of the decision process, while edges correspond to actions connecting these states.

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

using CEED, CEED.GenerativeDesigns

# As previously discussed, we provide a dataset of historical records, the target variable, along with an information-theoretic measure to quantify the uncertainty about the target variable.

# In what follows, we obtain three functions:
# - `sampler`: this is a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator;
# - `uncertainty`: this is a function of `evidence`,
# - `weights`: this represents a function of `evidence` that distributes probabilistic weights across the rows in the dataset.

# Note that internally, a state of the decision process is represented as a tuple `(evidence, costs)`.

(; sampler, uncertainty, weights) =
    DistanceBased(data; target= "HeartDisease", uncertainty=Entropy, similarity=Exponential(; λ = 5));

# You can specify the method for computing the distance using the `distance` keyword. By default, the Kronecker delta and quadratic distance will be utilised for categorical and continuous features, respectively. 

# Alternatively, you can provide a dictionary of `feature => distance` pairs. The implemented distance functionals are `DiscreteMetric(; λ)` and `QuadraticDistance(; λ, standardize=true)`. In that case, the specified distance will be applied to the respective feature, after which the distances will be collated across the range of features.

# You can also use the Mahalanobis distance (`MahalanobisDistance(; diagonal)`).

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
    thresholds=6,
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

# We have to provide the maximum number of concurrent experiments. Additionally, we can specify the tradeoff between monetary cost and execution time - in our case, we aim to minimize the execution time.

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
    thresholds=6,
    evidence,
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
