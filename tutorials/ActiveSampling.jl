# # Active Sampling for Generative Designs

# ## Background on Active Sampling

# In multi-objective optimization (MOO), particularly in the context of active learning and reinforcement learning (RL),
# conditional sampling plays a critical role in achieving optimized outcomes that align with specific desirable criteria.
# The essence of conditional sampling is to direct the generative process not only towards the global objectives but also to adhere to additional, domain-specific constraints or features. This approach is crucial for several reasons:

# MOO often involves balancing several competing objectives, such as minimizing cost while maximizing performance.
# Conditional sampling allows for the integration of additional constraints or preferences,
# ensuring that the optimization does not overly favor one objective at the expense of others.

# On a related note, many optimization problems have domain-specific constraints or desirable features that are not explicitly part of the primary objectives.
# For example, in drug design, beyond just optimizing for efficacy and safety, one might need to consider factors like solubility or synthesizability.
# Conditional sampling ensures that solutions not only meet the primary objectives but also align with these additional practical considerations.

# ## Application to Generative Designs

# In the context of CEEDesigns.jl, active sampling can be used to selectively prioritize historical observations.
# For example, if the goal is to understand a current trend or pattern,
# active sampling can be used to assign more weight to recent data points or deprioritize those that may not be relevant to the current context.

# This way, the sampled data will offer a more precise representation of the current state or trend.

# For details on the theoretical background of generative designs and notation, please see our [introductory tutorial](SimpleGenerative.md) and an [applied tutorial](GenerativeDesigns.jl).

# Here we will again assume that the generative process is based on sampling from a historical dataset, which gives measurements on $m$ features $X = \{x_1, \ldots, x_m\}$ for $l$ entities
# (with entities and features representing rows and columns, respectively). 
# Each experiment $e$ may yield measurements on some
# subset of features $(X_{e}\subseteq X)$.

# Given the current state, i.e., experimental evidence acquired thus far for the new entity, we assign probabilistic weights $w_{j}$ over historical entities which are similar to the new entity. These weights can be used to
# weight the values of $y$ or features associated with $e_{S^{\prime}}$ to construct approximations of $q(y|e_{S})$ and 
# $q(e_{S^{\prime}}|e_{S})$.

# In the context of active sampling, we aim to further adjust the weights, $w_{j}$, calculated by the algorithm. 

# This adjustment can be accomplished by introducing a "prior", which is essentially a vector of weights that will multiply the computed weights, $w_{j}$, in an element-wise manner.
# If we denote the "prior" weights as $p_{j}$, then the final weights assigned to the $j$-th row are computed as $w'_{j} = p_{j} * w_{j}$.

# Alternatively, we can use "feature-wise" priors, which are considered when a readout for the specific feature is available for the new entity. # It is important to note here that the distance, which forms the basis of the probabilistic weights, is inherently computed only over the observed features.

# To be more precise, for each experiment $e\in E$, we let $p^e_j$ denote the "prior" associated with that specific experiment.
# If $S$ represents the set of experiments that have been performed for the new compound so far, we compute the reweighted probabilistic weight as $w'_{j} = \product_{e\in S} p^e_j \cdot w_j$.

# We remark that this method can be used to filter out certain rows by setting their weight to zero.

# Considering feature-wise priors can offer a more detailed and nuanced understanding of the data. These priors can be used to dynamically adjust the weight of historical observations, based on the specific readouts considered for the observation across different features. 

# For more information about active sampling, refer to the following articles.

# - [Evolutionary Multiobjective Optimization via Efficient Sampling-Based Strategies](https://link.springer.com/article/10.1007/s40747-023-00990-z).
# - [Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization](https://arxiv.org/abs/2301.07784).
# - [Conditional gradient method for multiobjective optimization](https://link.springer.com/article/10.1007/s10589-020-00260-5)
# - [A practical guide to multi-objective reinforcement learning and planning](https://link.springer.com/article/10.1007/s10458-022-09552-y)

# ## Synthetic Data Example

using CEEDesigns, CEEDesigns.GenerativeDesigns

# We create a synthetic dataset with continuous variables, `x1`, `x2`, and `y`. Both `x1` and `x2` are modeled as independent random variables that follow a normal distribution.
# The target variable, `y`, is given as a weighted sum of `x1` and `x2`, with an additional noise component. The corrected version of your sentence should be: Consequently, if the value of `x2`, for example,
# falls into a "sparse" region, we want the algorithm to avoid overfitting and focus its attention more on the other variable.

using Random
using DataFrames

## Define the size of the dataset.
n = 1000;
## Generate `x1` and `x2` as independent random variables with normal distribution.
x1 = randn(n)
x2 = randn(n);
## Compute `y` as the sum of `x1`, `x2`, and the noise.
y = x1 .+ 0.2 * x2 .+ 0.1 * randn(n);
## Create a data frame.
data = DataFrame(; x1 = x1, x2 = x2, y = y);
data[1:10, :]

# ### Active Sampling

# We again use the default method `DistanceBased` to assign the probabilistic weights across historical observations.

# In addition, we will demonstrate the use of two additional keyword arguments of `DistanceBased`, related to active sampling:
# - `importance_weights`: this is a dictionary of pairs `colname` with either `weights` or a function `x -> weight`, which will be applied to each element of the column to obtain the vector of weights. If data for a given column is available in the current state, the product of the corresponding weights is used to adjust the similarity vector.
#
# For filtering out certain rows where the readouts fall outside a selected range, we can use the following keyword:
# - `filter_range`: this is a dictionary of pairs `colname => (lower bound, upper bound)`. If there's data in the current state for a specific column specified in this list, only historical observations within the defined range for that column are considered.

# In the current setup, we aim to adaptively filter out the values `x1` or `x2` that lie outside of one standard deviation from the mean.

filter_range = Dict()

filter_range = Dict("x1" => (-1, 1), "x2" => (-1, 1))

# We then call `DistanceBased` as follows:
(sampler_active, uncertainty_active, weights_active) = DistanceBased(
    data;
    target = "y",
    similarity = Exponential(; λ = 0.5),
    filter_range = filter_range,
);

# To compare behavior with and without active sampling, we call `DistanceBased` again:
(; sampler, uncertainty, weights) =
    DistanceBased(data; target = "y", similarity = Exponential(; λ = 0.5));

# We plot the weights $w_j$ assigned to historical observations for both cases - with active sampling and without. The actual observation is shown in orange.
evidence = Evidence("x1" => 5.0, "x2" => 0.5)
#
using Plots
## The plotting engine (GR) requires us to use RGB instead of RGBA.
rgba_to_rgb(a) = RGB(((1 - a) .* (1, 1, 1) .+ a .* (0.0, 0.5, 0.5))...)
alphas_active = max.(0.1, weights_active(evidence) ./ maximum(weights_active(evidence)))
p1 = scatter(
    data[!, "x1"],
    data[!, "x2"];
    color = map(a -> rgba_to_rgb(a), alphas_active),
    title = "weights\n(active sampling)",
    mscolor = nothing,
    colorbar = false,
    label = false,
)
scatter!(
    p1,
    [evidence["x1"]],
    [evidence["x2"]];
    color = :orange,
    mscolor = nothing,
    label = nothing,
)
p1
#
alphas = max.(0.1, weights(evidence) ./ maximum(weights(evidence)))
p2 = scatter(
    data[!, "x1"],
    data[!, "x2"];
    color = map(a -> rgba_to_rgb(a), alphas),
    title = "weights\n(no active sampling)",
    mscolor = nothing,
    colorbar = false,
    label = false,
)
scatter!(
    p2,
    [evidence["x1"]],
    [evidence["x2"]];
    color = :orange,
    mscolor = nothing,
    label = nothing,
)
p2
#
# As it turns out, when active sampling was not used, the algorithm tended to overfit to the closest yet sparse points, which did not represent the true distribution accurately.
# We can also compare the estimated uncertainty, which is computed as the variance of the posterior.
#
# Without using active sampling, we obtain:
round(uncertainty_active(evidence); digits = 1)
#
# While for active sampling, we get:
round(uncertainty(evidence); digits = 1)

# #### Experimental Designs for Uncertainty Reduction

# We compare the set of cost-efficient designs in cases where active sampling is used and where it is not.
#
# We specify the experiments along with the associated features:
experiments = Dict("x1" => 1.0, "x2" => 1.0, "y" => 6.0)

# We specify the initial state.
evidence = Evidence("x2" => 5.0)

# Next we compute the set of efficient designs.
designs = efficient_designs(
    experiments;
    sampler,
    uncertainty,
    thresholds = 5,
    evidence,
    mdp_options = (; max_parallel = 1),
);

designs_active = efficient_designs(
    experiments;
    sampler = sampler_active,
    uncertainty = uncertainty_active,
    thresholds = 5,
    evidence,
    mdp_options = (; max_parallel = 1),
);

# We can compare the fronts.
p = scatter(
    map(x -> x[1][1], designs),
    map(x -> x[1][2], designs);
    ylabel = "% uncertainty",
    label = "efficient designs (no active sampling)",
    title = "efficient front",
    color = :blue,
    mscolor = nothing,
)
scatter!(
    p,
    map(x -> x[1][1], designs_active),
    map(x -> x[1][2], designs_active);
    label = "efficient designs (active sampling)",
    color = :teal,
    mscolor = nothing,
)
p
