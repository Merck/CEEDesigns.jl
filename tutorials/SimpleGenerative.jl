using CEEDesigns, CEEDesigns.GenerativeDesigns
using DataFrames
using Combinatorics: powerset
using ScientificTypes
import Distributions
using POMDPs
using POMDPTools
using MCTS
using Plots
using StatsPlots, StatsBase

# --------------------------------------------------------------------------------
# In this file we want to develop 2 simple examples of using the generative designs
# solver. 
#   * One for continuous outcomes using the Variance based `uncertainty` function.
#   * One for discrete outcomes using the Entropy based `uncertainty` function. 


# --------------------------------------------------------------------------------
# 
# First example is for the continuous outcome.
# 
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# We want to generate synthetic data (see https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-regression)
# We'll sample the predictors from a 4D Gaussian copula, so that there is specified covariance
# between the predictors. Otherwise the similarity metric will not be very meaningful, because if all
# predictors are independent similarity will essentially be uniformly distributed.

# these are the 3 sample functions for synthetic regression data from
# https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-regression

make_friedman1 = function(n, noise=0)
    X = DataFrame(rand(Float64, (n,5)), :auto)
    ϵ = noise > 0 ? rand(Distributions.Normal(0,noise),size(X,1)) : 0
    X.y = @. 10 * sin(π * X[:,1] * X[:,2]) + 20 * (X[:,3] - 0.5)^2 + 10 * X[:,4] + 5 * X[:,5] + ϵ
    return X
end

make_friedman2 = function(U, noise=0)
    size(U,2) == 4 || error("input U must have 4 columns, has $(size(U,2))")
    n = size(U,1)
    X = DataFrame(zeros(Float64, n, 4), :auto)
    for i in 1:4
        X[:,i] .= U[:,i]
    end
    ϵ = noise > 0 ? rand(Distributions.Normal(0,noise),size(X,1)) : 0
    X.y = @. (X[:,1]^2 + (X[:,2] * X[:,3]  - 1 / (X[:,2] * X[:,4]))^2)^0.5 + ϵ
    return X
end

make_friedman3 = function(U, noise=0)
    size(U,2) == 4 || error("input U must have 4 columns, has $(size(U,2))")
    n = size(U,1)
    X = DataFrame(zeros(Float64, n, 4), :auto)
    for i in 1:4
        X[:,i] .= U[:,i]
    end
    ϵ = noise > 0 ? rand(Distributions.Normal(0,noise),size(X,1)) : 0
    X.y = @. atan((X[:,2] * X[:,3] - 1 / (X[:,2] * X[:,4])) / X[:,1]) + ϵ
    return X
end

using Copulas

# sample the "predictors" from a Gaussian copula with specified covariance matrix
# the strange marginals are from https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-regression
ρ12, ρ13, ρ14, ρ23, ρ24, ρ34 = 0.8, 0.5, 0.3, 0.5, 0.25, 0.4
Σ = [
    1 ρ12 ρ13 ρ14;
    ρ12 1 ρ23 ρ24;
    ρ13 ρ23 1 ρ34;
    ρ14 ρ24 ρ34 1
]

X1 = Distributions.Uniform(0,100)
X2 = Distributions.Uniform(40*π,560*π)
X3 = Distributions.Uniform(0,1)
X4 = Distributions.Uniform(1,11)

# generate the copula distribution
C = GaussianCopula(Σ)
D = SklarDist(C, (X1,X2,X3,X4))

X = rand(D, 1000)

data = make_friedman3(transpose(X), 0.01)

# --------------------------------------------------------------------------------
# Generate synthetic "historical data"

types = Dict(
    :x1 => ScientificTypes.Continuous,
    :x2 => ScientificTypes.Continuous,
    :x3 => ScientificTypes.Continuous,
    :x4 => ScientificTypes.Continuous,
    :y => ScientificTypes.Continuous
)
data = coerce(data, types);

# we can see what the outcome looks like against each predictor
p1 = scatter(data.x1, data.y, legend=false, xlab="x1")
p2 = scatter(data.x2, data.y, legend=false, xlab="x2")
p3 = scatter(data.x3, data.y, legend=false, xlab="x3")
p4 = scatter(data.x4, data.y, legend=false, xlab="x4")
plot(p1, p2, p3, p4, layout=(2,2), legend=false)

# let's see histogram of the outcome
histogram(data.y, legend=false, ylab="Count", xlab="y")


# --------------------------------------------------------------------------------
# set up the distance based method for weights and uncertainty evaluation

(; sampler, uncertainty, weights) = DistanceBased(data, "y", Variance, GenerativeDesigns.Exponential(; λ = 5));

# we may also look at the uncertainty from each marginal distribution
data_uncertainties = [i => uncertainty(Evidence(i => mean(data[:,i]))) for i in names(data)[1:4]]
sort!(data_uncertainties, by=x->x[2], rev=true)

sticks(
    eachindex(data_uncertainties), 
    [i[2] for i in data_uncertainties], 
    xformatter = i->data_uncertainties[Int(i)][1],
    label=false,
    ylab="Uncertainty"
)

# we can view the posterior distribution over the outcome conditioned on the current state
evidence = Evidence("x1" => mean(data.x1))
plot_weights = StatsBase.weights(weights(evidence))

p1 = StatsPlots.density(
    data.y, weights=plot_weights, 
    legend=false, ylabel="Density", title="q(y|eₛ), uncertainty $(round(uncertainty(evidence), digits=1))"
)
p2 = Plots.histogram(
    data.y, weights=plot_weights, 
    legend=false, ylabel="Density", title="q(y|eₛ)"
)
plot(p1, p2, layout=(1,2), legend=false)


# # we'll set up the costs of each experiment
# experiments_costs = Dict(
#     "e1" => (1,1) => ["x1"],
#     "e2" => (1,1) => ["x2"],
#     "e3" => (1,1) => ["x3"],
#     "e4" => (1,1) => ["x4"]
# )

# # make a nice dataframe display of the exp costs and what they measure
# experiments_costs_df = DataFrame(experiment=String[], time=Int[], cost=Int[], measurement=String[]);
# push!(experiments_costs_df,[[e, experiments_costs[e][1][1], experiments_costs[e][1][2], only(experiments_costs[e][2])] for e in keys(experiments_costs)]...);
# experiments_costs_df

# we'll set up the experimental costs such that experiments which have less marginal uncertinaty
# are more costly
observables_experiments = Dict(["x$i" => "e$i" for i in 1:4])
experiments_costs = Dict([observables_experiments[e[1]] => (i,i) => [e[1]] for (i,e) in enumerate(sort(data_uncertainties, by=x->x[2], rev=true))])

# add a final very expensive experiment directly on the target variable
experiments_costs["ey"] = (100,100) => ["y"]

experiments_costs_df = DataFrame(experiment=String[], time=Int[], cost=Int[], measurement=String[]);
push!(experiments_costs_df,[[e, experiments_costs[e][1][1], experiments_costs[e][1][2], only(experiments_costs[e][2])] for e in keys(experiments_costs)]...);
experiments_costs_df


# --------------------------------------------------------------------------------
# find efficient actions

# options
n_thresholds=7
evidence=Evidence()
solver = DPWSolver(; n_iterations = 500, tree_in_info = true)
repetitions = 5
mdp_options = (; max_parallel=length(experiments_costs), discount=1.0, costs_tradeoff=(0.5,0.5))

designs = efficient_designs(
    experiments_costs,
    sampler,
    uncertainty,
    n_thresholds,
    evidence;
    solver,
    mdp_options = mdp_options,
    repetitions = repetitions,
);

plot_front(designs; labels = make_labels(designs), ylabel = "% uncertainty")


# --------------------------------------------------------------------------------
# 
# Second example is for the discrete outcome.
# 
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# data for classification problem was generated by this function
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification

# using PyCall
# sklearn = PyCall.pyimport_conda("sklearn", "scikit-learn")
# py"""
# import sklearn as sk
# from sklearn.datasets import make_classification
# """

# X, y = py"make_classification(n_samples=1000,n_features=5,n_informative=4,n_redundant=1,n_repeated=0,n_classes=2,n_clusters_per_class=2,flip_y=0.1)"

# using DataFrames, CSV
# dat = DataFrame(X, :auto)
# dat.y = y
# CSV.write("./class.csv",dat)

using CSV

data = CSV.read("./data/class.csv", DataFrame)

types = Dict(
    :x1 => ScientificTypes.Continuous,
    :x2 => ScientificTypes.Continuous,
    :x3 => ScientificTypes.Continuous,
    :x4 => ScientificTypes.Continuous,
    :y => ScientificTypes.Multiclass
)
data = coerce(data, types);


# --------------------------------------------------------------------------------
# uncertainty and similarity based sampling

(; sampler, uncertainty, weights) = DistanceBased(data, "y", Entropy, GenerativeDesigns.Exponential(; λ = 5));

# we may also look at the uncertainty from each marginal distribution
# this is a bit nonsensical as the data generating function will create multimodal clusters
# so it will look artifically as if nothing is informative, but that is not the case.
data_uncertainties = [i => uncertainty(Evidence(i => mode(data[:,i]))) for i in names(data)[1:end-1]]
sort!(data_uncertainties, by=x->x[2], rev=true)

sticks(
    eachindex(data_uncertainties), 
    [i[2] for i in data_uncertainties], 
    xformatter = i->data_uncertainties[Int(i)][1],
    label=false,
    ylab="Uncertainty"
)

# we can view the posterior distribution over the outcome conditioned on the current state
evidence = Evidence("x1" => mean(data.x1))
plot_weights = StatsBase.weights(weights(evidence))

target_belief = countmap(data[!, "y"], plot_weights)
p = bar(
    0:1,
    [target_belief[0], target_belief[1]];
    xrot = 40,
    ylabel = "probability",
    title = "unc: $(round(uncertainty(evidence), digits=1))",
    kind = :bar,
    legend = false,
);
xticks!(p, 0:1, ["0", "1"]);
p

# # experimental costs
# experiments_costs = Dict(
#     "e1" => (1,1) => ["x1"],
#     "e2" => (1,1) => ["x2"],
#     "e3" => (1,1) => ["x3"],
#     "e4" => (1,1) => ["x4"]
# )

# # make a nice dataframe display of the exp costs and what they measure
# experiments_costs_df = DataFrame(experiment=String[], time=Int[], cost=Int[], measurement=String[]);
# push!(experiments_costs_df,[[e, experiments_costs[e][1][1], experiments_costs[e][1][2], only(experiments_costs[e][2])] for e in keys(experiments_costs)]...);
# experiments_costs_df

# we'll set up the experimental costs such that experiments which have less marginal uncertinaty
# are more costly
observables_experiments = Dict(["x$i" => "e$i" for i in 1:5])
experiments_costs = Dict([observables_experiments[e[1]] => (i,i) => [e[1]] for (i,e) in enumerate(sort(data_uncertainties, by=x->x[2], rev=true))])

# add a final very expensive experiment directly on the target variable
experiments_costs["ey"] = (100,100) => ["y"]

experiments_costs_df = DataFrame(experiment=String[], time=Int[], cost=Int[], measurement=String[]);
push!(experiments_costs_df,[[e, experiments_costs[e][1][1], experiments_costs[e][1][2], only(experiments_costs[e][2])] for e in keys(experiments_costs)]...);
experiments_costs_df


# --------------------------------------------------------------------------------
# find efficient actions

n_thresholds=7
evidence=Evidence()
solver = DPWSolver(; n_iterations = 500, tree_in_info = true)
repetitions = 5
mdp_options = (; max_parallel=length(experiments_costs), discount=1.0, costs_tradeoff=(0.5,0.5))

designs = efficient_designs(
    experiments_costs,
    sampler,
    uncertainty,
    n_thresholds,
    evidence;
    solver,
    mdp_options = mdp_options,
    repetitions = repetitions,
);

plot_front(designs; labels = make_labels(designs), ylabel = "% uncertainty")