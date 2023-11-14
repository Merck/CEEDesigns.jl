using Revise, CEED, CEED.GenerativeDesigns
using Combinatorics: powerset
using DataFrames
using ScientificTypes
using Distributions
using POMDPs
using POMDPTools
using MCTS
using Plots

# --------------------------------------------------------------------------------
# synthetic data gen; see https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-regression

make_friedman1 = function(n, noise=0)
    X = DataFrame(rand(Float64, (n,5)), :auto)
    ϵ = noise > 0 ? rand(Distributions.Normal(0,noise),size(X,1)) : 0
    X.y = @. 10 * sin(π * X[:,1] * X[:,2]) + 20 * (X[:,3] - 0.5)^2 + 10 * X[:,4] + 5 * X[:,5] + ϵ
    return X
end

make_friedman2 = function(n, noise=0)
    X = DataFrame(zeros(Float64, n, 4), :auto)
    X[:,1] .= rand(Distributions.Uniform(0,100), n)
    X[:,2] .= rand(Distributions.Uniform(40*π,560*π), n)
    X[:,3] .= rand(Distributions.Uniform(0,1), n)
    X[:,4] .= rand(Distributions.Uniform(1,11), n)
    ϵ = noise > 0 ? rand(Distributions.Normal(0,noise),size(X,1)) : 0
    X.y = @. (X[:,1]^2 + (X[:,2] * X[:,3]  - 1 / (X[:,2] * X[:,4]))^2)^0.5 + ϵ
    return X
end

make_friedman3 = function(n, noise=0)
    X = DataFrame(zeros(Float64, n, 4), :auto)
    X[:,1] .= rand(Distributions.Uniform(0,100), n)
    X[:,2] .= rand(Distributions.Uniform(40*π,560*π), n)
    X[:,3] .= rand(Distributions.Uniform(0,1), n)
    X[:,4] .= rand(Distributions.Uniform(1,11), n)
    ϵ = noise > 0 ? rand(Distributions.Normal(0,noise),size(X,1)) : 0
    X.y = @. atan((X[:,2] * X[:,3] - 1 / (X[:,2] * X[:,4])) / X[:,1]) + ϵ
    return X
end


# --------------------------------------------------------------------------------
# let's generate some synthetic "historical data"
n_hist = 1000

# data = DataFrame(
#     x1=rand(Distributions.Uniform(-10,10),n_hist),
#     x2=rand(Distributions.Exponential(), n_hist),
#     x3=rand(Distributions.Binomial(50,0.25),n_hist),
#     x4=rand(Distributions.Gamma(10),n_hist)
# )

# data.y = @. 50*sin(data.x1) + 2.5*data.x2^0.5 + 5*(data.x2*data.x3)^0.95 + (20*data.x4 / (1 + data.x4))


# data = DataFrame(
#     x1=rand(Distributions.Normal(),n_hist),
#     x2=rand(Distributions.Norma(), n_hist),
#     x3=rand(Distributions.Norma(),n_hist),
#     x4=rand(Distributions.Norma(),n_hist)
# )

# data.y = @. 5*data.x1 + 2*data.x2 + 3*data.x3 + 6*data.x4 + 1.5*(data.x1*data.x2) + 3.25*(data.x3*data.x4) + 0.5*(data.x1*data.x2*data.x4)

data = make_friedman3(n_hist)

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
# uncertainty and similarity based sampling

(; sampler, uncertainty, weights) = DistanceBased(data, "y", Variance, GenerativeDesigns.Exponential(; λ = 5));

# we can view the posterior distribution over the outcome conditioned on the current state
using StatsPlots, StatsBase

evidence = Evidence("x4" => mean(data.x4))
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

# we'll set up the costs such that experiments corresponding to more informative marginals are more expensive
experiments_costs = Dict(
    "e1" => (2,2) => ["x1"],
    "e2" => (3,3) => ["x2"],
    "e3" => (4,4) => ["x3"],
    "e4" => (1,1) => ["x4"]
)

# make a nice dataframe display of the exp costs and what they measure
experiments_costs_df = DataFrame(experiment=String[], time=Int[], cost=Int[], measurement=String[]);
push!(experiments_costs_df,[[e, experiments_costs[e][1][1], experiments_costs[e][1][2], only(experiments_costs[e][2])] for e in keys(experiments_costs)]...);
experiments_costs_df


# --------------------------------------------------------------------------------
# find efficient actions

costs=experiments_costs
# sampler
# uncertainty
n_thresholds=6
# evidence=Evidence()
# evidence=Evidence("x4"=>mean(data.x4))
solver = DPWSolver(; n_iterations = 500, tree_in_info = true)
repetitions = 5
realized_uncertainty = false
mdp_options = (; max_parallel=length(experiments_costs), discount=1.0, costs_tradeoff=[0.5,0.5])

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