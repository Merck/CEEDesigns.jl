using Revise, CEED, CEED.GenerativeDesigns
using Combinatorics: powerset
using DataFrames
using ScientificTypes
using Distributions
using POMDPs
using POMDPTools
using MCTS
using Plots

# let's generate some synthetic "historical data"
n_hist = 250

data = DataFrame(
    x1=rand(Distributions.Uniform(-10,10),n_hist),
    x2=rand(Distributions.Exponential(), n_hist),
    x3=rand(Distributions.Binomial(50,0.25),n_hist),
    x4=rand(Distributions.Gamma(10),n_hist)
)

data.y = @. 50*sin(data.x1) + 2.5*data.x2 + 5*data.x2*data.x3 + (20*data.x4 / (1 + data.x4))

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

# stuff to get the similarity based distance metric working
(; sampler, uncertainty, weights) = DistanceBased(data, "y", Variance, GenerativeDesigns.Exponential(; λ = 5));

# we can view the posterior distribution over the outcome conditioned on the current state
using StatsPlots, StatsBase

evidence = Evidence("x3" => mean(data.x3))
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

# we assume "simple" experiments, where each provides measurement of a single observable
experiments_costs = Dict(
    "e1" => (5.0,5) => ["x1"],
    "e2" => (10.0,10) => ["x2"],
    "e3" => (1.0,1) => ["x3"],
    "e4" => (2.0,1) => ["x4"]
)

# find efficient designs
costs=experiments_costs
# sampler
# uncertainty
n_thresholds=6
# evidence=Evidence()
evidence=Evidence("x3"=>mean(data.x3))
solver = DPWSolver(; n_iterations = 1000, tree_in_info = true)
repetitions = 5
realized_uncertainty = false
mdp_options = (; max_parallel=length(experiments_costs), discount=0.95, costs_tradeoff=[0.5,0.5])

designs = efficient_designs(
    experiments_costs,
    sampler,
    uncertainty,
    n_thresholds,
    evidence;
    solver,
    mdp_options = mdp_options,
    repetitions = 5,
);

plot_front(designs; labels = make_labels(designs), ylabel = "% uncertainty")


# # --------------------------------------------------------------------------------
# # looking into the internals

# # for efficient_design
# threshold=0.2
# costs_tradeoff=[0.5,0.5]

# # within efficient_design ...
# mdp = UncertaintyReductionMDP(
#     costs,
#     sampler,
#     uncertainty,
#     threshold,
#     evidence;
#     costs_tradeoff=costs_tradeoff,
#     mdp_options...,
# )

# actions(mdp, mdp.initial_state)

# planner = solve(solver, mdp)
# action, info = action_info(planner, mdp.initial_state)

# queue = [Sim(mdp, planner) for _ = 1:repetitions]