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
n_hist = 1000

data = DataFrame(
    x1=rand(Normal(),n_hist),
    x2=rand(Distributions.Exponential(), n_hist),
    x3=rand(Binomial(50,0.25),n_hist)
)

data.y = @. sin(data.x1) + 10*data.x2 + 2.5*data.x2*data.x3

types = Dict(
    :x1 => ScientificTypes.Continuous,
    :x2 => ScientificTypes.Continuous,
    :x3 => ScientificTypes.Continuous,
    :y => ScientificTypes.Continuous
)
data = coerce(data, types);