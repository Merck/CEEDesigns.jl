using Revise, CEED, CEED.StaticDesigns
using Combinatorics: powerset
using DataFrames
using Distributions
using POMDPs
using POMDPTools
using MCTS
using Plots

# let's generate some synthetic "historical data"
n_hist = 1000

data = DataFrame(
    x1=rand(Normal(),n_hist),
    x2=rand(Exponential(), n_hist),
    x3=rand(Binomial(50,0.25),n_hist)
)

data.y = @. sin(data.x2) + 10*data.x1 + 2.5*data.x3*data.x2
