module CEED

using DataFrames, Plots

export front, plot_front
export make_labels, plot_evals

# make Pareto fronts
include("fronts.jl")

# experimental designs
include("StaticDesigns/StaticDesigns.jl")
include("GenerativeDesigns/GenerativeDesigns.jl")

end
