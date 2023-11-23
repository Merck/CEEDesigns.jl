module GenerativeDesigns

using POMDPs
using POMDPTools

using Combinatorics
using DataFrames, ScientificTypes
using LinearAlgebra
using Statistics
using StatsBase: Weights, countmap, entropy, sample
using Random: default_rng, AbstractRNG
using MCTS

using ..CEED: front

export UncertaintyReductionMDP, DistanceBased
export QuadraticDistance, DiscreteDistance, MahalanobisDistance
export Exponential
export Variance, Entropy
export Evidence, State
export efficient_design, efficient_designs
export efficient_value

"""
Represent experimental evidence as an immutable dictionary.
"""
const Evidence = Base.ImmutableDict{String,Any}

function Base.merge(d::Evidence, dsrc::Dict)
    for (k, v) in dsrc
        d = Evidence(d, k, v)
    end

    return d
end

Evidence(p1::Pair, pairs::Pair...) = merge(Evidence(), Dict(p1, pairs...))

"""
Represent experimental state as a tuple of experimental costs and evidence.
"""
const State = NamedTuple{(:evidence, :costs),Tuple{Evidence,NTuple{2,Float64}}}

function Base.merge(state::State, evidence, costs)
    return State((merge(state.evidence, evidence), state.costs .+ costs))
end

include("distancebased.jl")

"""
Represent action as a named tuple `(; costs=(monetary cost, time), features)`.
"""
const ActionCost = NamedTuple{(:costs, :features),<:Tuple{NTuple{2,Float64},Vector{String}}}

const const_bigM = 1_000_000

const default_solver = DPWSolver(; n_iterations = 100_000, tree_in_info = true)

# minimize the expected experimental cost while ensuring the uncertainty remains below a specified threshold.
include("UncertaintyReductionMDP.jl")

# maximize the value of the experimental evidence (such as clinical utility), adjusted for experimental costs.
include("EfficientValueMDP.jl")

end
