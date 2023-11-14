using CEED, CEED.StaticDesigns
using Combinatorics: powerset
using DataFrames
using POMDPs
using POMDPTools
using MCTS

# mimic the output of `evaluate_experiments`
experiments = ["e1","e2","e3"]
experiments_val = Dict([e => rand() for e in experiments])

# make some synthetic "evaluation data" such that
# no subset which includes both e_i and e_j can be worse
# that either one alone, but with some randomness.
experiments_evals = Dict(
    map(Set.(collect(powerset(experiments)))) do s
        if length(s) > 0
            s => prod([experiments_val[i] for i in s])
        else
            return s => 1.0
        end 
    end
)

# better experiments are more costly
experiments_costs = Dict(
    sort(collect(keys(experiments_val)), by=k->experiments_val[k], rev=true) .=> tuple.(1:3,1:3)
)

# use data frame for nice printing
DataFrame(
    experiment=collect(keys(experiments_costs)),
    time=getindex.(values(experiments_costs),1),
    cost=getindex.(values(experiments_costs),2)
)

# plot looks ok
plot_evals(experiments_evals; f = x->sort(collect(keys(x)), by = k->x[k], rev=true), ylabel = "loss")

# use data frame for nice printing
DataFrame(
    S=collect.(collect(keys(experiments_evals))),
    value=collect(values(experiments_evals))
)

# # call `efficient_designs`
# max_parallel = 2
# tradeoff = (0.5, 0.5)
# mdp_kwargs = CEED.StaticDesigns.default_mdp_kwargs

# experimental_costs = Dict(e => v isa Pair ? v[1] : v for (e, v) in experiments_costs)

# evals = Dict{Set{String},CEED.StaticDesigns.ExperimentalEval}(
#     if e isa Number
#         s => (; loss = convert(Float64, e), filtration = 1.0)
#     else
#         s => (;
#             loss = convert(Float64, e.loss),
#             filtration = convert(Float64, e.filtration),
#         )
#     end for (s, e) in experiments_evals
# )

# # note that collect(evals) will give us the possible subsets S 
# # of experiments E. For each one, we want to find it's optimal arrangement.

# # the for loop within `efficient_designs`
# designs = []

# for design in collect(evals)
#     arrangement = CEED.StaticDesigns.optimal_arrangement(
#         experimental_costs,
#         evals,
#         design[1];
#         max_parallel,
#         tradeoff,
#         mdp_kwargs,
#     )

#     push!(
#         designs,
#         (
#             (arrangement.combined_cost, design[2].loss),
#             (;
#                 arrangement = arrangement.arrangement,
#                 monetary_cost = arrangement.monetary_cost,
#                 time = arrangement.time,
#             ),
#         ),
#     )
# end

# pareto_designs = front(x -> x[1], designs)

# plot_front(pareto_designs; labels = make_labels(pareto_designs), ylabel = "loss")

# or by using the function we have that wraps this
designs = efficient_designs(experiments_costs, experiments_evals, max_parallel=max_parallel, tradeoff=tradeoff)

plot_front(designs; labels = make_labels(pareto_designs), ylabel = "loss")