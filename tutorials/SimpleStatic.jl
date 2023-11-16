# # Static Designs

# This tutorial presents a synthetic example of using CEED to optimize static experimental design.
# We consider a situation where there are 3 experiments, and we draw a value of their "loss function"
# or "entropy" from the uniform distribution on the unit interval for each. 

# The "evaluation data" for subsets of experiments is the produt of those value for each individual experiment.
# Therefore, because smaller values are better, the subset containing multiple experiments is guaranteed to be 
# more "valuable" than any individual experiment.

using CEED, CEED.StaticDesigns
using Combinatorics: powerset
using DataFrames
using POMDPs, POMDPTools, MCTS

# First we mimic the output of `evaluate_experiments`.

experiments = ["e1","e2","e3"];
experiments_val = Dict([e => rand() for e in experiments]);

experiments_evals = Dict(
    map(Set.(collect(powerset(experiments)))) do s
        if length(s) > 0
            s => prod([experiments_val[i] for i in s])
        else
            return s => 1.0
        end 
    end
);

# Better experiments are more costly, both in terms of time and monetary cost. We print
# the data frame showing each experiment and its associated costs.

experiments_costs = Dict(
    sort(collect(keys(experiments_val)), by=k->experiments_val[k], rev=true) .=> tuple.(1:3,1:3)
);

DataFrame(
    experiment=collect(keys(experiments_costs)),
    time=getindex.(values(experiments_costs),1),
    cost=getindex.(values(experiments_costs),2)
)

# We can plot the experiments ordered by their "loss function".

plot_evals(experiments_evals; f = x->sort(collect(keys(x)), by = k->x[k], rev=true), ylabel = "loss")

# We print the data frame showing each subset of experiments and its overall loss value.

DataFrame(
    S=collect.(collect(keys(experiments_evals))),
    value=collect(values(experiments_evals))
)

# We use `efficient_designs` to solve the optimal arrangements.

max_parallel = 2;
tradeoff = (0.5, 0.5);

designs = efficient_designs(experiments_costs, experiments_evals, max_parallel=max_parallel, tradeoff=tradeoff);

plot_front(designs; labels = make_labels(designs), ylabel = "loss")