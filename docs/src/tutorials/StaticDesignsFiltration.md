```@meta
EditURL = "StaticDesignsFiltration.jl"
```

# [Heart Disease Triage With Early Droupout](@id static_designs_filtration)

Consider again a situation where a group of patients is tested for a specific disease.
It may be costly to conduct an experiment yielding the definitive answer. Instead, we want to utilize various proxy experiments that provide partial information about the presence of the disease.

Moreover, we may assume that for some patients, the evidence gathered from proxy experiments can be considered 'conclusive'. Effectively, some patients may not need any additional triage; they might be deemed healthy or require immediate commencement of treatment. By doing so, we could save significant resources.

## Theoretical Framework

We take as a basis the setup and notation from the basic framework presented in the [static experimental designs tutorial](@ref simple_static).

We again have a set of experiments $E$, but now assume that a set of extrinsic decision-making rules is imposed on the experimental readouts.
If the experimental evidence acquired for a given entity satisfies a specific criterion, that entity is then removed from the triage.
However, other entities within the batch will generally continue in the experimental process.
In general, the process of establishing such rules is largely dependent on the specific problem and requires comprehensive understanding of the subject area.

We denote the expected fraction of entities that remain in the triage after conducting a set $S$ of experiments as the filtration rate, $f_S$. In the context of disease triage, this can be interpreted as the fraction of patients for whom the experimental evidence does not provide a 'conclusive' result.

As previously, each experiment $e$ incurs a cost $(m_e, t_e)$. Again, we let $O_{S}$ denote an arrangement of experiments in $S$.

Given a subset $S$ of experiments and their arrangement $O_{S}$, the total (discounted) monetary cost and execution time of the experimental design is given as $m_o = \sum_{i=1}^{l} r_{S_{i-1}}\sum_{e\in o_i} m_e$ and $t_o = \sum_{i=1}^{l} \max \{ t_e : e\in o_i\}$, respectively.
Importantly, the new factor $r_{S_{i-1}}$ models the fact that a set of entities may have dropped out in the previous experiments, hence saving the resources on running the subsequent experiments.

We note that these computations are based on the assumption that monetary cost is associated with the analysis of a single experimental entity, such as a patient. Therefore, the total monetary cost obtained for a specific arrangement is effectively the ["expected"](https://en.wikipedia.org/wiki/Expected_value) monetary cost, adjusted for a single entity. Conversely, we suppose that all entities can be concurrently examined in a specific experiment. As such, the total execution time is equivalent to the longest time until all experiments within an arrangement are finished or all entities have been eliminated (which ocurrs when the filtration rate the experiments conducted so far is zero). Importantly, this distinctly differs from calculating the 'expected lifespan' of an entity in the triage.

For instance, consider the experiments $e_1,\, e_2,\, e_3$, and $e_4$ with associated costs $(1, 1)$, $(1, 3)$, $(1, 2)$, and $(1, 4)$, and filtration rates $0.9,\,0.8,\,0.7$, and $0.6$. For subsets of experiments, we simply assume that the filtration rates multiply, thereby treating the experiments as independent binary discriminators. In other words, the filtration rate for a set $S=\{ e_1, e_3 \}$ would equal $f_S = 0.9 * 0.7 = 0.63$.

If we conduct experiments $e_1$ through $e_4$ in sequence, this would correspond to an arrangement $o = (\{ e_1 \}, \{ e_2 \}, \{ e_3 \}, \{ e_4 \})$ with a total cost of $m_o = 1 + 0.9 * 1 + 0.72 * 1 + 0.504 * 1 = 3.124$ and $t_o = 10$.

However, if we decide to conduct $e_1$ in parallel with $e_3$, and $e_2$ with $e_4$, we would obtain an arrangement $o = (\{ e_1, e_3 \}, \{ e_2, e_4 \})$ with a total cost of $m_o = 2 + 0.63 * 2 = 3.26$, and $t_o = 3 + 4 = 7$.

Given the constraint on the maximum number of parallel experiments, we construct an arrangement $o$ of experiments $S$ such that, for a fixed tradeoff $\lambda$ between monetary cost and execution time, the expected combined cost $c_{(o, \lambda)} = \lambda m_o + (1-\lambda) t_o$ is minimized. Significantly, our objective is to leverage the filtration rates within the experimental arrangement.

### A Note on Optimal Arrangements

In situations when experiments within a set $S$ are executed sequentially, i.e., one after the other, it can be demonstrated that the experiments should be arranged in ascending order by $\frac{\lambda m_e + (1-\lambda) t_e}{1-f_e}$.

Continuing our example and assuming that the experiments are conducted sequentially, the optimal arrangement $o$ would be to run experiments $e_4$ through $e_1$, yielding $m_o = 2.356$.

When we allow simultaneous execution of experiments, the problem turns more complicated, and we currently are not aware of an 'analytical' solution for it. Instead, we proposed approximating the 'optimal' arrangement as the 'optimal' policy found for a particular Markov decision process, in which:

- _state_ is the set of experiments that have been conducted thus far,
- _actions_ are subsets of experiments which have not been conducted yet; the size of these subsets is restricted by the maximum number of parallel experiments;
- _reward_ is a combined monetary cost and execution time, discounted by the filtration rate of previously conducted experiments.

Provided we know the information values $v_S$, filtration rates $r_S$, and experimental costs $c_S$ for each set of experiments $S$, we find a collection of Pareto-efficient experimental designs that balance both the implied value of information and the experimental cost.

## Heart Disease Dataset

In this tutorial, we consider a dataset that includes 11 clinical features along with a binary variable indicating the presence of heart disease in patients. The dataset can be found at [Kaggle: Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It utilizes heart disease datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

````@example StaticDesignsFiltration
using CSV, DataFrames
data = CSV.File("data/heart_disease.csv") |> DataFrame
data[1:10, :]
````

In order to adapt the dataset to the current context, we can assume that, for every experiment, a medical specialist determined a range for 'conclusive' and 'inconclusive' outputs. This determination could be based on, say, optimizing the precision and recall factors of the resultant discriminative model. As an example, consider [A novel approach for heart disease prediction using strength scores with significant predictors](https://d-nb.info/1242792767/34) where rule mining for heart disease prediction is considered.\

It should be noted that the readout ranges defined below are entirely fictional.

````@example StaticDesignsFiltration
inconclusive_regions = Dict(
    "ChestPainType" => ["NAP", "ASY"],
    "RestingBP" => (50, 150),
    "ExerciseAngina" => ["N"],
    "FastingBS" => [0],
    "RestingECG" => ["Normal"],
    "MaxHR" => (50, 120),
    "Cholesterol" => (0, 240),
    "Oldpeak" => (-2.55, 2.55),
)
````

We apply the rules to derive a binary dataset where 'true' signifies that the readout was inconclusive, requiring them to remain in the triage.

````@example StaticDesignsFiltration
data_binary = DataFrame();
for colname in names(data[!, Not("HeartDisease")])
    if haskey(inconclusive_regions, colname)
        if inconclusive_regions[colname] isa Vector
            data_binary[!, colname] =
                map(x -> x ∈ inconclusive_regions[colname], data[!, colname])
        else
            lb, ub = inconclusive_regions[colname]
            data_binary[!, colname] = map(x -> lb <= x <= ub, data[!, colname])
        end
    else
        data_binary[!, colname] = trues(nrow(data))
    end
end

data_binary[1:10, :]
````

## Discriminative Power and Filtration Rates

In this scenario, we model the value of information $v_S$ acquired by conducting a set of experiments as the ratio of patients for whom the results across the experiments in $S$ were 'inconclusive', i.e., $|\cap_{e\in S}\{ \text{patient} : \text{inconclusive in } e \}| / |\text{patients}|$. Essentially, the very same measure is used here to estimate the filtration rate.

The CEEDesigns.jl package offers an additional flexibility by allowing an experiment to yield readouts over multiple features at the same time. In our scenario, we can consider the features `RestingECG`, `Oldpeak`, `ST_Slope`, and `MaxHR` to be obtained from a single experiment `ECG`.

We specify the experiments along with the associated features:

````@example StaticDesignsFiltration
experiments = Dict(
    # experiment => features
    "BloodPressure" => ["RestingBP"],
    "ECG" => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => ["Cholesterol"],
    "BloodSugar" => ["FastingBS"],
)
````

We may also provide additional zero-cost features, which are always available.

````@example StaticDesignsFiltration
zero_cost_features = ["Age", "Sex", "ChestPainType", "ExerciseAngina"]
````

For binary datasets, we may use `evaluate_experiments` from `CEEDesigns.StaticDesigns` to evaluate the discriminative power of subsets of experiments.

````@example StaticDesignsFiltration
using CEEDesigns, CEEDesigns.StaticDesigns
````

````@example StaticDesignsFiltration
perf_eval = evaluate_experiments(experiments, data_binary; zero_cost_features)
````

Note that for each subset of experiments, the function returns a named tuple with fields `loss` and `filtration`.

We plot discriminative power evaluated for subsets of experiments.

````@example StaticDesignsFiltration
plot_evals(perf_eval; ylabel = "discriminative power")
````

## Cost-Efficient Designs

We specify the cost associated with the execution of each experiment.

````@example StaticDesignsFiltration
costs = Dict(
    # experiment => cost
    "BloodPressure" => 1,
    "ECG" => 5,
    "BloodCholesterol" => 20,
    "BloodSugar" => 20,
)
````

We use the provided function `efficient_designs` to construct the set of cost-efficient experimental designs. Note that the filtration is enabled implicitly since we provided the filtration rates within `perf_eval`. Another form of `perf_eval` would be `subset of experiments => information measure`, in which case the filtration would equal one. That is, no dropout would be considered.

````@example StaticDesignsFiltration
designs = efficient_designs(costs, perf_eval)
````

````@example StaticDesignsFiltration
plot_front(designs; labels = make_labels(designs), ylabel = "discriminative power")
````

Let us compare the above with the efficient front with disabled filtration.

````@example StaticDesignsFiltration
# pass performance eval with discarded filtration rates (defaults to 1)
designs_no_filtration = efficient_designs(costs, Dict(k => v.loss for (k, v) in perf_eval))
````

````@example StaticDesignsFiltration
using Plots
p = scatter(
    map(x -> x[1][1], designs_no_filtration),
    map(x -> x[1][2], designs_no_filtration);
    label = "designs with filtration disabled",
    c = :blue,
    mscolor = nothing,
    fontsize = 16,
)
scatter!(
    p,
    map(x -> x[1][1], designs),
    map(x -> x[1][2], designs);
    xlabel = "combined cost",
    ylabel = "discriminative power",
    label = "designs with filtration enabled",
    c = :teal,
    mscolor = nothing,
    fontsize = 16,
)
````

## Arrangement of a Set of Experiments

Importantly, the total execution cost of an experiment is generally not the sum of costs associated to each individual experiment. In fact, a non-zero dropout rate (`filtration < 1`) discounts the expected cost of subsequent experiments. As discussed previously, we are not aware of an 'analytical' solution to this problem.

Instead, we approximate the 'optimal' arrangement as the 'optimal' policy for a particular Markov decision process. This can be accomplished using, for instance, the [Monte Carlo Tree Search algorithm](https://github.com/JuliaPOMDP/MCTS.jl) as implemented in the [POMDPs.jl](http://juliapomdp.github.io/POMDPs.jl/latest/) package.

The following is a visualisation of the DPW search tree that was used to find an optimal arrangement for a set of experiments yielding the highest information value.

````@example StaticDesignsFiltration
using MCTS, D3Trees

experiments = Set(vcat.(designs[end][2].arrangement...)[1])
(; planner) = CEEDesigns.StaticDesigns.optimal_arrangement(
    costs,
    perf_eval,
    experiments;
    mdp_kwargs = (; tree_in_info = true),
)
_, info = action_info(planner, Set{String}())

t = D3Tree(info[:tree]; init_expand = 2)
````

````@example StaticDesignsFiltration
plot_front(designs; labels = make_labels(designs), ylabel = "discriminative power")
````

### Parallel Experiments

We may exploit parallelism in the experimental arrangement. To that end, we first specify the monetary cost and execution time for each experiment, respectively.

````@example StaticDesignsFiltration
experiments_costs = Dict(
    # experiment => (monetary cost, execution time) => features
    "BloodPressure" => (1.0, 1.0) => ["RestingBP"],
    "ECG" => (5.0, 5.0) => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => (20.0, 20.0) => ["Cholesterol"],
    "BloodSugar" => (20.0, 20.0) => ["FastingBS"],
)
````

We provide the maximum number of concurrent experiments. Additionally, we specify the tradeoff between monetary cost and execution time - in our case, we aim to minimize the execution time.

Below, we demonstrate the flexibility of `efficient_designs` as it can both evaluate the performance of experiments and generate efficient designs. Internally, `evaluate_experiments` is called first, followed by `efficient_designs`. Keyword arguments to the respective functions has to wrapped in `eval_options` and `arrangement_options` named tuples, respectively.

````@example StaticDesignsFiltration
# Implicit, calculates accuracies automatically
designs = efficient_designs(
    experiments_costs,
    data_binary;
    eval_options = (; zero_cost_features),
    arrangement_options = (; max_parallel = 2, tradeoff = (0.0, 1)),
)
````

As we can see, the algorithm correctly suggests running experiments in parallel.

````@example StaticDesignsFiltration
plot_front(designs; labels = make_labels(designs), ylabel = "discriminative power")
````

