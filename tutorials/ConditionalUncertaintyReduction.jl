# # [Conditional Uncertainty Reduction: Case-Guided Sequential Experimental Design](@id conditional_uncertainty_reduction)

# This document describes the background behind conditional (constraint-aware) uncertainty-reduction MDP and applies this
# enhanced version of the baseline CEEDesigns.jl framework to a real-world problem of sequential assay planning for drug discovery.
# The tutorial implements the case-guided sequential experimental design methodology of [Chen et al., 2026](https://arxiv.org/abs/2601.14710),
# reproducing the CNS/brain-penetration case study from that paper using CEEDesigns.jl.

# In the experimental setup, our objective is to minimize the expected experimental cost while ensuring the uncertainty remains
# below a specified threshold and that a conditional likelihood criterion is satisfied.

# In our setting, the conditional likelihood is the posterior probability mass the generative model assigns to the target variable
# $y$ falling in a "desirable" range, given the evidence accumulated so far, i.e. $L(s) = P(y \in Y^\star \mid \text{evidence}(s))$.
# Concretely in this tutorial, $L(s) = P(0.5 \leq k_\text{puu} \leq 1 \mid \text{QSAR and assay readouts})$. Because the planner is
# free to stop only once $L(s) \geq \tau$, the policy is steered toward evidence states that are not only low-uncertainty but also
# confident the compound lies in the promising region. See the [Generative Experimental Designs tutorial](@ref simple_generative)
# for the underlying similarity-weighted belief.


# Consider a situation where we have a set of assays or experiments that can be performed sequentially to gather information
# about a compound. In this example, our objective is to efficiently determine a compound's brain penetration potential
# which is dependent on the compound's ability to cross the blood-brain barrier. We must create an assay plan by selecting
# from a set of cheap, fast, but less informative in vitro assays and an expensive, slow, but definitive in vivo assay that
# directly measures the unbound brain-to-plasma partition coefficient, $k_\text{puu}$.

# The following preamble loads the necessary packages and fixes the RNG seed for reproducibility.

using Plots #hide
using CEEDesigns, CEEDesigns.GenerativeDesigns #hide
using CEEDesigns: ensemble_to_dataframe, plot_ensemble_pareto #hide
using Random: seed! #hide
seed!(1) #hide

# ## Brain Penetration Assays Dataset

# In this tutorial, we consider a dataset that includes 220 compounds with complete measurements for all relevant assays
# (100 nM PgP, 1 µM PgP, 100 nM BCRP) and the target property k_puu (unbound brain-to-plasma partition coefficient).
# The dataset also includes associated Quantitative Structure-Activity Relationship (QSAR) model predictions, which
# provide initial estimates for the assay outcomes and other relevant properties such as Mean Residence Time (MRT).

# The dataset was originally published in [Chen et al., 2026](https://arxiv.org/abs/2601.14710) and can be found at
# [CNS Assays Dataset](https://github.com/MSDLLCpapers/IBMDPDesigns.jl/tree/main/CNS_example/data).

using CSV, DataFrames
data = CSV.File("data/cns_assays.csv") |> DataFrame
data[1:10, :]

# ## Experimental Setup

# When setting up the full experimental planning pipeline, the necessary components are:
# 1. A distance-based generative model derived from historical data, weighting the historical compounds by similarity.
# 2. A table mapping each assay to its operational costs, both monetary and temporal.
# 3. A conditional terminal condition: the MDP is terminal only when both
#    `H(s) ≤ ε` (uncertainty threshold) and `L(s) ≥ τ` (goal-likelihood threshold) are met.
# 4. A Monte Carlo Tree Search-Double Progressive Widening (MCTS-DPW) solver that searches over combinatorial assay batches.

# ### Feature Distance Weights

# We discount the informativeness gap (or 'fidelity gap') between cheap, in-silico predictions and expensive, physical assays by
# assigning different distance scales (per-feature $\lambda_k$) — a smaller $\lambda$ makes a feature less decisive for similarity,
# so two compounds that differ on that feature are still considered close neighbors in the weighted kernel.

in_silico = ["1uM_PgP_qsar", "100_nM_Mouse_BCRP_qsar", "qsar_mrt"]
physical = [
    "blood_frac_conc",
    "brain_conc",
    "brain_binding",
    "plasma_protein_binding",
    "kpuu",
    "100nM_PgP",
    "1uM_PgP",
    "100nM_BCRP",
];

# The distance dictionary assigns a per-feature weight λ_k that controls how strongly each feature influences the similarity kernel.
# Distances are variance-normalized then transformed into similarities via an exponential kernel. In-silico QSAR predictions are estimates
# and thus less reliable, so we down-weight them (λ = 50) relative to the physical assay results (λ = 200), ensuring that the available
# physical measurements have a stronger influence on the belief update.

distances = Dict{String, Any}()
for f in in_silico
    distances[f] = QuadraticDistance(; λ = 50)
end
for f in physical
    distances[f] = QuadraticDistance(; λ = 200)
end

# These specific values ($\lambda = 50$ for in-silico, $\lambda = 200$ for physical measurements) are set heuristically from empirical
# tuning on this dataset — they are not derived from any theoretical optimality result, and practitioners should re-tune them (e.g.
# via cross-validated predictive log-likelihood of $k_\text{puu}$) when applying the workflow to other data.

# ### Generative Model Construction

# We obtain three functions in what follows, which together define the implicit generative model:
#  -  `sampler` — this is a function of `(evidence, features, rng)` which samples a historical case from the dataset with probabilities
#      proportional to the similarity weights.  The `evidence` is the current experimental evidence for the compound of interest,
#      `features` is the set of features (assays) we want to sample from, and `rng` is a random number generator.
#  -  `uncertainty`: this is a function of `evidence`, computes the weighted variance H(s) of the target property kpuu over the
#      historical data.
#  -  `weights(evidence)` — this represents a function of `evidence` that gives the similarity weight vector w_i(s) for each
#      historical compound. It serves as a probability distribution for  sampling and is used to compute the goal-likelihood L(s).

(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "kpuu",
    uncertainty = Variance(),
    similarity = Exponential(; λ = 0.5),
    distance = distances,
);

# ### Experiment Costs

# Each experiment is specified as `name => (monetary_cost, time_in_days)`.

# The operational costs are defined as (400, 7) for each in vitro assay and (4000, 7) for the in vivo kpuu assay.

# The solver finds the Pareto-optimal sequences that trade off total cost against terminal uncertainty, recommending the cheapest
# plan that meets the confidence requirements.

experiments = Dict(
    "100nM_PgP" => (400.0, 7.0),
    "100nM_BCRP" => (400.0, 7.0),
    "1uM_PgP" => (400.0, 7.0),
    "kpuu" => (4000.0, 21.0),
);

# ### Solver Configuration

# The conditional terminal condition requires that the posterior probability of kpuu falling in the desirable range [0.5, 1.0]
# exceeds the belief threshold τ.

target_condition = Dict("kpuu" => [0.5, 1.0])
taus = [0.6, 0.9];

# The MCTS-DPW solver uses a very low value of `n_iterations` per MCTS run, to keep runtime manageable; in real applications this should
# be increased. Other options such as `exploration_constant = 5.0` controls the balance between exploration of new actions with exploitation
# of known high-value actions within the search tree, and `depth = 5`, again keeping runtime down. Setting `depth = 11` would allow the planner to
# look ahead through all possible assay orderings.

solver = GenerativeDesigns.DPWSolver(;
    n_iterations = 2_000,
    exploration_constant = 5.0,
    depth = 5,
    tree_in_info = true,
    keep_tree = true,
);

# ## Representative Compound Scenarios with Varying Initial Evidence

# We evaluate four representative scenarios designed to test the planner against a heuristic that deems compounds as promising
# (likely 0.5 ≤ kpuu ≤ 1) when QSAR_PgP < 2 AND QSAR_BCRP < 2 and not promising (kpuu < 0.5) when QSAR_PgP > 4 OR QSAR_BCRP > 4.

# | Scenario | PgP QSAR | BCRP QSAR | Challenge                         |
# |----------|----------|-----------|-----------------------------------|
# | 1        | < 2      | < 2       | Baseline confirmation             |
# | 2        | < 2      | > 4       | BCRP false negative               |
# | 3        | > 4      | < 2       | PgP false negative                |
# | 4        | > 4      | > 4       | Double false negative             |

# All selected compounds have true kpuu > 0.5.
# The `select_representative_rows` function below selects the compound with the lowest kpuu from each category, the hardest case, and
# constructs its initial evidence state from the three QSAR predictions (MRT, PgP, BCRP).

function select_representative_rows(data::DataFrame; num_instances::Int = 20)
    conditions = [
        (data[!, "1uM_PgP_qsar"] .< 2) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .< 2) .&
            (data[!, "kpuu"] .> 0.5),
        (data[!, "1uM_PgP_qsar"] .< 2) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .> 4) .&
            (data[!, "kpuu"] .> 0.5),
        (data[!, "1uM_PgP_qsar"] .> 4) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .< 2) .&
            (data[!, "kpuu"] .> 0.5),
        (data[!, "1uM_PgP_qsar"] .> 4) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .> 4) .&
            (data[!, "kpuu"] .> 0.5),
    ]

    selected_columns = [
        "1uM_PgP_qsar",
        "100_nM_Mouse_BCRP_qsar",
        "qsar_mrt",
        "kpuu",
        "100nM_PgP",
        "1uM_PgP",
        "100nM_BCRP",
    ]
    selected_rows = DataFrame()
    state_init_list = []

    for (i, condition) in enumerate(conditions)
        filtered_data = data[condition, :]
        if nrow(filtered_data) > 0
            sorted_data = sort(filtered_data, :kpuu)
            representative_rows = first(sorted_data, min(num_instances, nrow(sorted_data)))[
                !,
                selected_columns,
            ]
            selected_rows = vcat(selected_rows, representative_rows)

            if nrow(representative_rows) > 0
                row = first(eachrow(representative_rows))
                state_init = Evidence(
                    "qsar_mrt" => row["qsar_mrt"],
                    "1uM_PgP_qsar" => row["1uM_PgP_qsar"],
                    "100_nM_Mouse_BCRP_qsar" => row["100_nM_Mouse_BCRP_qsar"],
                )
                push!(state_init_list, state_init)
            end
        else
            continue
        end
    end

    return state_init_list, selected_rows
end

state_init_list, selected_data = select_representative_rows(data)

scenarios = [
    "Scenario 1: Low PgP, Low BCRP",
    "Scenario 2: Low PgP, High BCRP",
    "Scenario 3: High PgP, Low BCRP",
    "Scenario 4: High PgP, High BCRP",
];

# ## Ensemble Planning Across All Scenarios and Thresholds

# For each of the four scenarios we:
# 1. Compute separate Pareto fronts for each τ value (τ = 0.6 and τ = 0.9), mapping out the cost vs uncertainty trade-off
#    frontier under each goal-likelihood constraint.
# 2. Run a single ensemble of N = 5 independent MCTS planners using the strictest constraint (τ = 0.9). By *ensemble* we
# mean running $N$ independent MCTS-DPW planners on the same initial evidence. Each planner yields its own Pareto front
# of candidate designs; aggregating by majority vote over the selected action sets at each uncertainty level yields a
# more robust recommendation (the MLASP) than any single run, and the spread across runs gives an empirical measure of
# policy variance. The ensemble results are then evaluated at various levels of uncertainty threshold. In the following
# example, we generate 5 thresholds spaces evenly between 0 and 1, inclusive. Majority voting across runs at each
# uncertainty level yields the  Maximum-Likelihood Action-Sets Path (MLASP), the most robust assay recommendation.

all_designs = Dict()
all_ensemble_dfs = Dict()
all_plots = []

for (idx, evidence) in enumerate(state_init_list)

    ## Compute separate Pareto fronts for each τ value
    for tau in taus
        designs = conditional_efficient_designs(
            experiments;
            sampler = sampler,
            uncertainty = uncertainty,
            thresholds = 5,
            evidence = evidence,
            weights = weights,
            data = data,
            terminal_condition = (target_condition, tau),
            solver = solver,
            realized_uncertainty = true,
            mdp_options = (; max_parallel = 3, costs_tradeoff = (1.0, 0.0), bigM = 10_000),
        )

        if !haskey(all_designs, tau)
            all_designs[tau] = []
        end
        push!(all_designs[tau], designs)
    end

    ## Ensemble analysis with multiple tau values
    reference_tau = maximum(taus)
    ensemble_results = perform_ensemble_designs(
        experiments;
        sampler = sampler,
        uncertainty = uncertainty,
        thresholds = 5,
        evidence = evidence,
        weights = weights,
        data = data,
        terminal_condition = (target_condition, reference_tau),
        realized_uncertainty = true,
        solver = solver,
        mdp_options = (;
            max_parallel = 3,
            costs_tradeoff = (1.0, 0.0),
            bigM = 10_000, # Chosen to stay consistent with assay costs
        ),
        N = 5,
        thred_set = taus,
    )

    ## Process results for each tau value
    for tau in taus
        runs = ensemble_results[:belief => tau]
        df_ensemble = ensemble_to_dataframe(runs)

        if !haskey(all_ensemble_dfs, tau)
            all_ensemble_dfs[tau] = []
        end
        push!(all_ensemble_dfs[tau], df_ensemble)

        plt = plot_ensemble_pareto(df_ensemble, tau)
        plot!(plt; title = "$(scenarios[idx]) (τ = $(tau))")
        push!(all_plots, plt)
    end
end

# Let's see the plot for the first value of τ.

plot(all_plots[1])

# And the second.

plot(all_plots[2])

# ## Summary of Scenario Outcomes

# The summary table gives the following results for each scenario and belief threshold τ:
# - `P_kpuu_in_range` — posterior probability $P(k_\text{puu} \in [0.5, 1.0] \mid \text{QSAR})$ that the compound lies in
#   the desirable range given QSAR features alone (before any physical assays).
# - `Constraint` - whether the initial belief already meets the constraint P ≥ τ without any physical assays.
# - `Cost_Range` and `Unc_Range` - the cost and uncertainty ranges across the Pareto front of designs that meet the constraint.
# - `MLASP` - the Most Likely Action-Set Path or final recommendation which is constructed by majority vote over the actions
#    (assays) recommended by the ensemble policies at each stage. Provided is at each uncertainty threshold, the assay/s that
#    won the majority vote across the ensemble.

tau_summary = []

for tau in taus
    summary_data = []

    for (i, name) in enumerate(scenarios)
        ev = state_init_list[i]

        ## Initial evidence values
        evidence_cols = Dict()
        for (k, v) in pairs(ev)
            evidence_cols[k] = round(v; digits = 3)
        end

        init_unc = round(uncertainty(ev); digits = 3)

        cond_prob = conditional_likelihood(
            ev;
            compute_weights = weights,
            hist_data = data,
            target_condition = target_condition,
        )
        prob_kpuu = round(cond_prob; digits = 3)
        constraint_met = cond_prob >= tau ? "✓" : "✗"

        ## Pareto designs
        designs = all_designs[tau][i]
        costs = [perf[1] for (perf, _) in designs]
        uncs = [perf[2] for (perf, _) in designs]
        num_pareto = length(designs)
        cost_range = if !isempty(costs)
            "\$$(round(Int, minimum(costs))) – \$$(round(Int, maximum(costs)))"
        else
            "N/A"
        end
        unc_range = if !isempty(uncs)
            "$(round(minimum(uncs); digits = 3)) – $(round(maximum(uncs); digits = 3))"
        else
            "N/A"
        end

        ## Ensemble summary
        df = all_ensemble_dfs[tau][i]
        unique_actions = unique(df.Action_Set)
        num_unique_actions = length(unique_actions)

        ## MLASP
        mlasp_info = []
        for t in sort(unique(df.Threshold))
            sub = filter(r -> r.Threshold == t, df)
            best = sub[argmax(sub.Frequency), :]
            total_freq = sum(sub.Frequency)
            pct = round(100 * best.Frequency / total_freq; digits = 0)
            actions_str = isempty(best.Action_Set) ? "(none)" : best.Action_Set
            push!(mlasp_info, "ε=$(round(t; digits = 2)): $(actions_str) ($(Int(pct))%)")
        end

        push!(
            summary_data,
            (
                Scenario = replace(name, r"^Scenario \d+: " => ""),
                QSar_MRT = get(evidence_cols, "qsar_mrt", NaN),
                PgP_QSAR = get(evidence_cols, "1uM_PgP_qsar", NaN),
                BCRP_QSAR = get(evidence_cols, "100_nM_Mouse_BCRP_qsar", NaN),
                Init_Uncertainty = init_unc,
                P_kpuu_in_range = prob_kpuu,
                Constraint = constraint_met,
                Pareto_Designs = num_pareto,
                Cost_Range = cost_range,
                Unc_Range = unc_range,
                Unique_Actions = num_unique_actions,
                MLASP = join(mlasp_info, "; "),
            ),
        )
    end

    push!(tau_summary, summary_data)
end;

# We can examiune the results for the first value of τ:

DataFrame(tau_summary[1])

# And for the second:

DataFrame(tau_summary[2])
