# Regression / unit tests for `EfficientValueMDP` and `efficient_value`.
#
# These tests guard against:
#   * the previously incorrect `hasmethod(value, Tuple{Evidence, Vector{Float64}})`
#     assertion (now `Tuple{Evidence, NTuple{2, Float64}}`);
#   * `EfficientValueMDP` not being re-exported from `CEEDesigns.GenerativeDesigns`;
#   * regressions in the `repetitions = 0` / `repetitions > 0` and `tree_in_info`
#     code paths inside `efficient_value`;
#   * RNG reproducibility when the solver is constructed with a seeded RNG.

using Test
using Random: Xoshiro, AbstractRNG
using CSV, DataFrames
using ScientificTypes
using CEEDesigns, CEEDesigns.GenerativeDesigns

# ----- Export check (Severe #10) -----
# `EfficientValueMDP` must be reachable from `using CEEDesigns.GenerativeDesigns`.
@test isdefined(GenerativeDesigns, :EfficientValueMDP)
@test EfficientValueMDP === GenerativeDesigns.EfficientValueMDP

# ----- Data setup, mirroring `test_mahalanobis.jl` -----
data = CSV.File("GenerativeDesigns/data/heart_disease.csv") |> DataFrame

types = Dict(
    :MaxHR => Continuous,
    :Cholesterol => Continuous,
    :ChestPainType => Multiclass,
    :Oldpeak => Continuous,
    :HeartDisease => Multiclass,
    :Age => Continuous,
    :ST_Slope => Multiclass,
    :RestingECG => Multiclass,
    :RestingBP => Continuous,
    :Sex => Multiclass,
    :FastingBS => Continuous,
    :ExerciseAngina => Multiclass,
)
data = coerce(data, types)

(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy(),
    similarity = Exponential(; λ = 5),
)

experiments = Dict(
    "BloodPressure" => 1.0 => ["RestingBP"],
    "ECG" => 5.0 => ["RestingECG", "Oldpeak", "ST_Slope", "MaxHR"],
    "BloodCholesterol" => 20.0 => ["Cholesterol"],
    "BloodSugar" => 20.0 => ["FastingBS"],
    "HeartDisease" => 100.0,
)

evidence = Evidence("Age" => 35, "Sex" => "M")

# Correct value signature: (Evidence, NTuple{2, Float64}) -> Float64
value_correct = function (evidence, (monetary_cost, execution_time))
    return (1 - uncertainty(evidence)) - 0.005 * monetary_cost
end

# Incorrect value signature: (Evidence, Vector{Float64}) -> Float64
# This was previously *accepted* by the buggy assertion and *rejected* by the
# fixed one.
value_vector = function (evidence::Evidence, costs::Vector{Float64})
    return (1 - uncertainty(evidence)) - 0.005 * costs[1]
end

# ----- Constructor signature checks (regression for the assertion fix) -----

# (1) The correct (Evidence, NTuple{2, Float64}) signature must be accepted.
#     The buggy assertion `hasmethod(value, Tuple{Evidence, Vector{Float64}})`
#     would have rejected this perfectly valid `value`.
@test (
    EfficientValueMDP(experiments; sampler, value = value_correct, evidence) isa
        EfficientValueMDP
)

# (2) A `value` with the wrong (Evidence, Vector{Float64}) signature must be
#     rejected. The buggy assertion would have accepted this even though the
#     MDP itself would later pass `state.costs::NTuple{2, Float64}` and crash.
@test_throws AssertionError EfficientValueMDP(
    experiments;
    sampler,
    value = value_vector,
    evidence,
)

# ----- End-to-end happy paths -----

# Tiny solver for speed.
small_solver = GenerativeDesigns.DPWSolver(;
    n_iterations = 50,
    depth = 2,
    tree_in_info = true,
)

# repetitions > 0 with tree_in_info = true
design_with_reps_and_tree = efficient_value(
    experiments;
    sampler,
    value = value_correct,
    evidence,
    solver = small_solver,
    repetitions = 3,
)
@test design_with_reps_and_tree isa Tuple
@test length(design_with_reps_and_tree) == 2
@test design_with_reps_and_tree[1] isa Real
@test hasproperty(design_with_reps_and_tree[2], :stats)
@test hasproperty(design_with_reps_and_tree[2], :tree)
@test hasproperty(design_with_reps_and_tree[2], :arrangement)

# repetitions = 0 with tree_in_info = true
design_no_reps_with_tree = efficient_value(
    experiments;
    sampler,
    value = value_correct,
    evidence,
    solver = small_solver,
)
@test design_no_reps_with_tree isa Tuple
@test length(design_no_reps_with_tree) == 2
@test design_no_reps_with_tree[1] isa Real
@test !hasproperty(design_no_reps_with_tree[2], :stats)
@test hasproperty(design_no_reps_with_tree[2], :tree)
@test hasproperty(design_no_reps_with_tree[2], :arrangement)

# repetitions > 0 without tree_in_info
small_solver_no_tree = GenerativeDesigns.DPWSolver(;
    n_iterations = 50,
    depth = 2,
    tree_in_info = false,
)
design_with_reps_no_tree = efficient_value(
    experiments;
    sampler,
    value = value_correct,
    evidence,
    solver = small_solver_no_tree,
    repetitions = 3,
)
@test design_with_reps_no_tree isa Tuple
@test hasproperty(design_with_reps_no_tree[2], :stats)
@test !hasproperty(design_with_reps_no_tree[2], :tree)

# repetitions = 0 without tree_in_info
design_no_reps_no_tree = efficient_value(
    experiments;
    sampler,
    value = value_correct,
    evidence,
    solver = small_solver_no_tree,
)
@test design_no_reps_no_tree isa Tuple
@test !hasproperty(design_no_reps_no_tree[2], :stats)
@test !hasproperty(design_no_reps_no_tree[2], :tree)

# ----- RNG reproducibility -----
# Two calls that share an identically-seeded `Xoshiro` (one per call, so each
# solver gets its own fresh RNG state) must yield identical objective values.
function run_seeded()
    solver = GenerativeDesigns.DPWSolver(;
        n_iterations = 50,
        depth = 2,
        tree_in_info = false,
        rng = Xoshiro(42),
    )
    return efficient_value(
        experiments;
        sampler,
        value = value_correct,
        evidence,
        solver,
    )
end

result_a = run_seeded()
result_b = run_seeded()
@test result_a[1] == result_b[1]
@test result_a[2].arrangement == result_b[2].arrangement
