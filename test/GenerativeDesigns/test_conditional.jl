using Test
using CSV, DataFrames
using ScientificTypes
using Random
using Random: Xoshiro
import POMDPs

using CEEDesigns, CEEDesigns.GenerativeDesigns

@testset "ConditionalUncertaintyReductionMDP" begin
    # ------------------------------------------------------------------
    # Shared fixture data: heart_disease.csv with mixed scitypes.
    # ------------------------------------------------------------------
    raw = CSV.File("GenerativeDesigns/data/heart_disease.csv") |> DataFrame
    types = Dict(
        :MaxHR => Continuous,
        :Cholesterol => Continuous,
        :Oldpeak => Continuous,
        :Age => Continuous,
        :RestingBP => Continuous,
        :Sex => Multiclass,
        :FastingBS => Continuous,
    )
    raw_typed = coerce(raw, types)

    # Numeric-only view for the "happy path" tests (mirrors test_mahalanobis.jl).
    continuous_cols =
        filter(colname -> eltype(raw_typed[!, colname]) == Float64, names(raw_typed))
    data = raw_typed[!, continuous_cols ∪ ["HeartDisease"]]

    (; sampler, uncertainty, weights) = DistanceBased(
        data;
        target = "HeartDisease",
        uncertainty = Variance(),
        similarity = Exponential(; λ = 5),
        distance = SquaredMahalanobisDistance(; diagonal = 1),
    )

    experiments = Dict(
        "BloodPressure" => 1.0 => ["RestingBP"],
        "ECG" => 5.0 => ["Oldpeak", "MaxHR"],
        "BloodCholesterol" => 20.0 => ["Cholesterol"],
        "BloodSugar" => 20.0 => ["FastingBS"],
        "HeartDisease" => 100.0,
    )

    # tiny solver to keep the test cheap
    tiny_solver = GenerativeDesigns.DPWSolver(; n_iterations = 50, tree_in_info = true)

    # ------------------------------------------------------------------
    # Test 1: severe #13 regression — Multiclass column => ArgumentError
    # ------------------------------------------------------------------
    @testset "rejects non-numeric target_condition column (severe #13)" begin
        # Build a sampler/weights/uncertainty over the typed dataset which
        # still contains the Multiclass `Sex` column.
        sub = raw_typed[!, ["Age", "RestingBP", "Sex", "HeartDisease"]]
        r2 = DistanceBased(
            sub;
            target = "HeartDisease",
            uncertainty = Variance(),
            similarity = Exponential(; λ = 5),
        )
        s2, u2, w2 = r2.sampler, r2.uncertainty, r2.weights
        # Should fail at construction with an ArgumentError naming `Sex`.
        @test_throws ArgumentError ConditionalUncertaintyReductionMDP(
            Dict("BloodPressure" => 1.0 => ["RestingBP"]);
            sampler = s2,
            uncertainty = u2,
            threshold = 0.1,
            weights = w2,
            data = sub,
            terminal_condition = (Dict("Sex" => ("M", "M")), 0.5),
        )
        # Re-trigger to inspect the message; ArgumentError should mention "Sex".
        err = try
            ConditionalUncertaintyReductionMDP(
                Dict("BloodPressure" => 1.0 => ["RestingBP"]);
                sampler = s2,
                uncertainty = u2,
                threshold = 0.1,
                weights = w2,
                data = sub,
                terminal_condition = (Dict("Sex" => ("M", "M")), 0.5),
            )
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Sex", sprint(showerror, err))
    end

    # ------------------------------------------------------------------
    # Test 2: happy path — conditional_efficient_design returns Tuple shape
    # ------------------------------------------------------------------
    @testset "conditional_efficient_design happy path" begin
        target_condition = Dict("HeartDisease" => [0.0, 1.0])
        design = conditional_efficient_design(
            experiments;
            sampler,
            uncertainty,
            threshold = 0.5,
            evidence = Evidence(),
            weights,
            data,
            terminal_condition = (target_condition, 0.0),
            solver = tiny_solver,
        )
        @test design isa Tuple
        @test length(design) == 2
        @test design[1] isa Tuple                           # (cost, threshold) pair
        @test length(design[1]) == 2
        @test hasproperty(design[2], :arrangement)          # NamedTuple metadata
    end

    # ------------------------------------------------------------------
    # Test 3: thresholds = 1 is rejected (severe #6 regression)
    # ------------------------------------------------------------------
    @testset "conditional_efficient_designs rejects thresholds = 1 (severe #6)" begin
        target_condition = Dict("HeartDisease" => [0.0, 1.0])
        @test_throws ArgumentError conditional_efficient_designs(
            experiments;
            sampler,
            uncertainty,
            thresholds = 1,
            evidence = Evidence(),
            weights,
            data,
            terminal_condition = (target_condition, 0.0),
            solver = tiny_solver,
        )
    end

    # ------------------------------------------------------------------
    # Test 4: perform_ensemble_designs key shape (M7)
    # ------------------------------------------------------------------
    @testset "perform_ensemble_designs returns Float64-keyed Dict (M7)" begin
        target_condition = Dict("HeartDisease" => [0.0, 1.0])
        thred_set = [0.5, 0.9]
        results = perform_ensemble_designs(
            experiments;
            sampler,
            uncertainty,
            thresholds = 2,
            evidence = Evidence(),
            weights,
            data,
            terminal_condition = (target_condition, 0.0),
            solver = tiny_solver,
            N = 1,
            thred_set = thred_set,
        )
        @test results isa Dict
        @test Set(keys(results)) == Set(Float64.(thred_set))
        @test all(k -> k isa Float64, keys(results))
        for tau in thred_set
            @test haskey(results, Float64(tau))
            @test length(results[Float64(tau)]) == 1   # N = 1
        end
    end

    # ------------------------------------------------------------------
    # Test 5: RNG reproducibility — seeded ensemble runs match
    # ------------------------------------------------------------------
    @testset "RNG reproducibility with seeded sampler" begin
        # Build a *fresh* solver for each run (the default solver carries
        # mutable RNG state — see M9), and seed Random.GLOBAL_RNG before
        # each call so default RNGs read from the same stream.
        target_condition = Dict("HeartDisease" => [0.0, 1.0])
        function _run_once()
            Random.seed!(42)
            solver = GenerativeDesigns.DPWSolver(;
                n_iterations = 50,
                tree_in_info = true,
                rng = Xoshiro(42),
            )
            return perform_ensemble_designs(
                experiments;
                sampler = sampler,
                uncertainty = uncertainty,
                thresholds = 2,
                evidence = Evidence(),
                weights = weights,
                data = data,
                terminal_condition = (target_condition, 0.0),
                solver = solver,
                N = 1,
                thred_set = [0.5],
            )
        end

        results_a = _run_once()
        results_b = _run_once()

        # Same keys
        @test Set(keys(results_a)) == Set(keys(results_b))
        # Same first-axis costs/thresholds across the two runs
        for k in keys(results_a)
            front_a = results_a[k][1]
            front_b = results_b[k][1]
            @test length(front_a) == length(front_b)
            for i in eachindex(front_a)
                @test front_a[i][1] == front_b[i][1]   # the (cost, threshold) Tuple
            end
        end
    end

    # ------------------------------------------------------------------
    # Test 6: empty target_condition reduces to baseline behavior
    # ------------------------------------------------------------------
    @testset "empty target_condition: terminal iff uncertainty < threshold" begin
        # Force a deterministic uncertainty function so we can drive isterminal.
        const_unc = ev -> 0.05
        # constant weights & sampler suffice — the path should not consult them.
        const_w = ev -> ones(nrow(data)) ./ nrow(data)
        const_sampler = (ev, features, rng) -> Dict(features[1] => 0.0)

        # Empty target_condition + uncertainty (0.05) <= threshold (0.1) => terminal
        mdp_terminal = ConditionalUncertaintyReductionMDP(
            Dict("BloodPressure" => 1.0 => ["RestingBP"]);
            sampler = const_sampler,
            uncertainty = const_unc,
            threshold = 0.1,
            evidence = Evidence(),
            weights = const_w,
            data = data,
            terminal_condition = (Dict(), 0.0),
        )
        @test POMDPs.isterminal(mdp_terminal, mdp_terminal.initial_state) == true

        # Same MDP but threshold below uncertainty => not terminal
        mdp_nonterminal = ConditionalUncertaintyReductionMDP(
            Dict("BloodPressure" => 1.0 => ["RestingBP"]);
            sampler = const_sampler,
            uncertainty = const_unc,
            threshold = 0.01,
            evidence = Evidence(),
            weights = const_w,
            data = data,
            terminal_condition = (Dict(), 0.0),
        )
        @test POMDPs.isterminal(mdp_nonterminal, mdp_nonterminal.initial_state) == false
    end
end
