using Test
using DataFrames
using ScientificTypes
import POMDPs
using CEEDesigns, CEEDesigns.GenerativeDesigns

@testset "Pre-JOSS review fixes" begin

    # ---- S1: DiscreteDistance is a proper distance (match -> 0, mismatch -> λ) ----
    @testset "S1 DiscreteDistance direction" begin
        d = DiscreteDistance(; λ = 1)
        @test d("M", ["M", "F", "M"]) == [0.0, 1.0, 0.0]

        data = DataFrame(; target = [1.0, 2.0, 3.0, 4.0], Sex = ["M", "M", "F", "F"])
        coerce!(data, :Sex => Multiclass)
        (; weights) = DistanceBased(
            data;
            target = "target",
            uncertainty = Variance(),
            similarity = Exponential(; λ = 1.0),
            distance = Dict("Sex" => DiscreteDistance()),
        )
        w = weights(Evidence("Sex" => "M"))
        # rows 1,2 match the query (Sex=M) -> must be weighted strictly higher
        @test sum(w[1:2]) > sum(w[3:4])
    end

    # ---- C1: terminal_condition accepts Symbol keys (no runtime crash) ----
    @testset "C1 terminal_condition Symbol keys" begin
        data = DataFrame(; target = [0.2, 0.7, 0.9], x = [1, 2, 3])
        weights_fn = ev -> [0.2, 0.3, 0.5]
        unc = ev -> 0.0
        sampler = (ev, f, rng) -> Dict(f[1] => 1.0)
        costs = Dict("A" => 1.0)
        mdp = ConditionalUncertaintyReductionMDP(
            costs;
            sampler,
            uncertainty = unc,
            threshold = 0.1,
            weights = weights_fn,
            data = data,
            terminal_condition = (Dict(:target => [0.5, 1.0]), 0.75),  # SYMBOL key
        )
        # admissible mass = 0.3 + 0.5 = 0.8 >= 0.75 -> terminal, and must not throw
        @test POMDPs.isterminal(mdp, mdp.initial_state) == true
    end

    # ---- C2: Union{Missing,Real} constraint column accepted; missing excluded ----
    @testset "C2 Union{Missing} constraint column" begin
        data = DataFrame(; target = Union{Missing, Float64}[0.2, 0.7, missing], x = [1, 2, 3])
        weights_fn = ev -> [0.2, 0.3, 0.5]
        unc = ev -> 0.0
        sampler = (ev, f, rng) -> Dict(f[1] => 1.0)
        costs = Dict("A" => 1.0)
        mdp = ConditionalUncertaintyReductionMDP(
            costs;
            sampler,
            uncertainty = unc,
            threshold = 0.1,
            weights = weights_fn,
            data = data,
            terminal_condition = (Dict("target" => [0.5, 1.0]), 0.2),
        )
        @test mdp isa ConditionalUncertaintyReductionMDP
        # row1=0.2 (out), row2=0.7 (in -> 0.3), row3=missing (excluded)
        p = conditional_likelihood(
            Evidence();
            compute_weights = weights_fn,
            hist_data = data,
            target_condition = Dict("target" => (0.5, 1.0)),
        )
        @test p ≈ 0.3
    end

    # ---- M1: max_experiments counts experiments, not evidence entries ----
    @testset "M1 max_experiments counts experiments" begin
        sampler = (ev, f, rng) -> Dict(f[1] => 1.0)
        unc = ev -> 1.0  # never terminal via uncertainty
        costs = Dict("ECG" => ((1.0, 0.5) => ["MaxHR", "RestingECG"]))
        mdp = UncertaintyReductionMDP(
            costs;
            sampler,
            uncertainty = unc,
            threshold = 0.5,
            evidence = Evidence("Age" => 35, "Sex" => "M"),
            max_experiments = 2,
        )
        acts = POMDPs.actions(mdp, mdp.initial_state)
        @test acts != [["EOX"]]                 # 2 prior-evidence entries must NOT exhaust budget
        @test any(a -> "ECG" in a, acts)
    end

    # ---- M4: constant column raises a clear ArgumentError ----
    @testset "M4 constant column ArgumentError" begin
        data = DataFrame(; target = [1.0, 2.0, 3.0], c = [5.0, 5.0, 5.0])
        (; weights) = DistanceBased(
            data;
            target = "target",
            uncertainty = Variance(),
            distance = Dict("c" => QuadraticDistance()),
        )
        @test_throws ArgumentError weights(Evidence("c" => 5.0))
    end

    # ---- N3: importance_weights length is validated ----
    @testset "N3 importance_weights length" begin
        data = DataFrame(; target = [1.0, 2.0, 3.0], c = [5.0, 6.0, 7.0])
        @test_throws ArgumentError DistanceBased(
            data;
            target = "target",
            uncertainty = Variance(),
            importance_weights = Dict("c" => [1.0, 2.0]),  # length 2 != 3 rows
        )
    end

end
