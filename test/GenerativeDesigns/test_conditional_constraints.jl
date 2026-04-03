using Test
using DataFrames
using Random
import POMDPs
using CEEDesigns.GenerativeDesigns

@testset "Conditional constraints MDP" begin
    # tiny historical data
    data = DataFrame(target = [0.2, 0.7, 0.9], x = [1, 2, 3])

    # fixed posterior weights over rows
    weights_fn = evidence -> [0.2, 0.3, 0.5]

    # uncertainty function
    uncertainty_fn = evidence -> 0.0  # always below threshold

    # sampler returns a Dict (CEED style)
    sampler_fn = (evidence, features, rng) -> Dict(features[1] => 1.0)

    costs = Dict("A" => 1.0)

    # constraint: target in [0.5, 1.0] => rows 2 and 3 => weight = 0.3 + 0.5 = 0.8
    target_condition = Dict("target" => [0.5, 1.0])

    @testset "conditional_likelihood computes posterior mass in admissible range" begin
        p = conditional_likelihood(
            Evidence();
            compute_weights = weights_fn,
            hist_data = data,
            target_condition = target_condition,
        )
        @test p ≈ 0.8
    end

    @testset "terminal condition respects tau threshold" begin
        mdp = ConditionalUncertaintyReductionMDP(
            costs;
            sampler = sampler_fn,
            uncertainty = uncertainty_fn,
            threshold = 0.1,
            evidence = Evidence(),
            weights = weights_fn,
            data = data,
            terminal_condition = (target_condition, 0.75),  # should pass
        )

        @test POMDPs.isterminal(mdp, mdp.initial_state) == true

        mdp2 = ConditionalUncertaintyReductionMDP(
            costs;
            sampler = sampler_fn,
            uncertainty = uncertainty_fn,
            threshold = 0.1,
            evidence = Evidence(),
            weights = weights_fn,
            data = data,
            terminal_condition = (target_condition, 0.95),  # should fail
        )

        @test POMDPs.isterminal(mdp2, mdp2.initial_state) == false
    end

    @testset "transition updates evidence" begin
        mdp = ConditionalUncertaintyReductionMDP(
            costs;
            sampler = sampler_fn,
            uncertainty = evidence -> 1.0,  # not terminal
            threshold = 0.1,
            evidence = Evidence(),
            weights = weights_fn,
            data = data,
            terminal_condition = (target_condition, 0.95),
        )

        dist = POMDPs.transition(mdp, mdp.initial_state, ["A"])
        next_state = rand(MersenneTwister(1), dist)

        @test haskey(next_state.evidence, "A")
        @test next_state.evidence["A"] == 1.0
        @test next_state.costs == (1.0, 0.0)
    end

    @testset "reward is incremental" begin
        mdp = ConditionalUncertaintyReductionMDP(
            costs;
            sampler = sampler_fn,
            uncertainty = uncertainty_fn,
            threshold = 0.1,
            evidence = Evidence(),
            weights = weights_fn,
            data = data,
            terminal_condition = (target_condition, 0.75),
            costs_tradeoff = (1.0, 1.0),
        )

        previous_state = State((Evidence(), (1.0, 2.0)))
        state = State((merge(Evidence(), Dict("A" => 1.0)), (4.0, 5.0)))

        @test POMDPs.reward(mdp, previous_state, ["A"], state) == -6.0
    end
end
