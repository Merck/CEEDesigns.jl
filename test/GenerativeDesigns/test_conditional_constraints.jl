using Test
using DataFrames
using Random
import POMDPs
using CEEDesigns.GenerativeDesigns


@testset "Conditional constraints MDP" begin
    # tiny historical data
    data = DataFrame(target = [0.2, 0.7, 0.9], x = [1, 2, 3])

    # fixed weights over rows (must sum to 1 for probability interpretation)
    weights_fn = evidence -> [0.2, 0.3, 0.5]

    # uncertainty function (simple constant)
    uncertainty_fn = evidence -> 0.0  # always below threshold

    # sampler returns a Dict (CEED style)
    sampler_fn = (evidence, features, rng) -> Dict(features[1] => 1.0)

    costs = Dict("A" => 1.0)

    # constraint: target in [0.5, 1.0] => rows 2 and 3 => weight = 0.3 + 0.5 = 0.8
    target_condition = Dict("target" => [0.5, 1.0])

    mdp = ConditionalUncertaintyReductionMDP(
        costs;
        sampler = sampler_fn,
        uncertainty = uncertainty_fn,
        threshold = 0.1,
        evidence = Evidence(),
        weights = weights_fn,
        data = data,
        terminal_condition = (target_condition, 0.75),  # tau = 0.75, should pass
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
        terminal_condition = (target_condition, 0.95),  # tau too high, should fail
    )

    @test POMDPs.isterminal(mdp2, mdp2.initial_state) == false
end
