using Test
using DataFrames
using CEED.GenerativeDesigns: DistanceBased, Evidence
using ScientificTypes
using CEED, CEED.GenerativeDesigns

# Define the types for each column
types =
    Dict(:A => Continuous, :B => Continuous, :Target1 => Multiclass, :Target2 => Continuous)

# Sample data for testing with all numerical values
data = DataFrame(; A = 1:10, B = 11:20, Target1 = rand(1:2, 10), Target2 = rand(1:10, 10))

# Coerce the data to the correct types
data = coerce(data, types)

# Define a dummy uncertainty function
dummy_uncertainty(data; prior) = weights -> sum(weights)

# Define a dummy similarity function
dummy_similarity = x -> exp(-sum(x))

# Define target constraints for importance sampling with numerical conditions
target_constraints =
    Dict("Target1" => x -> x .== 1 ? 2.0 : 1.0, "Target2" => x -> x .> 5 ? 1.5 : 1.0)

# Define desirable ranges for each dimension
desirable_range = Dict("A" => (3, 7), "B" => (15, 18))
# Create the DistanceBased function with the new features
distance_based_result = DistanceBased(
    data,
    ["Target1", "Target2"],
    dummy_uncertainty,
    dummy_similarity,
    Dict();
    prior = ones(nrow(data)),
    desirable_range = desirable_range,
    importance_sampling = true,
    target_constraints = target_constraints,
)

# Test the weights computation considering the desirable range and importance sampling
@testset "DistanceBased Function Tests" begin
    evidence = Evidence("A" => 5, "B" => 16)
    weights = distance_based_result.weights(evidence)

    # Check if weights are zero outside the desirable range
    @test all(weights[1:2] .== 0.0)
    @test all(weights[8:10] .== 0.0)

    # Check if weights are adjusted according to the target constraints
    @test weights[3] >= weights[4]  # Assuming "Good" is more frequent in the first half
end
