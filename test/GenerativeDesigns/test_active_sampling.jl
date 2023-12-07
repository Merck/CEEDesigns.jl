using Test
using DataFrames
using CEEDesigns.GenerativeDesigns: DistanceBased, Evidence
using ScientificTypes
using CEEDesigns, CEEDesigns.GenerativeDesigns

# Define the types for each column
types = Dict(:A => Continuous, :B => Continuous, :Y => Continuous)

# Sample data for testing with all numerical values
data = DataFrame(; A = 1:10, B = 1:10, Y = rand(1:10, 10))

# Coerce the data to the correct types
data = coerce(data, types)

# Define a dummy uncertainty function
dummy_uncertainty(data; prior) = weights -> sum(weights)

# Define a dummy similarity function
dummy_similarity = x -> exp(-sum(x))

# Define desirable ranges for dimension "A"
filter_range = Dict("A" => (3, 8))
# Define importance weights for dimension "B"
importance_weights = Dict("B" => x -> 2 <= x <= 7)

# Create the DistanceBased function with the new features
(; weights) = DistanceBased(
    data;
    target = ["Y"],
    uncertainty = dummy_uncertainty,
    similarity = dummy_similarity,
    filter_range,
    importance_weights,
)

# Test the weights computation considering the desirable range and importance sampling
@testset "`DistanceBased` tests for active sampling" begin
    evidence_1 = Evidence("A" => 5)
    assigned_weights_1 = weights(evidence_1)

    # Check if weights are zero only in the given range
    @test all(assigned_weights_1[1:2] .== 0.0)
    @test all(assigned_weights_1[9:10] .== 0.0)
    @test all(assigned_weights_1[3:8] .> 0)

    evidence_2 = Evidence("B" => 5)
    assigned_weights_2 = weights(evidence_2)
    @show assigned_weights_2

    # Check if weights are zero only in the given range
    @test all(assigned_weights_2[1] == 0.0)
    @test all(assigned_weights_2[8:10] .== 0)
    @test all(assigned_weights_2[2:7] .> 0.0)
end
