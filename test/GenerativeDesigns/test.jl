# test with the sum distances
include("test_distances_sum.jl")

# test with the squared Mahalanobis distance
include("test_mahalanobis.jl")

# test active sampling
include("test_active_sampling.jl")

include("test_conditional_constraints.jl")

# regression / unit tests for `EfficientValueMDP`
include("test_efficient_value.jl")
