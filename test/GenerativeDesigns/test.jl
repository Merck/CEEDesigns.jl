# test with the sum distances
include("test_distances_sum.jl")

# regression tests for pre-JOSS review fixes (S1/C1/C2/M1/M4/N3)
include("test_review_fixes.jl")

# test with the squared Mahalanobis distance
include("test_mahalanobis.jl")

# test active sampling
include("test_active_sampling.jl")

include("test_conditional_constraints.jl")

# regression / unit tests for `EfficientValueMDP`
include("test_efficient_value.jl")

# regression / unit tests for `ConditionalUncertaintyReductionMDP`
include("test_conditional.jl")
