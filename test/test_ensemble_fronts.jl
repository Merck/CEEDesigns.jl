using Test
using DataFrames
using CEEDesigns
using CEEDesigns: ensemble_to_dataframe, plot_ensemble_pareto, _compute_dispersion

# Build a synthetic "run" matching the shape produced by the GenerativeDesigns
# MDP wrappers (`efficient_design` / `efficient_value`):
#   front  ::Vector of designs
#   design ::Tuple{Tuple{Utility,Threshold}, NamedTuple{(:arrangement, ...)}}
#   arrangement::Vector{Vector{String}}  (each step wraps the action set)
function make_design(utility, threshold, arrangement)
    return ((utility, threshold), (; arrangement = arrangement))
end

@testset "ensemble_to_dataframe label cleanliness (M6)" begin
    # One run with a single design whose arrangement matches the documented
    # shape `Vector{Vector{String}}`.
    arrangement = [["BloodPressure"], ["ECG", "Glucose"]]
    front = [make_design(1.5, 0.4, arrangement)]
    runs = [front]

    df = ensemble_to_dataframe(runs)

    @test nrow(df) == 1
    @test df.Action_Set[1] == "BloodPressure,ECG,Glucose"
    # Defensive: must NOT contain the bracket-laden Vector form.
    @test !occursin('[', df.Action_Set[1])
    @test !occursin(']', df.Action_Set[1])
    @test !occursin('"', df.Action_Set[1])

    # Equivalence: two runs whose actions appear in different order should
    # collapse onto the same Action_Set key.
    runs2 = [
        [make_design(1.0, 0.4, [["B"], ["A"]])],
        [make_design(2.0, 0.4, [["A"], ["B"]])],
    ]
    df2 = ensemble_to_dataframe(runs2)
    @test nrow(df2) == 1
    @test df2.Action_Set[1] == "A,B"
    @test df2.Frequency[1] == 2
    @test df2.Average_Utility[1] == 1.5
end

@testset "_compute_dispersion distributes duplicate Average_Utility (M11)" begin
    # Two rows with identical Average_Utility under the same threshold must NOT
    # both be assigned the slot-1 offset.
    df = DataFrame(;
        Threshold = [0.5, 0.5, 0.5],
        Action_Set = ["A", "B", "C"],
        Average_Utility = [1.0, 1.0, 2.0],
        Frequency = [3, 1, 2],
    )

    dispersion_offsets, x_coords, action_counts = _compute_dispersion(df)

    @test action_counts[0.5] == 3
    @test length(dispersion_offsets[0.5]) == 3
    # All three offset slots are distinct (the helper produces a
    # `range(-max_offset, max_offset; length = n_actions)` of length 3).
    @test length(unique(dispersion_offsets[0.5])) == 3
    # The x_coords vector must produce three distinct values, even though two
    # rows share Average_Utility. This is the regression: the old findfirst +
    # float-equality logic collapsed both duplicates onto offset slot 1, so
    # x_coords[1] == x_coords[2]. With the per-threshold counter fix they
    # consume slots 1 and 2 respectively.
    @test length(x_coords) == 3
    @test x_coords[1] != x_coords[2]
end

@testset "plot_ensemble_pareto guards empty input (M12)" begin
    empty_df = DataFrame(;
        Threshold = Float64[],
        Action_Set = String[],
        Average_Utility = Float64[],
        Frequency = Int[],
    )
    @test_throws ArgumentError plot_ensemble_pareto(empty_df, 0.5)
end
