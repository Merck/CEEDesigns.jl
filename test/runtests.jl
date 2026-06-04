using SafeTestsets, BenchmarkTools

@time begin
    @time @safetestset "fronts tests" begin
        include("fronts.jl")
    end
    @time @safetestset "ensemble_fronts tests" begin
        include("test_ensemble_fronts.jl")
    end
    @time @safetestset "`StaticDesigns` tests" begin
        include("StaticDesigns/test.jl")
    end
    @time @safetestset "`GenerativeDesigns` tests" begin
        include("GenerativeDesigns/test.jl")
    end
end
