begin
    using Plots
    plotly()
    evals1 = filter(x -> "histology" ∉ x[1], perf_eval)
    xs1 = sort!(collect(keys(evals1)); by = x -> -perf_eval[x])
    ys1 = map(xs1) do x
        return perf_eval[x] isa Number ? perf_eval[x] : perf_eval[x].loss
    end

    evals2 = filter(x -> "histology" ∈ x[1], perf_eval)
    xs2 = sort!(collect(keys(evals2)); by = x -> -perf_eval[setdiff(x, ["histology"])])
    ys2 = map(xs2) do x
        return perf_eval[x] isa Number ? perf_eval[x] : perf_eval[x].loss
    end

    p_evals = Plots.sticks(
        1:length(xs2),
        1 .- min.(ys1, ys2);
        ticks = 1:length(evals2),
        #xformatter=i -> isempty(xs2[Int(i)]) ? "∅" : join(xs1[Int(i)], ", "),
        guidefontsize = 8,
        tickfontsize = 8,
        ylabel = "accuracy",
        c = CEED.colorant"rgb(110,206,178)",
        label = "w/ histology",
        xrotation = 50,
    )

    Plots.sticks!(
        p_evals,
        1:length(xs1),
        1 .- ys1;
        ticks = 1:length(evals1),
        xformatter = i -> isempty(xs1[Int(i)]) ? "∅" : join(xs1[Int(i)], ", "),
        guidefontsize = 8,
        tickfontsize = 8,
        ylabel = "accuracy",
        c = CEED.colorant"rgb(104,140,232)",
        label = "w/o histology",
        width = 2,
        xrotation = 50,
    )
end
