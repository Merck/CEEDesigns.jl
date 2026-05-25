"""
    front(v)
    front(f, v; atol1=0, atol2=0)

Construct the Pareto front of `v` under joint minimization on the two
objectives `f(x)[1]` and `f(x)[2]`. When `f` is omitted, `identity` is used,
so each element of `v` is taken to be a `(c1, c2)` pair directly.

A point `p` is on the front iff no other point strictly dominates it: there
is no `q ∈ v` with `f(q)[1] ≤ f(p)[1]` and `f(q)[2] ≤ f(p)[2]` and at least
one inequality strict. Points with identical `(c1, c2)` are mutually
non-dominated and are all retained.

The optional tolerances `atol1`, `atol2` thin the front *after* the
dominance pass: a candidate is kept only if its first coordinate exceeds the
last kept point's by at least `atol1` *and* its second coordinate is at
least `atol2` smaller. With both tolerances `0` (the default), no thinning
occurs and the result is the exact Pareto front.

# Examples

```jldoctest
v = [(1, 2), (2, 3), (2, 1)]
front(v)

# output

2-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (2, 1)
```

```jldoctest
v = [(1, (1, 2)), (2, (2, 3)), (3, (2, 1))]
front(x -> x[2], v)

# output

2-element Vector{Tuple{Int64, Tuple{Int64, Int64}}}:
 (1, (1, 2))
 (3, (2, 1))
```

```jldoctest
v = [(1, 2), (2, 1.99), (3, 1)]
front(v; atol2 = 0.2)

# output

2-element Vector{Tuple{Int64, Float64}}:
 (1, 2.0)
 (3, 1.0)
```
"""
function front end

function front(v::T; atol1::Float64 = 0.0, atol2::Float64 = atol1) where {T <: AbstractVector}
    return front(identity, v; atol1, atol2)
end

function front(
        f::F,
        v::T;
        atol1::Float64 = 0.0,
        atol2::Float64 = atol1,
    ) where {F <: Function, T <: AbstractVector}
    # Sort lexicographically with a strict-weak comparator.
    v_sorted = sort(
        v;
        lt = (x, y) -> begin
            fx, fy = f(x), f(y)
            fx[1] == fy[1] ? fx[2] < fy[2] : fx[1] < fy[1]
        end,
    )

    # Online Pareto front (minimization on both axes). Because `v_sorted` is
    # ordered by ascending first coordinate then ascending second, a candidate
    # is non-dominated by all previously kept points iff its second coordinate
    # is strictly below the running minimum, OR it ties exactly with the most
    # recently kept point on both coordinates (mutually non-dominated tie).
    pareto = empty(v_sorted)
    min_y = Inf
    for p in v_sorted
        fp = f(p)
        if isempty(pareto)
            push!(pareto, p)
            min_y = fp[2]
        elseif fp[2] < min_y
            push!(pareto, p)
            min_y = fp[2]
        elseif fp[2] == min_y
            flast = f(pareto[end])
            if flast[1] == fp[1]
                push!(pareto, p)
            end
        end
    end

    # Tolerance thinning: keep a candidate only if it is meaningfully separated
    # from the last kept front point on both axes. With atol1 = atol2 = 0 (the
    # default), this pass is a no-op.
    if atol1 == 0.0 && atol2 == 0.0
        return pareto
    end
    result = empty(pareto)
    for p in pareto
        if isempty(result)
            push!(result, p)
            continue
        end
        fp = f(p)
        flast = f(result[end])
        if (fp[1] - flast[1]) >= atol1 && (flast[2] - fp[2]) >= atol2
            push!(result, p)
        end
    end
    return result
end

"""
    plot_front(designs; grad=cgrad(:Paired_12), xlabel, ylabel, labels=get_labels(designs))

Render scatter plot of efficient designs, as returned from `efficient_designs`.

You may optionally specify a color gradient, to draw the colors from.

# Examples

```julia
designs = efficient_designs(experiment, state)
plot_front(designs)
plot_front(designs; grad = cgrad(:Paired_12))
```
"""
function plot_front(
        designs;
        grad = cgrad(:Paired_12),
        xlabel = "combined cost",
        ylabel = "information measure",
        labels = make_labels(designs),
    )
    xs = map(x -> x[1][1], designs)
    ys = map(x -> x[1][2], designs)

    p = scatter(
        [xs[1]],
        [ys[1]];
        xlabel,
        ylabel,
        label = labels[1],
        c = grad[1],
        mscolor = nothing,
        fontsize = 16,
    )
    for i in 2:length(designs)
        scatter!(p, [xs[i]], [ys[i]]; label = labels[i], c = grad[i], mscolor = nothing)
    end

    return p
end

"""
    make_labels(designs)

Make labels used plotting of experimental designs.
"""
function make_labels(designs)
    return map(designs) do x
        if x[2] isa AbstractDict
            arrangement = get(x[2], "arrangement", nothing)
        elseif x[2] isa NamedTuple && haskey(x[2], :arrangement)
            arrangement = x[2].arrangement
        else
            arrangement = nothing
        end

        if isnothing(arrangement)
            "∅"
        else
            labels = ["$i: " * join(group, ", ") for (i, group) in enumerate(arrangement)]

            join(labels, "; ")
        end
    end
end

"""
    plot_evals(evals; f, ylabel="information measure")

Create a stick plot that visualizes the performance measures evaluated for subsets of experiments.

Argument `evals` should be the output of [`evaluate_experiments`](@ref CEEDesigns.StaticDesigns.evaluate_experiments) and the kwarg `f` (if provided) is a function that
should take as input `evals` and return a list of its keys in the order to be plotted on the x-axis.
By default they are sorted by length.
"""
function plot_evals(
        evals;
        ylabel = "information measure",
        f = x -> sort(collect(keys(x)); by = length),
    )
    xs = f(evals)
    ys = map(xs) do x
        return evals[x] isa Number ? evals[x] : evals[x].loss
    end
    xformatter = i -> isempty(xs[Int(i)]) ? "∅" : join(xs[Int(i)], ", ")

    return sticks(
        1:length(evals),
        ys;
        ticks = 1:length(evals),
        xformatter,
        guidefontsize = 8,
        tickfontsize = 8,
        ylabel,
        label = nothing,
        xrotation = 30,
    )
end
