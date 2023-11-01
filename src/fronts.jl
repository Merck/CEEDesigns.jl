"""
    front(v)
    front(f, v; atol1=0, atol2=0)

Construct a Pareto front of `v`. Elements of `v` will be masked by `f` in the computation.

The first and second (objective) coordinates have to differ by at least `atol1`, `atol2`, respectively, relatively to the latest point on the front.

# Examples

```jldocstest
v = [(1,2), (2,3), (2,1)]
front(v)

# output
[(1, 2), (2, 1)]
```

```jldoctest
v = [(1, (1, 2)), (2, (2, 3)), (3, (2, 1))]
front(x -> x[2], v)

# output

[(1, (1, 2)), (3, (2, 1))]
```

```jldoctest
v = [(1, 2), (2, 1.99), (3, 1)]
front(v; atol2 = 0.2)

# output

[(1, 2), (3, 1)]
```
"""
function front end

function front(v::T; atol1::Float64 = 0.0, atol2::Float64 = atol1) where {T<:AbstractVector}
    front(identity, v; atol1, atol2)
end

function front(
    f::F,
    v::T;
    atol1::Float64 = 0.0,
    atol2::Float64 = atol1,
) where {F<:Function,T<:AbstractVector}
    # dict sort
    v_sorted = sort(
        v;
        lt = (x, y) ->
            (f(x)[1] < f(y)[1] || (f(x)[1] == f(y)[1] && f(x)[2] <= f(y)[2])),
    )

    # check if the second coordinate drops below the second coordinate of the last non-dominated point
    ix_front = 1
    ix_current = 2
    while ix_current <= length(v_sorted)
        if (f(v_sorted[ix_current])[2] < f(v_sorted[ix_front])[2] - atol2) &&
           (f(v_sorted[ix_current])[1] > f(v_sorted[ix_front])[1] + atol1)
            ix_front = ix_current
            ix_current += 1
        else
            deleteat!(v_sorted, ix_current)
        end
    end

    v_sorted
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
    for i = 2:length(designs)
        if xs[i] < 10_000
            scatter!(p, [xs[i]], [ys[i]]; label = labels[i], c = grad[i], mscolor = nothing)
        end
    end

    p
end

"""
    make_labels(designs)

Make labels used plotting of experimental designs.
"""
function make_labels(designs)
    return map(designs) do x
        if !haskey(x[2], :arrangement) || isempty(x[2].arrangement)
            "∅"
        else
            labels =
                ["$i: " * join(group, ", ") for (i, group) in enumerate(x[2].arrangement)]

            join(labels, "; ")
        end
    end
end

"""
    plot_evals(eval; ylabel="information measure")

Create a stick plot that visualizes the performance measures evaluated for subsets of experiments.
"""
function plot_evals(evals; ylabel = "information measure", kwargs...)
    xs = sort!(collect(keys(evals)); by = x -> length(x))
    ys = map(xs) do x
        evals[x] isa Number ? evals[x] : evals[x].loss
    end
    xformatter = i -> isempty(xs[Int(i)]) ? "∅" : join(xs[Int(i)], ", ")

    sticks(
        1:length(evals),
        ys;
        ticks = 1:length(evals),
        xformatter,
        guidefontsize = 8,
        tickfontsize = 8,
        ylabel,
        c = :teal,
        label = nothing,
        xrotation = 30,
    )
end
