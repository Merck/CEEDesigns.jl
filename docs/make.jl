using Documenter, DocumenterMarkdown, Literate
using CEEDesigns

# Literate for tutorials
const literate_dir = joinpath(@__DIR__, "..", "tutorials")
const tutorials_src = [
    "SimpleStatic.jl",
    "StaticDesigns.jl",
    "StaticDesignsFiltration.jl",
    "GenerativeDesigns.jl",
]
const generated_dir = joinpath(@__DIR__, "src", "tutorials/")

# copy tutorials src
for file in tutorials_src
    println(joinpath(literate_dir, file), generated_dir)
    cp(joinpath(literate_dir, file), joinpath(generated_dir, file); force = true)
end

for dir in ["assets", "data"]
    cp(joinpath(literate_dir, dir), joinpath(generated_dir, dir); force = true)
end

for file in tutorials_src
    Literate.markdown(
        joinpath(generated_dir, file),
        generated_dir;
        documenter = true,
        credit = false,
    )
end

pages = [
    "index.md",
    "Tutorials" => [
        "tutorials/SimpleStatic.md",
        "tutorials/StaticDesigns.md",
        "tutorials/StaticDesignsFiltration.md",
        "tutorials/GenerativeDesigns.md",
    ],
    "api.md",
]

makedocs(;
    sitename = "CEEDesigns.jl",
    format = Documenter.HTML(;
        prettyurls = false,
        edit_link = "main",
        assets = ["assets/favicon.ico"],
    ),
    pages,
)

deploydocs(; repo = "github.com/Merck/CEEDesigns.jl.git")
