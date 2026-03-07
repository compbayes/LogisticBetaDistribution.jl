using Pkg

# Activate the docs environment in docs/Project.toml
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()  # optional but good to keep

@show Base.active_project()  # temporary debug, can be removed later

using LogisticBetaDistribution
using Documenter

DocMeta.setdocmeta!(LogisticBetaDistribution, :DocTestSetup, :(using LogisticBetaDistribution); recursive=true)

makedocs(;
    sitename="LogisticBetaDistribution.jl",
    modules=[LogisticBetaDistribution],
    authors="Mattias Villani",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://compbayes.github.io/LogisticBetaDistribution.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "home.md",
        "Implemented methods" => "methods.md",
        "Index" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/compbayes/LogisticBetaDistribution.jl",
)
