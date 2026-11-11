using Documenter, NonArchimedeanMachineLearning

makedocs(
    sitename="NonArchimedeanMachineLearning.jl",
    modules=[NonArchimedeanMachineLearning],
    warnonly=true,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        size_threshold=400000 # 400 KB
    ),
    pages=include("pages.jl")
)

deploydocs(
    repo="github.com/Paul-Lez/NonArchimedeanMachineLearning.jl.git",
    devbranch="main",
    push_preview=true
)
