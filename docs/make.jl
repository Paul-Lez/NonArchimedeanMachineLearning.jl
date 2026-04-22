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
    pages=[
        "Home" => "index.md",
        "Basic" => [
            "Functions" => "basic/functions.md",
            "Polydisc" => "basic/polydisc.md",
            "Tangent Vector" => "basic/tangent_vector.md",
            "Valuation" => "basic/valuation.md",
            "Valued Point" => "basic/valued_point.md",
        ],
        "Optimization" => [
            "Loss" => "optimization/loss.md",
            "Model" => "optimization/model.md",
            "Optim Setup" => "optimization/optim_setup.md",
            "Optimizers" => [
                "Gradient Descent" => "optimization/optimizers/gradient_descent.md",
                "Greedy Descent" => "optimization/optimizers/greedy_descent.md",
                "Random Descent" => "optimization/optimizers/random_descent.md",
                "Tree Search" => [
                    "DAG-MCTS" => "optimization/optimizers/tree_search/dag_mcts.md",
                    "DOO" => "optimization/optimizers/tree_search/doo.md",
                    "MCTS" => "optimization/optimizers/tree_search/mcts.md",
                    "Value Transforms" => "optimization/optimizers/tree_search/value_transforms.md",
                ]
            ]
        ],
        "Statistics" => [
            "Frechet" => "statistics/frechet.md",
            "Least Squares" => "statistics/least_squares.md",
        ],
        "Visualization" => [
            "Loss Landscape" => "visualization/loss_landscape.md",
            "Search Tree Viz" => "visualization/search_tree_viz.md",
        ]
    ]
)

deploydocs(
    repo="github.com/Paul-Lez/NAMLOptimization.jl.git",
    devbranch="main",
    push_preview=true
)
