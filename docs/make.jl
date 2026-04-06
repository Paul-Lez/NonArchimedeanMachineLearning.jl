using Documenter, NAML

makedocs(
    sitename="NAML.jl",
    modules=[NAML],
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
                    "Flat UCB" => "optimization/optimizers/tree_search/flat_ucb.md",
                    "HOO" => "optimization/optimizers/tree_search/hoo.md",
                    "MCTS" => "optimization/optimizers/tree_search/mcts.md",
                    "Modified UCT" => "optimization/optimizers/tree_search/modified_uct.md",
                    "UCT" => "optimization/optimizers/tree_search/uct.md",
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
