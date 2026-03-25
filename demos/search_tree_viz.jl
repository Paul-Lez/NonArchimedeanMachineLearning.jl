#!/usr/bin/env julia
# Quick test: DAG-MCTS on a 3-variable LinearAbsolutePolynomialSum, then visualize the search tree.

using Oscar
include("../src/NAML.jl")
using .NAML

using D3Trees

# Setup: p=2, precision=20
K = PadicField(2, 20)

# LinearAbsolutePolynomialSum: |a + b - 4| + |a - b + 2| + |c - a - 2|
# Variables are (a, b, c). Unique root at (1, 3, 3) — all p-adic integers.
l1 = LinearPolynomial([K(1), K(1), K(0)], K(-4))    # a + b - 4
l2 = LinearPolynomial([K(1), K(-1), K(0)], K(2))    # a - b + 2
l3 = LinearPolynomial([K(-1), K(0), K(1)], K(-2))   # c - a - 2
f = LinearAbsolutePolynomialSum([l1, l2, l3])

# Build loss directly from the function (no model/data needed)
VP = ValuationPolydisc{ValuedFieldPoint{2,20,PadicFieldElem}, Int, 3}
batch_eval = batch_evaluate_init(f, VP)
loss = Loss(
    params -> map(batch_eval, params),
    vs -> [directional_derivative(batch_eval, v) for v in vs],
)

# Initial parameter polydisc for (a, b, c)
param = ValuationPolydisc([K(0), K(0), K(0)], [0, 0, 0])

config = DAGMCTSConfig(
    num_simulations=3000,
    exploration_constant=1.41,
    degree=1,
    persist_table=true,
    selection_mode=BestValue,
)

optim = dag_mcts_descent_init(param, loss, config)

println("Running DAG-MCTS for 20 epochs...")
for i in 1:1
    step!(optim)
    println("  Epoch $i: loss = $(eval_loss(optim))")
end

println("\nBuilding D3Tree visualization...")
tree = visualize_search_tree(optim; max_depth=6, init_expand=2, svg_node_size=(120, 60))
inbrowser(tree, "Google Chrome")
println("  Nodes in tree: $(length(tree.children))")
println("  Root text: $(tree.text[1])")
println("  Root tooltip:\n    ", replace(tree.tooltip[1], "\n" => "\n    "))

println("\nDone! In a notebook you can just display `tree`, or call `inbrowser(tree)` to open it.")
