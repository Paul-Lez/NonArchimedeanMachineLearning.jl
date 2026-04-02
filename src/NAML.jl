module NAML

using Oscar
using LinearAlgebra
using Printf

# Include all source files
include("basic/valuation.jl")
include("basic/valued_point.jl")
include("basic/polydisc.jl")
include("basic/tangent_vector.jl")
include("basic/functions.jl")
include("optimization/model.jl")
include("optimization/optim_setup.jl")
include("optimization/optimizers/gradient_descent.jl")
include("optimization/optimizers/greedy_descent.jl")
include("optimization/optimizers/random_descent.jl")
include("optimization/loss.jl")
include("optimization/optimizers/tree_search/value_transforms.jl")
include("optimization/optimizers/tree_search/hoo.jl")
include("optimization/optimizers/tree_search/mcts.jl")
include("optimization/optimizers/tree_search/uct.jl")
include("optimization/optimizers/tree_search/modified_uct.jl")
include("optimization/optimizers/tree_search/flat_ucb.jl")
include("optimization/optimizers/tree_search/doo.jl")
include("optimization/optimizers/tree_search/dag_mcts.jl")
include("statistics/frechet.jl")
include("statistics/least_squares.jl")
include("visualization/loss_landscape.jl")
include("visualization/search_tree_viz.jl")

# Export types and functions

# From basic/valuation.jl
export valuation, unit

# From basic/valued_point.jl
export ValuedFieldPoint
export unwrap, lift
# Note: prime and precision are exported via polydisc.jl (prime) and Base (precision)

# From basic/polydisc.jl
export ValuationPolydisc, AbsPolydisc
export center, radius, dim, prime
# Note: join is not exported to avoid conflict with Base.join - use NAML.join explicitly
export dist, children, children_along_branch, concatenate
export canonical_center  # For computing canonical polydisc representation (used by hash)

# From basic/tangent_vector.jl
export ValuationTangent
# Note: zero and basis_vector not exported to avoid conflicts with Base - use NAML.zero, NAML.basis_vector

# From basic/functions.jl
export PolydiscFunction, AbsolutePolynomialSum, LinearAbsolutePolynomialSum, LinearPolynomial
export DifferentiableFunction
export PolydiscFunctionEvaluator  # Abstract evaluator type
export LinearPolynomialEvaluator, ConstantEvaluator
export AddEvaluator, SubEvaluator, MulEvaluator, DivEvaluator
export SMulEvaluator, CompEvaluator, SumEvaluator
export LambdaEvaluator, MPolyEvaluator
export directional_exponent, directional_derivative, grad, eval_abs
# Note: evaluate not exported to avoid conflicts with Oscar/AbstractAlgebra - use NAML.evaluate

# From optimization/model.jl
export AbstractModel, Model, ModelEvaluator
export var_indices, param_indices, set_abstract_model_variable, batch_evaluate_init

# From optimization/optim_setup.jl
export Loss, OptimSetup
export eval_loss, update_param!, step!, has_converged, optimize!

# From optimization/loss.jl
export MSE_loss_init, MPE_loss_init

# From optimization/optimizers/greedy_descent.jl
export greedy_descent, greedy_descent_init

# From optimization/optimizers/random_descent.jl (BASELINE ONLY - for experimental comparison)
export random_descent, random_descent_init

# From optimization/optimizers/gradient_descent.jl
export gradient_param, gradient_descent, gradient_descent_init

# From optimization/optimizers/tree_search/hoo.jl
export HOONode, HOOConfig, HOOState
export hoo_descent, hoo_descent_init
export get_tree_size, get_visited_nodes, get_leaf_nodes

# From optimization/optimizers/tree_search/value_transforms.jl
export sigmoid_transform, tanh_transform, negation_transform, inverse_transform
export DEFAULT_VALUE_TRANSFORM

# From optimization/optimizers/tree_search/mcts.jl
export MCTSNode, MCTSConfig, MCTSState
export SelectionMode, VisitCount, BestValue, BestLoss
export mcts_descent, mcts_descent_init

# From optimization/optimizers/tree_search/uct.jl
export UCTNode, UCTConfig, UCTState
export uct_descent, uct_descent_init

# From optimization/optimizers/tree_search/modified_uct.jl
export ModifiedUCTNode, ModifiedUCTConfig, ModifiedUCTState
export modified_uct_descent, modified_uct_descent_init

# From optimization/optimizers/tree_search/flat_ucb.jl
export FlatUCBNode, FlatUCBConfig, FlatUCBState
export flat_ucb_descent, flat_ucb_descent_init

# From optimization/optimizers/tree_search/doo.jl
export DOONode, DOOConfig, DOOState
export doo_descent, doo_descent_init
export get_best_node, get_best_value, get_leaf_count, get_all_leaves

# From optimization/optimizers/tree_search/dag_mcts.jl
export DAGMCTSNode, DAGMCTSConfig, DAGMCTSState
export dag_mcts_descent, dag_mcts_descent_init
export get_dag_stats, print_dag_stats, verify_transposition_table

# From statistics/frechet.jl
export frechet_mean

# From statistics/least_squares.jl
export make_ordinary_least_squares_loss, solve_linear_system

# From visualization/loss_landscape.jl
export ConvexHullTree, convex_hull
export sample_loss_landscape
export print_landscape_summary, plot_loss_landscape, export_landscape_csv
# Tree visualization
export plot_tree_with_loss, plot_tree_simple

# From visualization/search_tree_viz.jl
export visualize_search_tree

end # module NAML
