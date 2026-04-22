########### UCT (Upper Confidence Bounds applied to Trees) Optimizer ###########

# This file implements UCT optimization for non-Archimedean spaces.
# UCT is a bandit-based tree search algorithm that balances exploration and
# exploitation using upper confidence bounds derived from parent visit counts.

##################################################
# UCT Node Structure
##################################################

@doc raw"""
    UCTNode{S,T,N}

A node in the UCT search tree.

# Fields
- `polydisc::ValuationPolydisc{S,T,N}`: The polydisc at this node
- `parent::Union{UCTNode{S,T,N}, Nothing}`: Parent node (nothing for root)
- `children::Vector{UCTNode{S,T,N}}`: Child nodes that have been expanded
- `visits::Int`: Number of times this node has been visited (n_i in spec)
- `total_value::Float64`: Sum of all values backpropagated through this node
- `depth::Int`: Depth of this node in the tree (0 for root)
"""
mutable struct UCTNode{S, T, N}
    polydisc::ValuationPolydisc{S, T, N}
    parent::Union{UCTNode{S, T, N}, Nothing}
    children::Vector{UCTNode{S, T, N}}
    visits::Int
    total_value::Float64
    depth::Int
end

@doc raw"""
    UCTNode(polydisc::ValuationPolydisc{S,T,N}, parent=nothing, depth=0) where {S,T,N}

Create a new UCT node with the given polydisc, optional parent, and depth.
"""
function UCTNode(polydisc::ValuationPolydisc{S, T, N}, parent = nothing, depth = 0) where {
        S, T, N}
    return UCTNode{S, T, N}(
        polydisc,
        parent,
        UCTNode{S, T, N}[],
        0,
        0.0,
        depth
    )
end

@doc raw"""
    average_value(node::UCTNode)

Compute the empirical mean of a node (X_{i,n_i} in spec).
Returns 0.0 if node has not been visited.
"""
function average_value(node::UCTNode)
    return node.visits > 0 ? node.total_value / node.visits : 0.0
end

##################################################
# UCT Configuration
##################################################

@doc raw"""
    UCTConfig

Configuration parameters for the UCT optimizer.

# Fields
- `max_depth::Int`: Maximum tree depth (D in spec)
- `num_simulations::Int`: Number of UCT simulations to run per step
- `exploration_constant::Float64`: Exploration constant c in UCB formula (default √2)
- `degree::Int`: Degree for child polydisc generation
- `strict::Bool`: If true, use single-branch descent
- `value_transform::Function`: Transform from loss to value (default: loss -> 1/loss)
"""
struct UCTConfig
    max_depth::Int
    num_simulations::Int
    exploration_constant::Float64
    degree::Int
    strict::Bool
    value_transform::Function
end

@doc raw"""
    UCTConfig(; kwargs...)

Create a UCT configuration.

# Keyword Arguments
- `max_depth::Int=10`: Maximum tree depth
- `num_simulations::Int=100`: Number of simulations per step
- `exploration_constant::Float64=sqrt(2.0)`: UCB exploration constant
- `degree::Int=1`: Child generation degree
- `strict::Bool=false`: Whether to use single-branch descent
- `value_transform::Function=loss -> 1.0 / (loss + 1e-10)`: Loss to value transformation
"""
function UCTConfig(;
        max_depth::Int = 10,
        num_simulations::Int = 100,
        exploration_constant::Float64 = sqrt(2.0),
        degree::Int = 1,
        strict::Bool = false,
        value_transform::Function = loss -> 1.0 / (loss + 1e-10)
)
    return UCTConfig(
        max_depth,
        num_simulations,
        exploration_constant,
        degree,
        strict,
        value_transform
    )
end

##################################################
# UCT State (for tracking across optimization steps)
##################################################

@doc raw"""
    UCTState{S,T,N}

State maintained across UCT optimization steps.

# Fields
- `root::UCTNode{S,T,N}`: The current root node of the search tree
- `step_count::Int`: Number of optimization steps taken
"""
mutable struct UCTState{S, T, N}
    root::UCTNode{S, T, N}
    step_count::Int
end

##################################################
# UCB Score Computation
##################################################

@doc raw"""
    ucb_score(node::UCTNode, parent_visits::Int, exploration_constant::Float64)

Compute the Upper Confidence Bound (UCB) score for a node.

Formula from spec: B_{i,p,n_i} = X_{i,n_i} + c * √(2 * ln(p) / n_i)

where:
- X_{i,n_i} is the empirical mean (average value)
- c is the exploration constant
- p is the parent's visit count
- n_i is this node's visit count

Returns Inf if the node hasn't been visited (to ensure exploration).
"""
function ucb_score(node::UCTNode, parent_visits::Int, exploration_constant::Float64)
    if node.visits == 0
        return Inf  # Unvisited nodes have infinite priority (spec requirement)
    end

    exploitation = average_value(node)
    exploration = exploration_constant * sqrt(2.0 * log(parent_visits) / node.visits)

    return exploitation + exploration
end

@doc raw"""
    select_child_ucb(node::UCTNode, exploration_constant::Float64)

Select the child with the highest UCB score.
Uses parent's visit count (p in spec) for UCB computation.
"""
function select_child_ucb(node::UCTNode, exploration_constant::Float64)
    @assert !isempty(node.children) "Cannot select from node with no children"

    parent_visits = node.visits  # p in spec

    best_score = -Inf
    best_child = nothing

    for child in node.children
        score = ucb_score(child, parent_visits, exploration_constant)
        if score > best_score
            best_score = score
            best_child = child
        end
    end

    return best_child
end

##################################################
# UCT Core Operations
##################################################

@doc raw"""
    expand_node!(node::UCTNode{S,T,N}, config::UCTConfig) where {S,T,N}

Expand a node by generating all its child polydiscs.
Children are created at depth = parent.depth + 1.
"""
function expand_node!(node::UCTNode{S, T, N}, config::UCTConfig) where {S, T, N}
    if !isempty(node.children)
        return  # Already expanded
    end

    # Generate children using same function as other optimizers
    if config.strict
        child_polydiscs = children(node.polydisc, 1)
    else
        child_polydiscs = children(node.polydisc, config.degree)
    end

    # Create child nodes at depth + 1
    child_depth = node.depth + 1
    for polydisc in child_polydiscs
        child_node = UCTNode(polydisc, node, child_depth)
        push!(node.children, child_node)
    end
end

@doc raw"""
    traverse_to_leaf(root::UCTNode, config::UCTConfig)

Phase 1 (Traversal/Selection) from spec:
Starting from root, traverse to depth D by selecting children with highest UCB.
Expands nodes as needed during traversal.

Returns: (leaf_node, path)
- leaf_node: The node reached at depth D or earlier if tree ends
- path: Vector of all nodes visited (for backpropagation)
"""
function traverse_to_leaf(root::UCTNode, config::UCTConfig)
    path = [root]
    current = root

    # Loop from depth 1 to D (spec Phase 1, step 2)
    while current.depth < config.max_depth
        # Expand current node if needed
        if isempty(current.children)
            expand_node!(current, config)
        end

        # If still no children (shouldn't happen with polydisc), break
        if isempty(current.children)
            break
        end

        # Select child with highest UCB (spec Phase 1, step 2c-d)
        current = select_child_ucb(current, config.exploration_constant)
        push!(path, current)
    end

    return current, path
end

@doc raw"""
    evaluate_node(node::UCTNode{S,T,N}, loss::Loss, config::UCTConfig) where {S,T,N}

Phase 2 (Sampling) from spec:
Evaluate the loss at a node and transform to value.

Returns the transformed value (higher is better).
"""
function evaluate_node(node::UCTNode{S, T, N}, loss::Loss, config::UCTConfig) where {
        S, T, N}
    # Get reward x_n from environment (spec Phase 2)
    loss_value = loss.eval([node.polydisc])[1]

    # Transform to value (higher is better for UCT)
    return config.value_transform(loss_value)
end

@doc raw"""
    backpropagate!(path::Vector{UCTNode}, value::Float64)

Phase 3 (Backpropagation) from spec:
Update all nodes in the trajectory with the sampled value.

For each node i in path (spec Phase 3):
- Increment visit count: n_i ← n_i + 1
- Update empirical mean: X_{i,n_i} ← X_{i,n_i} + (x_n - X_{i,n_i}) / n_i

Note: The spec formula for updating the mean is equivalent to:
X_{i,n_i} = (X_{i,n_i-1} * (n_i - 1) + x_n) / n_i

We use total_value accumulation which is equivalent.
"""
function backpropagate!(path::Vector{<:UCTNode}, value::Float64)
    for node in path
        node.visits += 1
        # Update using incremental mean formula
        # X_new = X_old + (x - X_old) / n
        # Equivalently: total_new = total_old + x
        node.total_value += value
    end
end

@doc raw"""
    uct_simulation!(root::UCTNode, loss::Loss, config::UCTConfig)

Perform one complete UCT simulation (one round n in spec):

1. **Phase 1 (Traversal)**: Traverse from root to leaf at depth D using UCB
2. **Phase 2 (Sampling)**: Evaluate loss at the leaf
3. **Phase 3 (Backpropagation)**: Update all nodes in the path

Returns the value obtained from the simulation.
"""
function uct_simulation!(root::UCTNode, loss::Loss, config::UCTConfig)
    # Phase 1: Selection/Traversal
    leaf, path = traverse_to_leaf(root, config)

    # Phase 2: Sampling
    value = evaluate_node(leaf, loss, config)

    # Phase 3: Backpropagation
    backpropagate!(path, value)

    return value
end

##################################################
# Main UCT Algorithm
##################################################

@doc raw"""
    uct_search(root::UCTNode{S,T,N}, loss::Loss, config::UCTConfig) where {S,T,N}

Run UCT from a root node and return the best child.

Performs `config.num_simulations` iterations of the full UCT algorithm.
After all simulations, selects the child with the best average value.

Returns: `(best_polydisc, best_child_node, converged)`.
"""
function uct_search(root::UCTNode{S, T, N}, loss::Loss, config::UCTConfig) where {S, T, N}
    # Ensure root is expanded (needed to have children to select from)
    expand_node!(root, config)

    if isempty(root.children)
        # No children to explore, return root polydisc
        return root.polydisc, root, true
    end

    # Run simulations (spec: repeat for rounds n = 1, 2, ...)
    for _ in 1:config.num_simulations
        uct_simulation!(root, loss, config)
    end

    # Select best child by average value (pure exploitation)
    best_child = argmax(c -> average_value(c), root.children)

    return best_child.polydisc, best_child, false
end

##################################################
# UCT Optimizer Interface (compatible with OptimSetup)
##################################################

@doc raw"""
    uct_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, state::UCTState{S,T,N}, config::UCTConfig) where {S,T,N}

Perform one step of UCT optimization.

This function follows the same interface as `greedy_descent` and `mcts_descent`,
making it compatible with `OptimSetup`.

# Arguments
- `loss::Loss`: The loss function structure
- `param::ValuationPolydisc{S,T,N}`: Current parameter values
- `state::UCTState{S,T,N}`: UCT state (includes the search tree)
- `config::UCTConfig`: Configuration parameters

# Returns
`Tuple{ValuationPolydisc{S,T,N}, UCTState{S,T,N}, Bool}`: New parameters,
updated state, and convergence status
"""
function uct_descent(
        loss::Loss,
        param::ValuationPolydisc{S, T, N},
        state::UCTState{S, T, N},
        config::UCTConfig
) where {S, T, N}
    # Update root if param changed (shouldn't normally happen)
    if state.root.polydisc != param
        state.root = UCTNode(param, nothing, 0)
    end

    # Run UCT search
    best_polydisc, best_node, converged = uct_search(state.root, loss, config)

    # Update state: make the best child the new root (fresh tree each iteration)
    state.root = UCTNode(best_polydisc, nothing, 0)
    state.step_count += 1

    return best_polydisc, state, converged
end

@doc raw"""
    uct_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, config::UCTConfig=UCTConfig()) where {S,T,N}

Initialize an optimization setup for UCT.

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `config::UCTConfig`: UCT configuration (uses defaults if not provided)

# Returns
`OptimSetup`: Configured optimization setup for UCT
"""
function uct_descent_init(
        param::ValuationPolydisc{S, T, N},
        loss::Loss,
        config::UCTConfig = UCTConfig()
) where {S, T, N}
    # Initialize state with root at depth 0
    root = UCTNode(param, nothing, 0)
    state = UCTState{S, T, N}(root, 0)

    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> uct_descent(l, p, st, ctx),
        state,
        config,
        false
    )
end

##################################################
# Utility Functions
##################################################

@doc raw"""
    print_uct_tree_stats(node::UCTNode, max_depth::Int=3)

Print statistics about the UCT tree for debugging.
"""
function print_uct_tree_stats(node::UCTNode, max_depth::Int = 3)
    if node.depth > max_depth
        return
    end

    indent = "  " ^ node.depth
    println("$(indent)Depth $(node.depth): visits=$(node.visits), avg_value=$(round(average_value(node), digits=6)), children=$(length(node.children))")

    # Sort children by visits for display
    sorted_children = sort(node.children, by = c -> c.visits, rev = true)
    for child in sorted_children[1:min(3, length(sorted_children))]
        print_uct_tree_stats(child, max_depth)
    end
end

@doc raw"""
    get_uct_tree_size(node::UCTNode)

Count the total number of nodes in the UCT tree.
"""
function get_uct_tree_size(node::UCTNode)
    count = 1
    for child in node.children
        count += get_uct_tree_size(child)
    end
    return count
end
