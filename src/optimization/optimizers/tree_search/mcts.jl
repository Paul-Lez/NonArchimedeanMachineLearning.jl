########### Monte Carlo Tree Search (MCTS) Optimizer ###########

# This file implements MCTS-based optimization for non-Archimedean spaces.
# The key idea is to use MCTS to explore the tree of polydiscs and find
# the one that minimizes the loss function.

##################################################
# MCTS Node Structure
##################################################

@doc raw"""
    MCTSNode{S,T}

A node in the MCTS search tree.

# Fields
- `polydisc::ValuationPolydisc{S,T,N}`: The polydisc at this node
- `parent::Union{MCTSNode{S,T,N}, Nothing}`: Parent node (nothing for root)
- `children::Vector{MCTSNode{S,T,N}}`: Child nodes that have been expanded
- `visits::Int`: Number of times this node has been visited
- `total_value::Float64`: Sum of all values backpropagated through this node
- `is_expanded::Bool`: Whether this node's children have been generated
"""
mutable struct MCTSNode{S,T,N}
    polydisc::ValuationPolydisc{S,T,N}
    parent::Union{MCTSNode{S,T,N}, Nothing}
    children::Vector{MCTSNode{S,T,N}}
    visits::Int
    total_value::Float64
    is_expanded::Bool
end

@doc raw"""
    MCTSNode(polydisc::ValuationPolydisc{S,T,N}, parent=nothing) where {S,T,N}

Create a new MCTS node with the given polydisc and optional parent.
"""
function MCTSNode(polydisc::ValuationPolydisc{S,T,N}, parent=nothing) where {S,T,N}
    return MCTSNode{S,T,N}(
        polydisc,
        parent,
        MCTSNode{S,T,N}[],
        0,
        0.0,
        false
    )
end

@doc raw"""
    average_value(node::MCTSNode)

Compute the average value of a node (total_value / visits).
Returns 0.0 if node has not been visited.
"""
function average_value(node::MCTSNode)
    return node.visits > 0 ? node.total_value / node.visits : 0.0
end

##################################################
# MCTS Configuration
##################################################

@doc raw"""
    SelectionMode

Enum for MCTS child selection strategy after simulations complete.

# Values
- `VisitCount`: Select child with highest visit count (standard MCTS, robust)
- `BestValue`: Select child leading to best value in tree (greedy MCTS)
"""
@enum SelectionMode VisitCount BestValue

@doc raw"""
    MCTSConfig

Configuration parameters for the MCTS optimizer.

# Fields
- `num_simulations::Int`: Number of MCTS simulations to run per step
- `exploration_constant::Float64`: UCB1 exploration constant (usually √2 ≈ 1.41)
- `degree::Int`: Degree for child polydisc generation (passed to `children` function)
- `max_children::Union{Int, Nothing}`: Maximum number of children to consider per expansion (nothing = all)
- `strict::Bool`: If true, use single-branch descent; if false, use full children
- `value_transform::Function`: Transform from loss to value (default: loss -> 1/loss)
- `selection_mode::SelectionMode`: Strategy for selecting next step (VisitCount or BestValue)
"""
struct MCTSConfig
    num_simulations::Int
    exploration_constant::Float64
    degree::Int
    max_children::Union{Int, Nothing}
    strict::Bool
    value_transform::Function
    selection_mode::SelectionMode
end

@doc raw"""
    MCTSConfig(; kwargs...)

Create an MCTS configuration with sensible defaults.

# Keyword Arguments
- `num_simulations::Int=100`: Number of simulations per step
- `exploration_constant::Float64=1.41`: UCB1 exploration constant
- `degree::Int=1`: Child generation degree
- `max_children::Union{Int, Nothing}=nothing`: Max children to consider (nothing = all)
- `strict::Bool=false`: Whether to use single-branch descent
- `value_transform::Function=loss -> 1.0 / (loss + 1e-10)`: Loss to value transformation
- `selection_mode::SelectionMode=VisitCount`: Child selection strategy (VisitCount or BestValue)
"""
function MCTSConfig(;
    num_simulations::Int=100,
    exploration_constant::Float64=1.41,
    degree::Int=1,
    max_children::Union{Int, Nothing}=nothing,
    strict::Bool=false,
    value_transform::Function=loss -> 1.0 / (loss + 1e-10),
    selection_mode::SelectionMode=VisitCount
)
    return MCTSConfig(
        num_simulations,
        exploration_constant,
        degree,
        max_children,
        strict,
        value_transform,
        selection_mode
    )
end

##################################################
# MCTS State (for tracking across optimization steps)
##################################################

@doc raw"""
    MCTSState{S,T,N}

State maintained across MCTS optimization steps.

# Fields
- `root::MCTSNode{S,T,N}`: The current root node of the search tree
- `next_branch::Int`: Next branch index for strict mode
- `step_count::Int`: Number of optimization steps taken
"""
mutable struct MCTSState{S,T,N}
    root::MCTSNode{S,T,N}
    next_branch::Int
    step_count::Int
end

##################################################
# UCB1 Selection
##################################################

@doc raw"""
    ucb1_score(node::MCTSNode, parent_visits::Int, exploration_constant::Float64)

Compute the UCB1 score for a node.

UCB1(node) = average_value(node) + c * √(ln(parent_visits) / node_visits)

Higher scores indicate nodes that should be explored (either high value or underexplored).
"""
function ucb1_score(node::MCTSNode, parent_visits::Int, exploration_constant::Float64)
    if node.visits == 0
        return Inf  # Unvisited nodes have infinite priority
    end
    exploitation = average_value(node)
    exploration = exploration_constant * sqrt(log(parent_visits) / node.visits)
    return exploitation + exploration
end

@doc raw"""
    select_child(node::MCTSNode, exploration_constant::Float64)

Select the child with the highest UCB1 score.
"""
function select_child(node::MCTSNode, exploration_constant::Float64)
    @assert !isempty(node.children) "Cannot select from node with no children"

    best_score = -Inf
    best_child = nothing

    for child in node.children
        score = ucb1_score(child, node.visits, exploration_constant)
        if score > best_score
            best_score = score
            best_child = child
        end
    end

    return best_child
end

##################################################
# MCTS Core Operations
##################################################

@doc raw"""
    expand_node!(node::MCTSNode{S,T}, config::MCTSConfig) where {S,T}

Expand a node by generating its child polydiscs.

Uses the same children generation as greedy/gradient descent.
"""
function expand_node!(node::MCTSNode{S,T}, config::MCTSConfig) where {S,T}
    if node.is_expanded
        return
    end

    # Generate children using the same function as other optimizers
    if config.strict
        # In strict mode, we would need to track branch index
        # For now, use degree=1 single coordinate children
        child_polydiscs = children(node.polydisc, 1)
    else
        child_polydiscs = children(node.polydisc, config.degree)
    end

    # Optionally limit the number of children
    if !isnothing(config.max_children) && length(child_polydiscs) > config.max_children
        # Randomly sample children (could be improved with heuristics)
        indices = randperm(length(child_polydiscs))[1:config.max_children]
        child_polydiscs = child_polydiscs[indices]
    end

    # Create child nodes
    for polydisc in child_polydiscs
        child_node = MCTSNode(polydisc, node)
        push!(node.children, child_node)
    end

    node.is_expanded = true
end

@doc raw"""
    select_path(root::MCTSNode, exploration_constant::Float64)

Select a path from root to a leaf node using UCB1.

Returns the leaf node reached by following UCB1 selections.
"""
function select_path(root::MCTSNode, exploration_constant::Float64)
    node = root

    while node.is_expanded && !isempty(node.children)
        node = select_child(node, exploration_constant)
    end

    return node
end

@doc raw"""
    evaluate_node(node::MCTSNode{S,T}, loss::Loss, config::MCTSConfig) where {S,T}

Evaluate a node using the loss function.

Returns the transformed value (by default, 1/loss).
"""
function evaluate_node(node::MCTSNode{S,T}, loss::Loss, config::MCTSConfig) where {S,T}
    # Evaluate the loss at this polydisc
    loss_value = loss.eval([node.polydisc])[1]
    # Transform to value (higher is better for MCTS)
    return config.value_transform(loss_value)
end

@doc raw"""
    backpropagate!(node::MCTSNode, value::Float64)

Backpropagate a value from a leaf node up to the root.

Updates visits and total_value for all nodes on the path.
"""
function backpropagate!(node::MCTSNode, value::Float64)
    current = node
    while !isnothing(current)
        current.visits += 1
        current.total_value += value
        current = current.parent
    end
end

##################################################
# Main MCTS Algorithm
##################################################

@doc raw"""
    mcts_search(root::MCTSNode{S,T}, loss::Loss, config::MCTSConfig) where {S,T}

Run MCTS from a root node and return the best child.

Performs `config.num_simulations` iterations of:
1. Selection: Follow UCB1 to a leaf
2. Expansion: Expand the leaf if not terminal
3. Evaluation: Compute value using 1/loss
4. Backpropagation: Update all nodes on the path

Returns the child of root with the highest visit count (most promising).
"""
function mcts_search(root::MCTSNode{S,T}, loss::Loss, config::MCTSConfig) where {S,T}
    # Ensure root is expanded
    expand_node!(root, config)

    if isempty(root.children)
        # No children to explore, return root polydisc
        return root.polydisc, root
    end

    for _ in 1:config.num_simulations
        # Selection: traverse tree using UCB1 until we reach an unexpanded node
        leaf = select_path(root, config.exploration_constant)

        # Expansion: expand the node if it hasn't been expanded
        if !leaf.is_expanded
            expand_node!(leaf, config)
        end

        # Choose a child to evaluate (if any exist)
        if !isempty(leaf.children)
            # Pick a random unvisited child, or any child if all visited
            unvisited = [c for c in leaf.children if c.visits == 0]
            eval_node = isempty(unvisited) ? rand(leaf.children) : rand(unvisited)
        else
            eval_node = leaf
        end

        # Evaluation: compute value at the node
        value = evaluate_node(eval_node, loss, config)

        # Backpropagation: update statistics up the tree
        backpropagate!(eval_node, value)
    end

    # Select the best child according to configured selection mode
    best_child = select_best_child(root, config)

    return best_child.polydisc, best_child
end

@doc raw"""
    mcts_best_child_by_value(root::MCTSNode)

Return the child with the best average value (alternative to visit-count selection).
"""
function mcts_best_child_by_value(root::MCTSNode)
    if isempty(root.children)
        return nothing
    end
    return argmax(c -> average_value(c), root.children)
end

##################################################
# Selection Strategy Functions
##################################################

@doc raw"""
    find_best_node_in_tree(node::MCTSNode)

Recursively find the node with the best average value in the entire tree.

Returns the node with highest average_value, considering only visited nodes.
"""
function find_best_node_in_tree(node::MCTSNode)
    # If node has no visits, it can't be the best
    if node.visits == 0
        return nothing
    end

    best_node = node
    best_value = average_value(node)

    # Recursively search children
    for child in node.children
        child_best = find_best_node_in_tree(child)
        if !isnothing(child_best)
            child_value = average_value(child_best)
            if child_value > best_value
                best_node = child_best
                best_value = child_value
            end
        end
    end

    return best_node
end

@doc raw"""
    trace_to_root_child(node::MCTSNode, root::MCTSNode)

Trace back from a node to find which direct child of root lies on the path.

# Arguments
- `node::MCTSNode`: The node to trace back from
- `root::MCTSNode`: The root node

# Returns
The direct child of `root` that is an ancestor of `node`, or `node` itself if it's a direct child.
Returns `nothing` if `node` is the root or not in the tree.
"""
function trace_to_root_child(node::MCTSNode, root::MCTSNode)
    # If node is root, return nothing
    if node === root
        return nothing
    end

    # Trace back to find the child of root
    current = node
    while !isnothing(current.parent) && current.parent !== root
        current = current.parent
    end

    # Check if we found a child of root
    if isnothing(current.parent)
        # Node is not in this tree
        return nothing
    else
        # current.parent === root, so current is a child of root
        return current
    end
end

@doc raw"""
    select_best_child(root::MCTSNode, config::MCTSConfig)

Select the best child of root according to the configured selection mode.

# Selection Modes
- `VisitCount`: Returns child with highest visit count (standard MCTS, robust)
- `BestValue`: Finds node with best value in tree, returns root's child leading to it (greedy)

# Arguments
- `root::MCTSNode`: The root node with expanded children
- `config::MCTSConfig`: Configuration specifying selection mode

# Returns
The selected child node.
"""
function select_best_child(root::MCTSNode, config::MCTSConfig)
    if isempty(root.children)
        error("Cannot select from node with no children")
    end

    if config.selection_mode == VisitCount
        # Standard MCTS: select most visited child
        return argmax(c -> c.visits, root.children)
    elseif config.selection_mode == BestValue
        # Greedy MCTS: find best node in tree, trace back to root's child
        best_node = find_best_node_in_tree(root)

        if isnothing(best_node)
            # Fallback: no visited nodes, select first child
            return root.children[1]
        end

        # If best node is a direct child, return it
        if best_node.parent === root
            return best_node
        end

        # Otherwise, trace back to find which child of root leads to best_node
        root_child = trace_to_root_child(best_node, root)

        if isnothing(root_child)
            # Fallback: best_node is root itself or not in tree, select best direct child
            return argmax(c -> average_value(c), root.children)
        end

        return root_child
    else
        error("Unknown selection mode: $(config.selection_mode)")
    end
end

##################################################
# MCTS Optimizer Interface (compatible with OptimSetup)
##################################################

@doc raw"""
    mcts_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, state::MCTSState{S,T,N}, config::MCTSConfig) where {S,T,N}

Perform one step of MCTS optimization.

This function follows the same interface as `greedy_descent` and `gradient_descent`,
making it compatible with `OptimSetup`.

# Arguments
- `loss::Loss`: The loss function structure
- `param::ValuationPolydisc{S,T,N}`: Current parameter values
- `state::MCTSState{S,T,N}`: MCTS state (includes the search tree)
- `config::MCTSConfig`: Configuration parameters

# Returns
`Tuple{ValuationPolydisc{S,T,N}, MCTSState{S,T,N}}`: New parameters and updated state
"""
function mcts_descent(
    loss::Loss,
    param::ValuationPolydisc{S,T,N},
    state::MCTSState{S,T,N},
    config::MCTSConfig
) where {S,T,N}
    # Update root if param changed (shouldn't normally happen)
    if state.root.polydisc != param
        state.root = MCTSNode(param)
    end

    # Run MCTS search
    best_polydisc, best_node = mcts_search(state.root, loss, config)

    # Update state: make the best child the new root (tree reuse)
    state.root = MCTSNode(best_polydisc)  # Fresh node for next iteration
    state.step_count += 1

    return best_polydisc, state
end

@doc raw"""
    mcts_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, config::MCTSConfig=MCTSConfig()) where {S,T,N}

Initialize an optimization setup for MCTS.

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `config::MCTSConfig`: MCTS configuration (uses defaults if not provided)

# Returns
`OptimSetup`: Configured optimization setup for MCTS
"""
function mcts_descent_init(
    param::ValuationPolydisc{S,T,N},
    loss::Loss,
    config::MCTSConfig=MCTSConfig()
) where {S,T,N}
    # Initialize state
    root = MCTSNode(param)
    state = MCTSState{S,T,N}(root, 1, 0)

    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> mcts_descent(l, p, st, ctx),
        state,
        config
    )
end

##################################################
# Utility Functions
##################################################

@doc raw"""
    print_tree_stats(node::MCTSNode, depth::Int=0, max_depth::Int=3)

Print statistics about the MCTS tree for debugging.
"""
function print_tree_stats(node::MCTSNode, depth::Int=0, max_depth::Int=3)
    if depth > max_depth
        return
    end

    indent = "  " ^ depth
    println("$(indent)Node: visits=$(node.visits), avg_value=$(round(average_value(node), digits=4)), children=$(length(node.children))")

    # Sort children by visits for display
    sorted_children = sort(node.children, by=c -> c.visits, rev=true)
    for (i, child) in enumerate(sorted_children[1:min(3, length(sorted_children))])
        print_tree_stats(child, depth + 1, max_depth)
    end
end

@doc raw"""
    get_tree_size(node::MCTSNode)

Count the total number of nodes in the MCTS tree.
"""
function get_tree_size(node::MCTSNode)
    count = 1
    for child in node.children
        count += get_tree_size(child)
    end
    return count
end
