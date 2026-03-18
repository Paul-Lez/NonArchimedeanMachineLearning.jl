########### DAG-MCTS (Monte Carlo Tree Search with Transposition Tables) ###########

# This file implements MCTS adapted for Directed Acyclic Graphs (DAGs) using
# a global transposition table. This is essential for polydisc optimization where
# multiple sequences of coordinate refinements can lead to the same polydisc.
#
# Key differences from standard MCTS (mcts.jl):
# 1. Nodes do NOT have a single parent pointer - they can have multiple parents
# 2. A global transposition table maps polydisc state -> node instance
# 3. Backpropagation uses an explicit path stack instead of parent pointers
# 4. UCT formula uses global statistics from the shared node

##################################################
# DAG-MCTS Node Structure
##################################################

@doc raw"""
    DAGMCTSNode{S,T,N}

A node in the DAG-MCTS search graph.

Unlike standard MCTS nodes, DAG nodes can have multiple parents since the same
polydisc state can be reached via different action sequences.

# Fields
- `polydisc::ValuationPolydisc{S,T,N}`: The polydisc at this node
- `parents::Vector{DAGMCTSNode{S,T,N}}`: All parent nodes (can be multiple in a DAG)
- `children::Vector{DAGMCTSNode{S,T,N}}`: Child nodes indexed by action (1-based)
- `visits::Int`: Total visit count N(s) aggregated from all paths
- `total_value::Float64`: Sum of all values Q(s) backpropagated through this node
- `is_expanded::Bool`: Whether this node's children have been generated

# Design Decision (recorded for future experimentation)
We track all parents in a vector rather than having no parent information.
This is more expensive but useful for analysis and debugging. The alternative
would be to track no parent info and rely purely on the explicit path during traversal.
"""
mutable struct DAGMCTSNode{S,T,N}
    polydisc::ValuationPolydisc{S,T,N}
    parents::Vector{DAGMCTSNode{S,T,N}}
    children::Vector{DAGMCTSNode{S,T,N}}
    visits::Int
    total_value::Float64
    is_expanded::Bool
end

@doc raw"""
    DAGMCTSNode(polydisc::ValuationPolydisc{S,T,N}) where {S,T,N}

Create a new DAG-MCTS node with the given polydisc and no parents.
"""
function DAGMCTSNode(polydisc::ValuationPolydisc{S,T,N}) where {S,T,N}
    return DAGMCTSNode{S,T,N}(
        polydisc,
        DAGMCTSNode{S,T,N}[],
        DAGMCTSNode{S,T,N}[],
        0,
        0.0,
        false
    )
end

@doc raw"""
    average_value(node::DAGMCTSNode)

Compute the average value Q(s)/N(s) of a node.
Returns 0.0 if node has not been visited.
"""
function average_value(node::DAGMCTSNode)
    return node.visits > 0 ? node.total_value / node.visits : 0.0
end

##################################################
# DAG-MCTS Configuration
##################################################

@doc raw"""
    DAGMCTSConfig

Configuration parameters for the DAG-MCTS optimizer.

# Fields
- `num_simulations::Int`: Number of MCTS simulations to run per step
- `exploration_constant::Float64`: UCT exploration constant c (usually √2 ≈ 1.41)
- `degree::Int`: Degree for child polydisc generation (passed to `children` function)
- `value_transform::Function`: Transform from loss to value (default: loss -> -loss)
- `persist_table::Bool`: Whether to persist transposition table across optimization steps
- `selection_mode::SelectionMode`: Strategy for selecting next step (VisitCount or BestValue)

# Design Decision (recorded for future experimentation)
The `persist_table` option allows experimenting with:
- persist_table=true: Reuse learned information across steps (may grow large)
- persist_table=false: Fresh search each step (like standard MCTS)

# TODO: Add support for the following features from standard MCTS:
# - strict mode for single-branch descent
# - max_children limit
"""
struct DAGMCTSConfig
    num_simulations::Int
    exploration_constant::Float64
    degree::Int
    value_transform::Function
    persist_table::Bool
    selection_mode::SelectionMode
    track_parents::Bool
end

@doc raw"""
    DAGMCTSConfig(; kwargs...)

Create a DAG-MCTS configuration with sensible defaults.

# Keyword Arguments
- `num_simulations::Int=100`: Number of simulations per step
- `exploration_constant::Float64=1.41`: UCT exploration constant
- `degree::Int=1`: Child generation degree
- `value_transform::Function=loss -> -loss`: Loss to value transformation
- `persist_table::Bool=true`: Whether to persist transposition table across steps
- `selection_mode::SelectionMode=VisitCount`: Child selection strategy (VisitCount or BestValue)
- `track_parents::Bool=false`: Whether to track parent pointers (needed for debug verification; off by default for performance)
"""
function DAGMCTSConfig(;
    num_simulations::Int=100,
    exploration_constant::Float64=1.41,
    degree::Int=1,
    value_transform::Function=loss -> -loss,
    persist_table::Bool=true,
    selection_mode::SelectionMode=VisitCount,
    track_parents::Bool=false
)
    return DAGMCTSConfig(
        num_simulations,
        exploration_constant,
        degree,
        value_transform,
        persist_table,
        selection_mode,
        track_parents
    )
end

##################################################
# DAG-MCTS State (includes transposition table)
##################################################

@doc raw"""
    DAGMCTSState{S,T,N}

State maintained across DAG-MCTS optimization steps.

# Fields
- `root::DAGMCTSNode{S,T,N}`: The current root node of the search graph
- `transposition_table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}}`:
    Global table mapping hashed polydisc states to node instances
- `step_count::Int`: Number of optimization steps taken
- `best_node::Union{DAGMCTSNode{S,T,N}, Nothing}`: Running tracker of the best node by average value
- `best_value::Float64`: Best average value seen so far
- `best_root_child::Union{DAGMCTSNode{S,T,N}, Nothing}`: Which direct child of root leads to best_node
- `best_root_action::Int`: Action index of best_root_child
- `min_loss_node::Union{DAGMCTSNode{S,T,N}, Nothing}`: Node with minimum raw loss evaluation
- `min_loss::Float64`: Minimum raw loss seen so far
- `min_loss_root_child::Union{DAGMCTSNode{S,T,N}, Nothing}`: Which direct child of root leads to min_loss_node
"""
mutable struct DAGMCTSState{S,T,N}
    root::DAGMCTSNode{S,T,N}
    transposition_table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}}
    step_count::Int
    best_node::Union{DAGMCTSNode{S,T,N}, Nothing}
    best_value::Float64
    best_root_child::Union{DAGMCTSNode{S,T,N}, Nothing}
    best_root_action::Int
    min_loss_node::Union{DAGMCTSNode{S,T,N}, Nothing}
    min_loss::Float64
    min_loss_root_child::Union{DAGMCTSNode{S,T,N}, Nothing}
end

##################################################
# Transposition Table Operations
##################################################

@doc raw"""
    get_or_create_node!(table::Dict, polydisc::ValuationPolydisc{S,T,N}, parent::Union{DAGMCTSNode{S,T,N}, Nothing}=nothing) where {S,T,N}

Look up a polydisc in the transposition table, creating a new node if not found.

This is the core "Lookup & Link" operation for DAG-MCTS:
1. Wrap polydisc in HashedPolydisc (computes hash once)
2. Check if hashed polydisc is in transposition_table via `get!` (single hash probe)
3. If found: retrieve existing node instance; if not: create new node
4. Link the parent if provided

# Arguments
- `table`: The transposition table (keyed by HashedPolydisc)
- `polydisc`: The polydisc state to look up
- `parent`: Optional parent node to link

# Returns
`DAGMCTSNode{S,T,N}`: The node for this polydisc (existing or newly created)
"""
function get_or_create_node!(
    table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}},
    polydisc::ValuationPolydisc{S,T,N},
    parent::Union{DAGMCTSNode{S,T,N}, Nothing}=nothing
) where {S,T,N}
    key = HashedPolydisc(polydisc)
    node = get!(table, key) do
        DAGMCTSNode(polydisc)
    end

    # Link parent if provided and not already linked
    if !isnothing(parent) && !(parent in node.parents)
        push!(node.parents, parent)
    end

    return node
end

##################################################
# UCT Selection
##################################################

@doc raw"""
    uct_score(node::DAGMCTSNode, parent_visits::Int, exploration_constant::Float64)

Compute the UCT score for a node.

UCT(s) = Q(s)/N(s) + c * √(ln(N(parent)) / N(s))

where Q(s) and N(s) are the global statistics from the shared node instance,
and N(parent) is the visit count of the parent in the current traversal.

# Arguments
- `node`: The child node being scored
- `parent_visits`: Visit count of the parent node (N(parent))
- `exploration_constant`: The exploration constant c

# Returns
`Float64`: The UCT score (higher = should explore)
"""
function uct_score(node::DAGMCTSNode, parent_visits::Int, exploration_constant::Float64)
    if node.visits == 0
        return Inf  # Unvisited nodes have infinite priority
    end
    exploitation = average_value(node)
    exploration = exploration_constant * sqrt(log(parent_visits) / node.visits)
    return exploitation + exploration
end

@doc raw"""
    select_child(node::DAGMCTSNode, exploration_constant::Float64)

Select the child with the highest UCT score.

# Returns
Tuple of (action_index, child_node) for the best child
"""
function select_child(node::DAGMCTSNode, exploration_constant::Float64)
    @assert !isempty(node.children) "Cannot select from node with no children"

    best_score = -Inf
    best_action = nothing
    best_child = nothing

    for (action, child) in enumerate(node.children)
        score = uct_score(child, node.visits, exploration_constant)
        if score > best_score
            best_score = score
            best_action = action
            best_child = child
        end
    end

    return best_action, best_child
end

##################################################
# DAG-MCTS Core Operations
##################################################

@doc raw"""
    expand_node!(node::DAGMCTSNode{S,T,N}, table::Dict, config::DAGMCTSConfig) where {S,T,N}

Expand a node by generating its child polydiscs and linking them in the DAG.

Uses the transposition table to detect if any child polydisc already exists,
linking to the existing node if so (the "Lookup & Link" step).

# Arguments
- `node`: The node to expand
- `table`: The transposition table
- `config`: Configuration parameters

# Returns
Nothing (modifies node.children in place)
"""
function expand_node!(
    node::DAGMCTSNode{S,T,N},
    table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}},
    config::DAGMCTSConfig
) where {S,T,N}
    if node.is_expanded
        return
    end

    # Generate child polydiscs
    child_polydiscs = children(node.polydisc, config.degree)

    # Pre-size children vector and create/retrieve nodes for each child
    # When track_parents=false, pass nothing as parent to skip parent vector push
    n_children = length(child_polydiscs)
    resize!(node.children, n_children)
    parent_for_link = config.track_parents ? node : nothing
    for (i, child_polydisc) in enumerate(child_polydiscs)
        child_node = get_or_create_node!(table, child_polydisc, parent_for_link)
        node.children[i] = child_node
    end

    node.is_expanded = true
end

@doc raw"""
    select_path(root::DAGMCTSNode, exploration_constant::Float64)

Select a path from root to a leaf node using UCT.

Unlike standard MCTS, we maintain an explicit path stack since nodes
don't have single parent pointers.

# Arguments
- `root`: The root node to start from
- `exploration_constant`: UCT exploration constant

# Returns
`Vector{DAGMCTSNode}`: The path from root to leaf (inclusive)
"""
function select_path(root::DAGMCTSNode, exploration_constant::Float64)
    path = [root]
    node = root

    while node.is_expanded && !isempty(node.children)
        _, child = select_child(node, exploration_constant)
        push!(path, child)
        node = child
    end

    return path
end

@doc raw"""
    evaluate_node(node::DAGMCTSNode{S,T,N}, loss::Loss, config::DAGMCTSConfig) where {S,T,N}

Evaluate a node using the loss function and transform to value.

# Returns
`Float64`: The transformed value (higher is better)
"""
function evaluate_node(node::DAGMCTSNode{S,T,N}, loss::Loss, config::DAGMCTSConfig) where {S,T,N}
    loss_value = loss.eval([node.polydisc])[1]
    return config.value_transform(loss_value)
end

@doc raw"""
    backpropagate!(path::Vector{<:DAGMCTSNode}, value::Float64)

Backpropagate a value through all nodes in the explicit path.

This is the "Stack Method" for DAG backpropagation - we iterate through
the specific path taken during this traversal rather than following
parent pointers (which would be ambiguous in a DAG).

# Arguments
- `path`: Vector of nodes from root to leaf representing this traversal's path
- `value`: The transformed value to backpropagate
- `eval_node`: The leaf node where the loss was evaluated
- `loss_value`: The raw loss value at the leaf (before value_transform)

# Note
This updates the global N(s) and Q(s) statistics in each shared node instance.
"""
function backpropagate!(path::Vector{<:DAGMCTSNode}, value::Float64, state::DAGMCTSState,
                        eval_node::DAGMCTSNode=path[end], loss_value::Float64=NaN)
    # The root is path[1]; the root child is path[2] if path has length >= 2
    root_child = length(path) >= 2 ? path[2] : nothing
    for (i, node) in enumerate(path)
        node.visits += 1
        node.total_value += value
        # Update running best-node tracker (skip root: we need a node strictly
        # below root so that select_best_child_dag can identify which root child
        # lies above it)
        if i == 1
            continue
        end
        avg = average_value(node)
        if avg > state.best_value
            state.best_value = avg
            state.best_node = node
            if !isnothing(root_child)
                state.best_root_child = root_child
            end
        end
    end
    # Track minimum raw loss at the evaluated leaf (for BestLoss selection)
    if !isnan(loss_value) && loss_value < state.min_loss
        state.min_loss = loss_value
        state.min_loss_node = eval_node
        if !isnothing(root_child)
            state.min_loss_root_child = root_child
        end
    end
end

##################################################
# Main DAG-MCTS Algorithm
##################################################

@doc raw"""
    dag_mcts_simulation!(root::DAGMCTSNode{S,T,N}, table::Dict, loss::Loss, config::DAGMCTSConfig) where {S,T,N}

Perform one complete DAG-MCTS simulation.

The four phases of MCTS adapted for DAG:
1. **Selection**: Follow UCT to a leaf, maintaining explicit path stack
2. **Expansion**: Expand leaf using transposition table for deduplication
3. **Evaluation**: Compute value at the leaf (or a child)
4. **Backpropagation**: Update statistics along the explicit path

# Returns
`Float64`: The value obtained from this simulation
"""
function dag_mcts_simulation!(
    root::DAGMCTSNode{S,T,N},
    table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}},
    loss::Loss,
    config::DAGMCTSConfig,
    state::DAGMCTSState{S,T,N}
) where {S,T,N}
    # Phase 1: Selection - traverse using UCT, maintaining path stack
    path = select_path(root, config.exploration_constant)
    leaf = path[end]

    # Phase 2: Expansion - expand using transposition table
    if !leaf.is_expanded
        expand_node!(leaf, table, config)
    end

    # Choose a node to evaluate
    if !isempty(leaf.children)
        # Pick an unvisited child, or random if all visited
        unvisited = [c for c in leaf.children if c.visits == 0]
        if !isempty(unvisited)
            eval_node = rand(unvisited)
        else
            eval_node = rand(leaf.children)
        end
        push!(path, eval_node)
    else
        eval_node = leaf
    end

    # Phase 3: Evaluation
    loss_value = loss.eval([eval_node.polydisc])[1]
    value = config.value_transform(loss_value)

    # Phase 4: Backpropagation using explicit path stack
    backpropagate!(path, value, state, eval_node, loss_value)

    return value
end

@doc raw"""
    find_best_node_in_dag(root::DAGMCTSNode, table::Dict)

Recursively find the node with the best average value in the entire DAG.

Returns the node with highest average_value, considering only visited nodes.
"""
function find_best_node_in_dag(root::DAGMCTSNode{S,T,N}, table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}}) where {S,T,N}
    best_node = nothing
    best_value = -Inf

    for node in values(table)
        if node.visits > 0
            val = average_value(node)
            if val > best_value
                best_value = val
                best_node = node
            end
        end
    end

    return best_node
end

@doc raw"""
    trace_to_root_child(target::DAGMCTSNode, root::DAGMCTSNode, table::Dict)

Trace back from a node to find which direct child of root lies on a path to it.

Uses parent pointers to trace upward from the target to root, which is O(depth)
instead of O(nodes). In a DAG, nodes can have multiple parents, so at each step
we pick any parent that is an ancestor of root (BFS upward).

# Returns
The direct child of `root` that can reach `node`, or `nothing` if not found.
"""
function trace_to_root_child(
    target::DAGMCTSNode{S,T,N},
    root::DAGMCTSNode{S,T,N},
    table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}}
) where {S,T,N}
    # If target is root, return nothing
    if target === root
        return nothing
    end

    # If target is a direct child of root, return it
    for child in root.children
        if child === target
            return child
        end
    end

    # Collect root's direct children into a set for O(1) lookup
    root_children = Set{UInt}(objectid(child) for child in root.children)

    # Trace upward from target using parent pointers
    # Use BFS over parents to handle the DAG structure
    # visited tracks nodes by identity to avoid cycles
    visited = Set{UInt}()
    queue = DAGMCTSNode{S,T,N}[target]
    push!(visited, objectid(target))

    while !isempty(queue)
        current = popfirst!(queue)

        # Check if current is a direct child of root
        if objectid(current) in root_children
            return current
        end

        # Add unvisited parents to queue
        for parent in current.parents
            pid = objectid(parent)
            if pid !== objectid(root) && !(pid in visited)
                push!(visited, pid)
                push!(queue, parent)
            end
        end
    end

    return nothing
end

@doc raw"""
    select_best_child_dag(root::DAGMCTSNode, table::Dict, config::DAGMCTSConfig)

Select the best child of root according to the configured selection mode.

# Selection Modes
- `VisitCount`: Returns child with highest visit count (standard MCTS, robust)
- `BestValue`: Finds node with best average value in DAG, returns root's child leading to it (greedy)
- `BestLoss`: Finds leaf with minimum raw loss evaluation, returns root's child leading to it

# Returns
The selected child node.
"""
function select_best_child_dag(
    root::DAGMCTSNode{S,T,N},
    table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}},
    config::DAGMCTSConfig,
    state::DAGMCTSState{S,T,N}
) where {S,T,N}
    if isempty(root.children)
        error("Cannot select from node with no children")
    end

    if config.selection_mode == VisitCount
        # Standard MCTS: select most visited child
        best_child = nothing
        best_visits = -1
        for child in root.children
            if child.visits > best_visits
                best_visits = child.visits
                best_child = child
            end
        end
        return best_child

    elseif config.selection_mode == BestValue
        # Greedy MCTS: find the node with the current best average value in the DAG.
        # We perform a fresh scan to avoid staleness (average_value is not monotonic).
        best_node = find_best_node_in_dag(root, table)

        if isnothing(best_node)
            # No non-root node visited yet, select first child
            return first(root.children)
        end

        # If best node is a direct child, return it
        for child in root.children
            if child === best_node
                return child
            end
        end

        # Trace back to find which child of root leads to best_node
        root_child = trace_to_root_child(best_node, root, table)

        if !isnothing(root_child)
            return root_child
        end

        # Fallback: if tracing fails (e.g. parents not tracked), pick child with best avg value
        return argmax(c -> average_value(c), root.children)

    elseif config.selection_mode == BestLoss
        # Select root child whose subtree contains the leaf with minimum raw loss
        min_node = state.min_loss_node

        if isnothing(min_node)
            return first(root.children)
        end

        # If min-loss node is a direct child, return it
        for child in root.children
            if child === min_node
                return child
            end
        end

        # Use tracked root child (set during backpropagation) — O(1)
        if !isnothing(state.min_loss_root_child)
            return state.min_loss_root_child
        end

        # Fallback: trace back using parent pointers if available
        root_child = trace_to_root_child(min_node, root, table)
        if !isnothing(root_child)
            return root_child
        end

        error("BestLoss selection failed: min_loss_node exists but is not reachable from root. " *
              "min_loss_root_child=$(state.min_loss_root_child), min_loss=$(state.min_loss)")
    else
        error("Unknown selection mode: $(config.selection_mode)")
    end
end

@doc raw"""
    dag_mcts_search(root::DAGMCTSNode{S,T,N}, table::Dict, loss::Loss, config::DAGMCTSConfig) where {S,T,N}

Run DAG-MCTS from a root node and return the best child.

Performs `config.num_simulations` iterations of DAG-MCTS.
Returns the best child according to the configured selection mode.

# Returns
Tuple of (best_polydisc, best_node)
"""
function dag_mcts_search(
    root::DAGMCTSNode{S,T,N},
    table::Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}},
    loss::Loss,
    config::DAGMCTSConfig,
    state::DAGMCTSState{S,T,N}
) where {S,T,N}
    # Ensure root is expanded
    expand_node!(root, table, config)

    if isempty(root.children)
        return root.polydisc, root, true
    end

    # Run simulations
    for _ in 1:config.num_simulations
        dag_mcts_simulation!(root, table, loss, config, state)
    end

    # Select best child according to selection mode
    best_child = select_best_child_dag(root, table, config, state)

    return best_child.polydisc, best_child, false
end

##################################################
# DAG-MCTS Optimizer Interface (compatible with OptimSetup)
##################################################

@doc raw"""
    dag_mcts_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, state::DAGMCTSState{S,T,N}, config::DAGMCTSConfig) where {S,T,N}

Perform one step of DAG-MCTS optimization.

This function follows the same interface as other optimizers (greedy_descent,
mcts_descent, etc.), making it compatible with `OptimSetup`.

# Arguments
- `loss::Loss`: The loss function structure
- `param::ValuationPolydisc{S,T,N}`: Current parameter values
- `state::DAGMCTSState{S,T,N}`: DAG-MCTS state (includes transposition table)
- `config::DAGMCTSConfig`: Configuration parameters

# Returns
`Tuple{ValuationPolydisc{S,T,N}, DAGMCTSState{S,T,N}}`: New parameters and updated state
"""
function dag_mcts_descent(
    loss::Loss,
    param::ValuationPolydisc{S,T,N},
    state::DAGMCTSState{S,T,N},
    config::DAGMCTSConfig
) where {S,T,N}
    # Run DAG-MCTS search
    best_polydisc, best_node, converged = dag_mcts_search(
        state.root,
        state.transposition_table,
        loss,
        config,
        state
    )

    # Update state for next step
    if config.persist_table
        # Keep transposition table, just update root
        # The new root should already be in the table
        new_root = get_or_create_node!(state.transposition_table, best_polydisc)
        state.root = new_root
        # Reset best-node tracking cache. The new root has different children,
        # so best_root_child is invalid. And best_value/best_node must be reset
        # because the stale best_value (from the old root's search) can prevent
        # backpropagate! from ever re-establishing best_root_child — the old
        # best_node's average drifts with new visits and may never exceed the
        # stale threshold. The actual information is preserved in the persisted
        # table's visit counts and values.
        state.best_node = nothing
        state.best_value = -Inf
        state.best_root_child = nothing
        state.best_root_action = 0
        state.min_loss_node = nothing
        state.min_loss = Inf
        state.min_loss_root_child = nothing
    else
        # Fresh search: clear table and create new root
        empty!(state.transposition_table)
        new_root = DAGMCTSNode(best_polydisc)
        state.transposition_table[HashedPolydisc(best_polydisc)] = new_root
        state.root = new_root
        state.best_node = nothing
        state.best_value = -Inf
        state.best_root_child = nothing
        state.best_root_action = 0
        state.min_loss_node = nothing
        state.min_loss = Inf
        state.min_loss_root_child = nothing
    end

    state.step_count += 1

    return best_polydisc, state, converged
end

@doc raw"""
    dag_mcts_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, config::DAGMCTSConfig=DAGMCTSConfig()) where {S,T,N}

Initialize an optimization setup for DAG-MCTS.

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `config::DAGMCTSConfig`: DAG-MCTS configuration (uses defaults if not provided)

# Returns
`OptimSetup`: Configured optimization setup for DAG-MCTS

# Example
```julia
# Set up DAG-MCTS optimizer with persistent transposition table
config = DAGMCTSConfig(num_simulations=200, persist_table=true)
optim = dag_mcts_descent_init(param, loss, config)

# Run optimization
for i in 1:100
    step!(optim)
    if i % 10 == 0
        println("Step \$i: Loss = ", eval_loss(optim))
        println("  Unique nodes: ", length(optim.state.transposition_table))
    end
end
```
"""
function dag_mcts_descent_init(
    param::ValuationPolydisc{S,T,N},
    loss::Loss,
    config::DAGMCTSConfig=DAGMCTSConfig()
) where {S,T,N}
    # Initialize transposition table with root
    table = Dict{HashedPolydisc{S,T,N}, DAGMCTSNode{S,T,N}}()
    root = DAGMCTSNode(param)
    table[HashedPolydisc(param)] = root

    # Initialize state
    state = DAGMCTSState{S,T,N}(root, table, 0, nothing, -Inf, nothing, 0, nothing, Inf, nothing)

    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> dag_mcts_descent(l, p, st, ctx),
        state,
        config,
        false
    )
end

##################################################
# Utility Functions
##################################################

@doc raw"""
    get_dag_stats(state::DAGMCTSState)

Get statistics about the DAG structure.

# Returns
NamedTuple with:
- `unique_nodes`: Number of unique nodes in transposition table
- `total_visits`: Sum of all visit counts
- `multi_parent_nodes`: Number of nodes with multiple parents (true transpositions)
"""
function get_dag_stats(state::DAGMCTSState)
    unique_nodes = length(state.transposition_table)
    total_visits = sum(n.visits for n in values(state.transposition_table))

    # Compute multi-parent count: use parent vectors if available, else forward scan
    has_parents = any(n -> !isempty(n.parents), values(state.transposition_table))
    if has_parents
        multi_parent = count(n -> length(n.parents) > 1, values(state.transposition_table))
    else
        # Forward scan: count how many nodes appear as children of 2+ distinct parents
        child_parent_count = Dict{UInt, Int}()
        for node in values(state.transposition_table)
            if node.is_expanded
                for child in node.children
                    cid = objectid(child)
                    child_parent_count[cid] = get(child_parent_count, cid, 0) + 1
                end
            end
        end
        multi_parent = count(v -> v > 1, values(child_parent_count))
    end

    return (
        unique_nodes = unique_nodes,
        total_visits = total_visits,
        multi_parent_nodes = multi_parent
    )
end

@doc raw"""
    print_dag_stats(state::DAGMCTSState, max_depth::Int=3)

Print statistics about the DAG-MCTS structure for debugging.
"""
function print_dag_stats(state::DAGMCTSState, max_depth::Int=3)
    stats = get_dag_stats(state)
    println("DAG-MCTS Statistics:")
    println("  Unique nodes in table: $(stats.unique_nodes)")
    println("  Total visits: $(stats.total_visits)")
    println("  Nodes with multiple parents: $(stats.multi_parent_nodes)")
    println("  Step count: $(state.step_count)")

    if stats.multi_parent_nodes > 0
        println("  Transposition ratio: $(round(stats.multi_parent_nodes / stats.unique_nodes * 100, digits=1))%")
    end
end

@doc raw"""
    verify_transposition_table(state::DAGMCTSState)

Verify the integrity of the transposition table.

Checks that:
1. All nodes in the table have polydiscs equal to their keys (via isequal)
2. All children of expanded nodes are in the table
3. Parent relationships are consistent

# Returns
`Bool`: true if table is consistent, false otherwise
"""
function verify_transposition_table(state::DAGMCTSState)
    table = state.transposition_table

    for (hashed_polydisc, node) in table
        # Check polydisc matches (use isequal for consistency with Dict semantics)
        if !isequal(node.polydisc, hashed_polydisc.polydisc)
            @warn "Node polydisc doesn't match table key"
            return false
        end

        # Check children are in table
        if node.is_expanded
            for child in node.children
                child_key = HashedPolydisc(child.polydisc)
                if !haskey(table, child_key)
                    @warn "Child not in transposition table"
                    return false
                end
                # Check the table returns the same instance
                retrieved = table[child_key]
                if retrieved !== child
                    @warn "Child in table is different instance"
                    return false
                end
            end
        end

        # Check parent relationships (only if parents were tracked)
        if !isempty(node.parents)
            for parent in node.parents
                found = false
                for child in parent.children
                    if child === node
                        found = true
                        break
                    end
                end
                if !found
                    @warn "Parent-child relationship inconsistent"
                    return false
                end
            end
        end
    end

    return true
end
