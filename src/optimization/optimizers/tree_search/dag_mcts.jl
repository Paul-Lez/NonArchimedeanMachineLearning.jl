"""
DAG-aware Monte Carlo Tree Search with a transposition table for shared
polydisc states.
"""

##################################################
# DAG-MCTS Variants and Node Structures
##################################################

abstract type AbstractDAGMCTSNode{S, T, N} end
abstract type AbstractDAGMCTSState{S, T, N} end

@doc raw"""
    DAGMCTSVariant

Selects the internal DAG-MCTS algorithm.

# Values
- `PathStatsDAGMCTS`: Search statistics are stored on
  root-to-node paths and selection uses DUCB. Node evaluations are shared through
  a transposition cache.
- `NodeStatsDAGMCTS`: Search statistics are stored
  directly on shared DAG nodes.
- `UCT1DAGMCTS`: Child selection follows UCT1 from Childs, Brodeur, and
  Kocsis: each shared position stores local move statistics ``Q(s,a)`` and
  ``N(s,a)``.
- `UCT2DAGMCTS`: Child selection follows UCT2 from Childs, Brodeur, and
  Kocsis: exploitation uses the shared child-node value ``Q(g(s,a))`` while
  the exploration term still uses the local move count ``N(s,a)``.
- `UCTMaxDAGMCTS`: Child selection is like node-stat DAG-MCTS, but the
  exploitation term uses the maximum value observed for the child instead of
  its average value.
"""
@enum DAGMCTSVariant PathStatsDAGMCTS NodeStatsDAGMCTS UCT1DAGMCTS UCT2DAGMCTS UCTMaxDAGMCTS

@doc raw"""
    DAGMCTSEdgeStats

Local move statistics for an action from a shared DAG node.

These are the paper's ``N_{s,a}`` and accumulated value used to compute
``Q_{s,a}``. They are kept separate from child node statistics because UCT2
uses local move visits in the exploration term even when the child node has
additional visits through transpositions.
"""
mutable struct DAGMCTSEdgeStats
    visits::Int
    total_value::Float64
    max_value::Float64
end

DAGMCTSEdgeStats() = DAGMCTSEdgeStats(0, 0.0, -Inf)
DAGMCTSEdgeStats(visits::Int, total_value::Float64) =
    DAGMCTSEdgeStats(visits, total_value, visits > 0 ? total_value / visits : -Inf)

average_value(stats::DAGMCTSEdgeStats) =
    stats.visits > 0 ? stats.total_value / stats.visits : 0.0

max_value(stats::DAGMCTSEdgeStats) =
    stats.visits > 0 ? stats.max_value : 0.0

@doc raw"""
    DAGMCTSNode{S,T,N}

A node in the node-statistics DAG-MCTS search graph.

Unlike standard MCTS nodes, DAG nodes can have multiple parents since the same
polydisc state can be reached via different action sequences.

# Fields
- `polydisc::ValuationPolydisc{S,T,N}`: The polydisc at this node
- `parents::Vector{DAGMCTSNode{S,T,N}}`: All parent nodes (can be multiple in a DAG)
- `children::Vector{DAGMCTSNode{S,T,N}}`: Child nodes indexed by action (1-based)
- `edge_stats::Vector{DAGMCTSEdgeStats}`: Per-action ``N(s,a)``/``Q(s,a)``
  statistics, parallel to `children`
- `visits::Int`: Total visit count N(s) aggregated from all paths
- `total_value::Float64`: Sum of all values Q(s) backpropagated through this node
- `max_value::Float64`: Maximum value backpropagated through this shared node
- `is_expanded::Bool`: Whether this node's children have been generated
- `is_terminal::Bool`: True if expansion produces no children (precision limit reached)
- `is_solved::Bool`: True if terminal, or expanded with all children solved
- `proven_value::Float64`: Exact value once solved (NaN if unsolved)
- `unsolved_children_count::Int`: Number of children not yet marked solved

# Design Decision (recorded for future experimentation)
We track all parents in a vector rather than having no parent information.
This is more expensive but useful for analysis and debugging. The alternative
would be to track no parent info and rely purely on the explicit path during traversal.
"""
mutable struct DAGMCTSNode{S, T, N} <: AbstractDAGMCTSNode{S, T, N}
    polydisc::ValuationPolydisc{S, T, N}
    parents::Vector{DAGMCTSNode{S, T, N}}
    children::Vector{DAGMCTSNode{S, T, N}}
    edge_stats::Vector{DAGMCTSEdgeStats}
    visits::Int
    total_value::Float64
    max_value::Float64
    is_expanded::Bool
    is_terminal::Bool          # true if expansion produces no children (precision limit reached)
    is_solved::Bool            # true if terminal, or expanded with all children solved
    proven_value::Float64      # exact value once solved (NaN if unsolved)
    unsolved_children_count::Int  # number of children not yet marked solved
end

@doc raw"""
    DAGMCTSNode(polydisc::ValuationPolydisc{S,T,N}) where {S,T,N}

Create a new DAG-MCTS node with the given polydisc and no parents.
"""
function DAGMCTSNode(polydisc::ValuationPolydisc{S, T, N}) where {S, T, N}
    return DAGMCTSNode{S, T, N}(
        polydisc,
        DAGMCTSNode{S, T, N}[],
        DAGMCTSNode{S, T, N}[],
        DAGMCTSEdgeStats[],
        0,
        0.0,
        -Inf,
        false,
        false,   # is_terminal
        false,   # is_solved
        NaN,     # proven_value
        0        # unsolved_children_count
    )
end

function DAGMCTSNode{S, T, N}(
        polydisc::ValuationPolydisc{S, T, N},
        parents::Vector{DAGMCTSNode{S, T, N}},
        children::Vector{DAGMCTSNode{S, T, N}},
        visits::Int,
        total_value::Float64,
        is_expanded::Bool,
        is_terminal::Bool,
        is_solved::Bool,
        proven_value::Float64,
        unsolved_children_count::Int
) where {S, T, N}
    return DAGMCTSNode{S, T, N}(
        polydisc,
        parents,
        children,
        [DAGMCTSEdgeStats() for _ in eachindex(children)],
        visits,
        total_value,
        visits > 0 ? total_value / visits : -Inf,
        is_expanded,
        is_terminal,
        is_solved,
        proven_value,
        unsolved_children_count
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

function max_value(node::DAGMCTSNode)
    return node.visits > 0 ? node.max_value : 0.0
end

@doc raw"""
    DAGMCTSPathStats

Visit/value statistics for a single root-to-node path in the path-style
DAG-MCTS variant.
"""
mutable struct DAGMCTSPathStats
    visits::Int
    total_value::Float64
end

DAGMCTSPathStats() = DAGMCTSPathStats(0, 0.0)

average_value(stats::DAGMCTSPathStats) =
    stats.visits > 0 ? stats.total_value / stats.visits : 0.0

const DAGMCTSPathKey{S, T, N} = Tuple{Vararg{HashedPolydisc{S, T, N}}}

@doc raw"""
    DAGMCTSPathNode{S,T,N}

A node in the path-style DAG-MCTS search graph.

This node stores only graph structure plus lightweight diagnostic aggregate
statistics. DUCB selection and backpropagation use `DAGMCTSPathStats` keyed by
root-to-node paths, not these node aggregates.
"""
mutable struct DAGMCTSPathNode{S, T, N} <: AbstractDAGMCTSNode{S, T, N}
    polydisc::ValuationPolydisc{S, T, N}
    parents::Vector{DAGMCTSPathNode{S, T, N}}
    children::Vector{DAGMCTSPathNode{S, T, N}}
    visits::Int
    total_value::Float64
    is_expanded::Bool
    is_terminal::Bool
end

function DAGMCTSPathNode(polydisc::ValuationPolydisc{S, T, N}) where {S, T, N}
    return DAGMCTSPathNode{S, T, N}(
        polydisc,
        DAGMCTSPathNode{S, T, N}[],
        DAGMCTSPathNode{S, T, N}[],
        0,
        0.0,
        false,
        false
    )
end

function average_value(node::DAGMCTSPathNode)
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
- `value_transform::Function`: Transform from loss to value (default: sigmoid_transform())
- `persist_table::Bool`: Whether to persist the transposition table across optimization steps
- `selection_mode::SelectionMode`: Strategy for selecting the next step
  (`VisitCount`, `BestValue`, or `BestLoss`)
- `track_parents::Bool`: Whether to retain parent pointers for debugging and verification
- `variant::DAGMCTSVariant`: Internal DAG-MCTS algorithm. The default is
  `PathStatsDAGMCTS`.

# Design Decision (recorded for future experimentation)
The `persist_table` option allows experimenting with:
- persist_table=true: Reuse learned information across steps (may grow large)
- persist_table=false: Fresh search each step
"""
struct DAGMCTSConfig
    num_simulations::Int
    exploration_constant::Float64
    degree::Int
    value_transform::Function
    persist_table::Bool
    selection_mode::SelectionMode
    track_parents::Bool
    variant::DAGMCTSVariant
end

@doc raw"""
    DAGMCTSConfig(; kwargs...)

Create a DAG-MCTS configuration with default settings.

# Keyword Arguments
- `num_simulations::Int=100`: Number of simulations per step
- `exploration_constant::Float64=1.41`: UCT exploration constant
- `degree::Int=1`: Child generation degree
- `value_transform::Function=DEFAULT_VALUE_TRANSFORM`: Loss to value transformation (see `sigmoid_transform`, `tanh_transform`, `negation_transform`)
- `persist_table`: Whether to persist the transposition table across steps.
  Defaults to `false` for the path-stat variant and `true` for the
  node-stat variant.
- `selection_mode`: Child selection strategy (`VisitCount`, `BestValue`, or
  `BestLoss`). Defaults to `BestValue` for the path-stat variant
  and `VisitCount` for the node-stat variant.
- `track_parents::Bool=false`: Whether to track parent pointers (needed for debug verification; off by default for performance)
- `variant::DAGMCTSVariant=PathStatsDAGMCTS`: Select the dag-mcts implementation.
  Use `UCT1DAGMCTS`, `UCT2DAGMCTS`, or `UCTMaxDAGMCTS` for transposition-aware
  node-stat variants.
"""
function DAGMCTSConfig(;
        num_simulations::Int = 100,
        exploration_constant::Float64 = 1.41,
        degree::Int = 1,
        value_transform::Function = DEFAULT_VALUE_TRANSFORM,
        persist_table::Union{Bool, Nothing} = nothing,
        selection_mode::Union{SelectionMode, Nothing} = nothing,
        track_parents::Bool = false,
        variant::DAGMCTSVariant = PathStatsDAGMCTS
)
    node_state_variant = variant in (NodeStatsDAGMCTS, UCT1DAGMCTS, UCT2DAGMCTS,
        UCTMaxDAGMCTS)
    resolved_persist_table = isnothing(persist_table) ? node_state_variant :
                             persist_table
    resolved_selection_mode = isnothing(selection_mode) ?
                              (variant == PathStatsDAGMCTS ? BestValue : VisitCount) :
                              selection_mode

    return DAGMCTSConfig(
        num_simulations,
        exploration_constant,
        degree,
        value_transform,
        resolved_persist_table,
        resolved_selection_mode,
        track_parents,
        variant
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
mutable struct DAGMCTSState{S, T, N} <: AbstractDAGMCTSState{S, T, N}
    root::DAGMCTSNode{S, T, N}
    transposition_table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}}
    step_count::Int
    best_node::Union{DAGMCTSNode{S, T, N}, Nothing}
    best_value::Float64
    best_root_child::Union{DAGMCTSNode{S, T, N}, Nothing}
    best_root_action::Int
    min_loss_node::Union{DAGMCTSNode{S, T, N}, Nothing}
    min_loss::Float64
    min_loss_root_child::Union{DAGMCTSNode{S, T, N}, Nothing}
end

@doc raw"""
    DAGMCTSPathState{S,T,N}

State for the path-style DAG-MCTS variant.

The graph is stored in `transposition_table`, objective evaluations are cached
in `evaluation_cache`, and MCTS statistics are stored in `path_stats` keyed by
root-to-node paths. Expansion state is also path-local: a shared DAG node may
already have children because another path expanded it, while the corresponding
MCTS tree node for this path would still be unexpanded.
"""
mutable struct DAGMCTSPathState{S, T, N} <: AbstractDAGMCTSState{S, T, N}
    root::DAGMCTSPathNode{S, T, N}
    transposition_table::Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}}
    evaluation_cache::Dict{HashedPolydisc{S, T, N}, Float64}
    loss_cache::Dict{HashedPolydisc{S, T, N}, Float64}
    path_stats::Dict{DAGMCTSPathKey{S, T, N}, DAGMCTSPathStats}
    expanded_paths::Set{DAGMCTSPathKey{S, T, N}}
    step_count::Int
    min_loss_node::Union{DAGMCTSPathNode{S, T, N}, Nothing}
    min_loss::Float64
    min_loss_root_child::Union{DAGMCTSPathNode{S, T, N}, Nothing}
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
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}},
        polydisc::ValuationPolydisc{S, T, N},
        parent::Union{DAGMCTSNode{S, T, N}, Nothing} = nothing
) where {S, T, N}
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

@doc raw"""
    get_or_create_path_node!(table, polydisc, parent=nothing)

Look up or create a node for the path-stat DAG-MCTS variant.
"""
function get_or_create_path_node!(
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}},
        polydisc::ValuationPolydisc{S, T, N},
        parent::Union{DAGMCTSPathNode{S, T, N}, Nothing} = nothing
) where {S, T, N}
    key = HashedPolydisc(polydisc)
    node = get!(table, key) do
        DAGMCTSPathNode(polydisc)
    end

    if !isnothing(parent) && !(parent in node.parents)
        push!(node.parents, parent)
    end

    return node
end

function _extend_path_key(
        key::DAGMCTSPathKey{S, T, N},
        child::DAGMCTSPathNode{S, T, N}
) where {S, T, N}
    return (key..., HashedPolydisc(child.polydisc))
end

function _root_path_key(root::DAGMCTSPathNode{S, T, N}) where {S, T, N}
    return (HashedPolydisc(root.polydisc),)
end

function _is_path_expanded(
        state::DAGMCTSPathState{S, T, N},
        path_key::DAGMCTSPathKey{S, T, N}
) where {S, T, N}
    return path_key in state.expanded_paths
end

function _mark_path_expanded!(
        state::DAGMCTSPathState{S, T, N},
        path_key::DAGMCTSPathKey{S, T, N}
) where {S, T, N}
    push!(state.expanded_paths, path_key)
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
    exploration = exploration_constant * sqrt(log(max(parent_visits, 1)) / node.visits)
    return exploitation + exploration
end

@doc raw"""
    uct1_score(edge_stats::DAGMCTSEdgeStats, parent_visits::Int, exploration_constant::Float64)

Compute the UCT1 score from Childs, Brodeur, and Kocsis:

```math
Q(s,a) + c \sqrt{\log N(s) / N(s,a)}.
```

Unvisited actions are assigned `Inf`.
"""
function uct1_score(
        edge_stats::DAGMCTSEdgeStats,
        parent_visits::Int,
        exploration_constant::Float64
)
    if edge_stats.visits == 0
        return Inf
    end
    exploitation = average_value(edge_stats)
    exploration = exploration_constant * sqrt(log(max(parent_visits, 1)) / edge_stats.visits)
    return exploitation + exploration
end

@doc raw"""
    uct2_score(child::DAGMCTSNode, edge_stats::DAGMCTSEdgeStats, parent_visits::Int, exploration_constant::Float64)

Compute the UCT2 score from Childs, Brodeur, and Kocsis:

```math
Q(g(s,a)) + c \sqrt{\log N(s) / N(s,a)}.
```

The exploitation term uses the shared child node's value estimate, while the
exploration term uses the local action count. Unvisited actions are assigned
`Inf`.
"""
function uct2_score(
        child::DAGMCTSNode,
        edge_stats::DAGMCTSEdgeStats,
        parent_visits::Int,
        exploration_constant::Float64
)
    if edge_stats.visits == 0
        return Inf
    end
    exploitation = average_value(child)
    exploration = exploration_constant * sqrt(log(max(parent_visits, 1)) / edge_stats.visits)
    return exploitation + exploration
end

@doc raw"""
    uctmax_score(node::DAGMCTSNode, parent_visits::Int, exploration_constant::Float64)

Compute a UCT score whose exploitation term is the maximum value observed for
the child node rather than the average value:

```math
\max Q(s) + c \sqrt{\log N(parent) / N(s)}.
```

Unvisited nodes are assigned `Inf`.
"""
function uctmax_score(node::DAGMCTSNode, parent_visits::Int,
        exploration_constant::Float64)
    if node.visits == 0
        return Inf
    end
    exploitation = max_value(node)
    exploration = exploration_constant * sqrt(log(max(parent_visits, 1)) / node.visits)
    return exploitation + exploration
end

function _ensure_edge_stats!(node::DAGMCTSNode)
    if length(node.edge_stats) < length(node.children)
        append!(node.edge_stats,
            (DAGMCTSEdgeStats() for _ in 1:(length(node.children) - length(node.edge_stats))))
    elseif length(node.edge_stats) > length(node.children)
        resize!(node.edge_stats, length(node.children))
    end
    return node.edge_stats
end

@doc raw"""
    select_child(node::DAGMCTSNode, exploration_constant::Float64, variant::DAGMCTSVariant=NodeStatsDAGMCTS)

Select the child with the highest UCT score.

# Returns
Tuple of (action_index, child_node) for the best child
"""
function select_child(
        node::DAGMCTSNode,
        exploration_constant::Float64,
        variant::DAGMCTSVariant = NodeStatsDAGMCTS
)
    @assert !isempty(node.children) "Cannot select from node with no children"

    edge_stats = _ensure_edge_stats!(node)
    best_score = -Inf
    best_action = nothing
    best_child = nothing

    for (action, child) in enumerate(node.children)
        if child.is_solved
            # Solved children use proven_value directly (no exploration bonus)
            score = child.proven_value
        elseif variant == UCT1DAGMCTS
            score = uct1_score(edge_stats[action], node.visits, exploration_constant)
        elseif variant == UCT2DAGMCTS
            score = uct2_score(child, edge_stats[action], node.visits, exploration_constant)
        elseif variant == UCTMaxDAGMCTS
            score = uctmax_score(child, node.visits, exploration_constant)
        else
            score = uct_score(child, node.visits, exploration_constant)
        end
        if score > best_score
            best_score = score
            best_action = action
            best_child = child
        end
    end

    return best_action, best_child
end

@doc raw"""
    ducb_score(path_stats, path_key, child, exploration_constant)

Compute the path-statistics DAG upper confidence bound

```math
DUCB(P, y) = Q(P \cdot y) + c \sqrt{\log N(P) / N(P \cdot y)}.
```

Unvisited path extensions are assigned `Inf`.
"""
function ducb_score(
        path_stats::Dict{DAGMCTSPathKey{S, T, N}, DAGMCTSPathStats},
        path_key::DAGMCTSPathKey{S, T, N},
        child::DAGMCTSPathNode{S, T, N},
        exploration_constant::Float64
) where {S, T, N}
    child_key = _extend_path_key(path_key, child)
    child_stats = get(path_stats, child_key, DAGMCTSPathStats())
    if child_stats.visits == 0
        return Inf
    end

    parent_stats = get(path_stats, path_key, DAGMCTSPathStats())
    parent_visits = max(parent_stats.visits, 1)
    exploitation = average_value(child_stats)
    exploration = exploration_constant * sqrt(log(parent_visits) / child_stats.visits)
    return exploitation + exploration
end

function select_child(
        node::DAGMCTSPathNode{S, T, N},
        path_key::DAGMCTSPathKey{S, T, N},
        path_stats::Dict{DAGMCTSPathKey{S, T, N}, DAGMCTSPathStats},
        exploration_constant::Float64
) where {S, T, N}
    @assert !isempty(node.children) "Cannot select from node with no children"

    best_score = -Inf
    best_action = 1
    best_child = node.children[1]

    for (action, child) in enumerate(node.children)
        score = ducb_score(path_stats, path_key, child, exploration_constant)
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
        node::DAGMCTSNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}},
        config::DAGMCTSConfig
) where {S, T, N}
    if node.is_expanded
        return
    end

    # Generate child polydiscs
    child_polydiscs = children(node.polydisc, config.degree)

    # Pre-size children vector and create/retrieve nodes for each child
    # When track_parents=false, pass nothing as parent to skip parent vector push
    n_children = length(child_polydiscs)
    resize!(node.children, n_children)
    resize!(node.edge_stats, n_children)
    parent_for_link = config.track_parents ? node : nothing
    for (i, child_polydisc) in enumerate(child_polydiscs)
        child_node = get_or_create_node!(table, child_polydisc, parent_for_link)
        node.children[i] = child_node
        node.edge_stats[i] = DAGMCTSEdgeStats()
    end

    node.is_expanded = true

    # Terminal detection: no children means precision limit reached
    if isempty(node.children)
        node.is_terminal = true
        node.is_solved = true
        # proven_value will be set by the caller after evaluating the loss
        node.unsolved_children_count = 0
    else
        # Count unsolved children (some may already be solved via transposition table)
        node.unsolved_children_count = count(c -> !c.is_solved, node.children)

        # If all children happen to already be solved (transposition), mark this node solved too
        if node.unsolved_children_count == 0
            node.is_solved = true
            node.proven_value = maximum(child.proven_value for child in node.children)
        end
    end
end

function expand_node!(
        node::DAGMCTSPathNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}},
        config::DAGMCTSConfig
) where {S, T, N}
    if node.is_expanded
        return
    end

    child_polydiscs = children(node.polydisc, config.degree)

    n_children = length(child_polydiscs)
    resize!(node.children, n_children)
    parent_for_link = config.track_parents ? node : nothing
    for (i, child_polydisc) in enumerate(child_polydiscs)
        child_node = get_or_create_path_node!(table, child_polydisc, parent_for_link)
        node.children[i] = child_node
    end

    node.is_expanded = true
    node.is_terminal = isempty(node.children)
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
function select_path(
        root::DAGMCTSNode,
        exploration_constant::Float64,
        variant::DAGMCTSVariant = NodeStatsDAGMCTS
)
    path = [root]
    node = root

    while node.is_expanded && !isempty(node.children) && node.visits > 0 && !node.is_solved
        _, child = select_child(node, exploration_constant, variant)
        push!(path, child)
        node = child
    end

    return path
end

function select_path(
        root::DAGMCTSPathNode{S, T, N},
        state::DAGMCTSPathState{S, T, N},
        exploration_constant::Float64
) where {S, T, N}
    path = DAGMCTSPathNode{S, T, N}[root]
    path_key = (HashedPolydisc(root.polydisc),)
    node = root

    while _is_path_expanded(state, path_key) && !isempty(node.children)
        _, child = select_child(node, path_key, state.path_stats, exploration_constant)
        push!(path, child)
        path_key = _extend_path_key(path_key, child)
        node = child
    end

    return path, path_key
end

@doc raw"""
    evaluate_node(node::DAGMCTSNode{S,T,N}, loss::Loss, config::DAGMCTSConfig) where {S,T,N}

Evaluate a node using the loss function and transform to value.

# Returns
`Float64`: The transformed value (higher is better)
"""
function evaluate_node(node::DAGMCTSNode{S, T, N}, loss::Loss, config::DAGMCTSConfig) where {
        S, T, N}
    loss_value = loss.eval([node.polydisc])[1]
    return config.value_transform(loss_value)
end

function evaluate_node(
        node::DAGMCTSPathNode{S, T, N},
        loss::Loss,
        config::DAGMCTSConfig
) where {S, T, N}
    loss_value = loss.eval([node.polydisc])[1]
    return config.value_transform(loss_value)
end

function cached_evaluate_node!(
        state::DAGMCTSPathState{S, T, N},
        node::DAGMCTSPathNode{S, T, N},
        loss::Loss,
        config::DAGMCTSConfig
) where {S, T, N}
    key = HashedPolydisc(node.polydisc)
    if haskey(state.evaluation_cache, key)
        return state.evaluation_cache[key], state.loss_cache[key]
    end

    loss_value = loss.eval([node.polydisc])[1]
    value = config.value_transform(loss_value)
    state.evaluation_cache[key] = value
    state.loss_cache[key] = loss_value
    return value, loss_value
end

@doc raw"""
    backpropagate!(path::Vector{<:DAGMCTSNode}, value::Float64)

Backpropagate a value through all nodes in the explicit path.

This is the Path-wise approach for DAG backpropagation - we iterate through
the specific path taken during this traversal rather than following
parent pointers (this is ambiguous in a DAG!).

# Arguments
- `path`: Vector of nodes from root to leaf representing this traversal's path
- `value`: The transformed value to backpropagate
- `eval_node`: The leaf node where the loss was evaluated
- `loss_value`: The raw loss value at the leaf (before value_transform)

# Note
This updates the global N(s) and Q(s) statistics in each shared node instance.
"""
function backpropagate!(path::Vector{<:DAGMCTSNode}, value::Float64, state::DAGMCTSState,
        eval_node::DAGMCTSNode = path[end], loss_value::Float64 = NaN)
    # The root is path[1]; the root child is path[2] if path has length >= 2
    root_child = length(path) >= 2 ? path[2] : nothing
    for i in 1:(length(path) - 1)
        parent = path[i]
        child = path[i + 1]
        action = findfirst(c -> c === child, parent.children)
        if !isnothing(action)
            _ensure_edge_stats!(parent)
            stats = parent.edge_stats[action]
            stats.visits += 1
            stats.total_value += value
            stats.max_value = max(stats.max_value, value)
        end
    end

    for (i, node) in enumerate(path)
        node.visits += 1
        node.total_value += value
        node.max_value = max(node.max_value, value)
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

function backpropagate!(
        path::Vector{DAGMCTSPathNode{S, T, N}},
        path_key::DAGMCTSPathKey{S, T, N},
        value::Float64,
        state::DAGMCTSPathState{S, T, N},
        eval_node::DAGMCTSPathNode{S, T, N} = path[end],
        loss_value::Float64 = NaN
) where {S, T, N}
    for i in eachindex(path)
        prefix_key = path_key[1:i]
        stats = get!(state.path_stats, prefix_key) do
            DAGMCTSPathStats()
        end
        stats.visits += 1
        stats.total_value += value

        # Diagnostic aggregate only. DUCB never reads node-level statistics.
        node = path[i]
        node.visits += 1
        node.total_value += value
    end

    if !isnan(loss_value) && loss_value < state.min_loss
        state.min_loss = loss_value
        state.min_loss_node = eval_node
        state.min_loss_root_child = length(path) >= 2 ? path[2] : nothing
    end
end

##################################################
# Solved Status Propagation
##################################################

@doc raw"""
    check_solved!(node::DAGMCTSNode)

Check if a DAG node should be marked as solved.
A node is solved when expanded AND all children are solved.
(DAG-MCTS always generates all children, so `is_expanded` implies fully expanded.)

Sets `proven_value` to the max of children's `proven_values`.

Returns `true` if the node was newly marked as solved.
"""
function check_solved!(node::DAGMCTSNode)
    # If the node has already been marked as solved or isn't expanded, then we return false
    # since there's no change in status
    if node.is_solved || !node.is_expanded
        return false
    end
    if node.unsolved_children_count > 0
        return false
    end
    node.is_solved = true
    node.proven_value = maximum(child.proven_value for child in node.children)
    return true
end

@doc raw"""
    propagate_solved_up_dag!(node::DAGMCTSNode, path::Vector{<:DAGMCTSNode})

Propagate solved status upward in the DAG after `node` has been marked solved.

When parent pointers are available (`track_parents=true`), uses BFS to notify
ALL parents across the DAG. When parent pointers are unavailable, propagates
only along the explicit simulation path (pragmatic fallback — some parents may
not be notified immediately, but will catch up in future simulations).
"""
function propagate_solved_up_dag!(node::DAGMCTSNode, path::Vector{<:DAGMCTSNode})
    if !isempty(node.parents)
        # Parent pointers available: BFS to notify ALL parents
        queue = [node]
        while !isempty(queue)
            solved_child = popfirst!(queue)
            for parent in solved_child.parents
                if parent.is_solved
                    continue
                end
                parent.unsolved_children_count -= 1
                if check_solved!(parent)
                    push!(queue, parent)
                end
            end
        end
    else
        # No parent pointers: propagate along the explicit path only
        node_idx = findlast(n -> n === node, path)
        if isnothing(node_idx) || node_idx <= 1
            return
        end
        for i in (node_idx - 1):-1:1
            parent = path[i]
            if parent.is_solved
                break
            end
            parent.unsolved_children_count -= 1
            if !check_solved!(parent)
                break
            end
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
        root::DAGMCTSNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}},
        loss::Loss,
        config::DAGMCTSConfig,
        state::DAGMCTSState{S, T, N}
) where {S, T, N}
    # Phase 1: Selection - traverse using UCT, maintaining path stack
    path = select_path(root, config.exploration_constant, config.variant)
    leaf = path[end]

    # If we reached a solved node, use its proven value directly
    if leaf.is_solved
        backpropagate!(path, leaf.proven_value, state, leaf)
        return leaf.proven_value
    end

    # Phase 2: Expansion - expand using transposition table
    if !leaf.is_expanded
        expand_node!(leaf, table, config)
    end

    # Handle terminal leaf: evaluate, set proven_value, propagate solved status
    if leaf.is_terminal
        if isnan(leaf.proven_value)
            loss_value = loss.eval([leaf.polydisc])[1]
            value = config.value_transform(loss_value)
            leaf.proven_value = value
            backpropagate!(path, value, state, leaf, loss_value)
        else
            backpropagate!(path, leaf.proven_value, state, leaf)
        end
        propagate_solved_up_dag!(leaf, path)
        return leaf.proven_value
    end

    # Handle node that became solved during expansion (all children already solved via transposition)
    if leaf.is_solved
        backpropagate!(path, leaf.proven_value, state, leaf)
        propagate_solved_up_dag!(leaf, path)
        return leaf.proven_value
    end

    # Choose a node to evaluate, preferring unsolved unvisited children
    if !isempty(leaf.children)
        unvisited = [c for c in leaf.children if c.visits == 0 && !c.is_solved]
        if !isempty(unvisited)
            eval_node = rand(unvisited)
        else
            unsolved = [c for c in leaf.children if !c.is_solved]
            # The first branch should never occur here, so we should remove it at some point.
            eval_node = isempty(unsolved) ? rand(leaf.children) : rand(unsolved)
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
function find_best_node_in_dag(root::DAGMCTSNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}}) where {S, T, N}
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
        target::DAGMCTSNode{S, T, N},
        root::DAGMCTSNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}}
) where {S, T, N}
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
    # visited tracks nodes by identity to avoid cycles (Note: do we actually need to do this?)
    visited = Set{UInt}()
    queue = DAGMCTSNode{S, T, N}[target]
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
- `VisitCount`: Returns child with highest visit count (standard MCTS)
- `BestValue`: Finds node with best average value in DAG, returns root's child leading to it (greedy)
- `BestLoss`: Finds leaf with minimum raw loss evaluation, returns root's child leading to it

# Returns
The selected child node.
"""
function select_best_child_dag(
        root::DAGMCTSNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}},
        config::DAGMCTSConfig,
        state::DAGMCTSState{S, T, N}
) where {S, T, N}
    if isempty(root.children)
        error("Cannot select from node with no children")
    end

    # If root is solved, select child with best proven value
    if root.is_solved
        return argmax(c -> c.proven_value, root.children)
    end

    if config.selection_mode == VisitCount
        # Standard MCTS: select most visited child. UCT1/UCT2 use local
        # action counts N(root,a), not the child's global transposition visits.
        if config.variant in (UCT1DAGMCTS, UCT2DAGMCTS)
            _ensure_edge_stats!(root)
            best_action = 1
            best_visits = root.edge_stats[1].visits
            for action in 2:length(root.edge_stats)
                visits = root.edge_stats[action].visits
                if visits > best_visits
                    best_action = action
                    best_visits = visits
                end
            end
            return root.children[best_action]
        end

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
Tuple of `(best_polydisc, best_node, converged)`.
"""
function dag_mcts_search(
        root::DAGMCTSNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}},
        loss::Loss,
        config::DAGMCTSConfig,
        state::DAGMCTSState{S, T, N}
) where {S, T, N}
    # Ensure root is expanded
    expand_node!(root, table, config)

    # Handle terminal root
    if root.is_terminal
        loss_value = loss.eval([root.polydisc])[1]
        root.proven_value = config.value_transform(loss_value)
        return root.polydisc, root, true
    end

    if isempty(root.children)
        return root.polydisc, root, true
    end

    # Run simulations
    for _ in 1:config.num_simulations
        # Early exit if root is fully solved
        if root.is_solved
            break
        end
        dag_mcts_simulation!(root, table, loss, config, state)
    end

    # Select best child according to selection mode
    best_child = select_best_child_dag(root, table, config, state)

    # TODO: we should handle the case where the root is solved in a smarter way?
    return best_child.polydisc, best_child, root.is_terminal
end

@doc raw"""
    dag_mcts_path_simulation!(root, table, loss, config, state)

Run one path-style DAG-MCTS simulation.

Selection uses DUCB on path extensions, evaluation is cached by polydisc in the
transposition cache, and backpropagation updates every prefix of the sampled
path.
"""
function dag_mcts_path_simulation!(
        root::DAGMCTSPathNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}},
        loss::Loss,
        config::DAGMCTSConfig,
        state::DAGMCTSPathState{S, T, N}
) where {S, T, N}
    path, path_key = select_path(root, state, config.exploration_constant)
    leaf = path[end]

    if !leaf.is_expanded
        expand_node!(leaf, table, config)
    end
    _mark_path_expanded!(state, path_key)

    if leaf.is_terminal
        value, loss_value = cached_evaluate_node!(state, leaf, loss, config)
        backpropagate!(path, path_key, value, state, leaf, loss_value)
        return value
    end

    unvisited = [child for child in leaf.children
                 if get(state.path_stats, _extend_path_key(path_key, child),
                     DAGMCTSPathStats()).visits == 0]
    eval_node = if isempty(unvisited)
        error("selected an unexpanded non-terminal path whose child paths were already all visited")
    else
        rand(unvisited)
    end
    push!(path, eval_node)
    path_key = _extend_path_key(path_key, eval_node)

    value, loss_value = cached_evaluate_node!(state, eval_node, loss, config)
    backpropagate!(path, path_key, value, state, eval_node, loss_value)

    return value
end

function _path_child_stats(
        state::DAGMCTSPathState{S, T, N},
        root_key::DAGMCTSPathKey{S, T, N},
        child::DAGMCTSPathNode{S, T, N}
) where {S, T, N}
    child_key = _extend_path_key(root_key, child)
    return get(state.path_stats, child_key, DAGMCTSPathStats())
end

function _best_value_path(
        node::DAGMCTSPathNode{S, T, N},
        path_key::DAGMCTSPathKey{S, T, N},
        state::DAGMCTSPathState{S, T, N}
) where {S, T, N}
    stats = get(state.path_stats, path_key, nothing)
    if isnothing(stats) || stats.visits == 0
        return nothing, -Inf
    end

    best_path = path_key
    best_value = average_value(stats)

    for child in node.children
        child_key = _extend_path_key(path_key, child)
        child_best_path, child_best_value = _best_value_path(child, child_key, state)
        if !isnothing(child_best_path) && child_best_value > best_value
            best_path = child_best_path
            best_value = child_best_value
        end
    end

    return best_path, best_value
end

function _root_child_on_path(
        path_key::DAGMCTSPathKey{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}}
) where {S, T, N}
    # BestValue searches over all visited paths, but the optimiser can only
    # descend one level. Return the current root child that starts this path.
    if length(path_key) < 2
        error("Cannot identify a root child from a path with length $(length(path_key))")
    end
    root_child_key = path_key[2]
    if !haskey(table, root_child_key)
        error("BestValue selection failed: root child path key is not in the transposition table")
    end
    return table[root_child_key]
end

function select_best_child_dag(
        root::DAGMCTSPathNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}},
        config::DAGMCTSConfig,
        state::DAGMCTSPathState{S, T, N}
) where {S, T, N}
    if isempty(root.children)
        error("Cannot select from node with no children")
    end

    root_key = _root_path_key(root)

    if config.selection_mode == VisitCount
        return argmax(c -> _path_child_stats(state, root_key, c).visits, root.children)
    elseif config.selection_mode == BestValue
        best_path, _ = _best_value_path(root, root_key, state)
        if isnothing(best_path)
            return first(root.children)
        end

        if best_path == root_key
            # MCTS can report the root itself as the best visited node. Since
            # the optimiser must still descend one level, choose the best
            # already-visited root child as the next move.
            visited_children = [child for child in root.children
                                if _path_child_stats(state, root_key, child).visits > 0]
            if isempty(visited_children)
                error("BestValue selection found the root as the best path, " *
                      "but no visited root children are available")
            end
            return argmax(c -> average_value(_path_child_stats(state, root_key, c)),
                visited_children)
        end

        # Otherwise a deeper path won, so descend via the root child that starts
        # that path.
        return _root_child_on_path(best_path, table)
    elseif config.selection_mode == BestLoss
        if !isnothing(state.min_loss_root_child)
            return state.min_loss_root_child
        end
        return first(root.children)
    else
        error("Unknown selection mode: $(config.selection_mode)")
    end
end

@doc raw"""
    dag_mcts_path_search(root, table, loss, config, state)

Run the path-statistics DAG-MCTS update rule and return the selected
root child.
"""
function dag_mcts_path_search(
        root::DAGMCTSPathNode{S, T, N},
        table::Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}},
        loss::Loss,
        config::DAGMCTSConfig,
        state::DAGMCTSPathState{S, T, N}
) where {S, T, N}
    root_key = _root_path_key(root)
    if !root.is_expanded
        expand_node!(root, table, config)
    end
    _mark_path_expanded!(state, root_key)

    if root.is_terminal || isempty(root.children)
        cached_evaluate_node!(state, root, loss, config)
        return root.polydisc, root, true
    end

    for _ in 1:config.num_simulations
        if root.is_terminal
            break
        end
        dag_mcts_path_simulation!(root, table, loss, config, state)
    end

    best_child = select_best_child_dag(root, table, config, state)
    return best_child.polydisc, best_child, root.is_terminal
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
`Tuple{ValuationPolydisc{S,T,N}, DAGMCTSState{S,T,N}, Bool}`: New parameters,
updated state, and convergence status
"""
function dag_mcts_descent(
        loss::Loss,
        param::ValuationPolydisc{S, T, N},
        state::DAGMCTSState{S, T, N},
        config::DAGMCTSConfig
) where {S, T, N}
    # Run DAG-MCTS search
    best_polydisc, best_node,
    converged = dag_mcts_search(
        state.root,
        state.transposition_table,
        loss,
        config,
        state
    )

    # Update state for next step
    if config.persist_table
        # DAG-first architecture: the DAG persists via node.children pointers.
        # We re-root the tree and rebuild the transposition table as an ephemeral
        # index of only the reachable subtree. This prevents unbounded table growth
        # while preserving cached search information below the new root.
        new_root = get_or_create_node!(state.transposition_table, best_polydisc)
        state.root = new_root
        rebuild_table_from_subtree!(state, config)
    else
        # Fresh search: clear table and create new root
        empty!(state.transposition_table)
        new_root = DAGMCTSNode(best_polydisc)
        state.transposition_table[HashedPolydisc(best_polydisc)] = new_root
        state.root = new_root
    end

    # Reset best-node tracking (always needed: new root means old trackers are stale)
    state.best_node = nothing
    state.best_value = -Inf
    state.best_root_child = nothing
    state.best_root_action = 0
    state.min_loss_node = nothing
    state.min_loss = Inf
    state.min_loss_root_child = nothing

    state.step_count += 1

    return best_polydisc, state, converged
end

function _reset_path_search_state!(state::DAGMCTSPathState{S, T, N}) where {S, T, N}
    empty!(state.path_stats)
    state.path_stats[_root_path_key(state.root)] = DAGMCTSPathStats()
    empty!(state.expanded_paths)
    state.min_loss_node = nothing
    state.min_loss = Inf
    state.min_loss_root_child = nothing
end

function _reroot_path_search_state!(
        state::DAGMCTSPathState{S, T, N},
        old_root_key::DAGMCTSPathKey{S, T, N},
        new_root_hash::HashedPolydisc{S, T, N}
) where {S, T, N}
    old_prefix = (old_root_key[1], new_root_hash)
    new_root_key = _root_path_key(state.root)

    new_path_stats = Dict{DAGMCTSPathKey{S, T, N}, DAGMCTSPathStats}()
    for (path_key, stats) in state.path_stats
        if length(path_key) >= 2 && path_key[1:2] == old_prefix
            new_path_stats[path_key[2:end]] = stats
        end
    end
    if !haskey(new_path_stats, new_root_key)
        new_path_stats[new_root_key] = DAGMCTSPathStats()
    end
    state.path_stats = new_path_stats

    new_expanded_paths = Set{DAGMCTSPathKey{S, T, N}}()
    for path_key in state.expanded_paths
        if length(path_key) >= 2 && path_key[1:2] == old_prefix
            push!(new_expanded_paths, path_key[2:end])
        end
    end
    state.expanded_paths = new_expanded_paths

    state.min_loss_node = nothing
    state.min_loss = Inf
    state.min_loss_root_child = nothing
end

function dag_mcts_descent(
        loss::Loss,
        param::ValuationPolydisc{S, T, N},
        state::DAGMCTSPathState{S, T, N},
        config::DAGMCTSConfig
) where {S, T, N}
    old_root_key = _root_path_key(state.root)
    best_polydisc, best_node,
    converged = dag_mcts_path_search(
        state.root,
        state.transposition_table,
        loss,
        config,
        state
    )

    if config.persist_table
        new_root = get_or_create_path_node!(state.transposition_table, best_polydisc)
        new_root_hash = HashedPolydisc(best_polydisc)
        state.root = new_root
        rebuild_table_from_subtree!(state, config)
        _reroot_path_search_state!(state, old_root_key, new_root_hash)
    else
        empty!(state.transposition_table)
        empty!(state.evaluation_cache)
        empty!(state.loss_cache)
        new_root = DAGMCTSPathNode(best_polydisc)
        state.transposition_table[HashedPolydisc(best_polydisc)] = new_root
        state.root = new_root
        _reset_path_search_state!(state)
    end

    state.step_count += 1

    return best_polydisc, state, converged
end

@doc raw"""
    rebuild_table_from_subtree!(state::DAGMCTSState{S,T,N}, config::DAGMCTSConfig) where {S,T,N}

Rebuild the transposition table to contain only nodes reachable from `state.root`.

Performs a BFS traversal down from the root, collecting all reachable nodes into a
fresh dictionary. The old table is replaced, allowing Julia's GC to reclaim any
unreachable nodes. If `config.track_parents` is enabled, parent arrays of surviving
nodes are filtered to remove references to pruned ancestors.

This is the core of the "DAG-first" architecture: the DAG structure is maintained
by `node.children` pointers, and the transposition table is merely an ephemeral
index rebuilt each step.
"""
function rebuild_table_from_subtree!(state::DAGMCTSState{S, T, N}, config::DAGMCTSConfig) where {
        S, T, N}
    new_table = Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}}()

    # BFS from root to collect all reachable nodes
    queue = DAGMCTSNode{S, T, N}[state.root]
    new_table[HashedPolydisc(state.root.polydisc)] = state.root

    while !isempty(queue)
        curr = popfirst!(queue)
        if curr.is_expanded
            for child in curr.children
                key = HashedPolydisc(child.polydisc)
                if !haskey(new_table, key)
                    new_table[key] = child
                    push!(queue, child)
                end
            end
        end
    end

    # If parent tracking is enabled, filter out references to pruned ancestors
    if config.track_parents
        reachable_ids = Set{UInt}(objectid(node) for node in values(new_table))
        for node in values(new_table)
            filter!(p -> objectid(p) in reachable_ids, node.parents)
        end
    end

    state.transposition_table = new_table
end

function rebuild_table_from_subtree!(
        state::DAGMCTSPathState{S, T, N},
        config::DAGMCTSConfig
) where {S, T, N}
    new_table = Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}}()

    queue = DAGMCTSPathNode{S, T, N}[state.root]
    new_table[HashedPolydisc(state.root.polydisc)] = state.root

    while !isempty(queue)
        curr = popfirst!(queue)
        if curr.is_expanded
            for child in curr.children
                key = HashedPolydisc(child.polydisc)
                if !haskey(new_table, key)
                    new_table[key] = child
                    push!(queue, child)
                end
            end
        end
    end

    if config.track_parents
        reachable_ids = Set{UInt}(objectid(node) for node in values(new_table))
        for node in values(new_table)
            filter!(p -> objectid(p) in reachable_ids, node.parents)
        end
    end

    new_evaluation_cache = Dict{HashedPolydisc{S, T, N}, Float64}()
    new_loss_cache = Dict{HashedPolydisc{S, T, N}, Float64}()
    for key in keys(new_table)
        if haskey(state.evaluation_cache, key)
            new_evaluation_cache[key] = state.evaluation_cache[key]
        end
        if haskey(state.loss_cache, key)
            new_loss_cache[key] = state.loss_cache[key]
        end
    end

    state.transposition_table = new_table
    state.evaluation_cache = new_evaluation_cache
    state.loss_cache = new_loss_cache
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
function _dag_mcts_node_descent_init(
        param::ValuationPolydisc{S, T, N},
        loss::Loss,
        config::DAGMCTSConfig
) where {S, T, N}
    # Initialize transposition table with root
    table = Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}}()
    root = DAGMCTSNode(param)
    table[HashedPolydisc(param)] = root

    # Initialize state
    state = DAGMCTSState{S, T, N}(
        root, table, 0, nothing, -Inf, nothing, 0, nothing, Inf, nothing)

    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> dag_mcts_descent(l, p, st, ctx),
        state,
        config,
        false
    )
end

function _dag_mcts_path_descent_init(
        param::ValuationPolydisc{S, T, N},
        loss::Loss,
        config::DAGMCTSConfig
) where {S, T, N}
    table = Dict{HashedPolydisc{S, T, N}, DAGMCTSPathNode{S, T, N}}()
    root = DAGMCTSPathNode(param)
    root_key = HashedPolydisc(param)
    table[root_key] = root

    path_stats = Dict{DAGMCTSPathKey{S, T, N}, DAGMCTSPathStats}()
    path_stats[(root_key,)] = DAGMCTSPathStats()
    expanded_paths = Set{DAGMCTSPathKey{S, T, N}}()

    state = DAGMCTSPathState{S, T, N}(
        root,
        table,
        Dict{HashedPolydisc{S, T, N}, Float64}(),
        Dict{HashedPolydisc{S, T, N}, Float64}(),
        path_stats,
        expanded_paths,
        0,
        nothing,
        Inf,
        nothing
    )

    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> dag_mcts_descent(l, p, st, ctx),
        state,
        config,
        false
    )
end

function dag_mcts_descent_init(
        param::ValuationPolydisc{S, T, N},
        loss::Loss,
        config::DAGMCTSConfig = DAGMCTSConfig()
) where {S, T, N}
    if config.variant in (NodeStatsDAGMCTS, UCT1DAGMCTS, UCT2DAGMCTS, UCTMaxDAGMCTS)
        return _dag_mcts_node_descent_init(param, loss, config)
    elseif config.variant == PathStatsDAGMCTS
        return _dag_mcts_path_descent_init(param, loss, config)
    else
        error("Unknown DAG-MCTS variant: $(config.variant)")
    end
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

    solved_count = count(n -> n.is_solved, values(state.transposition_table))
    terminal_count = count(n -> n.is_terminal, values(state.transposition_table))

    return (
        unique_nodes = unique_nodes,
        total_visits = total_visits,
        multi_parent_nodes = multi_parent,
        solved_nodes = solved_count,
        terminal_nodes = terminal_count
    )
end

function get_dag_stats(state::DAGMCTSPathState)
    unique_nodes = length(state.transposition_table)
    total_visits = sum(stats.visits for stats in values(state.path_stats))

    has_parents = any(n -> !isempty(n.parents), values(state.transposition_table))
    if has_parents
        multi_parent = count(n -> length(n.parents) > 1, values(state.transposition_table))
    else
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

    terminal_count = count(n -> n.is_terminal, values(state.transposition_table))

    return (
        unique_nodes = unique_nodes,
        total_visits = total_visits,
        multi_parent_nodes = multi_parent,
        solved_nodes = terminal_count,
        terminal_nodes = terminal_count,
        path_count = length(state.path_stats),
        expanded_path_count = length(state.expanded_paths),
        cached_evaluations = length(state.evaluation_cache)
    )
end

@doc raw"""
    print_dag_stats(state::DAGMCTSState, max_depth::Int=3)

Print statistics about the DAG-MCTS structure for debugging.
"""
function print_dag_stats(state::DAGMCTSState, max_depth::Int = 3)
    stats = get_dag_stats(state)
    println("DAG-MCTS Statistics:")
    println("  Unique nodes in table: $(stats.unique_nodes)")
    println("  Total visits: $(stats.total_visits)")
    println("  Nodes with multiple parents: $(stats.multi_parent_nodes)")
    println("  Solved nodes: $(stats.solved_nodes)")
    println("  Terminal nodes: $(stats.terminal_nodes)")
    println("  Step count: $(state.step_count)")

    if stats.multi_parent_nodes > 0
        println("  Transposition ratio: $(round(stats.multi_parent_nodes / stats.unique_nodes * 100, digits=1))%")
    end
end

function print_dag_stats(state::DAGMCTSPathState, max_depth::Int = 3)
    stats = get_dag_stats(state)
    println("DAG-MCTS Statistics:")
    println("  Variant: PathStatsDAGMCTS")
    println("  Unique nodes in table: $(stats.unique_nodes)")
    println("  Path statistics entries: $(stats.path_count)")
    println("  Expanded path entries: $(stats.expanded_path_count)")
    println("  Cached evaluations: $(stats.cached_evaluations)")
    println("  Total path visits: $(stats.total_visits)")
    println("  Nodes with multiple parents: $(stats.multi_parent_nodes)")
    println("  Terminal nodes: $(stats.terminal_nodes)")
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
            if length(node.edge_stats) != length(node.children)
                @warn "edge_stats length does not match children length"
                return false
            end
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

        # Check solved-status consistency
        if node.is_terminal && !node.is_solved
            @warn "Terminal node is not marked as solved"
            return false
        end
        if node.is_solved && isnan(node.proven_value)
            @warn "Solved node has NaN proven_value"
            return false
        end
        if node.is_expanded && !node.is_terminal
            actual_unsolved = count(c -> !c.is_solved, node.children)
            if actual_unsolved != node.unsolved_children_count
                @warn "unsolved_children_count mismatch: expected $actual_unsolved, got $(node.unsolved_children_count)"
                return false
            end
        end
        if node.is_solved && !node.is_terminal && node.is_expanded
            if any(c -> !c.is_solved, node.children)
                @warn "Solved non-terminal node has unsolved children"
                return false
            end
        end
    end

    return true
end

function verify_transposition_table(state::DAGMCTSPathState)
    table = state.transposition_table

    for (hashed_polydisc, node) in table
        if !isequal(node.polydisc, hashed_polydisc.polydisc)
            @warn "Node polydisc doesn't match table key"
            return false
        end

        if node.is_expanded
            if node.is_terminal && !isempty(node.children)
                @warn "Terminal node has children"
                return false
            end
            for child in node.children
                child_key = HashedPolydisc(child.polydisc)
                if !haskey(table, child_key)
                    @warn "Child not in transposition table"
                    return false
                end
                if table[child_key] !== child
                    @warn "Child in table is different instance"
                    return false
                end
            end
        end

        if !isempty(node.parents)
            for parent in node.parents
                if !(any(child -> child === node, parent.children))
                    @warn "Parent-child relationship inconsistent"
                    return false
                end
            end
        end
    end

    for path_key in keys(state.path_stats)
        if isempty(path_key)
            @warn "Empty path key in path statistics"
            return false
        end
        for key in path_key
            if !haskey(table, key)
                @warn "Path statistics reference a node outside the transposition table"
                return false
            end
        end
    end

    for path_key in state.expanded_paths
        if isempty(path_key)
            @warn "Empty path key in expanded path set"
            return false
        end
        for key in path_key
            if !haskey(table, key)
                @warn "Expanded path references a node outside the transposition table"
                return false
            end
        end
    end

    for key in keys(state.evaluation_cache)
        if !haskey(table, key)
            @warn "Evaluation cache contains a node outside the transposition table"
            return false
        end
    end

    for key in keys(state.loss_cache)
        if !haskey(table, key)
            @warn "Loss cache contains a node outside the transposition table"
            return false
        end
    end

    return true
end
