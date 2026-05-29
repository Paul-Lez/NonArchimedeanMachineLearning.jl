"""
Deterministic Optimistic Optimization (DOO) algorithm.

Based on Rémi Munos (2011): "Optimistic Optimization of a Deterministic Function
without the Knowledge of its Smoothness", Section 3.

DOO is a hierarchical tree search algorithm for global optimization that:
- Uses a deterministic partition of the search space
- Maintains optimistic upper bounds (b-values) for each node
- Selects and expands the leaf with maximum b-value
- Does not require smoothness parameters (unlike HOO)

Key Algorithm:
1. Compute b-value: b(h,j) = f(x_{h,j}) + δ(h)
   where δ(h) is a known decreasing sequence of diameter bounds
2. Select leaf with maximum b-value
3. Expand selected leaf by generating children
4. Repeat until budget exhausted

Differences from HOO:
- DOO: b = f(x) + δ(h) (deterministic)
- HOO: b = μ̂ + √(2ln(n)/N) + ν₁ρʰ (stochastic + smoothness)

In strict mode, this follows a degree-d coordinate-wise cell decomposition: at
depth h, the node is expanded along one fixed d-subset of coordinates from a
cyclic enumeration of all d-subsets of {1, ..., n}. The optimistic radius uses
the number of complete per-coordinate refinements guaranteed at that depth.
"""

# TODO: Add PriorityQueue from DataStructures.jl for O(log N) leaf selection
# Currently using O(N) scan through leaves vector

"""
    DOONode{S,T,N}

Node in the DOO (Deterministic Optimistic Optimization) tree.

Fields:
- `polydisc::ValuationPolydisc{S,T,N}`: Region represented by this node
- `depth::Int`: Depth in tree (root has depth 0)
- `position::Int`: Position index among siblings
- `parent::Union{DOONode{S,T,N}, Nothing}`: Parent node reference
- `children::Vector{DOONode{S,T,N}}`: Expanded children
- `value::Union{Float64, Nothing}`: Evaluated function value (after value_transform)
- `is_expanded::Bool`: Whether node has been expanded
"""
mutable struct DOONode{S, T, N}
    polydisc::ValuationPolydisc{S, T, N}
    depth::Int
    position::Int
    parent::Union{DOONode{S, T, N}, Nothing}
    children::Vector{DOONode{S, T, N}}
    value::Union{Float64, Nothing}
    is_expanded::Bool
end

function DOONode(polydisc::ValuationPolydisc{S, T, N}, depth::Int, position::Int,
        parent::Union{DOONode{S, T, N}, Nothing}) where {S, T, N}
    DOONode{S, T, N}(polydisc, depth, position, parent, DOONode{S, T, N}[], nothing, false)
end

"""
    DOOConfig

Configuration for DOO algorithm.

Fields:
- `delta::Function`: Diameter function δ(h) providing upper bound on cell diameter at depth h.
                     Should be a decreasing function of depth.
                     NOTE: User will define this based on specific problem structure.
- `degree::Int`: Number of coordinates refined by each expansion (default: 1)
- `strict::Bool`: If true, use the coordinate-wise cell decomposition (default: false)
- `value_transform::Function`: Transform loss to value for maximization (default: loss -> -loss)

Theoretical Notes:
- DOO convergence depends on how well δ(h) bounds the actual cell diameters
- δ(h) should decrease at rate matching the partition refinement
- For binary splits: δ(h) = 0.5^h is typical
- For p-adic polydiscs: δ(h) depends on prime and radius shrinkage.
  In strict degree-d mode, use
  δ_d(h) = δ(binomial(n - 1, d - 1) * floor(h / binomial(n, d))).
  The factor `binomial(n - 1, d - 1)` is the number of d-subsets containing
  any fixed coordinate in one full cycle through all d-subsets.

Note: DOO does not need an explicit max_depth parameter. The tree search naturally
terminates when the polydisc `children()` function returns empty at the precision
boundary of the p-adic field.
"""
struct DOOConfig
    delta::Function
    degree::Int
    strict::Bool
    value_transform::Function

    function DOOConfig(;
            delta::Function,
            degree::Int = 1,
            strict::Bool = false,
            value_transform::Function = loss -> -loss
    )
        new(delta, degree, strict, value_transform)
    end
end

"""
    DOOState{S,T,N}

State for DOO optimization.

Fields:
- `root::DOONode{S,T,N}`: Root of search tree
- `total_samples::Int`: Total function evaluations performed
- `next_branch::Int`: Starting subset index for strict mode
- `step_count::Int`: Number of optimization steps taken
- `leaves::Vector{DOONode{S,T,N}}`: Vector of unexpanded leaf nodes
- `branch_sets::Vector{Vector{Int}}`: Precomputed strict-mode degree-d coordinate subsets

Note: leaves vector currently requires O(N) scan to find maximum b-value.
TODO: Replace with PriorityQueue from DataStructures.jl for O(log N) performance.
"""
mutable struct DOOState{S, T, N}
    root::DOONode{S, T, N}
    total_samples::Int
    next_branch::Int
    step_count::Int
    leaves::Vector{DOONode{S, T, N}}
    branch_sets::Vector{Vector{Int}}

    function DOOState{S, T, N}(root::DOONode{S, T, N},
            branch_sets::Vector{Vector{Int}} = Vector{Vector{Int}}()) where {S, T, N}
        new{S, T, N}(root, 0, 1, 0, [root], branch_sets)
    end
end

"""
    doo_bound_depth(node::DOONode, config::DOOConfig)

Depth argument used in the DOO optimism term.

For the usual simultaneous cell decomposition this is the node depth h. In
strict degree-d mode, one full cycle through all `binomial(n, d)` d-subsets
refines every coordinate `binomial(n - 1, d - 1)` times.
"""
function doo_bound_depth(node::DOONode{S, T, N}, config::DOOConfig) where {S, T, N}
    if !config.strict
        return node.depth
    end

    degree = config.degree
    cycle_length = binomial(N, degree)
    per_coordinate_refinements = binomial(N - 1, degree - 1)
    return per_coordinate_refinements * (node.depth ÷ cycle_length)
end

"""
    strict_branch_sets(::Val{N}, degree::Int) where N

Deterministic enumeration of coordinate subsets for strict degree-d DOO.
"""
function strict_branch_sets(::Val{N}, degree::Int) where {N}
    return [collect(Int, subset) for subset in AbstractAlgebra.combinations(N, degree)]
end

"""
    strict_branch_indices(node::DOONode, state::DOOState)

Coordinate to refine for the coordinate-wise strict DOO tree.

The coordinate subset is a function of node depth rather than global step
count: root children use `state.next_branch` as the starting subset index, and
deeper nodes cycle through the deterministic d-subset enumeration.
"""
function strict_branch_indices(node::DOONode{S, T, N},
        state::DOOState{S, T, N}) where {S, T, N}
    return state.branch_sets[mod1(state.next_branch + node.depth, length(state.branch_sets))]
end

"""
    b_value(node::DOONode, config::DOOConfig)

Compute the b-value (optimistic upper bound) for a node.

Formula: b(h,j) = value + δ(h)

where:
- value = value_transform(loss) is the transformed function value at the node
- δ(h) is the diameter bound at depth h. In strict degree-d mode this uses
  δ(binomial(n - 1, d - 1) * floor(h / binomial(n, d))).

The b-value represents the best possible value that could be achieved
within the node's region, assuming the function could vary by at most δ(h).

Returns: Float64 b-value (Inf for unexplored nodes)
"""
function b_value(node::DOONode, config::DOOConfig)
    if node.value === nothing
        return Inf  # Unexplored nodes have infinite optimistic potential
    end
    return node.value + config.delta(doo_bound_depth(node, config))
end

"""
    select_best_leaf(state::DOOState, config::DOOConfig)

Select the unexpanded leaf with maximum b-value.

This implements the selection step of DOO: choose the leaf node that
has the highest optimistic upper bound on its potential value.

Returns: DOONode with maximum b-value, or nothing if no leaves available

Performance: O(N) where N is number of leaves.
TODO: Use PriorityQueue for O(log N) selection.
"""
function select_best_leaf(state::DOOState{S, T, N}, config::DOOConfig) where {S, T, N}
    if isempty(state.leaves)
        return nothing
    end

    best_node = nothing
    best_b = -Inf

    for node in state.leaves
        b = b_value(node, config)
        if b > best_b
            best_b = b
            best_node = node
        end
    end

    return best_node
end

"""
    expand_node!(node::DOONode, loss::Loss, config::DOOConfig, state::DOOState)

Expand a node by generating and evaluating its children.

Algorithm:
1. Generate child polydiscs (using children() or children_along_branch())
2. Create child nodes
3. Evaluate loss on the children
4. Transform loss to value
5. Update parent's children list
6. Increment sample counter

Returns: Vector of newly created child nodes (empty if node cannot be expanded)
"""
function expand_node!(node::DOONode{S, T, N}, loss::Loss, config::DOOConfig,
        state::DOOState{S, T, N}) where {S, T, N}
    if node.is_expanded
        return DOONode{S, T, N}[]
    end

    # Generate children polydiscs
    if config.strict
        # Expand along the coordinate subset prescribed by this node's depth.
        branch_indices = strict_branch_indices(node, state)
        child_polydiscs = children_along_branches(node.polydisc, branch_indices)
    else
        # Full expansion along all branches
        child_polydiscs = children(node.polydisc, config.degree)
    end

    if isempty(child_polydiscs)
        node.is_expanded = true
        return DOONode{S, T, N}[]
    end

    # Create and evaluate child nodes
    child_losses = loss.eval(child_polydiscs)
    children_nodes = DOONode{S, T, N}[]
    for (i, child_disc) in enumerate(child_polydiscs)
        # Create child node
        child = DOONode(child_disc, node.depth + 1, i, node)

        # Transform loss to value (for maximization framework)
        child.value = config.value_transform(child_losses[i])

        # Add to parent's children
        push!(node.children, child)
        push!(children_nodes, child)
    end

    state.total_samples += length(children_nodes)
    node.is_expanded = true
    return children_nodes
end

"""
    doo_descent(loss::Loss, param::ValuationPolydisc{S,T,N},
                state::DOOState{S,T,N}, config::DOOConfig) where {S,T,N}

Perform one step of DOO optimization.

Algorithm:
1. Select leaf with maximum b-value (optimistic upper bound)
2. Remove selected leaf from leaves list
3. Expand selected leaf (generate and evaluate children)
4. Update state (step count)
5. If it has no children, report convergence
6. Add new children to leaves list
7. Return best-valued node's polydisc as new parameter

Returns: `(new_param::ValuationPolydisc, updated_state::DOOState, converged::Bool)`.
"""
function doo_descent(loss::Loss, param::ValuationPolydisc{S, T, N},
        state::DOOState{S, T, N}, config::DOOConfig) where {S, T, N}
    # Select leaf with maximum b-value
    best_leaf = select_best_leaf(state, config)

    if best_leaf === nothing
        # No unexpanded leaves remain — fully converged
        return (param, state, true)
    end

    # Remove selected leaf from leaves list
    filter!(n -> n !== best_leaf, state.leaves)

    # Expand the selected leaf
    new_children = expand_node!(best_leaf, loss, config, state)

    # Update optimization state
    state.step_count += 1

    if isempty(new_children)
        best_node = get_best_node(state)
        return (best_node.polydisc, state, true)
    end

    # Add new children to leaves list
    append!(state.leaves, new_children)

    # Return the best-valued node found so far as the new parameter
    best_node = get_best_node(state)

    return (best_node.polydisc, state, false)
end

"""
    doo_descent_init(param::ValuationPolydisc{S,T}, loss::Loss,
                     next_branch::Int, config::DOOConfig) where {S,T}

Initialize DOO optimizer.

Creates the initial search tree with root node, evaluates root,
and returns an OptimSetup configured for DOO optimization.

Arguments:
- `param`: Initial parameter polydisc (becomes root of search tree)
- `loss`: Loss function with eval and grad methods
- `next_branch`: Starting subset index for strict mode
- `config`: DOO configuration

Returns: OptimSetup instance ready for optimization via step!()
"""
function doo_descent_init(param::ValuationPolydisc{S, T, N}, loss::Loss,
        next_branch::Int, config::DOOConfig) where {S, T, N}
    @req 1 <= config.degree <= N "degree must be between 1 and the dimension of the polydisc"

    # Create root node
    root = DOONode(param, 0, 0, nothing)

    # Evaluate loss at root
    # Note: loss.eval expects an array of polydiscs
    root_loss = loss.eval([param])[1]
    root.value = config.value_transform(root_loss)

    # Create initial state with root as only leaf
    branch_sets = config.strict ? strict_branch_sets(Val(N), config.degree) : Vector{Vector{Int}}()
    state = DOOState{S, T, N}(root, branch_sets)
    state.next_branch = next_branch
    state.total_samples = 1

    # Create descent function closure
    descent = (l, p, s, c) -> doo_descent(l, p, s, c)

    # Return OptimSetup
    return OptimSetup(loss, param, descent, state, config, false)
end

# Utility functions

"""
    get_tree_size(state::DOOState)

Get total number of nodes in the DOO tree (including root and all descendants).
"""
function get_tree_size(state::DOOState{S, T, N}) where {S, T, N}
    function count_nodes(node::DOONode{S, T, N})
        count = 1
        for child in node.children
            count += count_nodes(child)
        end
        return count
    end
    return count_nodes(state.root)
end

"""
    get_leaf_count(state::DOOState)

Get number of unexpanded leaf nodes currently in the leaves list.
"""
function get_leaf_count(state::DOOState)
    return length(state.leaves)
end

"""
    get_all_leaves(state::DOOState)

Get all leaf nodes (expanded or not) in the tree by traversing from root.

This differs from state.leaves which only tracks unexpanded leaves.
"""
function get_all_leaves(state::DOOState{S, T, N}) where {S, T, N}
    leaves = DOONode{S, T, N}[]

    function collect_leaves(node::DOONode{S, T, N})
        if isempty(node.children)
            push!(leaves, node)
        else
            for child in node.children
                collect_leaves(child)
            end
        end
    end

    collect_leaves(state.root)
    return leaves
end

"""
    get_best_node(state::DOOState)

Get the node with the best (highest) value evaluated so far.

Since value = value_transform(loss), higher value means lower loss.
This returns the node corresponding to the best solution found.
"""
function get_best_node(state::DOOState{S, T, N}) where {S, T, N}
    best_node = nothing
    best_value = -Inf

    function search_best(node::DOONode{S, T, N})
        if node.value !== nothing && node.value > best_value
            best_node = node
            best_value = node.value
        end
        for child in node.children
            search_best(child)
        end
    end

    search_best(state.root)
    return best_node
end

"""
    get_best_value(state::DOOState)

Get the best value found so far (after value_transform).
"""
function get_best_value(state::DOOState)
    node = get_best_node(state)
    return node === nothing ? nothing : node.value
end
