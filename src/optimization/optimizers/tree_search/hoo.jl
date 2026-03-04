########### Hierarchical Optimistic Optimization (HOO) ###########

# This file implements the HOO algorithm from "X-Armed Bandits" by Bubeck et al. (2011).
# HOO is a tree-based global optimization algorithm that maintains optimistic upper bounds
# to balance exploration and exploitation in a hierarchical search space.

##################################################
# HOO Node Structure
##################################################

@doc raw"""
    HOONode{S,T,N}

A node in the HOO search tree.

# Fields
- `polydisc::ValuationPolydisc{S,T,N}`: The polydisc (region) at this node
- `depth::Int`: Depth h in the tree (root has depth 0)
- `position::Int`: Position i among siblings at this depth
- `parent::Union{HOONode{S,T,N}, Nothing}`: Parent node (nothing for root)
- `children::Vector{HOONode{S,T,N}}`: Child nodes that have been expanded
- `visits::Int`: Number of times this node has been visited (N(h,i))
- `sum_values::Float64`: Sum of function values sampled at this node (for computing μ̂)
- `is_expanded::Bool`: Whether this node's children have been generated
"""
mutable struct HOONode{S,T,N}
    polydisc::ValuationPolydisc{S,T,N}
    depth::Int
    position::Int
    parent::Union{HOONode{S,T,N}, Nothing}
    children::Vector{HOONode{S,T,N}}
    visits::Int
    sum_values::Float64
    is_expanded::Bool
end

@doc raw"""
    HOONode(polydisc::ValuationPolydisc{S,T,N}, depth::Int=0, position::Int=0, parent=nothing) where {S,T,N}

Create a new HOO node with the given polydisc, depth, position, and optional parent.
"""
function HOONode(
    polydisc::ValuationPolydisc{S,T,N},
    depth::Int=0,
    position::Int=0,
    parent=nothing
) where {S,T,N}
    return HOONode{S,T,N}(
        polydisc,
        depth,
        position,
        parent,
        HOONode{S,T,N}[],
        0,
        0.0,
        false
    )
end

@doc raw"""
    empirical_mean(node::HOONode)

Compute the empirical mean μ̂(h,i) of a node (sum_values / visits).
Returns 0.0 if node has not been visited.
"""
function empirical_mean(node::HOONode)
    return node.visits > 0 ? node.sum_values / node.visits : 0.0
end

##################################################
# HOO Configuration
##################################################

@doc raw"""
    HOOConfig

Configuration parameters for the HOO optimizer.

# Fields
- `rho::Float64`: Shrinkage rate ρ ∈ (0,1) - diameter shrinks as ρ^h
- `nu1::Float64`: Smoothness parameter ν₁ - how much function can vary within region
- `max_depth::Int`: Maximum tree depth to explore
- `degree::Int`: Degree for child polydisc generation (passed to `children` function)
- `strict::Bool`: If true, use single-branch descent; if false, use full children
- `value_transform::Function`: Transform from loss to value (default: identity, since we minimize)
"""
struct HOOConfig
    rho::Float64
    nu1::Float64
    max_depth::Int
    degree::Int
    strict::Bool
    value_transform::Function
end

@doc raw"""
    HOOConfig(; kwargs...)

Create an HOO configuration with sensible defaults.

# Keyword Arguments
- `rho::Float64=0.5`: Shrinkage rate (each level halves the region size)
- `nu1::Float64=0.1`: Smoothness parameter (assume relatively smooth function)
- `max_depth::Int=10`: Maximum tree depth
- `degree::Int=1`: Child generation degree
- `strict::Bool=false`: Whether to use single-branch descent
- `value_transform::Function=loss -> 1.0 / (loss + 1e-10)`: Loss to reward transformation (default: 1/loss for minimization)
"""
function HOOConfig(;
    rho::Float64=0.5,
    nu1::Float64=0.1,
    max_depth::Int=10,
    degree::Int=1,
    strict::Bool=false,
    value_transform::Function=loss -> 1.0 / (loss + 1e-10)
)
    @assert 0 < rho < 1 "rho must be in (0,1)"
    @assert nu1 > 0 "nu1 must be positive"
    @assert max_depth > 0 "max_depth must be positive"

    return HOOConfig(
        rho,
        nu1,
        max_depth,
        degree,
        strict,
        value_transform
    )
end

##################################################
# HOO State (for tracking across optimization steps)
##################################################

@doc raw"""
    HOOState{S,T,N}

State maintained across HOO optimization steps.

# Fields
- `root::HOONode{S,T,N}`: The root node of the search tree
- `total_samples::Int`: Total number of samples taken (n in the B-value formula)
- `next_branch::Int`: Next branch index for strict mode
- `step_count::Int`: Number of optimization steps taken
"""
mutable struct HOOState{S,T,N}
    root::HOONode{S,T,N}
    total_samples::Int
    next_branch::Int
    step_count::Int
end

##################################################
# U-value and B-value Computation (Optimistic Upper Bounds)
##################################################

@doc raw"""
    compute_u_value(node::HOONode, total_samples::Int, config::HOOConfig)

Compute the U-value (leaf upper confidence bound) for a node.

The U-value formula is:
```math
U(h,i) = μ̂(h,i) + √(\frac{2 \log(n)}{N(h,i)}) + ν₁ \cdot ρ^h
```

where:
- μ̂(h,i) is the empirical mean of samples taken in this region
- n is the total number of samples taken
- N(h,i) is the number of visits to this node
- ν₁ is the smoothness parameter
- ρ^h accounts for the shrinking diameter of regions at depth h

For unvisited nodes (N(h,i) = 0), returns Inf to ensure exploration.

# Note on Minimization vs Maximization
The original HOO paper maximizes rewards. Since we're minimizing loss:
- We transform the loss using `value_transform` to get a reward
- Higher U/B-values indicate more promising regions
"""
function compute_u_value(node::HOONode, total_samples::Int, config::HOOConfig)
    if node.visits == 0
        return Inf  # Unvisited nodes have infinite priority for exploration
    end

    # Empirical mean μ̂(h,i)
    mu_hat = empirical_mean(node)

    # Transform loss to reward (default: 1/(loss + ε) to match paper's maximization setup)
    reward = config.value_transform(mu_hat)

    # Confidence interval: √(2 * log(n) / N(h,i))
    if total_samples <= 1
        confidence_term = 0.0
    else
        confidence_term = sqrt(2.0 * log(total_samples) / node.visits)
    end

    # Diameter term: ν₁ * ρ^h
    diameter_term = config.nu1 * (config.rho ^ node.depth)

    # U-value formula from the paper
    # U(h,i) = μ̂(h,i) + √(2·log(n)/N(h,i)) + ν₁·ρ^h
    u_value = reward + confidence_term + diameter_term

    return u_value
end

@doc raw"""
    compute_b_value(node::HOONode, total_samples::Int, config::HOOConfig)

Compute the B-value (hierarchical optimistic bound) for a node.

The B-value implements the min-max backup from children:
```math
B(h,i) = \min\{ U(h,i), \max\{ B(h+1, 2i-1), B(h+1, 2i) \} \}
```

This tightens the U-value bound by considering information from children.
For nodes without children (not expanded), B-value equals U-value.
For unvisited nodes, returns Inf.
"""
function compute_b_value(node::HOONode, total_samples::Int, config::HOOConfig)
    # Compute U-value for this node
    u_value = compute_u_value(node, total_samples, config)

    # If node has no children, B-value equals U-value
    if isempty(node.children)
        return u_value
    end

    # Compute max of children's B-values (recursive)
    max_child_b = -Inf
    for child in node.children
        child_b = compute_b_value(child, total_samples, config)
        max_child_b = max(max_child_b, child_b)
    end

    # B-value is min of U-value and max of children's B-values
    return min(u_value, max_child_b)
end

##################################################
# HOO Core Operations
##################################################

@doc raw"""
    expand_node!(node::HOONode{S,T,N}, config::HOOConfig) where {S,T,N}

Expand a node by generating its child polydiscs.

Creates child nodes at depth h+1, each representing a subregion of the parent.
"""
function expand_node!(node::HOONode{S,T,N}, config::HOOConfig) where {S,T,N}
    if node.is_expanded
        return
    end

    # Don't expand beyond max depth
    if node.depth >= config.max_depth
        node.is_expanded = true  # Mark as expanded but with no children
        return
    end

    # Generate children using the same function as other optimizers
    if config.strict
        # In strict mode, descend one coordinate at a time
        child_polydiscs = children_along_branch(node.polydisc, 1)  # Adjust as needed
    else
        child_polydiscs = children(node.polydisc, config.degree)
    end

    # Create child nodes
    for (i, polydisc) in enumerate(child_polydiscs)
        child_node = HOONode(polydisc, node.depth + 1, i, node)
        push!(node.children, child_node)
    end

    node.is_expanded = true
end

@doc raw"""
    select_node(root::HOONode, total_samples::Int, config::HOOConfig)

Select a node to sample by following the path of highest B-values.

According to the HOO algorithm specification:
1. Start at root and initialize path P = {root}
2. While current node is expanded (in the tree):
   - Compute B-values for all children
   - Move to child with highest B-value (random tie-breaking)
   - Add selected child to path
3. Stop when we reach a node that hasn't been expanded yet
4. The final node is where we'll sample

Returns a tuple: (selected_node, path)
where path is a vector of all nodes traversed from root to selected node.
"""
function select_node(root::HOONode, total_samples::Int, config::HOOConfig)
    node = root
    path = HOONode[root]

    # Traverse tree following highest B-values
    while node.is_expanded && !isempty(node.children)
        # Find child with highest B-value
        best_b_value = -Inf
        best_children = HOONode[]

        for child in node.children
            b_value = compute_b_value(child, total_samples, config)

            if b_value > best_b_value
                best_b_value = b_value
                best_children = [child]
            elseif b_value == best_b_value
                # Tie-breaking: collect all children with equal B-values
                push!(best_children, child)
            end
        end

        # Random tie-breaking if multiple children have the same B-value
        if length(best_children) > 1
            node = rand(best_children)
        else
            node = best_children[1]
        end

        # Add to path
        push!(path, node)
    end

    return node, path
end

@doc raw"""
    backpropagate!(path::Vector{HOONode}, function_value::Float64, config::HOOConfig)

Back-propagate the sampled value through all nodes in the path.

According to the HOO specification (Phase 3):
1. Update statistics (visits and empirical mean) for ALL nodes in the traversed path
2. The sampled value is added to all ancestor nodes

This ensures that parent nodes accumulate statistics from all samples in their subtrees.
"""
function backpropagate!(path::Vector{HOONode}, function_value::Float64, config::HOOConfig)
    # Update all nodes in the path
    for node in path
        node.visits += 1
        node.sum_values += function_value
    end
end

##################################################
# Main HOO Algorithm
##################################################

@doc raw"""
    hoo_iteration(root::HOONode{S,T,N}, loss::Loss, state::HOOState{S,T,N}, config::HOOConfig) where {S,T,N}

Perform one iteration of the HOO algorithm.

Complete HOO iteration following the specification:
1. **Phase 1 - Traversal**: Select a node by following highest B-values
2. **Phase 2 - Sampling**: Evaluate loss at selected node and expand it
3. **Phase 3 - Back-propagation**: Update statistics for all nodes in path

Returns the selected node and its loss value.
"""
function hoo_iteration(
    root::HOONode{S,T,N},
    loss::Loss,
    state::HOOState{S,T,N},
    config::HOOConfig
) where {S,T,N}
    # Phase 1: Select node to sample (returns node and path from root)
    selected_node, path = select_node(root, state.total_samples + 1, config)

    # Phase 2: Sampling
    # Evaluate loss at this node
    loss_value = loss.eval([selected_node.polydisc])[1]

    # Expand the selected node (add it to the tree)
    # This creates its children with infinite B-values (unvisited)
    expand_node!(selected_node, config)

    # Phase 3: Back-propagation
    # Update statistics for ALL nodes in the path
    backpropagate!(path, loss_value, config)

    # Increment total samples
    state.total_samples += 1

    # Note: B-values are computed on-demand during next select_node call
    # The recursive compute_b_value function automatically propagates updates

    return selected_node, loss_value
end

@doc raw"""
    get_best_node(root::HOONode)

Find the node with the lowest empirical mean (best observed loss).

Recursively searches the tree to find the node with the minimum μ̂(h,i).
Only considers nodes that have been visited at least once.
"""
function get_best_node(root::HOONode)
    best_node = root
    best_mean = empirical_mean(root)

    function traverse(node)
        if node.visits > 0
            mean = empirical_mean(node)
            if mean < best_mean || (mean == best_mean && node.depth > best_node.depth)
                # Prefer deeper nodes if means are equal (more refined region)
                best_mean = mean
                best_node = node
            end
        end

        for child in node.children
            traverse(child)
        end
    end

    traverse(root)
    return best_node
end

##################################################
# HOO Optimizer Interface (compatible with OptimSetup)
##################################################

@doc raw"""
    hoo_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, state::HOOState{S,T,N}, config::HOOConfig) where {S,T,N}

Perform one step of HOO optimization.

This function follows the same interface as `greedy_descent`, `gradient_descent`, and `mcts_descent`,
making it compatible with `OptimSetup`.

Each "step" performs multiple HOO iterations (samples) and returns the best polydisc found so far.

# Arguments
- `loss::Loss`: The loss function structure
- `param::ValuationPolydisc{S,T,N}`: Current parameter values (used to verify state)
- `state::HOOState{S,T,N}`: HOO state (includes the search tree and statistics)
- `config::HOOConfig`: Configuration parameters

# Returns
`Tuple{ValuationPolydisc{S,T,N}, HOOState{S,T,N}}`: Best parameters found and updated state
"""
function hoo_descent(
    loss::Loss,
    param::ValuationPolydisc{S,T,N},
    state::HOOState{S,T,N},
    config::HOOConfig
) where {S,T,N}
    # Perform multiple HOO iterations per optimization step
    # (You can make this configurable if needed)
    num_iterations = 10

    for _ in 1:num_iterations
        hoo_iteration(state.root, loss, state, config)
    end

    # Find the best node (lowest empirical mean)
    best_node = get_best_node(state.root)

    # Update step count
    state.step_count += 1

    # Return the best polydisc found so far
    return best_node.polydisc, state, false
end

@doc raw"""
    hoo_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, config::HOOConfig=HOOConfig()) where {S,T,N}

Initialize an optimization setup for HOO.

Initialization according to HOO specification:
1. Create root node representing entire domain
2. Sample at root to initialize statistics
3. Expand root to create children with infinite B-values

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values (root region)
- `loss::Loss`: The loss function structure
- `config::HOOConfig`: HOO configuration (uses defaults if not provided)

# Returns
`OptimSetup`: Configured optimization setup for HOO

# Example
```julia
# Set up HOO optimizer
config = HOOConfig(rho=0.5, nu1=0.1, max_depth=10)
optim = hoo_descent_init(param, loss, config)

# Run optimization
for i in 1:100
    step!(optim)
    if i % 10 == 0
        println("Step ", i, ": Loss = ", eval_loss(optim))
    end
end
```
"""
function hoo_descent_init(
    param::ValuationPolydisc{S,T,N},
    loss::Loss,
    config::HOOConfig=HOOConfig()
) where {S,T,N}
    # Initialize root node at depth 0, position 1 (as per spec: (h,i) = (0,1))
    root = HOONode(param, 0, 1, nothing)

    # Initialize state
    state = HOOState{S,T,N}(root, 0, 1, 0)

    # Sample at the root to initialize
    loss_value = loss.eval([root.polydisc])[1]
    root.visits = 1
    root.sum_values = loss_value
    state.total_samples = 1

    # Expand root to create initial children
    # According to spec: "Initialize root B-values: B_{1,2} = B_{2,2} = +∞"
    # This happens automatically when children are created (unvisited nodes have B = +∞)
    expand_node!(root, config)

    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> hoo_descent(l, p, st, ctx),
        state,
        config,
        false
    )
end

##################################################
# Utility Functions
##################################################

@doc raw"""
    print_tree_stats(node::HOONode, max_depth::Int=3, depth::Int=0)

Print statistics about the HOO tree for debugging.

Displays depth, position, visits, empirical mean, and number of children for each node.
"""
function print_tree_stats(node::HOONode, max_depth::Int=3, depth::Int=0)
    if depth > max_depth
        return
    end

    indent = "  " ^ depth
    mean_str = node.visits > 0 ? @sprintf("%.6f", empirical_mean(node)) : "N/A"
    println("$(indent)Node (h=$(node.depth), i=$(node.position)): " *
            "visits=$(node.visits), μ̂=$(mean_str), children=$(length(node.children))")

    # Sort children by visits for display
    sorted_children = sort(node.children, by=c -> c.visits, rev=true)
    for child in sorted_children[1:min(5, length(sorted_children))]
        print_tree_stats(child, max_depth, depth + 1)
    end
end

@doc raw"""
    get_tree_size(node::HOONode)

Count the total number of nodes in the HOO tree.
"""
function get_tree_size(node::HOONode)
    count = 1
    for child in node.children
        count += get_tree_size(child)
    end
    return count
end

@doc raw"""
    get_leaf_nodes(node::HOONode)

Collect all leaf nodes (nodes with no children) in the tree.
"""
function get_leaf_nodes(node::HOONode)
    leaves = HOONode[]

    function collect_leaves(n)
        if isempty(n.children)
            push!(leaves, n)
        else
            for child in n.children
                collect_leaves(child)
            end
        end
    end

    collect_leaves(node)
    return leaves
end

@doc raw"""
    get_visited_nodes(node::HOONode)

Collect all nodes that have been visited (visits > 0).
"""
function get_visited_nodes(node::HOONode)
    visited = HOONode[]

    function collect_visited(n)
        if n.visits > 0
            push!(visited, n)
        end
        for child in n.children
            collect_visited(child)
        end
    end

    collect_visited(node)
    return visited
end
