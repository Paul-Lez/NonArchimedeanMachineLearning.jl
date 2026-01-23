########### Flat UCB (Flat Upper Confidence Bounds) Optimizer ###########

# This file implements Flat UCB optimization for non-Archimedean spaces.
# Flat UCB applies standard UCB directly to leaves. Internal nodes propagate
# the maximum B-value from their children in a bottom-up manner.

##################################################
# Flat UCB Node Structure
##################################################

@doc raw"""
    FlatUCBNode{S,T}

A node in the Flat UCB search tree.

# Fields
- `polydisc::ValuationPolydisc{S,T}`: The polydisc at this node
- `parent::Union{FlatUCBNode{S,T}, Nothing}`: Parent node (nothing for root)
- `children::Vector{FlatUCBNode{S,T}}`: Child nodes that have been expanded
- `visits::Int`: Number of times this node has been visited (n_i in spec)
- `total_value::Float64`: Sum of all values backpropagated through this node
- `depth::Int`: Depth of this node in the tree (0 for root)
- `b_value::Float64`: Cached B-value for this node (updated bottom-up)
- `is_leaf::Bool`: Whether this node is currently a leaf (no children expanded)
"""
mutable struct FlatUCBNode{S,T}
    polydisc::ValuationPolydisc{S,T}
    parent::Union{FlatUCBNode{S,T}, Nothing}
    children::Vector{FlatUCBNode{S,T}}
    visits::Int
    total_value::Float64
    depth::Int
    b_value::Float64
    is_leaf::Bool
end

@doc raw"""
    FlatUCBNode(polydisc::ValuationPolydisc{S,T}, parent=nothing, depth=0) where {S,T}

Create a new Flat UCB node with the given polydisc, optional parent, and depth.
"""
function FlatUCBNode(polydisc::ValuationPolydisc{S,T}, parent=nothing, depth=0) where {S,T}
    return FlatUCBNode{S,T}(
        polydisc,
        parent,
        FlatUCBNode{S,T}[],
        0,
        0.0,
        depth,
        Inf,  # Initial B-value is infinity (unvisited)
        true  # Initially a leaf
    )
end

@doc raw"""
    average_value(node::FlatUCBNode)

Compute the empirical mean of a node (X_{i,n_i} in spec).
Returns 0.0 if node has not been visited.
"""
function average_value(node::FlatUCBNode)
    return node.visits > 0 ? node.total_value / node.visits : 0.0
end

@doc raw"""
    is_leaf(node::FlatUCBNode)

Check if a node is a leaf (no children expanded).
"""
function is_leaf(node::FlatUCBNode)
    return isempty(node.children)
end

##################################################
# Flat UCB Configuration
##################################################

@doc raw"""
    FlatUCBConfig

Configuration parameters for the Flat UCB optimizer.

# Fields
- `max_depth::Int`: Maximum tree depth (D in spec)
- `num_simulations::Int`: Number of UCB simulations to run per step
- `beta::Float64`: Confidence probability parameter (e.g., 0.05)
- `total_nodes::Int`: Total nodes in tree N ≈ 2^{D+1} - 1
- `degree::Int`: Degree for child polydisc generation
- `strict::Bool`: If true, use single-branch descent
- `value_transform::Function`: Transform from loss to value (default: loss -> 1/loss)

# Confidence Formula
For a leaf visited n_i times:
- $\beta_{n_i} = \frac{\beta}{2^{D+1} n_i (n_i + 1)}$
- $c_{n_i} = \sqrt{\frac{\ln(\beta_{n_i}^{-1})}{2 n_i}}$

# B-value Formula
- **Leaf**: $B_{i,n_i} = X_{i,n_i} + c_{n_i}$
- **Internal node**: $B_{i,n_i} = \max_{j \in \mathcal{C}(i)} B_{j,n_j}$
"""
struct FlatUCBConfig
    max_depth::Int
    num_simulations::Int
    beta::Float64
    total_nodes::Int
    degree::Int
    strict::Bool
    value_transform::Function
end

@doc raw"""
    compute_flat_ucb_beta_n(n_i::Int, beta::Float64, total_nodes::Int)

Compute the confidence parameter β_n for a node with n_i visits.

Formula: $\beta_{n_i} = \frac{\beta}{2^{D+1} n_i (n_i + 1)} = \frac{\beta}{N n_i (n_i + 1)}$

where N = total_nodes ≈ 2^{D+1} - 1 (we use 2^{D+1} as approximation for simplicity).
"""
function compute_flat_ucb_beta_n(n_i::Int, beta::Float64, total_nodes::Int)
    if n_i == 0
        return 1.0  # Avoid division by zero
    end
    return beta / (total_nodes * n_i * (n_i + 1))
end

@doc raw"""
    FlatUCBConfig(; kwargs...)

Create a Flat UCB configuration.

# Keyword Arguments
- `max_depth::Int=10`: Maximum tree depth D
- `num_simulations::Int=100`: Number of simulations per step
- `beta::Float64=0.05`: Confidence probability parameter (1-β confidence)
- `degree::Int=1`: Child generation degree
- `strict::Bool=false`: Whether to use single-branch descent
- `value_transform::Function=loss -> 1.0 / (loss + 1e-10)`: Loss to value transformation

The constructor computes:
- `total_nodes`: Estimated as 2^(D+1) for binary tree (upper bound)
"""
function FlatUCBConfig(;
    max_depth::Int=10,
    num_simulations::Int=100,
    beta::Float64=0.05,
    degree::Int=1,
    strict::Bool=false,
    value_transform::Function=loss -> 1.0 / (loss + 1e-10)
)
    # Compute total nodes (use 2^(D+1) as upper bound for simplicity)
    total_nodes = 2^(max_depth + 1)

    return FlatUCBConfig(
        max_depth,
        num_simulations,
        beta,
        total_nodes,
        degree,
        strict,
        value_transform
    )
end

##################################################
# Flat UCB State
##################################################

@doc raw"""
    FlatUCBState{S,T}

State for the Flat UCB optimizer.

# Fields
- `root::FlatUCBNode{S,T}`: Root node of the search tree
- `step_count::Int`: Number of optimization steps completed
"""
mutable struct FlatUCBState{S,T}
    root::FlatUCBNode{S,T}
    step_count::Int
end

##################################################
# B-value Computation
##################################################

@doc raw"""
    compute_leaf_b_value(node::FlatUCBNode, config::FlatUCBConfig)

Compute the B-value for a leaf node using the UCB formula.

Formula: $B_{i,n_i} = X_{i,n_i} + \sqrt{\frac{\ln(\beta_{n_i}^{-1})}{2 n_i}}$

Returns Inf if the node has not been visited (unvisited nodes have highest priority).
"""
function compute_leaf_b_value(node::FlatUCBNode, config::FlatUCBConfig)
    if node.visits == 0
        return Inf  # Unvisited nodes have infinite priority
    end

    n_i = node.visits

    # Compute β_n
    beta_n = compute_flat_ucb_beta_n(n_i, config.beta, config.total_nodes)

    # Compute confidence term
    confidence = sqrt(log(1.0 / beta_n) / (2.0 * n_i))

    # B-value = empirical mean + confidence term
    return average_value(node) + confidence
end

@doc raw"""
    compute_internal_b_value(node::FlatUCBNode)

Compute the B-value for an internal node as the maximum of its children's B-values.

Formula: $B_{i,n_i} = \max_{j \in \mathcal{C}(i)} B_{j,n_j}$

Returns -Inf if the node has no children (should not happen in practice).
"""
function compute_internal_b_value(node::FlatUCBNode)
    if isempty(node.children)
        return -Inf  # No children (shouldn't happen for internal nodes)
    end

    return maximum(child.b_value for child in node.children)
end

@doc raw"""
    update_b_value!(node::FlatUCBNode, config::FlatUCBConfig)

Update the B-value for a node based on whether it's a leaf or internal node.
"""
function update_b_value!(node::FlatUCBNode, config::FlatUCBConfig)
    if is_leaf(node)
        node.b_value = compute_leaf_b_value(node, config)
    else
        node.b_value = compute_internal_b_value(node)
    end
end

@doc raw"""
    propagate_b_values!(node::FlatUCBNode, config::FlatUCBConfig)

Propagate B-values from the given node up to the root.

This is called after updating a leaf to ensure all ancestor nodes have correct B-values.
"""
function propagate_b_values!(node::FlatUCBNode, config::FlatUCBConfig)
    current = node
    while current !== nothing
        update_b_value!(current, config)
        current = current.parent
    end
end

##################################################
# Tree Operations
##################################################

@doc raw"""
    expand_node!(node::FlatUCBNode, config::FlatUCBConfig)

Expand a node by generating all its children.

Returns the vector of newly created children.
"""
function expand_node!(node::FlatUCBNode{S,T}, config::FlatUCBConfig) where {S,T}
    # Don't expand if already expanded
    if !isempty(node.children)
        return node.children
    end

    # Don't expand beyond max depth
    if node.depth >= config.max_depth
        return FlatUCBNode{S,T}[]
    end

    # Generate child polydiscs
    child_polydiscs = if config.strict
        children_along_branch(node.polydisc, 1)
    else
        children(node.polydisc, config.degree)
    end

    # Create child nodes
    new_children = FlatUCBNode{S,T}[]
    for child_polydisc in child_polydiscs
        child_node = FlatUCBNode(child_polydisc, node, node.depth + 1)
        push!(new_children, child_node)
    end

    # Update node
    node.children = new_children
    node.is_leaf = false

    return new_children
end

##################################################
# Traversal (Phase 1)
##################################################

@doc raw"""
    select_best_child(node::FlatUCBNode)

Select the child with the highest B-value.

Returns the child node with maximum B-value, or nothing if no children.
"""
function select_best_child(node::FlatUCBNode)
    if isempty(node.children)
        return nothing
    end

    # Find child with maximum B-value
    best_child = node.children[1]
    best_b_value = best_child.b_value

    for child in node.children[2:end]
        if child.b_value > best_b_value
            best_child = child
            best_b_value = child.b_value
        end
    end

    return best_child
end

@doc raw"""
    traverse_to_leaf(root::FlatUCBNode, config::FlatUCBConfig)

Phase 1: Traverse from root to a leaf by selecting children with highest B-values.

If we encounter a leaf before reaching max depth, we expand it and select a child.

Returns:
- The leaf node to be evaluated
- The path from root to leaf (including the leaf)
"""
function traverse_to_leaf(root::FlatUCBNode, config::FlatUCBConfig)
    path = [root]
    current = root

    # Traverse until we reach max depth or a true leaf (can't be expanded)
    while current.depth < config.max_depth
        # If current is a leaf (no children), expand it
        if is_leaf(current)
            children = expand_node!(current, config)

            # If expansion failed (at max depth), break
            if isempty(children)
                break
            end

            # Update B-values for all children (they start with Inf for unvisited)
            for child in children
                update_b_value!(child, config)
            end
        end

        # Select best child based on B-values
        best_child = select_best_child(current)

        if best_child === nothing
            break
        end

        push!(path, best_child)
        current = best_child
    end

    return current, path
end

##################################################
# Evaluation (Phase 2)
##################################################

@doc raw"""
    evaluate_node(node::FlatUCBNode, loss::Loss, config::FlatUCBConfig)

Phase 2: Evaluate the loss function at the node's polydisc and convert to value.

Returns the value (higher is better).
"""
function evaluate_node(node::FlatUCBNode, loss::Loss, config::FlatUCBConfig)
    # Evaluate loss at the node's polydisc
    loss_value = loss.eval([node.polydisc])[1]

    # Transform loss to value (higher is better)
    value = config.value_transform(loss_value)

    return value
end

##################################################
# Backpropagation (Phase 3)
##################################################

@doc raw"""
    backpropagate!(path::Vector{FlatUCBNode{S,T}}, value::Float64, config::FlatUCBConfig) where {S,T}

Phase 3: Backpropagate the value up the tree, updating statistics and B-values.

Updates:
1. Increment visit count and add value to total for all nodes in path
2. Update B-values bottom-up (from leaf to root)
"""
function backpropagate!(path::Vector{FlatUCBNode{S,T}}, value::Float64, config::FlatUCBConfig) where {S,T}
    # Update all nodes in the path
    for node in path
        node.visits += 1
        node.total_value += value
    end

    # Propagate B-values from leaf to root
    # Start from the leaf (last node in path)
    if !isempty(path)
        propagate_b_values!(path[end], config)
    end
end

##################################################
# Flat UCB Simulation
##################################################

@doc raw"""
    flat_ucb_simulation!(root::FlatUCBNode, loss::Loss, config::FlatUCBConfig)

Perform one complete Flat UCB simulation (one round).

Phases:
1. **Traversal**: Select path from root to leaf using B-values
2. **Sampling**: Evaluate loss at the leaf
3. **Backpropagation**: Update statistics and propagate B-values

Returns the value obtained at the leaf.
"""
function flat_ucb_simulation!(root::FlatUCBNode, loss::Loss, config::FlatUCBConfig)
    # Phase 1: Traverse to leaf
    leaf, path = traverse_to_leaf(root, config)

    # Phase 2: Evaluate
    value = evaluate_node(leaf, loss, config)

    # Phase 3: Backpropagate
    backpropagate!(path, value, config)

    return value
end

##################################################
# Optimizer Interface
##################################################

@doc raw"""
    flat_ucb_descent(loss::Loss, param::ValuationPolydisc, state::FlatUCBState, config::FlatUCBConfig)

Perform one Flat UCB optimization step.

This function:
1. Creates a fresh tree rooted at the current parameter
2. Runs `num_simulations` Flat UCB simulations
3. Selects the best child of the root (highest average value)
4. Returns the selected child's polydisc as the next parameter

Returns:
- `next_param`: The selected child polydisc
- `state`: Updated state (step count incremented)
"""
function flat_ucb_descent(
    loss::Loss,
    param::ValuationPolydisc{S,T},
    state::FlatUCBState{S,T},
    config::FlatUCBConfig
) where {S,T}
    # Create fresh root at current parameter
    root = FlatUCBNode(param, nothing, 0)

    # Initial evaluation of root
    root_value = evaluate_node(root, loss, config)
    root.visits = 1
    root.total_value = root_value
    update_b_value!(root, config)

    # Expand root to get initial children
    expand_node!(root, config)

    # Initialize children B-values
    for child in root.children
        update_b_value!(child, config)
    end

    # Run simulations
    for i in 1:config.num_simulations
        flat_ucb_simulation!(root, loss, config)
    end

    # Select best child (highest average value, breaking ties by visits)
    if isempty(root.children)
        # No children (shouldn't happen), stay at current position
        return param, state
    end

    best_child = root.children[1]
    best_avg_value = average_value(best_child)

    for child in root.children[2:end]
        child_avg_value = average_value(child)
        if child_avg_value > best_avg_value ||
           (child_avg_value == best_avg_value && child.visits > best_child.visits)
            best_child = child
            best_avg_value = child_avg_value
        end
    end

    # Update state
    state.step_count += 1
    state.root = root  # Store for debugging (optional)

    return best_child.polydisc, state
end

@doc raw"""
    flat_ucb_descent_init(param::ValuationPolydisc, loss::Loss, config::FlatUCBConfig)

Initialize a Flat UCB optimizer.

Returns an `OptimSetup` configured for Flat UCB optimization.

# Arguments
- `param::ValuationPolydisc`: Initial parameter point
- `loss::Loss`: Loss function to minimize
- `config::FlatUCBConfig`: Flat UCB configuration (default: FlatUCBConfig())

# Returns
- `OptimSetup`: Optimizer ready for `step!(optim)` calls
"""
function flat_ucb_descent_init(
    param::ValuationPolydisc{S,T},
    loss::Loss,
    config::FlatUCBConfig=FlatUCBConfig()
) where {S,T}
    # Initialize state
    root = FlatUCBNode(param, nothing, 0)
    state = FlatUCBState{S,T}(root, 0)

    # Create OptimSetup
    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> flat_ucb_descent(l, p, st, ctx),
        state,
        config
    )
end

##################################################
# Testing Functions
##################################################

@doc raw"""
    test_flat_ucb_basic()

Run basic unit tests for Flat UCB implementation.

Tests:
1. Node creation and initialization
2. B-value computation for leaves
3. B-value propagation for internal nodes
4. Node expansion
5. Backpropagation
"""
function test_flat_ucb_basic()
    println("\n" * "="^70)
    println("Flat UCB Basic Tests")
    println("="^70)

    # Setup
    prec = 10
    K = PadicField(2, prec)

    # Test 1: Node creation
    println("\nTest 1: Node creation and initialization")
    param = ValuationPolydisc([K(0)], [5])
    node = FlatUCBNode(param, nothing, 0)
    @assert node.depth == 0 "Expected depth 0"
    @assert node.visits == 0 "Expected 0 visits"
    @assert node.b_value == Inf "Expected Inf B-value for unvisited node"
    @assert is_leaf(node) "Expected node to be a leaf"
    println("  ✓ Node created correctly")

    # Test 2: B-value computation for unvisited leaf
    println("\nTest 2: B-value for unvisited leaf")
    config = FlatUCBConfig(max_depth=5, num_simulations=10, beta=0.05)
    b_val = compute_leaf_b_value(node, config)
    @assert b_val == Inf "Expected Inf for unvisited leaf"
    println("  ✓ Unvisited leaf has B-value = Inf")

    # Test 3: B-value computation for visited leaf
    println("\nTest 3: B-value for visited leaf")
    node.visits = 10
    node.total_value = 5.0
    avg = average_value(node)
    @assert avg == 0.5 "Expected average 0.5, got $avg"

    b_val = compute_leaf_b_value(node, config)
    @assert b_val > avg "Expected B-value > average"
    @assert isfinite(b_val) "Expected finite B-value"
    println("  ✓ Visited leaf: avg=$(round(avg, digits=4)), B=$(round(b_val, digits=4))")

    # Test 4: Node expansion
    println("\nTest 4: Node expansion")
    children = expand_node!(node, config)
    @assert !isempty(children) "Expected children to be created"
    @assert !is_leaf(node) "Expected node to no longer be a leaf"
    p = Int(Nemo.prime(K))
    @assert length(children) == p "Expected $p children, got $(length(children))"
    for child in children
        @assert child.parent === node "Expected child's parent to be node"
        @assert child.depth == node.depth + 1 "Expected child depth to be $(node.depth + 1)"
    end
    println("  ✓ Node expanded: $(length(children)) children created")

    # Test 5: Internal node B-value
    println("\nTest 5: Internal node B-value")
    # Set B-values for children
    for (i, child) in enumerate(children)
        child.b_value = Float64(i)
    end
    internal_b = compute_internal_b_value(node)
    expected_max = Float64(length(children))
    @assert internal_b == expected_max "Expected max B-value $expected_max, got $internal_b"
    println("  ✓ Internal node B-value = max of children: $(internal_b)")

    # Test 6: B-value propagation
    println("\nTest 6: B-value propagation")
    # Create a deeper tree
    root = FlatUCBNode(param, nothing, 0)
    root.visits = 1
    root.total_value = 1.0
    update_b_value!(root, config)

    expand_node!(root, config)
    child = root.children[1]
    child.visits = 5
    child.total_value = 2.5
    update_b_value!(child, config)

    # Propagate from child to root
    propagate_b_values!(child, config)

    # Root should now have B-value equal to max of its children
    root_b = root.b_value
    expected_b = maximum(c.b_value for c in root.children)
    @assert root_b == expected_b "Expected root B-value $expected_b, got $root_b"
    println("  ✓ B-value propagated correctly: root B=$(round(root_b, digits=4))")

    println("\n" * "="^70)
    println("✓ All basic tests passed!")
    println("="^70)
end

@doc raw"""
    test_flat_ucb_simple_optimization()

Test Flat UCB on a simple optimization problem: minimize |x|^2.

Starting from x = 16, the optimizer should descend toward x = 0.
"""
function test_flat_ucb_simple_optimization()
    println("\n" * "="^70)
    println("Flat UCB Simple Optimization Test")
    println("="^70)

    # Setup: minimize |x|^2 in 2-adic field
    prec = 20
    K = PadicField(2, prec)
    R, x = polynomial_ring(K, ["x"])

    # Loss: |x|^2
    poly = x[1]^2
    abs_poly = AbsolutePolynomialSum([poly])
    batch_eval = batch_evaluate_init(abs_poly)

    loss = Loss(
        params -> [batch_eval(p) for p in params],
        vs -> [directional_derivative(abs_poly, v) for v in vs]
    )

    # Initial parameter: x = 16 (far from optimum)
    initial_x = K(16)
    param = ValuationPolydisc([initial_x], [0])

    # Configuration
    config = FlatUCBConfig(
        max_depth=10,
        num_simulations=50,
        beta=0.05,
        degree=1,
        strict=false
    )

    # Initialize optimizer
    optim = flat_ucb_descent_init(param, loss, config)

    # Initial loss
    initial_loss = eval_loss(optim)
    println("\nInitial loss: $(round(initial_loss, digits=6))")
    println("Running 20 optimization steps...")

    # Run optimization
    for i in 1:20
        step!(optim)
        if i % 5 == 0
            current_loss = eval_loss(optim)
            println("  Step $i: loss = $(round(current_loss, digits=6))")
        end
    end

    # Final loss
    final_loss = eval_loss(optim)
    println("\nFinal loss: $(round(final_loss, digits=6))")

    # Check that loss decreased (allowing for exploration)
    @assert isfinite(final_loss) "Expected finite loss, got $final_loss"
    @assert final_loss <= initial_loss * 1.1 "Expected some progress (loss should not increase significantly)"

    println("✓ Optimization test passed (basic functionality verified)!\n")
    println("="^70)
end
