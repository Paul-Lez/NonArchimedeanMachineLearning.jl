"""
Loss Landscape Visualization for Non-Archimedean Machine Learning

This module provides tools for visualizing loss landscapes on polydisc parameter spaces.
The main workflow is:

1. **Convex Hull Computation**: Given a set of polydiscs (representing parameter regions),
   compute their convex hull as a tree structure where nodes are polydiscs and edges
   represent containment relationships.

2. **Loss Sampling**: Sample loss values along geodesics (shortest paths) between
   connected nodes in the convex hull tree.

3. **Tree Visualization**: Display the convex hull tree with edges colored by loss
   values, making it easy to see which regions of parameter space have low/high loss.

Key insight: In p-adic geometry, the "space" of polydiscs naturally forms a tree
(the Bruhat-Tits tree), and optimization can navigate this tree to find minima.

# Main Functions

- `convex_hull`: Compute the convex hull tree of a set of polydiscs
- `sample_loss_landscape`: Sample a loss function along edges of the convex hull
- `plot_tree_with_loss`: Visualize the tree with edges colored by loss
- `plot_tree_simple`: Visualize just the tree structure without loss coloring

# Example

```julia
using Plots
K = PadicField(2, 20)

# Define polydiscs in parameter space
d1 = ValuationPolydisc([K(0)], [5])
d2 = ValuationPolydisc([K(8)], [5])
d3 = ValuationPolydisc([K(16)], [5])

# Build convex hull
tree = convex_hull([d1, d2, d3])

# Define loss function
loss_fn(disc) = ... # your loss function

# Sample loss along tree edges
landscape = sample_loss_landscape(tree, loss_fn, 20)

# Visualize
plot_tree_with_loss(tree, landscape, colormap=:viridis)
```
"""

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

@doc raw"""
    ConvexHullTree{S,T,N}

A tree structure representing the convex hull of a set of polydiscs.

The convex hull consists of the input polydiscs and all their joins, with
edges determined by the containment relation.

# Fields
- `nodes::Vector{ValuationPolydisc{S,T,N}}`: All nodes in the tree (input discs + joins)
- `children::Dict{Int,Vector{Int}}`: Maps node index to indices of immediate children
- `parents::Dict{Int,Vector{Int}}`: Maps node index to indices of immediate parents
- `leaf_indices::Vector{Int}`: Indices of the original input discs
"""
struct ConvexHullTree{S,T,N}
    nodes::Vector{ValuationPolydisc{S,T,N}}
    children::Dict{Int,Vector{Int}}
    parents::Dict{Int,Vector{Int}}
    leaf_indices::Vector{Int}
end

# =============================================================================
# 2. CONVEX HULL COMPUTATION
# =============================================================================

@doc raw"""
    find_polydisc_index(discs::Vector{ValuationPolydisc{S,T,N}}, p::ValuationPolydisc{S,T,N}) where {S,T,N}

Find the index of a polydisc in a vector using Berkovich equality, or return 0 if not found.

Uses the `==` operator which checks Berkovich equality: two polydiscs are equal if they
have the same radius and their centers differ by elements with valuation >= radius.

# Arguments
- `discs::Vector{ValuationPolydisc{S,T,N}}`: Vector of polydiscs to search
- `p::ValuationPolydisc{S,T,N}`: Polydisc to find

# Returns
`Int`: Index of the first equal polydisc in the vector, or 0 if not found
"""
function find_polydisc_index(discs::Vector{ValuationPolydisc{S,T,N}}, p::ValuationPolydisc{S,T,N}) where {S,T,N}
    for (i, disc) in enumerate(discs)
        if disc == p  # Uses Base.== which implements Berkovich equality
            return i
        end
    end
    return 0
end

@doc raw"""
    compute_all_joins(initial_discs::Vector{ValuationPolydisc{S,T,N}}) where {S,T,N}

Compute all polydiscs in the convex hull by iteratively computing joins.

Starting from the initial list of polydiscs, repeatedly computes pairwise joins
until no new polydiscs are generated.

# Arguments
- `initial_discs::Vector{ValuationPolydisc{S,T,N}}`: Initial list of polydiscs

# Returns
`Vector{ValuationPolydisc{S,T,N}}`: All polydiscs in the convex hull
"""
function compute_all_joins(initial_discs::Vector{ValuationPolydisc{S,T,N}}) where {S,T,N}
    # Start with the initial discs
    all_discs = copy(initial_discs)

    # Keep track of which pairs we've already joined
    processed_pairs = Set{Tuple{Int,Int}}()

    # Iteratively compute joins until no new discs are added
    changed = true
    while changed
        changed = false
        n = length(all_discs)

        # Try all pairs
        for i in 1:n
            for j in (i+1):n
                # Skip if we've already processed this pair
                if (i, j) in processed_pairs
                    continue
                end

                # Mark this pair as processed
                push!(processed_pairs, (i, j))

                # Compute the join
                j_disc = join(all_discs[i], all_discs[j])

                # Check if this is a new disc
                if find_polydisc_index(all_discs, j_disc) == 0
                    push!(all_discs, j_disc)
                    changed = true
                end
            end
        end
    end

    return all_discs
end

@doc raw"""
    radius_strictly_smaller(r1::NTuple{N,T}, r2::NTuple{N,T}) where {N,T}

Check if r1 is strictly smaller than r2 in the valuation sense.

A radius is "smaller" if it has higher valuation values (representing a smaller disc).
Returns true if all components of r1 >= r2 and at least one is strictly greater.

# Arguments
- `r1::NTuple{N,T}`: First radius tuple (valuation)
- `r2::NTuple{N,T}`: Second radius tuple (valuation)

# Returns
`Bool`: true if r1 represents a strictly smaller disc than r2
"""
function radius_strictly_smaller(r1::NTuple{N,T}, r2::NTuple{N,T}) where {N,T}
    # r1 is smaller if all components have valuation >= r2 (higher valuation = smaller)
    # and at least one is strictly greater
    r1_vec = collect(r1)
    r2_vec = collect(r2)
    all_geq = all(r1_vec .>= r2_vec)
    some_gt = any(r1_vec .> r2_vec)
    return all_geq && some_gt
end

@doc raw"""
    is_immediate_parent(parent_idx::Int, child_idx::Int, nodes::Vector{ValuationPolydisc{S,T,N}}) where {S,T,N}

Check if parent is an immediate parent of child (no intermediate node).

Returns true if:
1. nodes[child_idx] ⊆ nodes[parent_idx] (containment)
2. child has strictly smaller radius than parent (higher valuation = smaller disc)
3. There is no intermediate node k with child ⊆ k ⊆ parent

# Arguments
- `parent_idx::Int`: Index of potential parent
- `child_idx::Int`: Index of potential child
- `nodes::Vector{ValuationPolydisc{S,T,N}}`: All nodes in the tree

# Returns
`Bool`: true if parent is an immediate parent of child
"""
function is_immediate_parent(parent_idx::Int, child_idx::Int, nodes::Vector{ValuationPolydisc{S,T,N}}) where {S,T,N}
    parent = nodes[parent_idx]
    child = nodes[child_idx]

    # Check if child is contained in parent
    if !(child <= parent)
        return false
    end

    # Check if they are equal (not a parent-child relationship)
    if child == parent  # Uses Base.== which implements Berkovich equality
        return false
    end

    # Critical: child must have strictly smaller radius (higher valuation = smaller disc)
    # This prevents cycles where two nodes mutually contain each other
    if !radius_strictly_smaller(child.radius, parent.radius)
        return false
    end

    # Check if there's an intermediate node
    for k_idx in eachindex(nodes)
        if k_idx == parent_idx || k_idx == child_idx
            continue
        end

        k = nodes[k_idx]

        # Check if child ⊆ k ⊆ parent with proper radius ordering
        if (child <= k) && (k <= parent) &&
           radius_strictly_smaller(child.radius, k.radius) &&
           radius_strictly_smaller(k.radius, parent.radius)
            return false
        end
    end

    return true
end

@doc raw"""
    build_tree_structure(nodes::Vector{ValuationPolydisc{S,T,N}}, num_initial::Int) where {S,T,N}

Build parent-child relationships for the convex hull tree.

# Arguments
- `nodes::Vector{ValuationPolydisc{S,T,N}}`: All nodes (initial discs first, then joins)
- `num_initial::Int`: Number of initial discs (these are the leaves)

# Returns
`Tuple{Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}}`: (children dict, parents dict)
"""
function build_tree_structure(nodes::Vector{ValuationPolydisc{S,T,N}}, num_initial::Int) where {S,T,N}
    n = length(nodes)
    children = Dict{Int,Vector{Int}}()
    parents = Dict{Int,Vector{Int}}()

    # Initialize empty vectors for all nodes
    for i in 1:n
        children[i] = Int[]
        parents[i] = Int[]
    end

    # For each pair of nodes, check if one is an immediate parent of the other
    for i in 1:n
        for j in 1:n
            if i == j
                continue
            end

            if is_immediate_parent(i, j, nodes)
                push!(children[i], j)
                push!(parents[j], i)
            end
        end
    end

    return children, parents
end

@doc raw"""
    convex_hull(discs::Vector{ValuationPolydisc{S,T,N}}) where {S,T,N}

Compute the convex hull tree of a list of polydiscs.

The convex hull is a tree whose nodes are the input polydiscs and all their joins,
with edges determined by the containment relation.

# Arguments
- `discs::Vector{ValuationPolydisc{S,T,N}}`: List of polydiscs

# Returns
`ConvexHullTree{S,T,N}`: The convex hull tree structure

# Example
```julia
K = PadicField(2, 20)
d1 = ValuationPolydisc([K(0)], [3])
d2 = ValuationPolydisc([K(4)], [3])
d3 = ValuationPolydisc([K(8)], [3])
tree = convex_hull([d1, d2, d3])
```
"""
function convex_hull(discs::Vector{ValuationPolydisc{S,T,N}}) where {S,T,N}
    # Compute all joins
    all_nodes = compute_all_joins(discs)

    # The first n nodes are the initial discs (leaves)
    num_initial = length(discs)
    leaf_indices = collect(1:num_initial)

    # Build the tree structure
    children, parents = build_tree_structure(all_nodes, num_initial)

    return ConvexHullTree(all_nodes, children, parents, leaf_indices)
end

# =============================================================================
# 3. GEODESIC INTERPOLATION AND LOSS SAMPLING
# =============================================================================

@doc raw"""
    valuation_to_radius(val::Real, p::Integer)

Convert a valuation to an actual radius.

For a p-adic disc with valuation radius `val`, the actual radius is `p^(-val)`.

# Arguments
- `val::Real`: The valuation (can be integer or float)
- `p::Integer`: The prime

# Returns
`Float64`: The actual radius as a real number
"""
function valuation_to_radius(val::Real, p::Integer)
    return Float64(p)^(-Float64(val))
end

@doc raw"""
    radius_to_valuation(r::Real, p::Integer)

Convert an actual radius to a valuation.

For a p-adic disc with actual radius `r`, the valuation is `-log_p(r)`.

# Arguments
- `r::Real`: The actual radius
- `p::Integer`: The prime

# Returns
`Float64`: The valuation (possibly non-integer)
"""
function radius_to_valuation(r::Real, p::Integer)
    return -log(Float64(p), Float64(r))
end

@doc raw"""
    geodesic_interpolation(d1::ValuationPolydisc{S,T,N}, d2::ValuationPolydisc{S,T,N}, x::Real) where {S,T,N}

Compute the polydisc at parameter `x` along the geodesic from `d1` to `d2`.

For two discs with the same center, the geodesic is linear interpolation of radii.
If d1 = D(a, r) and d2 = D(a, s) with r ≤ s, then at parameter x ∈ [0,1],
the interpolated disc is D(a, (1-x)r + xs).

Note: This function works with actual radii (not valuations) for interpolation.
The input discs may have integer valuation radii, but the output will have
Float64 radii representing the interpolated actual radius.

# Arguments
- `d1::ValuationPolydisc{S,T,N}`: First polydisc (must be contained in d2)
- `d2::ValuationPolydisc{S,T,N}`: Second polydisc (must contain d1)
- `x::Real`: Interpolation parameter in [0, 1]

# Returns
`ValuationPolydisc{S,Float64,N}`: Interpolated polydisc with Float64 radii

# Example
```julia
K = PadicField(2, 20)
d1 = ValuationPolydisc([K(0)], [5])  # Small disc, valuation 5
d2 = ValuationPolydisc([K(0)], [2])  # Large disc, valuation 2
d_half = geodesic_interpolation(d1, d2, 0.5)  # Midpoint
```
"""
function geodesic_interpolation(d1::ValuationPolydisc{S,T,N}, d2::ValuationPolydisc{S,T,N}, x::Real) where {S,T,N}
    # Verify preconditions
    @assert dim(d1) == dim(d2) "Polydiscs must have the same dimension"
    @assert all(valuation.(collect(d1.center) .- collect(d2.center)) .>= collect(d2.radius)) "d1 and d2 must have the same center (up to Berkovich equality)"
    @assert all(collect(d1.radius) .>= collect(d2.radius)) "d1 must be contained in d2 (higher valuation = smaller disc)"
    @assert 0 <= x <= 1 "Parameter x must be in [0, 1]"

    # Get the prime and convert to Julia Int
    p = Int(prime(d1))

    # Convert valuation radii to actual radii
    radii_1 = [valuation_to_radius(r, p) for r in d1.radius]
    radii_2 = [valuation_to_radius(r, p) for r in d2.radius]

    # Linear interpolation of radii
    radii_interp = [(1 - x) * r1 + x * r2 for (r1, r2) in zip(radii_1, radii_2)]

    # Return polydisc with interpolated radii (as Float64, not converted back to valuation)
    return ValuationPolydisc(collect(d1.center), radii_interp)
end

@doc raw"""
    sample_loss_landscape(tree::ConvexHullTree{S,T,N}, f::Function, num_samples::Int=10) where {S,T,N}

Sample function values along geodesics in the convex hull tree.

For each parent-child edge in the tree, this function samples the function `f` at
`num_samples` points along the geodesic connecting them. The geodesic parameter
ranges from 0 (at the child) to 1 (at the parent).

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree structure
- `f::Function`: Function to evaluate, takes a `ValuationPolydisc` and returns a scalar
- `num_samples::Int`: Number of sample points along each geodesic (default: 10)

# Returns
`Dict{Tuple{Int,Int}, Vector{Tuple{Float64,Float64}}}`: Dictionary mapping
`(parent_idx, child_idx)` pairs to vectors of `(x, y)` pairs, where:
- `x ∈ [0,1]` is the geodesic parameter (0 = child, 1 = parent)
- `y` is the function value at that point

# Example
```julia
K = PadicField(2, 20)
d1 = ValuationPolydisc([K(0)], [3])
d2 = ValuationPolydisc([K(4)], [3])
tree = convex_hull([d1, d2])

# Define a simple function
f(disc) = sum(disc.radius)

# Sample the landscape
landscape = sample_loss_landscape(tree, f, 5)
```
"""
function sample_loss_landscape(
    tree::ConvexHullTree{S,T,N},
    f::Function,
    num_samples::Int=10
) where {S,T,N}
    @assert num_samples >= 2 "num_samples must be at least 2"

    # Dictionary to store results
    landscape = Dict{Tuple{Int,Int}, Vector{Tuple{Float64,Float64}}}()

    # Iterate through all parent nodes
    for parent_idx in keys(tree.children)
        # Iterate through all children of this parent
        for child_idx in tree.children[parent_idx]
            # Get the actual polydiscs
            parent_disc = tree.nodes[parent_idx]
            child_disc = tree.nodes[child_idx]

            # Sample along the geodesic from child (x=0) to parent (x=1)
            samples = Tuple{Float64,Float64}[]
            for i in 0:(num_samples-1)
                x = i / (num_samples - 1)

                # geodesic_interpolation expects d1 ⊆ d2
                # child is contained in parent
                interp_disc = geodesic_interpolation(child_disc, parent_disc, x)

                # Evaluate the function
                y = f(interp_disc)

                push!(samples, (x, y))
            end

            # Store in dictionary
            landscape[(parent_idx, child_idx)] = samples
        end
    end

    return landscape
end

# =============================================================================
# 4. OUTPUT AND REPORTING
# =============================================================================

@doc raw"""
    print_landscape_summary(tree::ConvexHullTree{S,T,N}, landscape::Dict) where {S,T,N}

Print a text-based summary of the loss landscape.

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree structure
- `landscape::Dict`: Loss landscape data from sample_loss_landscape

# Example
```julia
tree = convex_hull([d1, d2, d3])
landscape = sample_loss_landscape(tree, loss_func, 10)
print_landscape_summary(tree, landscape)
```
"""
function print_landscape_summary(tree::ConvexHullTree{S,T,N}, landscape::Dict) where {S,T,N}
    println("=== Loss Landscape Summary ===\n")
    println("Tree Statistics:")
    println("  Total nodes: ", length(tree.nodes))
    println("  Leaf nodes: ", length(tree.leaf_indices))
    println("  Internal nodes: ", length(tree.nodes) - length(tree.leaf_indices))
    println("  Edges (parent-child pairs): ", length(landscape))
    println()

    # Collect all loss values
    all_losses = Float64[]
    for samples in values(landscape)
        append!(all_losses, [loss for (x, loss) in samples])
    end

    println("Loss Statistics:")
    println("  Min loss: ", minimum(all_losses))
    println("  Max loss: ", maximum(all_losses))
    println("  Mean loss: ", sum(all_losses) / length(all_losses))
    println()

    println("Edges by loss range:")
    for ((parent_idx, child_idx), samples) in sort(collect(landscape), by=first)
        edge_losses = [loss for (x, loss) in samples]
        edge_min = minimum(edge_losses)
        edge_max = maximum(edge_losses)
        println("  Edge ($child_idx → $parent_idx): loss ∈ [$(round(edge_min, digits=4)), $(round(edge_max, digits=4))]")
    end
end

@doc raw"""
    plot_loss_landscape(tree::ConvexHullTree{S,T,N}, landscape::Dict; kwargs...) where {S,T,N}

Plot the loss landscape using Plots.jl (if available).

Creates a visualization showing:
- Nodes as points
- Edges colored by loss value
- Loss values along geodesics

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree structure
- `landscape::Dict`: Loss landscape data from sample_loss_landscape

# Keyword Arguments
- `title::String`: Plot title (default: "Loss Landscape")
- `colormap::Symbol`: Color scheme (default: :viridis)
- `show_node_labels::Bool`: Whether to show node indices (default: true)
- `line_width::Real`: Width of edge lines (default: 2)

# Returns
Plot object from Plots.jl

# Example
```julia
using Plots
tree = convex_hull([d1, d2, d3])
landscape = sample_loss_landscape(tree, loss_func, 10)
plt = plot_loss_landscape(tree, landscape, title="My Loss Landscape")
savefig(plt, "landscape.png")
```

# Note
Requires Plots.jl to be installed and loaded:
```julia
using Plots
```
"""
function plot_loss_landscape(tree::ConvexHullTree{S,T,N}, landscape::Dict;
                            title::String="Loss Landscape",
                            colormap::Symbol=:viridis,
                            show_node_labels::Bool=true,
                            line_width::Real=2) where {S,T,N}

    # Check if Plots is available
    if !isdefined(Main, :Plots)
        error("Plots.jl is not loaded. Please run: using Plots")
    end

    Plots = Main.Plots

    # Collect all loss values for color scaling
    all_losses = Float64[]
    for samples in values(landscape)
        append!(all_losses, [loss for (x, loss) in samples])
    end
    min_loss = minimum(all_losses)
    max_loss = maximum(all_losses)

    # Create plot
    plt = Plots.plot(title=title,
                    xlabel="Geodesic Parameter",
                    ylabel="Loss Value",
                    legend=:outerright,
                    size=(800, 600))

    # Plot each edge with its samples
    for ((parent_idx, child_idx), samples) in landscape
        xs = [x for (x, loss) in samples]
        losses = [loss for (x, loss) in samples]

        # Normalize loss for color
        mean_loss = sum(losses) / length(losses)

        label = "Edge $child_idx → $parent_idx"
        Plots.plot!(plt, xs, losses,
                   label=label,
                   linewidth=line_width,
                   marker=:circle,
                   markersize=3,
                   alpha=0.7)
    end

    return plt
end

@doc raw"""
    export_landscape_csv(tree::ConvexHullTree{S,T,N}, landscape::Dict, filename::String) where {S,T,N}

Export loss landscape data to CSV format for external plotting.

The CSV format is:
```
parent_idx,child_idx,geodesic_param,loss_value
```

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree structure
- `landscape::Dict`: Loss landscape data from sample_loss_landscape
- `filename::String`: Output CSV file path

# Example
```julia
tree = convex_hull([d1, d2, d3])
landscape = sample_loss_landscape(tree, loss_func, 10)
export_landscape_csv(tree, landscape, "landscape_data.csv")
```
"""
function export_landscape_csv(tree::ConvexHullTree{S,T,N}, landscape::Dict, filename::String) where {S,T,N}
    open(filename, "w") do io
        println(io, "parent_idx,child_idx,geodesic_param,loss_value")

        for ((parent_idx, child_idx), samples) in sort(collect(landscape), by=first)
            for (x, loss) in samples
                println(io, "$parent_idx,$child_idx,$x,$loss")
            end
        end
    end

    println("Exported landscape data to: $filename")
end

# =============================================================================
# 5. TREE VISUALIZATION
# =============================================================================

@doc raw"""
    find_root(tree::ConvexHullTree)

Find the root node of the convex hull tree (the node with no parents).

# Returns
`Int`: Index of the root node
"""
function find_root(tree::ConvexHullTree)
    for i in 1:length(tree.nodes)
        if isempty(tree.parents[i])
            return i
        end
    end
    error("No root found - tree may be malformed")
end

@doc raw"""
    find_all_roots(tree::ConvexHullTree)

Find all root nodes of the convex hull tree (nodes with no parents).

# Returns
`Vector{Int}`: Indices of all root nodes
"""
function find_all_roots(tree::ConvexHullTree)
    return [i for i in 1:length(tree.nodes) if isempty(tree.parents[i])]
end

@doc raw"""
    extract_spanning_tree(tree::ConvexHullTree{S,T,N}) where {S,T,N}

Extract a spanning tree from the convex hull DAG.

The convex hull can form a DAG (directed acyclic graph) where nodes have multiple
parents. This function extracts a proper tree by:
1. Creating a virtual root if there are multiple roots
2. Keeping only one parent per node (the one with smallest radius sum, breaking ties by index)

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree (potentially a DAG)

# Returns
`Tuple{Dict{Int,Vector{Int}}, Dict{Int,Int}, Int}`: (children, parent, root_idx)
- `children`: Maps node index to list of children in the spanning tree
- `parent`: Maps node index to its single parent (-1 for root)
- `root_idx`: Index of the root node (or -1 for virtual root)
"""
function extract_spanning_tree(tree::ConvexHullTree{S,T,N}) where {S,T,N}
    n = length(tree.nodes)

    # Initialize new parent/children structures
    new_parent = Dict{Int, Int}()
    new_children = Dict{Int, Vector{Int}}()
    for i in 1:n
        new_parent[i] = -1  # -1 means no parent yet
        new_children[i] = Int[]
    end

    # Find all roots
    roots = find_all_roots(tree)

    # Determine the single root
    if length(roots) == 1
        root_idx = roots[1]
    elseif length(roots) > 1
        # Multiple roots: pick the one with smallest radius sum as "canonical" root
        # Others will become children of this root
        root_idx = roots[1]
        min_radius_sum = sum(collect(tree.nodes[root_idx].radius))
        for r in roots[2:end]
            r_sum = sum(collect(tree.nodes[r].radius))
            if r_sum < min_radius_sum
                min_radius_sum = r_sum
                root_idx = r
            end
        end
        # Make other roots children of the canonical root
        for r in roots
            if r != root_idx
                new_parent[r] = root_idx
                push!(new_children[root_idx], r)
            end
        end
    else
        error("No roots found in tree")
    end

    # BFS to assign unique parents to all nodes
    visited = Set{Int}([root_idx])
    for r in roots
        if r != root_idx
            push!(visited, r)
        end
    end

    queue = [root_idx]
    append!(queue, [r for r in roots if r != root_idx])

    while !isempty(queue)
        current = popfirst!(queue)

        # Get original children
        for child in tree.children[current]
            if child ∉ visited
                # Assign this as the parent
                new_parent[child] = current
                push!(new_children[current], child)
                push!(visited, child)
                push!(queue, child)
            end
        end
    end

    # Handle any disconnected nodes (shouldn't happen, but just in case)
    for i in 1:n
        if i ∉ visited
            # Find any parent that was visited
            for p in tree.parents[i]
                if p in visited
                    new_parent[i] = p
                    push!(new_children[p], i)
                    push!(visited, i)
                    break
                end
            end
        end
    end

    return new_children, new_parent, root_idx
end

@doc raw"""
    compute_subtree_width(tree::ConvexHullTree, node_idx::Int, leaf_width::Float64=1.0)

Recursively compute the width needed for a subtree rooted at the given node.

# Arguments
- `tree::ConvexHullTree`: The tree structure
- `node_idx::Int`: Index of the root of the subtree
- `leaf_width::Float64`: Width allocated to each leaf node (default: 1.0)

# Returns
`Float64`: Total width needed for the subtree
"""
function compute_subtree_width(tree::ConvexHullTree, node_idx::Int, leaf_width::Float64=1.0)
    children_list = tree.children[node_idx]

    if isempty(children_list)
        # Leaf node
        return leaf_width
    else
        # Internal node: sum of children's widths
        return sum(compute_subtree_width(tree, c, leaf_width) for c in children_list)
    end
end

@doc raw"""
    compute_tree_layout(tree::ConvexHullTree{S,T,N}) where {S,T,N}

Compute (x, y) positions for all nodes in the tree for visualization.

Uses a radius-based layout where:
- Y-coordinate is determined by radius (smaller radius = higher position)
- Nodes with the same radius are at the same vertical level
- X-coordinates spread nodes to avoid overlap while respecting tree structure

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree

# Returns
`Tuple{Dict{Int, Tuple{Float64, Float64}}, Dict{Int,Vector{Int}}, Dict{Int,Int}, Int}`:
- positions: Maps node index to (x, y) position
- spanning_children: Children in the spanning tree
- spanning_parent: Parent in the spanning tree
- root_idx: Index of the root
"""
function compute_tree_layout(tree::ConvexHullTree{S,T,N}) where {S,T,N}
    n = length(tree.nodes)

    # Extract spanning tree from the DAG
    spanning_children, spanning_parent, root_idx = extract_spanning_tree(tree)

    # Get radius values for each node (sum for multi-dimensional polydiscs)
    radius_values = [sum(collect(tree.nodes[i].radius)) for i in 1:n]

    # Get unique radius values and sort them (smallest radius = top of tree)
    unique_radii = sort(unique(radius_values))

    # Map radius to y-coordinate (smallest radius = highest y)
    radius_to_y = Dict{Float64, Float64}()
    max_level = length(unique_radii) - 1
    for (level, r) in enumerate(unique_radii)
        radius_to_y[Float64(r)] = Float64(max_level - (level - 1))
    end

    # Compute subtree widths for proper x-spacing
    subtree_width = Dict{Int, Float64}()

    function compute_width(node_idx)
        children = spanning_children[node_idx]
        if isempty(children)
            subtree_width[node_idx] = 1.0
        else
            subtree_width[node_idx] = sum(compute_width(c) for c in children)
        end
        return subtree_width[node_idx]
    end
    compute_width(root_idx)

    # Compute positions: x from tree structure, y from radius
    positions = Dict{Int, Tuple{Float64, Float64}}()

    function layout_node(node_idx, x_left, x_right)
        r = Float64(radius_values[node_idx])
        y = radius_to_y[r]
        x = (x_left + x_right) / 2
        positions[node_idx] = (x, y)

        children = spanning_children[node_idx]
        if !isempty(children)
            total_width = subtree_width[node_idx]
            available_width = x_right - x_left
            current_x = x_left

            for child in children
                child_width = available_width * (subtree_width[child] / total_width)
                layout_node(child, current_x, current_x + child_width)
                current_x += child_width
            end
        end
    end

    total_width = subtree_width[root_idx]
    layout_node(root_idx, 0.0, total_width)

    return positions, spanning_children, spanning_parent, root_idx
end

@doc raw"""
    loss_to_color(loss::Float64, min_loss::Float64, max_loss::Float64, colormap::Symbol=:viridis)

Convert a loss value to an RGB color using a colormap.

# Arguments
- `loss::Float64`: The loss value
- `min_loss::Float64`: Minimum loss for normalization
- `max_loss::Float64`: Maximum loss for normalization
- `colormap::Symbol`: Color scheme (default: :viridis)

# Returns
RGB color from the colormap
"""
function loss_to_color(loss::Float64, min_loss::Float64, max_loss::Float64, colormap::Symbol=:viridis)
    Plots = Main.Plots
    cgrad = Plots.cgrad(colormap)

    # Log scale: gives large colour variation near small loss values.
    # Guard against non-positive values with a small floor.
    floor_val    = 1e-12
    log_loss     = log(max(loss,     floor_val))
    log_min_loss = log(max(min_loss, floor_val))
    log_max_loss = log(max(max_loss, floor_val))

    t = log_max_loss ≈ log_min_loss ? 0.5 :
        (log_loss - log_min_loss) / (log_max_loss - log_min_loss)
    return cgrad[clamp(t, 0.0, 1.0)]
end

@doc raw"""
    plot_tree_with_loss(tree::ConvexHullTree{S,T,N}, landscape::Dict;
                        title::String="Loss Landscape on Tree",
                        colormap::Symbol=:viridis,
                        show_node_labels::Bool=true,
                        line_width::Real=4,
                        node_size::Real=8,
                        figsize::Tuple{Int,Int}=(800, 600)) where {S,T,N}

Plot the convex hull tree with edges colored by loss values.

Creates a proper tree visualization where:
- Nodes are positioned in a hierarchical tree layout (root at top)
- Each edge segment is colored according to the loss value at that point
- A colorbar shows the loss scale

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree structure
- `landscape::Dict`: Loss landscape data from `sample_loss_landscape`

# Keyword Arguments
- `title::String`: Plot title (default: "Loss Landscape on Tree")
- `colormap::Symbol`: Color scheme (default: :viridis)
- `show_node_labels::Bool`: Whether to show node indices (default: true)
- `line_width::Real`: Width of edge lines (default: 4)
- `node_size::Real`: Size of node markers (default: 8)
- `figsize::Tuple{Int,Int}`: Figure size in pixels (default: (800, 600))

# Returns
Plot object from Plots.jl

# Example
```julia
using Plots
tree = NonArchimedeanMachineLearning.convex_hull([d1, d2, d3])
landscape = sample_loss_landscape(tree, loss_func, 20)
plt = plot_tree_with_loss(tree, landscape)
savefig(plt, "tree_landscape.png")
```
"""
function plot_tree_with_loss(tree::ConvexHullTree{S,T,N}, landscape::Dict;
                             title::String="Loss Landscape on Tree",
                             colormap::Symbol=:viridis,
                             show_node_labels::Bool=true,
                             leaf_labels::Vector{String}=String[],
                             line_width::Real=4,
                             node_size::Real=8,
                             figsize::Tuple{Int,Int}=(800, 600)) where {S,T,N}

    # Check if Plots is available
    if !isdefined(Main, :Plots)
        error("Plots.jl is not loaded. Please run: using Plots")
    end

    Plots = Main.Plots

    # Compute tree layout (returns positions and spanning tree structure)
    positions, spanning_children, spanning_parent, root_idx = compute_tree_layout(tree)

    # Build a set of edges in the spanning tree for quick lookup
    spanning_edges = Set{Tuple{Int,Int}}()
    for (parent_idx, children) in spanning_children
        for child_idx in children
            push!(spanning_edges, (parent_idx, child_idx))
        end
    end

    # Collect all loss values for color scaling (only from spanning tree edges)
    all_losses = Float64[]
    for ((parent_idx, child_idx), samples) in landscape
        # Only use edges that are in the spanning tree
        if (parent_idx, child_idx) in spanning_edges
            append!(all_losses, [loss for (x, loss) in samples])
        end
    end

    # If no spanning tree edges have loss data, use all edges
    if isempty(all_losses)
        for samples in values(landscape)
            append!(all_losses, [loss for (x, loss) in samples])
        end
    end

    min_loss = minimum(all_losses)
    max_loss = maximum(all_losses)

    # Create plot with equal aspect ratio so y-levels (radius levels) are visually correct
    plt = Plots.plot(title=title,
                     size=figsize,
                     legend=false,
                     axis=false,
                     grid=false,
                     framestyle=:none,
                     aspect_ratio=:equal)

    # Draw edges with color-mapped segments (only spanning tree edges)
    for (parent_idx, children) in spanning_children
        for child_idx in children
            # Look for this edge in the landscape data
            edge_key = (parent_idx, child_idx)
            if haskey(landscape, edge_key)
                samples = landscape[edge_key]
                parent_pos = positions[parent_idx]
                child_pos = positions[child_idx]

                # Interpolate positions along the edge for each sample
                # x=0 is at child, x=1 is at parent
                for i in 1:(length(samples)-1)
                    x1, loss1 = samples[i]
                    x2, loss2 = samples[i+1]

                    # Interpolate positions
                    pos1 = (child_pos[1] + x1 * (parent_pos[1] - child_pos[1]),
                            child_pos[2] + x1 * (parent_pos[2] - child_pos[2]))
                    pos2 = (child_pos[1] + x2 * (parent_pos[1] - child_pos[1]),
                            child_pos[2] + x2 * (parent_pos[2] - child_pos[2]))

                    # Average loss for this segment
                    avg_loss = (loss1 + loss2) / 2
                    color = loss_to_color(avg_loss, min_loss, max_loss, colormap)

                    # Draw segment
                    Plots.plot!(plt, [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                               linewidth=line_width,
                               color=color,
                               label=nothing)
                end
            else
                # No loss data for this edge, draw in gray
                parent_pos = positions[parent_idx]
                child_pos = positions[child_idx]
                Plots.plot!(plt, [parent_pos[1], child_pos[1]], [parent_pos[2], child_pos[2]],
                           linewidth=line_width,
                           color=:gray,
                           label=nothing)
            end
        end
    end

    # Draw nodes as tiny dots that blend with edges
    xs = [positions[i][1] for i in 1:length(tree.nodes)]
    ys = [positions[i][2] for i in 1:length(tree.nodes)]

    # Make nodes nearly invisible - just small dots without heavy strokes
    Plots.scatter!(plt, xs, ys,
                   markersize=1,
                   color=:black,
                   markerstrokewidth=0,
                   alpha=0.3,
                   label=nothing)

    # Add node labels if requested
    if show_node_labels
        for i in 1:length(tree.nodes)
            x, y = positions[i]
            Plots.annotate!(plt, x, y + 0.15, Plots.text(string(i), 8, :center))
        end
    end

    # Add leaf labels if provided (one label per leaf, in input order)
    if !isempty(leaf_labels)
        n_labels = min(length(leaf_labels), length(tree.leaf_indices))
        for k in 1:n_labels
            leaf_idx = tree.leaf_indices[k]
            x, y = positions[leaf_idx]
            Plots.annotate!(plt, x, y - 0.25, Plots.text(leaf_labels[k], 9, :center, :bold))
        end
    end

    # Add colorbar by creating a dummy heatmap
    # Create a small gradient for the colorbar
    z_range = range(min_loss, max_loss, length=100)
    Plots.scatter!(plt, [NaN], [NaN],
                   zcolor=[min_loss],
                   clims=(min_loss, max_loss),
                   colorbar=true,
                   colorbar_title="Loss",
                   color=colormap,
                   markershape=:none,
                   label=nothing)

    return plt
end

@doc raw"""
    plot_tree_simple(tree::ConvexHullTree{S,T,N};
                     title::String="Convex Hull Tree",
                     show_node_labels::Bool=true,
                     line_width::Real=2,
                     node_size::Real=10,
                     figsize::Tuple{Int,Int}=(800, 600)) where {S,T,N}

Plot the convex hull tree structure without loss coloring.

Useful for visualizing just the tree structure before sampling loss.

# Arguments
- `tree::ConvexHullTree{S,T,N}`: The convex hull tree structure

# Keyword Arguments
- `title::String`: Plot title (default: "Convex Hull Tree")
- `show_node_labels::Bool`: Whether to show node indices (default: true)
- `line_width::Real`: Width of edge lines (default: 2)
- `node_size::Real`: Size of node markers (default: 10)
- `figsize::Tuple{Int,Int}`: Figure size in pixels (default: (800, 600))

# Returns
Plot object from Plots.jl
"""
function plot_tree_simple(tree::ConvexHullTree{S,T,N};
                          title::String="Convex Hull Tree",
                          show_node_labels::Bool=true,
                          line_width::Real=2,
                          node_size::Real=10,
                          figsize::Tuple{Int,Int}=(800, 600)) where {S,T,N}

    # Check if Plots is available
    if !isdefined(Main, :Plots)
        error("Plots.jl is not loaded. Please run: using Plots")
    end

    Plots = Main.Plots

    # Compute tree layout (returns positions and spanning tree structure)
    positions, spanning_children, spanning_parent, root_idx = compute_tree_layout(tree)

    # Create plot with equal aspect ratio so y-levels (radius levels) are visually correct
    plt = Plots.plot(title=title,
                     size=figsize,
                     legend=false,
                     axis=false,
                     grid=false,
                     framestyle=:none,
                     aspect_ratio=:equal)

    # Draw edges (only from spanning tree)
    for (parent_idx, children) in spanning_children
        for child_idx in children
            parent_pos = positions[parent_idx]
            child_pos = positions[child_idx]

            Plots.plot!(plt, [parent_pos[1], child_pos[1]], [parent_pos[2], child_pos[2]],
                       linewidth=line_width,
                       color=:gray,
                       label=nothing)
        end
    end

    # Draw nodes as tiny dots that blend with edges
    xs = [positions[i][1] for i in 1:length(tree.nodes)]
    ys = [positions[i][2] for i in 1:length(tree.nodes)]

    # Make nodes nearly invisible - just small dots without heavy strokes
    Plots.scatter!(plt, xs, ys,
                   markersize=1,
                   color=:black,
                   markerstrokewidth=0,
                   alpha=0.3,
                   label=nothing)

    # Add node labels if requested
    if show_node_labels
        for i in 1:length(tree.nodes)
            x, y = positions[i]
            Plots.annotate!(plt, x, y + 0.15, Plots.text(string(i), 8, :center))
        end
    end

    return plt
end
