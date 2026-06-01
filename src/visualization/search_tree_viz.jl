"""
Interactive D3-based visualizations for tree-search and DAG optimizer states.
"""

using D3Trees

function _loaded_wglmakie()
    if !isdefined(Main, :WGLMakie)
        error("WGLMakie.jl is not loaded. Please run: using WGLMakie")
    end
    return Main.WGLMakie
end

##################################################
# Internal accessors to normalize across node types
##################################################

const _TreeNode{S,
    T,
    N} = Union{
    MCTSNode{S, T, N}, DAGMCTSNode{S, T, N}, DAGMCTSPathNode{S, T, N},
    DOONode{S, T, N}
}

const _TreeState{S,
    T,
    N} = Union{
    MCTSState{S, T, N}, DAGMCTSState{S, T, N}, DAGMCTSPathState{S, T, N},
    DOOState{S, T, N}
}

_polydisc(node::_TreeNode) = node.polydisc

_visits(node::_TreeNode) = node.visits
_visits(node::DOONode) = isnothing(node.value) ? 0 : 1

_avg_value(node::MCTSNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::DAGMCTSNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::DAGMCTSPathNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::DOONode) = something(node.value, 0.0)

_node_children(node::_TreeNode) = node.children

_node_type_name(::MCTSNode) = "MCTS"
_node_type_name(::DAGMCTSNode) = "DAG-MCTS"
_node_type_name(::DAGMCTSPathNode) = "DAG-MCTS path"
_node_type_name(::DOONode) = "DOO"

function _has_depth(node::_TreeNode)
    hasfield(typeof(node), :depth)
end

function _depth_val(node::_TreeNode)
    _has_depth(node) ? node.depth : nothing
end

_num_parents(node::DAGMCTSNode) = length(node.parents)
_num_parents(node::DAGMCTSPathNode) = length(node.parents)
_num_parents(::_TreeNode) = -1  # not applicable

##################################################
# 3D cone layout types
##################################################

@doc raw"""
    SearchTreeConeLayout

Computed 3D layout for a search tree or DAG drawn on a cone-like surface.

# Fields
- `nodes::Vector{Any}`: Search nodes in breadth-first order.
- `children::Vector{Vector{Int}}`: Directed edges, using indices into `nodes`.
- `depths::Vector{Int}`: Search depth for each node.
- `angles::Vector{Float64}`: Angular position for each node, in radians.
- `positions::Vector{NTuple{3,Float64}}`: Cartesian `(x, y, z)` coordinates.
"""
struct SearchTreeConeLayout
    nodes::Vector{Any}
    children::Vector{Vector{Int}}
    depths::Vector{Int}
    angles::Vector{Float64}
    positions::Vector{NTuple{3, Float64}}
end

##################################################
# Flatten tree/DAG to indexed arrays
##################################################

function _flatten_search_tree(root::_TreeNode; max_depth::Int = 10, max_nodes::Int = 10000)
    children_vec = Vector{Int}[]
    nodes_vec = Any[]
    node_to_idx = Dict{UInt, Int}()
    depths = Int[]

    # BFS queue: (node, depth)
    queue = Tuple{typeof(root), Int}[(root, 0)]
    node_to_idx[objectid(root)] = 1
    push!(nodes_vec, root)
    push!(children_vec, Int[])
    push!(depths, 0)

    head = 1
    while head <= length(queue)
        node, d = queue[head]
        head += 1
        idx = node_to_idx[objectid(node)]

        d >= max_depth && continue
        length(nodes_vec) >= max_nodes && break

        for child in _node_children(node)
            cid = objectid(child)
            if haskey(node_to_idx, cid)
                # DAG: node already visited, just add edge
                push!(children_vec[idx], node_to_idx[cid])
            else
                new_idx = length(nodes_vec) + 1
                if new_idx > max_nodes
                    break
                end
                node_to_idx[cid] = new_idx
                push!(nodes_vec, child)
                push!(children_vec, Int[])
                push!(depths, d + 1)
                push!(children_vec[idx], new_idx)
                push!(queue, (child, d + 1))
            end
        end
    end

    return children_vec, nodes_vec, depths
end

function _validate_search_tree_limits(max_depth::Int, max_nodes::Int)
    max_depth >= 0 || throw(ArgumentError("max_depth must be nonnegative"))
    max_nodes >= 1 || throw(ArgumentError("max_nodes must be positive"))
end

@doc raw"""
    build_fixed_search_tree(root::ValuationPolydisc; max_depth=3, degree=1, max_nodes=10000)

Build a deterministic finite search tree by repeatedly applying `children`.

This is useful for visual experiments before wiring the plot to a live optimizer.
The returned root is an `MCTSNode`; visits and values are left at their default
values because no search has been run.

# Arguments
- `root::ValuationPolydisc`: Starting polydisc, usually the optimizer's initial
  parameter region.
- `max_depth::Int=3`: Number of refinement levels to generate.
- `degree::Int=1`: Child-generation degree passed to `children`.
- `max_nodes::Int=10000`: Hard cap to keep accidental full trees manageable.

# Example
```julia
K = PadicField(2, 20)
root_disc = ValuationPolydisc([K(0), K(0)], [0, 0])
root = build_fixed_search_tree(root_disc; max_depth=4)
```
"""
function build_fixed_search_tree(root::ValuationPolydisc{S, T, N};
        max_depth::Int = 3,
        degree::Int = 1,
        max_nodes::Int = 10000) where {S, T, N}
    _validate_search_tree_limits(max_depth, max_nodes)
    1 <= degree <= N || throw(ArgumentError("degree must be between 1 and the polydisc dimension"))

    root_node = MCTSNode(root)
    queue = Tuple{MCTSNode{S, T, N}, Int}[(root_node, 0)]
    node_count = 1
    head = 1

    while head <= length(queue) && node_count < max_nodes
        node, depth = queue[head]
        head += 1
        depth >= max_depth && continue

        child_discs = children(node.polydisc, degree)
        isempty(child_discs) && (node.is_terminal = true)

        added_children = 0
        for child_disc in child_discs
            node_count >= max_nodes && break
            child = MCTSNode(child_disc, node)
            push!(node.children, child)
            push!(queue, (child, depth + 1))
            node_count += 1
            added_children += 1
        end

        node.is_expanded = added_children == length(child_discs)
        node.unsolved_children_count = length(node.children)
    end

    return root_node
end

@doc raw"""
    build_fixed_search_dag(root::ValuationPolydisc; max_depth=3, degree=1, max_nodes=10000)

Build a deterministic finite search DAG by repeatedly applying `children` and
deduplicating Berkovich-equal polydiscs.

This uses the same `HashedPolydisc` transposition-table keys as DAG-MCTS. If two
different refinement orders reach the same polydisc, the result contains one
shared `DAGMCTSNode` with multiple parents, rather than duplicate tree nodes.

# Arguments
- `root::ValuationPolydisc`: Starting polydisc.
- `max_depth::Int=3`: Number of refinement levels to generate.
- `degree::Int=1`: Child-generation degree passed to `children`.
- `max_nodes::Int=10000`: Hard cap on unique polydisc nodes.

# Example
```julia
K = PadicField(2, 20)
root_disc = ValuationPolydisc([K(0), K(0)], [0, 0])
root = build_fixed_search_dag(root_disc; max_depth=4)
```
"""
function build_fixed_search_dag(root::ValuationPolydisc{S, T, N};
        max_depth::Int = 3,
        degree::Int = 1,
        max_nodes::Int = 10000) where {S, T, N}
    _validate_search_tree_limits(max_depth, max_nodes)
    1 <= degree <= N || throw(ArgumentError("degree must be between 1 and the polydisc dimension"))

    root_node = DAGMCTSNode(root)
    table = Dict{HashedPolydisc{S, T, N}, DAGMCTSNode{S, T, N}}(
        HashedPolydisc(root) => root_node)
    queue = Tuple{DAGMCTSNode{S, T, N}, Int}[(root_node, 0)]
    head = 1

    while head <= length(queue)
        node, depth = queue[head]
        head += 1
        depth >= max_depth && continue

        child_discs = children(node.polydisc, degree)
        isempty(child_discs) && (node.is_terminal = true)

        added_edges = 0
        for child_disc in child_discs
            key = HashedPolydisc(child_disc)
            child_was_known = haskey(table, key)
            if child_was_known
                child = table[key]
            else
                length(table) >= max_nodes && break
                child = DAGMCTSNode(child_disc)
                table[key] = child
                push!(queue, (child, depth + 1))
            end

            if !(child in node.children)
                push!(node.children, child)
                added_edges += 1
            end
            if !(node in child.parents)
                push!(child.parents, node)
            end
        end

        node.is_expanded = added_edges == length(child_discs)
        node.unsolved_children_count = length(node.children)
    end

    return root_node
end

function _parents_from_children(children_vec::Vector{Vector{Int}})
    parents = [Int[] for _ in children_vec]
    for (idx, child_indices) in enumerate(children_vec)
        for child_idx in child_indices
            push!(parents[child_idx], idx)
        end
    end
    return parents
end

function _nodes_by_depth(depths::Vector{Int})
    isempty(depths) && return Vector{Int}[]
    max_depth = maximum(depths)
    layers = [Int[] for _ in 0:max_depth]
    for (idx, depth) in enumerate(depths)
        push!(layers[depth + 1], idx)
    end
    return layers
end

function _rank_from_layers(layers::Vector{Vector{Int}})
    rank = zeros(Int, sum(length, layers))
    for layer in layers
        for (pos, idx) in enumerate(layer)
            rank[idx] = pos
        end
    end
    return rank
end

function _neighbor_barycenter(neighbors::Vector{Int}, rank::Vector{Int}, fallback::Real)
    isempty(neighbors) && return Float64(fallback)
    return sum(rank[n] for n in neighbors) / length(neighbors)
end

function _sort_layer_by_neighbors!(layers::Vector{Vector{Int}},
        neighbors::Vector{Vector{Int}},
        depth_range)
    rank = _rank_from_layers(layers)
    for depth in depth_range
        layer = layers[depth + 1]
        sort!(layer, by = idx -> (_neighbor_barycenter(neighbors[idx], rank, rank[idx]), rank[idx]))
    end
    return layers
end

function _crossing_reduced_layers(children_vec::Vector{Vector{Int}}, depths::Vector{Int};
        iterations::Int = 8)
    layers = _nodes_by_depth(depths)
    length(layers) <= 2 && return layers

    parents = _parents_from_children(children_vec)
    max_depth = length(layers) - 1
    for _ in 1:max(0, iterations)
        _sort_layer_by_neighbors!(layers, parents, 1:max_depth)
        _sort_layer_by_neighbors!(layers, children_vec, (max_depth - 1):-1:0)
    end
    return layers
end

function _compute_cone_angles(children_vec::Vector{Vector{Int}}, depths::Vector{Int};
        crossing_reduction_iters::Int = 8)
    n = length(children_vec)
    n == 0 && return Float64[]
    n == 1 && return [0.0]

    layers = _crossing_reduced_layers(children_vec, depths; iterations = crossing_reduction_iters)
    angles = zeros(Float64, n)

    for layer in layers
        m = length(layer)
        if m == 1
            angles[layer[1]] = 0.0
        elseif m > 1
            for (pos, idx) in enumerate(layer)
                angles[idx] = 2π * (pos - 1) / m
            end
        end
    end

    return angles
end

_cone_radius(depth::Real, radial_scale::Real, radial_exponent::Real) =
    depth == 0 ? 0.0 : Float64(radial_scale) * Float64(depth)^Float64(radial_exponent)

function _cone_position(depth::Integer, angle::Real;
        level_height::Real = 1.0,
        radial_scale::Real = 0.35,
        radial_exponent::Real = 2.0)
    r = _cone_radius(depth, radial_scale, radial_exponent)
    z = -Float64(level_height) * Float64(depth)
    return (r * cos(angle), r * sin(angle), z)
end

@doc raw"""
    cone_search_tree_layout(root_or_state; max_depth=10, max_nodes=10000,
                            level_height=1.0, radial_scale=0.35,
                            radial_exponent=2.0,
                            crossing_reduction_iters=8)

Compute a 3D cone layout for a tree-search node, state, or optimizer.

Each search depth is assigned a height `z = -level_height * depth`. The radial
distance from the cone axis grows as `radial_scale * depth^radial_exponent`.
The default exponent `2.0` mirrors the surface obtained by rotating
``z = \sqrt{y}`` around the vertical axis, after using depth as the vertical
coordinate.
"""
function cone_search_tree_layout(node::_TreeNode{S, T, N};
        max_depth::Int = 10,
        max_nodes::Int = 10000,
        level_height::Real = 1.0,
        radial_scale::Real = 0.35,
        radial_exponent::Real = 2.0,
        crossing_reduction_iters::Int = 8) where {S, T, N}
    _validate_search_tree_limits(max_depth, max_nodes)
    children_vec, nodes_vec,
    depths = _flatten_search_tree(node; max_depth = max_depth, max_nodes = max_nodes)
    angles = _compute_cone_angles(children_vec, depths;
        crossing_reduction_iters = crossing_reduction_iters)
    positions = [_cone_position(depths[i], angles[i];
                     level_height = level_height,
                     radial_scale = radial_scale,
                     radial_exponent = radial_exponent)
                 for i in eachindex(nodes_vec)]

    return SearchTreeConeLayout(nodes_vec, children_vec, depths, angles, positions)
end

function cone_search_tree_layout(state::_TreeState; kwargs...)
    cone_search_tree_layout(state.root; kwargs...)
end

function cone_search_tree_layout(optim::OptimSetup; kwargs...)
    cone_search_tree_layout(optim.state; kwargs...)
end

function cone_search_tree_layout(root::ValuationPolydisc; max_depth::Int = 3, degree::Int = 1,
        max_nodes::Int = 10000, deduplicate::Bool = true, kwargs...)
    graph = deduplicate ?
            build_fixed_search_dag(root; max_depth = max_depth, degree = degree, max_nodes = max_nodes) :
            build_fixed_search_tree(root; max_depth = max_depth, degree = degree, max_nodes = max_nodes)
    return cone_search_tree_layout(graph; max_depth = max_depth, max_nodes = max_nodes, kwargs...)
end

##################################################
# Styling utilities
##################################################

function _value_to_color(avg::Float64, min_val::Float64, max_val::Float64, visits::Int)
    if visits == 0
        return "#cccccc"
    end
    range = max_val - min_val
    t = range > 0 ? clamp((avg - min_val) / range, 0.0, 1.0) : 0.5
    # Blue (240°) = good (low value) → Red (0°) = bad (high value)
    hue = round(Int, 240 * (1 - t))
    return "hsl($hue, 70%, 55%)"
end

function _truncate_padic(s::String; maxlen::Int = 16)
    length(s) <= maxlen ? s : s[1:maxlen] * "…"
end

const _NAMED_RGB_COLORS = Dict{Symbol, NTuple{3, Float64}}(
    :black => (0.0, 0.0, 0.0),
    :white => (1.0, 1.0, 1.0),
    :red => (1.0, 0.0, 0.0),
    :lightblue => (173 / 255, 216 / 255, 230 / 255),
    :blue => (0.0, 0.0, 1.0)
)

function _normalize_rgb(color)
    if color isa Symbol
        haskey(_NAMED_RGB_COLORS, color) ||
            throw(ArgumentError("unsupported named color: $color"))
        return _NAMED_RGB_COLORS[color]
    elseif color isa AbstractString
        s = startswith(color, "#") ? color[2:end] : color
        length(s) == 6 || throw(ArgumentError("hex colors must have the form #rrggbb"))
        return (parse(Int, s[1:2], base = 16) / 255,
            parse(Int, s[3:4], base = 16) / 255,
            parse(Int, s[5:6], base = 16) / 255)
    elseif color isa Tuple && length(color) == 3 && all(c -> c isa Real, color)
        scale = all(c -> 0 <= c <= 1, color) ? 1.0 : 255.0
        return ntuple(i -> clamp(Float64(color[i]) / scale, 0.0, 1.0), 3)
    else
        throw(ArgumentError("expected a Symbol, #rrggbb string, or RGB tuple"))
    end
end

function _rgb_to_hex(rgb::NTuple{3, Float64})
    parts = [string(round(Int, 255 * clamp(c, 0.0, 1.0)), base = 16, pad = 2)
             for c in rgb]
    return "#" * Base.join(parts)
end

function _interpolate_rgb(low::NTuple{3, Float64}, high::NTuple{3, Float64}, t::Real)
    u = clamp(Float64(t), 0.0, 1.0)
    return ntuple(i -> (1 - u) * low[i] + u * high[i], 3)
end

function _hsv_to_rgb(h::Real, s::Real, v::Real)
    hue = 6 * mod(Float64(h), 1.0)
    sat = clamp(Float64(s), 0.0, 1.0)
    val = clamp(Float64(v), 0.0, 1.0)
    c = val * sat
    x = c * (1 - abs(mod(hue, 2) - 1))
    m = val - c

    r, g, b = if hue < 1
        (c, x, 0.0)
    elseif hue < 2
        (x, c, 0.0)
    elseif hue < 3
        (0.0, c, x)
    elseif hue < 4
        (0.0, x, c)
    elseif hue < 5
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end

    return (r + m, g + m, b + m)
end

function _rainbow_rgb(t::Real)
    # Violet/blue for low visits, through green/yellow/orange, to red.
    hue = (1 - clamp(Float64(t), 0.0, 1.0)) * 0.78
    return _hsv_to_rgb(hue, 1.0, 1.0)
end

function _visit_color_fraction(visits::Real, max_visits::Real; scale = :linear)
    max_visits <= 0 && return 0.0

    if scale == :linear
        return clamp(Float64(visits) / Float64(max_visits), 0.0, 1.0)
    elseif scale == :sqrt
        return sqrt(clamp(Float64(visits) / Float64(max_visits), 0.0, 1.0))
    elseif scale == :log
        return clamp(log1p(max(Float64(visits), 0.0)) / log1p(Float64(max_visits)), 0.0, 1.0)
    elseif scale isa Function
        return clamp(Float64(scale(visits, max_visits)), 0.0, 1.0)
    else
        throw(ArgumentError("visit_color_scale must be :linear, :sqrt, :log, or a function"))
    end
end

function _visit_color_rgbs(nodes;
        low_visit_color = "#add8e6",
        high_visit_color = "#ff0000",
        visit_color_max = nothing,
        visit_color_scale = :log,
        visit_color_scheme = :rainbow)
    visits = [_visits(node) for node in nodes]
    max_visits = isnothing(visit_color_max) ? maximum(visits; init = 0) : visit_color_max

    if visit_color_scheme == :rainbow
        return [_rainbow_rgb(_visit_color_fraction(visits[i], max_visits;
                    scale = visit_color_scale)) for i in eachindex(nodes)]
    elseif visit_color_scheme == :gradient
        low = _normalize_rgb(low_visit_color)
        high = _normalize_rgb(high_visit_color)
        return [_interpolate_rgb(low, high,
                    _visit_color_fraction(visits[i], max_visits; scale = visit_color_scale))
                for i in eachindex(nodes)]
    elseif visit_color_scheme isa Function
        return [_normalize_rgb(visit_color_scheme(
                    _visit_color_fraction(visits[i], max_visits; scale = visit_color_scale)))
                for i in eachindex(nodes)]
    else
        throw(ArgumentError("visit_color_scheme must be :rainbow, :gradient, or a function"))
    end
end

function _visit_colors(nodes; kwargs...)
    return _rgb_to_hex.(_visit_color_rgbs(nodes; kwargs...))
end

##################################################
# Build D3Tree arrays
##################################################

function _build_d3_arrays(nodes_vec, children_vec, depths)
    n = length(nodes_vec)

    visits_vec = [_visits(nd) for nd in nodes_vec]
    values_vec = [_avg_value(nd) for nd in nodes_vec]
    max_visits = maximum(visits_vec; init = 1)
    visited_values = [values_vec[i] for i in 1:n if visits_vec[i] > 0]
    min_val = isempty(visited_values) ? 0.0 : minimum(visited_values)
    max_val = isempty(visited_values) ? 1.0 : maximum(visited_values)

    text_arr = Vector{String}(undef, n)
    tooltip_arr = Vector{String}(undef, n)
    style_arr = Vector{String}(undef, n)
    link_style_arr = Vector{String}(undef, n)

    for i in 1:n
        node = nodes_vec[i]
        v = visits_vec[i]
        avg = values_vec[i]

        # Polydisc info (used in both label and tooltip)
        p = _polydisc(node)
        c_strs = [_truncate_padic(string(c isa ValuedFieldPoint ? unwrap(c) : c))
                  for c in center(p)]
        r_strs = [string(r) for r in radius(p)]
        center_str = Base.join(c_strs, ", ")
        radius_str = Base.join(r_strs, ", ")

        # Text label (compact — polydisc info is in the tooltip)
        if v > 0
            text_arr[i] = "n=$(v)\nv=$(round(avg; digits=4))"
        else
            text_arr[i] = "n=0"
        end

        parts = String[]
        push!(parts, "D([$center_str], [$radius_str])")
        push!(parts, "$(_node_type_name(node)) node")
        d = _depth_val(node)
        if !isnothing(d)
            push!(parts, "depth: $d")
        else
            push!(parts, "depth: $(depths[i])")
        end
        push!(parts, "visits: $v")
        if v > 0
            push!(parts, "avg value: $(round(avg; digits=6))")
        end
        push!(parts, "children: $(length(children_vec[i]))")
        np = _num_parents(node)
        if np >= 0
            push!(parts, "parents: $np")
        end
        tooltip_arr[i] = Base.join(parts, "\n")

        # Style
        color = _value_to_color(avg, min_val, max_val, v)
        bw = v > 0 ? clamp(1.0 + 3.0 * log(1 + v) / log(1 + max_visits), 1.0, 5.0) : 0.5
        dash = v > 0 ? "0" : "4"
        style_arr[i] = "fill:$color;stroke-width:$(round(bw; digits=1))px;stroke:#333;stroke-dasharray:$dash"

        # Link style
        link_w = v > 0 ? clamp(1.0 + 3.0 * v / max_visits, 1.0, 5.0) : 0.5
        link_style_arr[i] = "stroke-width:$(round(link_w; digits=1))px"
    end

    return text_arr, tooltip_arr, style_arr, link_style_arr
end

function _cone_surface_arrays(max_depth::Int;
        level_height::Real = 1.0,
        radial_scale::Real = 0.35,
        radial_exponent::Real = 2.0,
        depth_samples::Int = 80,
        angle_samples::Int = 120)
    depths = range(0, max_depth; length = max(2, depth_samples))
    angles = range(0, 2π; length = max(4, angle_samples))

    xs = [(_cone_radius(depth, radial_scale, radial_exponent) * cos(angle))
          for depth in depths, angle in angles]
    ys = [(_cone_radius(depth, radial_scale, radial_exponent) * sin(angle))
          for depth in depths, angle in angles]
    zs = [(-Float64(level_height) * Float64(depth)) for depth in depths, _ in angles]
    return xs, ys, zs
end

function _plot_gradient_edge!(Plots, plt,
        p1::NTuple{3, Float64},
        p2::NTuple{3, Float64},
        color1::NTuple{3, Float64},
        color2::NTuple{3, Float64};
        steps::Int,
        edge_width::Real)
    nsteps = max(1, steps)
    for step in 1:nsteps
        t0 = (step - 1) / nsteps
        t1 = step / nsteps
        q0 = ntuple(i -> (1 - t0) * p1[i] + t0 * p2[i], 3)
        q1 = ntuple(i -> (1 - t1) * p1[i] + t1 * p2[i], 3)
        color = _rgb_to_hex(_interpolate_rgb(color1, color2, (t0 + t1) / 2))
        Plots.plot!(plt,
            [q0[1], q1[1]],
            [q0[2], q1[2]],
            [q0[3], q1[3]];
            linecolor = color,
            linewidth = edge_width,
            label = false)
    end
    return plt
end

_should_draw_node(node, visited_only::Bool) = !visited_only || _visits(node) > 0

function _push_gradient_edge_segments!(points, colors,
        point_ctor,
        p1::NTuple{3, Float64},
        p2::NTuple{3, Float64},
        color1::NTuple{3, Float64},
        color2::NTuple{3, Float64};
        steps::Int)
    nsteps = max(1, steps)
    for step in 1:nsteps
        t0 = (step - 1) / nsteps
        t1 = step / nsteps
        q0 = ntuple(i -> (1 - t0) * p1[i] + t0 * p2[i], 3)
        q1 = ntuple(i -> (1 - t1) * p1[i] + t1 * p2[i], 3)
        push!(points, point_ctor(q0[1], q0[2], q0[3]))
        push!(points, point_ctor(q1[1], q1[2], q1[3]))
        push!(colors, _rgb_to_hex(_interpolate_rgb(color1, color2, t0)))
        push!(colors, _rgb_to_hex(_interpolate_rgb(color1, color2, t1)))
    end
    return points, colors
end

##################################################
# Public API
##################################################

@doc raw"""
    plot_search_tree_cone(root_or_state; kwargs...)

Plot a tree-search node, state, or optimizer on a 3D cone-like surface.

This is the first-pass geometric visualization for 2-adic two-dimensional
search trees/DAGs. It also works for higher-dimensional tree-search states,
but the current layout encodes the search graph structure rather than every
p-adic coordinate geometrically.

# Keyword Arguments
- `max_depth::Int=10`: Maximum tree depth to include.
- `max_nodes::Int=10000`: Maximum number of nodes to include.
- `level_height::Real=1.0`: Vertical spacing between search levels.
- `radial_scale::Real=0.35`: Radial scale of the cone.
- `radial_exponent::Real=2.0`: Radial growth exponent. The default follows
  the rotated ``z = \sqrt{y}`` profile.
- `crossing_reduction_iters::Int=8`: Number of parent/child barycenter
  ordering sweeps used to place connected DAG nodes near each other.
- `show_axes::Bool=false`: Draw axis labels, ticks, grid, and frame.
- `show_surface::Bool=false`: Draw a faint cone surface behind the tree.
- `visited_only::Bool=false`: If true, draw only nodes with positive visit
  count, and only edges whose endpoints are both drawn.
- `color_by_visits::Bool=true`: Color nodes and edges by node visit counts.
- `visit_color_scheme=:rainbow`: Use `:rainbow`, `:gradient`, or a function
  from color fraction to an RGB-compatible color.
- `low_visit_color="#add8e6"`: Low-visit color when
  `visit_color_scheme=:gradient`.
- `high_visit_color="#ff0000"`: High-visit color when
  `visit_color_scheme=:gradient`.
- `visit_color_max=nothing`: Optional fixed maximum visit count for comparing
  multiple snapshots on the same scale.
- `visit_color_scale=:log`: Use `:linear`, `:sqrt`, `:log`, or a function
  `(visits, max_visits) -> fraction`.
- `edge_gradient_steps::Int=10`: Number of color segments used for each edge
  gradient.
- `node_color=:white`: Marker color for nodes.
- `edge_color=:black`: Color for parent-child lines.

The function requires `Plots` to be loaded by the caller:

```julia
using Plots
plt = plot_search_tree_cone(root)
```
"""
function plot_search_tree_cone(node::_TreeNode{S, T, N};
        max_depth::Int = 10,
        max_nodes::Int = 10000,
        level_height::Real = 1.0,
        radial_scale::Real = 0.35,
        radial_exponent::Real = 2.0,
        crossing_reduction_iters::Int = 8,
        show_axes::Bool = false,
        show_surface::Bool = false,
        surface_alpha::Real = 0.02,
        surface_color = :lightgray,
        node_size::Real = 4,
        visited_only::Bool = false,
        color_by_visits::Bool = true,
        low_visit_color = "#add8e6",
        high_visit_color = "#ff0000",
        visit_color_max = nothing,
        visit_color_scale = :log,
        visit_color_scheme = :rainbow,
        edge_gradient_steps::Int = 10,
        node_color = :white,
        node_stroke_color = :black,
        edge_color = :black,
        edge_width::Real = 1.2,
        title::String = "",
        camera = (45, 30),
        size = (900, 700),
        kwargs...) where {S, T, N}
    if !isdefined(Main, :Plots)
        error("Plots.jl is not loaded. Please run: using Plots")
    end

    Plots = Main.Plots
    layout = cone_search_tree_layout(node;
        max_depth = max_depth,
        max_nodes = max_nodes,
        level_height = level_height,
        radial_scale = radial_scale,
        radial_exponent = radial_exponent,
        crossing_reduction_iters = crossing_reduction_iters)

    plt = Plots.plot(; title = title,
        legend = false,
        xlabel = show_axes ? "x" : "",
        ylabel = show_axes ? "y" : "",
        zlabel = show_axes ? "search depth" : "",
        axis = show_axes,
        grid = show_axes,
        ticks = show_axes ? :auto : false,
        framestyle = show_axes ? :box : :none,
        camera = camera,
        size = size,
        kwargs...)

    if show_surface
        xs, ys, zs = _cone_surface_arrays(maximum(layout.depths; init = 0);
            level_height = level_height,
            radial_scale = radial_scale,
            radial_exponent = radial_exponent)
        Plots.surface!(plt, xs, ys, zs;
            color = surface_color,
            opacity = surface_alpha,
            colorbar = false,
            label = false)
    end

    node_rgbs = color_by_visits ?
                _visit_color_rgbs(layout.nodes;
        low_visit_color = low_visit_color,
        high_visit_color = high_visit_color,
        visit_color_max = visit_color_max,
        visit_color_scale = visit_color_scale,
        visit_color_scheme = visit_color_scheme) :
                NTuple{3, Float64}[]
    node_colors = color_by_visits ? _rgb_to_hex.(node_rgbs) : String[]
    draw_node = [_should_draw_node(node, visited_only) for node in layout.nodes]

    for (idx, child_indices) in enumerate(layout.children)
        draw_node[idx] || continue
        p1 = layout.positions[idx]
        for child_idx in child_indices
            draw_node[child_idx] || continue
            p2 = layout.positions[child_idx]
            if color_by_visits
                _plot_gradient_edge!(Plots, plt, p1, p2, node_rgbs[idx], node_rgbs[child_idx];
                    steps = edge_gradient_steps,
                    edge_width = edge_width)
            else
                Plots.plot!(plt,
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    [p1[3], p2[3]];
                    linecolor = edge_color,
                    linewidth = edge_width,
                    label = false)
            end
        end
    end

    drawn_indices = findall(draw_node)
    if !isempty(drawn_indices)
        xs = [layout.positions[i][1] for i in drawn_indices]
        ys = [layout.positions[i][2] for i in drawn_indices]
        zs = [layout.positions[i][3] for i in drawn_indices]
        Plots.scatter!(plt, xs, ys, zs;
            markersize = node_size,
            markercolor = color_by_visits ? node_colors[drawn_indices] : node_color,
            markerstrokewidth = 0,
            label = false)
    end

    return plt
end

function plot_search_tree_cone(state::_TreeState; kwargs...)
    plot_search_tree_cone(state.root; kwargs...)
end

function plot_search_tree_cone(optim::OptimSetup; kwargs...)
    plot_search_tree_cone(optim.state; kwargs...)
end

function plot_search_tree_cone(root::ValuationPolydisc; max_depth::Int = 3, degree::Int = 1,
        max_nodes::Int = 10000, deduplicate::Bool = true, kwargs...)
    graph = deduplicate ?
            build_fixed_search_dag(root; max_depth = max_depth, degree = degree, max_nodes = max_nodes) :
            build_fixed_search_tree(root; max_depth = max_depth, degree = degree, max_nodes = max_nodes)
    return plot_search_tree_cone(graph; max_depth = max_depth, max_nodes = max_nodes, kwargs...)
end

@doc raw"""
    plot_search_tree_cone_interactive(root_or_state; kwargs...)

Create an interactive WGLMakie 3D cone plot for a tree-search node, state, or
optimizer.

This uses the same layout and visit-colour machinery as `plot_search_tree_cone`,
but returns a WGLMakie `Figure` that can be rotated, panned, and zoomed in a
browser or notebook. Load WGLMakie before calling:

```julia
using WGLMakie
fig = plot_search_tree_cone_interactive(optim; visited_only=true)
save_search_tree_cone_interactive_html("search_tree.html", fig)
```

The keyword arguments mirror `plot_search_tree_cone`; `edge_gradient_steps`
defaults to `4` here to keep large interactive DAGs responsive. Pass
`offline=true` before saving a standalone HTML file, or use
`serve_search_tree_cone_interactive` for a live local browser view.
"""
function plot_search_tree_cone_interactive(node::_TreeNode{S, T, N};
        max_depth::Int = 10,
        max_nodes::Int = 10000,
        level_height::Real = 1.0,
        radial_scale::Real = 0.35,
        radial_exponent::Real = 2.0,
        crossing_reduction_iters::Int = 8,
        show_axes::Bool = false,
        show_surface::Bool = false,
        surface_alpha::Real = 0.02,
        surface_color = :lightgray,
        node_size::Real = 8,
        visited_only::Bool = false,
        color_by_visits::Bool = true,
        low_visit_color = "#add8e6",
        high_visit_color = "#ff0000",
        visit_color_max = nothing,
        visit_color_scale = :log,
        visit_color_scheme = :rainbow,
        edge_gradient_steps::Int = 4,
        node_color = :white,
        edge_color = :black,
        edge_width::Real = 1.5,
        title::String = "",
        size = (900, 700),
        inspect::Bool = true,
        offline::Bool = false) where {S, T, N}
    WGLMakie = _loaded_wglmakie()
    offline && WGLMakie.Page(exportable = true, offline = true)
    layout = cone_search_tree_layout(node;
        max_depth = max_depth,
        max_nodes = max_nodes,
        level_height = level_height,
        radial_scale = radial_scale,
        radial_exponent = radial_exponent,
        crossing_reduction_iters = crossing_reduction_iters)

    fig = WGLMakie.Figure(size = size)
    ax = WGLMakie.Axis3(fig[1, 1];
        title = title,
        xlabel = show_axes ? "x" : "",
        ylabel = show_axes ? "y" : "",
        zlabel = show_axes ? "search depth" : "",
        aspect = :data)
    if !show_axes
        WGLMakie.hidedecorations!(ax; grid = true)
        WGLMakie.hidespines!(ax)
    end

    if show_surface
        xs, ys, zs = _cone_surface_arrays(maximum(layout.depths; init = 0);
            level_height = level_height,
            radial_scale = radial_scale,
            radial_exponent = radial_exponent)
        WGLMakie.surface!(ax, xs, ys, zs;
            color = (surface_color, surface_alpha),
            transparency = true)
    end

    node_rgbs = color_by_visits ?
                _visit_color_rgbs(layout.nodes;
        low_visit_color = low_visit_color,
        high_visit_color = high_visit_color,
        visit_color_max = visit_color_max,
        visit_color_scale = visit_color_scale,
        visit_color_scheme = visit_color_scheme) :
                NTuple{3, Float64}[]
    node_colors = color_by_visits ? _rgb_to_hex.(node_rgbs) : String[]
    draw_node = [_should_draw_node(node, visited_only) for node in layout.nodes]

    edge_points = WGLMakie.Point3f[]
    edge_colors = String[]
    for (idx, child_indices) in enumerate(layout.children)
        draw_node[idx] || continue
        p1 = layout.positions[idx]
        for child_idx in child_indices
            draw_node[child_idx] || continue
            p2 = layout.positions[child_idx]
            if color_by_visits
                _push_gradient_edge_segments!(edge_points, edge_colors, WGLMakie.Point3f,
                    p1, p2, node_rgbs[idx], node_rgbs[child_idx];
                    steps = edge_gradient_steps)
            else
                push!(edge_points, WGLMakie.Point3f(p1[1], p1[2], p1[3]))
                push!(edge_points, WGLMakie.Point3f(p2[1], p2[2], p2[3]))
                push!(edge_colors, edge_color)
                push!(edge_colors, edge_color)
            end
        end
    end

    if !isempty(edge_points)
        WGLMakie.linesegments!(ax, edge_points;
            color = edge_colors,
            linewidth = edge_width)
    end

    drawn_indices = findall(draw_node)
    if !isempty(drawn_indices)
        points = [WGLMakie.Point3f(layout.positions[i][1], layout.positions[i][2],
                      layout.positions[i][3]) for i in drawn_indices]
        WGLMakie.scatter!(ax, points;
            markersize = node_size,
            color = color_by_visits ? node_colors[drawn_indices] : node_color,
            strokewidth = 0)
    end

    inspect && WGLMakie.DataInspector(fig)
    return fig
end

function plot_search_tree_cone_interactive(state::_TreeState; kwargs...)
    plot_search_tree_cone_interactive(state.root; kwargs...)
end

function plot_search_tree_cone_interactive(optim::OptimSetup; kwargs...)
    plot_search_tree_cone_interactive(optim.state; kwargs...)
end

function plot_search_tree_cone_interactive(root::ValuationPolydisc; max_depth::Int = 3,
        degree::Int = 1, max_nodes::Int = 10000, deduplicate::Bool = true, kwargs...)
    graph = deduplicate ?
            build_fixed_search_dag(root; max_depth = max_depth, degree = degree, max_nodes = max_nodes) :
            build_fixed_search_tree(root; max_depth = max_depth, degree = degree, max_nodes = max_nodes)
    return plot_search_tree_cone_interactive(graph; max_depth = max_depth, max_nodes = max_nodes,
        kwargs...)
end

@doc raw"""
    save_search_tree_cone_interactive_html(filename, fig; title="Search Tree Cone")

Save a WGLMakie figure returned by `plot_search_tree_cone_interactive` as a
standalone HTML document.

This uses WGLMakie's Bonito static exporter when available, which writes a full
HTML document rather than an HTML fragment. That makes the exported file more
reliable when opened directly in a browser.

```julia
using WGLMakie

fig = plot_search_tree_cone_interactive(optim; visited_only=true, offline=true)
save_search_tree_cone_interactive_html("search_tree.html", fig)
```
"""
function save_search_tree_cone_interactive_html(filename::AbstractString, fig;
        title::AbstractString = "Search Tree Cone")
    WGLMakie = _loaded_wglmakie()
    if isdefined(WGLMakie, :Bonito)
        Bonito = getproperty(WGLMakie, :Bonito)
        if isdefined(Bonito, :export_static) && isdefined(Bonito, :App)
            Bonito.export_static(String(filename), Bonito.App(fig; title = String(title)))
            return filename
        end
    end

    WGLMakie.Page(exportable = true, offline = true)
    WGLMakie.save(String(filename), fig)
    return filename
end

@doc raw"""
    serve_search_tree_cone_interactive(fig; host="127.0.0.1", port=8765,
        title="Search Tree Cone", block=false)

Serve a WGLMakie figure returned by `plot_search_tree_cone_interactive` through
a local Bonito server.

This is the most reliable way to inspect the 3D plot interactively in a browser:
the Julia process stays alive, WGLMakie can use its normal browser connection,
and the plot is available at the returned `url`.

```julia
using WGLMakie

fig = plot_search_tree_cone_interactive(optim; visited_only=true)
served = serve_search_tree_cone_interactive(fig)
println(served.url)
wait(served.server)
```
"""
function serve_search_tree_cone_interactive(fig;
        host::AbstractString = "127.0.0.1",
        port::Integer = 8765,
        title::AbstractString = "Search Tree Cone",
        block::Bool = false)
    WGLMakie = _loaded_wglmakie()
    if !isdefined(WGLMakie, :Bonito)
        error("Could not find Bonito through WGLMakie. Please make sure WGLMakie.jl is loaded.")
    end

    Bonito = getproperty(WGLMakie, :Bonito)
    app = Bonito.App(() -> fig; title = String(title))
    server = Bonito.Server(app, String(host), Int(port))

    browser_host = host in ("127.0.0.1", "0.0.0.0") ? "localhost" : String(host)
    url = string(server.protocol, browser_host, ":", server.port, "/")

    if block
        println("Serving interactive search-tree plot at ", url)
        println("Press Ctrl+C to stop the server.")
        Base.wait(server)
    end

    return (server = server, url = url)
end

@doc raw"""
    visualize_search_tree(root_or_state; max_depth=10, max_nodes=10000, init_expand=3, kwargs...)

Create an interactive D3Tree visualization from any NonArchimedeanMachineLearning tree search node or state.

# Arguments
- `root_or_state`: Any search tree node (MCTSNode, DAGMCTSNode, DOONode),
  state (MCTSState, DAGMCTSState, DOOState),
  or OptimSetup with a tree search state.
- `max_depth::Int=10`: Maximum tree depth to include.
- `max_nodes::Int=10000`: Maximum number of nodes to include.
- `init_expand::Int=3`: Number of levels expanded initially in the visualization.
- `title::String`: Title for the visualization (auto-generated from node type if omitted).
- `svg_height::Int=800`: Height of the SVG in pixels.
- Additional keyword arguments are forwarded to `D3Tree`.

# Returns
A `D3Tree` object. Displays interactively in Jupyter/VSCode, or use `inbrowser(tree)`.

# Examples
```julia
# After running MCTS optimization
optim = mcts_descent_init(param, loss, config)
for i in 1:50; step!(optim); end

tree = visualize_search_tree(optim)
# or: tree = visualize_search_tree(optim.state)
# or: tree = visualize_search_tree(optim.state.root)
```
"""
function visualize_search_tree(node::_TreeNode{S, T, N};
        max_depth::Int = 10,
        max_nodes::Int = 10000,
        init_expand::Int = 3,
        title::String = "$(_node_type_name(node)) Search Tree",
        svg_height::Int = 800,
        kwargs...) where {S, T, N}
    children_vec, nodes_vec,
    depths = _flatten_search_tree(node; max_depth = max_depth, max_nodes = max_nodes)
    text, tooltip, style, link_style = _build_d3_arrays(nodes_vec, children_vec, depths)

    return D3Tree(children_vec;
        text = text,
        tooltip = tooltip,
        style = style,
        link_style = link_style,
        title = title,
        init_expand = init_expand,
        svg_height = svg_height,
        kwargs...
    )
end

function visualize_search_tree(state::_TreeState; kwargs...)
    visualize_search_tree(state.root; kwargs...)
end

function visualize_search_tree(optim::OptimSetup; kwargs...)
    visualize_search_tree(optim.state; kwargs...)
end
