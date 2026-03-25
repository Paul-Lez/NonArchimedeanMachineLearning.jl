########### Search Tree Visualization with D3Trees ###########

# Interactive visualization of NAML tree search optimizer states using D3Trees.jl.

using D3Trees

##################################################
# Internal accessors to normalize across node types
##################################################

const _TreeNode{S,T,N} = Union{
    MCTSNode{S,T,N}, UCTNode{S,T,N}, DAGMCTSNode{S,T,N},
    HOONode{S,T,N}, ModifiedUCTNode{S,T,N}, FlatUCBNode{S,T,N},
    DOONode{S,T,N}
}

const _TreeState{S,T,N} = Union{
    MCTSState{S,T,N}, UCTState{S,T,N}, DAGMCTSState{S,T,N},
    HOOState{S,T,N}, ModifiedUCTState{S,T,N}, FlatUCBState{S,T,N},
    DOOState{S,T,N}
}

_polydisc(node::_TreeNode) = node.polydisc

_visits(node::_TreeNode) = node.visits
_visits(node::DOONode) = isnothing(node.value) ? 0 : 1

_avg_value(node::MCTSNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::UCTNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::DAGMCTSNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::ModifiedUCTNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::FlatUCBNode) = node.visits > 0 ? node.total_value / node.visits : 0.0
_avg_value(node::HOONode) = node.visits > 0 ? node.sum_values / node.visits : 0.0
_avg_value(node::DOONode) = something(node.value, 0.0)

_node_children(node::_TreeNode) = node.children

_node_type_name(::MCTSNode) = "MCTS"
_node_type_name(::UCTNode) = "UCT"
_node_type_name(::DAGMCTSNode) = "DAG-MCTS"
_node_type_name(::HOONode) = "HOO"
_node_type_name(::ModifiedUCTNode) = "Mod-UCT"
_node_type_name(::FlatUCBNode) = "Flat-UCB"
_node_type_name(::DOONode) = "DOO"

function _has_depth(node::_TreeNode)
    hasfield(typeof(node), :depth)
end

function _depth_val(node::_TreeNode)
    _has_depth(node) ? node.depth : nothing
end

_num_parents(node::DAGMCTSNode) = length(node.parents)
_num_parents(::_TreeNode) = -1  # not applicable

##################################################
# Flatten tree/DAG to indexed arrays
##################################################

function _flatten_search_tree(root::_TreeNode; max_depth::Int=10, max_nodes::Int=10000)
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

function _truncate_padic(s::String; maxlen::Int=16)
    length(s) <= maxlen ? s : s[1:maxlen] * "…"
end

##################################################
# Build D3Tree arrays
##################################################

function _build_d3_arrays(nodes_vec, children_vec, depths)
    n = length(nodes_vec)

    visits_vec = [_visits(nd) for nd in nodes_vec]
    values_vec = [_avg_value(nd) for nd in nodes_vec]
    max_visits = maximum(visits_vec; init=1)
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
        c_strs = [_truncate_padic(string(c isa ValuedFieldPoint ? unwrap(c) : c)) for c in center(p)]
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

##################################################
# Public API
##################################################

@doc raw"""
    visualize_search_tree(root_or_state; max_depth=10, max_nodes=10000, init_expand=3, kwargs...)

Create an interactive D3Tree visualization from any NAML tree search node or state.

# Arguments
- `root_or_state`: Any search tree node (MCTSNode, UCTNode, etc.), state (MCTSState, etc.),
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
optim = mcts_descent_init(param, loss, 1, config)
for i in 1:50; step!(optim); end

tree = visualize_search_tree(optim)
# or: tree = visualize_search_tree(optim.state)
# or: tree = visualize_search_tree(optim.state.root)
```
"""
function visualize_search_tree(node::_TreeNode{S,T,N};
        max_depth::Int=10,
        max_nodes::Int=10000,
        init_expand::Int=3,
        title::String="$(_node_type_name(node)) Search Tree",
        svg_height::Int=800,
        kwargs...) where {S,T,N}
    children_vec, nodes_vec, depths = _flatten_search_tree(node; max_depth=max_depth, max_nodes=max_nodes)
    text, tooltip, style, link_style = _build_d3_arrays(nodes_vec, children_vec, depths)

    return D3Tree(children_vec;
        text=text,
        tooltip=tooltip,
        style=style,
        link_style=link_style,
        title=title,
        init_expand=init_expand,
        svg_height=svg_height,
        kwargs...
    )
end

function visualize_search_tree(state::_TreeState; kwargs...)
    visualize_search_tree(state.root; kwargs...)
end

function visualize_search_tree(optim::OptimSetup; kwargs...)
    visualize_search_tree(optim.state; kwargs...)
end
