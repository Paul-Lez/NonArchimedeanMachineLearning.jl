

"""
Data types:
- DAG: stores current nodes under consideration. Internally this should be an array that stores all nodes. Each node has arrays that point to the others via their indices.
- Node type.
"""


mutable struct Node{S}
    # Store the data of the node. 
    state::S
    # Indices of the parents in the DAG.
    parent::Vector{Int}
    # Indices of the children in the DAG.
    children::Vector{Int}
    # The index of the node in the DAG.
    index::Int
end
# TODO: do we also want to handle transposition tables in this structure?

@Base.kwdef struct FlattenedDag{S}
    nodes::Vector{Union{Nothing, Node{S}}}
    staticSize::Bool=false
end

function get_parents(dag::FlattenedDag{S}, node::Node{S}) where S
    parents = dag.nodes[node.parent]
    if any(x -> x === nothing, parents)
        error("One or more parent nodes are missing in the DAG.")
    end
    return Vector{Node{S}}(parents)
end

function get_children(dag::FlattenedDag{S}, node::Node{S}) where S
    children = dag.nodes[node.children]
    if any(x -> x === nothing, children)
        error("One or more child nodes are missing in the DAG.")
    end
    return Vector{Node{S}}(children)
end

# TODO: here do we want to check that the parents actually lie in the DAG?
function set_parents!(dag::FlattenedDag{S}, node::Node{S}, parents::Vector{Node{S}}) where S
    map(x -> check_node(dag, x), parents)
    node.parent = map(x -> x.index, parents)
end

# TODO: here do we want to check that the children actually lie in the DAG?
function set_children!(dag::FlattenedDag{S}, node::Node{S}, children::Vector{Node{S}}) where S
    map(x -> check_node(dag, x), children)
    node.children = map(x -> x.index, children)
end

function set_node!(dag::FlattenedDag{S}, node::Node{S}) where S
    if node.index > length(dag.nodes)
        if dag.staticSize
            error("Node index $(node.index) exceeds current static size $(length(dag.nodes)).")
        end
        resize!(dag.nodes, node.index)
    end
    if dag.nodes[node.index] !== nothing
        error("Node with index $(node.index) already exists in the DAG.")
    end
    for i in node.parent
        parentNode = dag.nodes[i]
        if parentNode === nothing
            error("Parent node with index $i does not exist in the DAG.")
        end
        push!(parentNode.children, node.index)
    end
    for i in node.children
        childNode = dag.nodes[i]
        if childNode === nothing
            error("Child node with index $i does not exist in the DAG.")    
        end
        push!(childNode.parent, node.index)
    end
    dag.nodes[node.index] = node
end

# remove a node and delete it from all the parents and children that refer to it.
function remove_node!(dag::FlattenedDag{S}, node::Node{S}) where S
    dag.nodes[node.index] = nothing
    for nodeIdx in node.parent
        parentNode = dag.nodes[nodeIdx]
        if parentNode !== nothing
            deleteat!(parentNode.children, findfirst(==(node.index), parentNode.children))
        end
    end
    for nodeIdx in node.children
        childNode = dag.nodes[nodeIdx]
        if childNode !== nothing
            deleteat!(childNode.parent, findfirst(==(node.index), childNode.parent))
        end
    end
end

function get_node(dag::FlattenedDag{S}, index::Int) where S
    node = dag.nodes[index]
    if node === nothing
        error("Node with index $index does not exist in the DAG.")
    end
    return node
end

# Check the node is valid in the DAG.
function check_node(dag::FlattenedDag{S}, node::Node{S}) where S
    if node.index > length(dag.nodes) || dag.nodes[node.index] === nothing
        error("Node with index $(node.index) does not exist in the DAG.")
    end
    if dag.nodes[node.index] !== node
        error("Node with index $(node.index) does not match the node in the DAG.")
    end
end

function check_index(dag::FlattenedDag{S}, index::Int) where S
    if index > length(dag.nodes) || dag.nodes[index] === nothing
        error("Node with index $index does not exist in the DAG.")
    end
end

function mk_empty_flattened_dag(::Type{S}, size::Int=0) where S
    return FlattenedDag{S}(Vector{Union{Nothing, Node{S}}}(size))
end

@doc raw"""
Create a new node with the given state and index, and add it to the DAG, after checking that no node already has this index.
"""
function mk_node_with_index(dag::FlattenedDag{S}, state::S, index::Int; parent::Vector{Int}=Int[], children::Vector{Int}=Int[]) where S
    node = Node{S}(state, parent, children, index)
    set_node!(dag, node)
    return node
end

@doc raw"""
Create a new node with the given state and add it to the DAG.
If the DAG is static, an error is thrown if there is no space for a new node.
Otherwise, the DAG is resized to accommodate the new node.
"""
function mk_node(dag::FlattenedDag{S}, state::S; parent::Vector{Int}=Int[], children::Vector{Int}=Int[]) where S
    idx = findfirst(x -> x === nothing, dag.nodes)
    if idx === nothing
        if dag.staticSize
            error("Cannot add new node to static DAG.")
        end
        idx = length(dag.nodes) + 1
        resize!(dag.nodes, idx)
        dag.nodes[idx] = nothing
    end
    return mk_node_with_index(dag, state, idx; parent=parent, children=children)
end

function mk_dag_from_root(root::Node{S}, size::Int=1) where S
    dag = mk_empty_flattened_dag{S}(size)
    set_node!(dag, root)
    return dag
end

function get_state(node::Node{S}) where S
    return node.state
end

function set_state!(node::Node{S}, state::S) where S
    node.state = state
end

function add_child!(dag::FlattenedDag{S}, parent::Node{S}, child::Node{S}) where S
    push!(parent.children, child.index)
    push!(child.parent, parent.index)
end

@doc raw"""
Restrict the DAG to the span of the given node, i.e. all nodes "reachable" from the given node.
Note: this modifies the dag (and the nodes of the dag) in place.
"""
function restrict_to_span!(dag::FlattenedDag{S}, node::Node{S}) where S
    # Get sub-DAG rooted at the given node.
    # newDagNodes = fill(Union{Nothing, Node{S}}(nothing), length(dag.nodes))
    # newDag = FlattenedDag{S}(newDagNodes)
    newDagIdx = Int[node.index]
    # set_node!(newDag, node)
    queue = [node]
    while !isempty(queue)
        currentNode = popfirst!(queue)
        for childNode in get_children(dag, currentNode)
            @assert childNode !== nothing "Child node is missing in the DAG."
            # add the child node to the new DAG if it is not already in it
            if !(childNode.index in newDagIdx)
                push!(newDagIdx, childNode.index)
                # set_node!(newDag, childNode)
                push!(queue, childNode)
            end
            # remove parents from the child node that are not in the new DAG
            childNode.parent = filter(x -> x in newDagIdx, childNode.parent)
        end
    end
    # Remove nodes that are not in the new DAG
    for idx in iterate(dag.nodes)
        if !idx in newDagIdx
            dag.nodes[idx] = nothing
        end
    end
end
