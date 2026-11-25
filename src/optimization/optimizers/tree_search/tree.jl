

"""
Data types:
- DAG: stores current nodes under consideration. Internally this should be an array that stores all nodes. Each node has arrays that point to the others via their indices.
- Node type.
"""

abstract type AbstractNode{S} end

mutable struct Node{S} <: AbstractNode{S}
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

# Maybe here the optimal thing to do is to have an absract structure too?
@Base.kwdef struct FlattenedDag{S, T<:AbstractNode{S}}
    nodes::Vector{Union{Nothing, T}}
    staticSize::Bool=false
end

function get_parents(dag::FlattenedDag{S, T}, node::T) where {S, T<:AbstractNode{S}}
    parents = dag.nodes[node.parent]
    if any(x -> x === nothing, parents)
        error("One or more parent nodes are missing in the DAG.")
    end
    return Vector{T}(parents)
end

function get_children(dag::FlattenedDag{S, T}, node::T) where {S, T<:AbstractNode{S}}
    children = dag.nodes[node.children]
    if any(x -> x === nothing, children)
        error("One or more child nodes are missing in the DAG.")
    end
    return Vector{T}(children)
end

# TODO: here do we want to check that the parents actually lie in the DAG?
function set_parents!(dag::FlattenedDag{S, T}, node::T, parents::Vector{T}) where {S, T<:AbstractNode{S}}
    map(x -> check_node(dag, x), parents)
    node.parent = map(x -> x.index, parents)
end

# TODO: here do we want to check that the children actually lie in the DAG?
function set_children!(dag::FlattenedDag{S, T}, node::T, children::Vector{T}) where {S, T<:AbstractNode{S}}
    map(x -> check_node(dag, x), children)
    node.children = map(x -> x.index, children)
end

function set_node!(dag::FlattenedDag{S, T}, node::T) where {S, T<:AbstractNode{S}}
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
function remove_node!(dag::FlattenedDag{S, T}, node::T) where {S, T<:AbstractNode{S}}
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

function get_node(dag::FlattenedDag{S, T}, index::Int) where {S, T<:AbstractNode{S}}
    node = dag.nodes[index]
    if node === nothing
        error("Node with index $index does not exist in the DAG.")
    end
    return node
end

# Check the node is valid in the DAG.
function check_node(dag::FlattenedDag{S, T}, node::T) where {S, T<:AbstractNode{S}}
    if node.index > length(dag.nodes) || dag.nodes[node.index] === nothing
        error("Node with index $(node.index) does not exist in the DAG.")
    end
    if dag.nodes[node.index] !== node
        error("Node with index $(node.index) does not match the node in the DAG.")
    end
end

function check_index(dag::FlattenedDag{S, T}, index::Int) where {S, T<:AbstractNode{S}}
    if index > length(dag.nodes) || dag.nodes[index] === nothing
        error("Node with index $index does not exist in the DAG.")
    end
end

function mk_empty_flattened_dag(::Type{S}, ::Type{T}, size::Int=0) where {S, T<:AbstractNode{S}}
    return FlattenedDag{S, T}(Vector{Union{Nothing, T}}(size))
end

# This one is tricky.

@doc raw"""
Create a new node with the given state and index, and add it to the DAG, after checking that no node already has this index.
"""
function mk_node_with_index(dag::FlattenedDag{S, T}, state::S, index::Int; parent::Vector{Int}=Int[], children::Vector{Int}=Int[]) where {S, T<:AbstractNode{S}}
    # Here, we assume that T is a concrete type that can be constructed with the given arguments.
    node = T(state, parent, children, index)
    set_node!(dag, node)
    return node
end

@doc raw"""
Create a new node with the given state and add it to the DAG.
If the DAG is static, an error is thrown if there is no space for a new node.
Otherwise, the DAG is resized to accommodate the new node.
"""
function mk_node(dag::FlattenedDag{S, T}, state::S; parent::Vector{Int}=Int[], children::Vector{Int}=Int[]) where {S, T<:AbstractNode{S}}
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

function mk_dag_from_root(root::T, size::Int=1) where {S, T<:AbstractNode{S}}
    dag = mk_empty_flattened_dag{S, T}(size)
    set_node!(dag, root)
    return dag
end

function get_state(node::Node{S}) where S
    return node.state
end

function set_state!(node::Node{S}, state::S) where S
    node.state = state
end

function add_child!(dag::FlattenedDag{S, Node{S}}, parent::Node{S}, child::Node{S}) where {S}
    !(child.index in parent.children) && push!(parent.children, child.index)
    !(parent.index in child.parent) && push!(child.parent, parent.index)
end

@doc raw"""
Restrict the DAG to the span of the given node, i.e. all nodes "reachable" from the given node.
Note: this modifies the dag (and the nodes of the dag) in place.
"""
function restrict_to_span!(dag::FlattenedDag{S, T}, node::T) where {S, T<:AbstractNode{S}}
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

mutable struct MCTSState{S} <: AbstractNode{S}
    # Store the data of the node. 
    state::S
    # The value of the node.
    value::Float64
    # The number of times the node has been visited.
    visits::Int
    # The index of the node in the DAG.
    index::Int
    # The indices of the parents in the DAG.
    parent::Vector{Int}
    # The indices of the children in the DAG.
    children::Vector{Int}
    # Whether or not the node is terminal
    is_terminal::Bool
    # Whether or not the node is solved
    is_solved::Bool
    # Whether or not the node is expanded
    is_expanded::Bool
    # The value of the node, as computed by the evaluation function
    evaluation::Float64
end

get_state(state::MCTSState{S}) where S = state.state
set_state!(state::MCTSState{S}, new_state::S) where S = (state.state = new_state)

get_value(state::MCTSState{S}) where S = state.value
set_value!(state::MCTSState{S}, new_value::Float64) where S = (state.value = new_value)

get_visits(state::MCTSState{S}) where S = state.visits
set_visits!(state::MCTSState{S}, new_visits::Int) where S = (state.visits = new_visits)

get_index(state::MCTSState{S}) where S = state.index
set_index!(state::MCTSState{S}, new_index::Int) where S = (state.index = new_index)

get_parent(state::MCTSState{S}) where S = state.parent
set_parent!(state::MCTSState{S}, new_parent::Vector{Int}) where S = (state.parent = new_parent)

get_children(state::MCTSState{S}) where S = state.children
set_children!(state::MCTSState{S}, new_children::Vector{Int}) where S = (state.children = new_children)

is_terminal(state::MCTSState{S}) where S = state.is_terminal
set_terminal!(state::MCTSState{S}, new_terminal::Bool) where S = (state.is_terminal = new_terminal)

is_solved(state::MCTSState{S}) where S = state.is_solved
set_solved!(state::MCTSState{S}, new_solved::Bool) where S = (state.is_solved = new_solved)

is_expanded(state::MCTSState{S}) where S = state.is_expanded
set_expanded!(state::MCTSState{S}, new_expanded::Bool) where S = (state.is_expanded = new_expanded)

get_evaluation(state::MCTSState{S}) where S = state.evaluation
set_evaluation!(state::MCTSState{S}, new_evaluation::Float64) where S = (state.evaluation = new_evaluation)



"""
MCTS:
To implement:
- expand!: here, we probably want to assume that there is a notion of expanding the state of the node.
- select_path: choose child from root to leaf, according to some criterion (which should not be hardcoded here).
- backpropagate!
- evaluate (do we want this? Or should we assume that this is already implemented? Design decision here!)
- run_mcts: do N simulations. Here perhaps we should let N be a parameter?
- find_best_node: find the best node in the tree according to some criteria (do we really want this?)


Then: compatibility layer with the OptimSetup infrastructure.
"""

@doc raw"""
    expand!(dag::FlattenedDag{S}, node::Node{S})
Expand the given node in the DAG by generating its children states and adding them to the DAG.
This also marks the node as expanded, and checks whether it is terminal (and thus solved).
"""
function expand!(dag::FlattenedDag{S}, node::Node{S}) where S
    children = expand(get_state(node))
    for childState in children
        chilNode = mk_node(dag, childState; parent=[node.index])
        add_child!(dag, node, chilNode)
    end
    if isempty(children)
        # TODO: create the suitable MCTS state and implement this function!
        set_solved!(node, true)
        # TODO: do we want these two to be separate?
        set_terminal!(node, true)
    end
    # TODO : implement this too!
    set_expanded!(node, true)
end

@doc raw"""
    select_path(dag::FlattenedDag{S}, node::Node{S})

Select a path down the DAG starting from the given node, according to some selection criterion,
which is passsed as a parameter to this function.
"""
function select_path(dag::FlattenedDag{S}, node::Node{S}) where S
    path = [node]
    currentNode = node
    # TODO: the original code was written in terms of !node.is_solved too. Do we want to keep it that way?
    # We can either choose to explore a different node than the solved one, or to continue exploring the solved node.
    # (which will just make it more likely that we pick the solved one when we make one step)
    while is_expanded(currentNode) && !isempty(currentNode.children)
        currentNode = select_child(dag, currentNode)
    end
    return currentNode
end

function evaluate(node::Node{S}) where S
    # Possibly we should also store this value in the node?
    evaluate(get_state(node))
end

function check_solved!(node::Node{S}) where S
    if is_terminal(node)
        set_solved!(node, true)
    end

    # TODO: Is this what we want? Are we fine assuming that the children are already all up to date?
    set_solved!(node, all(is_solved(get_children(node))))
end

function backpropagate!(dag::FlattenedDag{S}, node::Node{S}, value::Float64) where S
    queue = [node]
    while !isempty(queue)
        currentNode = popfirst!(queue)
        parents = get_parents(dag, currentNode)
        append!(queue, parents)
        new_value = get_value(currentNode) + value
        new_visits = get_visits(currentNode) + 1
        set_value!(currentNode, new_value)
        set_visits!(currentNode, new_visits)
        # TODO: implement this!
        check_solved!(currentNode)
    end
end

function run_mcts(dag::FlattenedDag{S}, root::Node{S}, num_simulations::Int) where S
    for _ in 1:num_simulations
        path = select_path(dag, root)
        if !is_expanded(path)
            expand!(dag, path)
        end
        value = evaluate(path)
        set_evaluation!(get_state(path), value)
        backpropagate!(dag, path, value)
    end
end
