include("../../src/basic/polydisc.jl")

# This file contains some API for embedding phylogenetic trees into the disc space. 
mutable struct Node
    name
    children::Vector{Node}
end

function Base.:(==)(n1::Node, n2::Node)
    return (n1.name == n2.name) && (n1.children == n2.children)
end

# This is buggy so we need to fix this
function parse_newick(str::String)
    stack = Vector{Node}()  # Create an empty stack to keep track of nodes
    current_node = Node("", [])  # Create a root node with an empty name and no children
    push!(stack, current_node)  # Push the root node onto the stack
    parent_node = nothing
    for char in str
        if char == '('  # If the character is '(', create a new node and make it a child of the current node
            new_node = Node("", [])
            push!(current_node.children, new_node)
            push!(stack, new_node)  # Push the new node onto the stack
            parent_node = current_node
            current_node = new_node  # Update the current node to the new node
        elseif char == ')'  # If the character is ')', move back to the parent node
            current_node = parent_node  # Pop the top node from the stack and make it the current node
            parent_node = nothing 
            # previous: current_node = pop!(stack)
        elseif char == ','  # If the character is ',', create a new node and make it a sibling of the current node
            new_node = Node("", [])
            push!(parent_node.children, new_node)  # Add the new node as a child of the parent node
            current_node = new_node  # Update the current node to the new node
        elseif char == ';'
            break
        else  # If the character is not '(', ')', or ',', append it to the name of the current node
            current_node.name *= char
        end
    end

    return stack[1]#.children[1]  # Return the first child of the root node
end

function disc_embedding(tree, basepoint::ValuationPolydisc{S, T}) where S where T
    """
    disc_embedding(tree, basepoint) recursively computes an embedding of tree into the 
    disc space sending the root node of tree to the point basepoint.
    """
    # If the tree only has one node then return a single node: the basepoint
    if isempty(tree.children)
        return Node(basepoint, Vector{ValuationPolydisc{S, T}}([]))
    else
        pts = children(basepoint)
        node_embeddings = Vector()
        sizehint!(node_embeddings, length(tree.children))
        # otherwise, iterate though the children of the root node
        for i in Base.eachindex(tree.children)
            # assign a child disc of the basepoint to each of these, and compute the corresponding embedding
            node_as_disc = disc_embedding(tree.children[i], pts[i])
            push!(node_embeddings, node_as_disc)
        end 
        return Node(basepoint, node_embeddings)
    end 
end

function field_embedding(tree, basepoint::ValuationPolydisc{S, T}) where S where T
    return true
end 