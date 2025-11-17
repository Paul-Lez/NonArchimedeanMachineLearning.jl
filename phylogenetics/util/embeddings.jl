include("../../src/basic/polydisc.jl")

@doc raw"""
    Node

A node structure for representing phylogenetic trees.

Used to represent tree structures that can be embedded into disc spaces. Each node
has a name identifier and a vector of children nodes.

# Fields
- `name`: Identifier for the node (typically a string for leaf names, or a polydisc for embedded nodes)
- `children::Vector{Node}`: Vector of child nodes in the tree
"""
mutable struct Node
    name
    children::Vector{Node}
end

@doc raw"""
    Base.:(==)(n1::Node, n2::Node)

Check equality of two tree nodes.

Two nodes are equal if they have the same name and their children vectors are equal.

# Arguments
- `n1::Node`: First node
- `n2::Node`: Second node

# Returns
`Bool`: True if nodes are equal
"""
function Base.:(==)(n1::Node, n2::Node)
    return (n1.name == n2.name) && (n1.children == n2.children)
end

@doc raw"""
    parse_newick(str::String)

Parse a Newick format string into a tree structure.

Parses phylogenetic tree notation in Newick format (e.g., `"(A,(B,C));"`).
Returns a tree as nested `Node` structures.

# Arguments
- `str::String`: A string in Newick format

# Returns
`Node`: The root node of the parsed tree

# Note
This implementation has known bugs and may not handle all Newick format variations correctly.
Use with caution and consider improvements for full format support.
"""
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

@doc raw"""
    disc_embedding(tree, basepoint::ValuationPolydisc{S, T}) where S where T

Recursively embed a phylogenetic tree into disc (polydisc) space.

Maps the tree structure into the hierarchical p-adic disc structure, with the root
mapped to the given basepoint and children mapped to child discs.

# Arguments
- `tree::Node`: The tree to embed (nodes have string names)
- `basepoint::ValuationPolydisc{S, T}`: The polydisc to which the root is mapped

# Returns
`Node`: The embedded tree where each node's name is replaced by its corresponding polydisc

# Algorithm
- Leaf nodes are mapped to their basepoint with no children
- Internal nodes are mapped recursively: each child subtree is embedded in one of the
  child discs of the parent's basepoint
"""
function disc_embedding(tree, basepoint::ValuationPolydisc{S, T}) where S where T
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

@doc raw"""
    field_embedding(tree, basepoint::ValuationPolydisc{S, T}) where S where T

Embed a tree into the p-adic field (stub implementation).

Currently unimplemented. This function is intended to provide an alternative embedding
using field-based methods rather than disc-based methods.

# Arguments
- `tree::Node`: The tree to embed
- `basepoint::ValuationPolydisc{S, T}`: The basepoint for embedding

# Returns
Currently returns `true` (placeholder)

# TODO
Full implementation needed.
"""
function field_embedding(tree, basepoint::ValuationPolydisc{S, T}) where S where T
    return true
end