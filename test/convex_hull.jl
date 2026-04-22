## Test file for convex hull computation

include("../src/NonArchimedeanMachineLearning.jl")
using .NonArchimedeanMachineLearning
using Oscar

println("=== Testing Convex Hull Computation ===\n")

# Setup: 2-adic field with precision 20
prec = 20
K = PadicField(2, prec)

println("Test 1: Three polydiscs in a line")
println("-" ^ 40)
# Create three polydiscs: D(0,3), D(4,3), D(8,3)
# These should form a simple tree with their join at the root
d1 = ValuationPolydisc([K(0)], [3])
d2 = ValuationPolydisc([K(4)], [3])
d3 = ValuationPolydisc([K(8)], [3])

discs_3 = [d1, d2, d3]
tree_3 = NonArchimedeanMachineLearning.convex_hull(discs_3)

println("Number of input discs: ", length(discs_3))
println("Number of nodes in convex hull: ", length(tree_3.nodes))
println("Leaf indices: ", tree_3.leaf_indices)
println()

# Display all nodes
println("All nodes:")
for (i, node) in enumerate(tree_3.nodes)
    println("  Node $i: center=$(node.center), radius=$(node.radius)")
end
println()

# Display tree structure
println("Tree structure (parent -> children):")
for (parent, children_list) in sort(collect(tree_3.children))
    if !isempty(children_list)
        println("  Node $parent -> ", children_list)
    end
end
println()

println("Tree structure (child -> parents):")
for (child, parents_list) in sort(collect(tree_3.parents))
    if !isempty(parents_list)
        println("  Node $child <- ", parents_list)
    end
end
println("\n")

println("Test 2: Four polydiscs")
println("-" ^ 40)
# Create four polydiscs to form a more complex tree
d1 = ValuationPolydisc([K(0)], [4])
d2 = ValuationPolydisc([K(8)], [4])
d3 = ValuationPolydisc([K(16)], [4])
d4 = ValuationPolydisc([K(24)], [4])

discs_4 = [d1, d2, d3, d4]
tree_4 = NonArchimedeanMachineLearning.convex_hull(discs_4)

println("Number of input discs: ", length(discs_4))
println("Number of nodes in convex hull: ", length(tree_4.nodes))
println("Leaf indices: ", tree_4.leaf_indices)
println()

# Display all nodes
println("All nodes:")
for (i, node) in enumerate(tree_4.nodes)
    println("  Node $i: center=$(node.center), radius=$(node.radius)")
end
println()

# Display tree structure
println("Tree structure (parent -> children):")
for (parent, children_list) in sort(collect(tree_4.children))
    if !isempty(children_list)
        println("  Node $parent -> ", children_list)
    end
end
println()

println("Tree structure (child -> parents):")
for (child, parents_list) in sort(collect(tree_4.parents))
    if !isempty(parents_list)
        println("  Node $child <- ", parents_list)
    end
end
println("\n")

println("Test 3: Verification of join computation")
println("-" ^ 40)
# Manually verify a join
j_12 = NonArchimedeanMachineLearning.join(d1, d2)
println("Join of d1 and d2:")
println("  d1: center=$(d1.center), radius=$(d1.radius)")
println("  d2: center=$(d2.center), radius=$(d2.radius)")
println("  join: center=$(j_12.center), radius=$(j_12.radius)")
println()

# Check if this join is in the tree
join_idx = find_polydisc_index(tree_4.nodes, j_12)
if join_idx > 0
    println("✓ Join found in tree at index $join_idx")
else
    println("✗ Join NOT found in tree")
end
println()

println("=== Tests Complete ===")
