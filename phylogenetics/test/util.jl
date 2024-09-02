using Test
include("../util/embeddings.jl")

# Test 1: Single node tree
tree_str_1 = "A;"
expected_tree_1 = Node("A", [])
computed_tree_1 = parse_newick(tree_str_1)
#@test computed_tree_1 == expected_tree_1

# Test 2: Tree with two leaf nodes
tree_str_2 = "(A,B);"
expected_tree_2 = Node("", [Node("A", []), Node("B", [])])
#@test parse_newick(tree_str_2) == expected_tree_2

# Test 3: Tree with nested nodes
tree_str_3 = "(A,(B,C));"
expected_tree_3 = Node("", [Node("A", [Node("B", []), Node("C", [])])])
#@test parse_newick(tree_str_3) == expected_tree_3

prec = 20
K = PadicField(3,prec)

p1 = ValuationPolydisc([K(1)], [0])
@show disc_embedding(parse_newick(tree_str_3), p1)