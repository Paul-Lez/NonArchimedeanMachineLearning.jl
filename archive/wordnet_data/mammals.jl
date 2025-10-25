using CSV, DataFrames
using Graphs

4
###
# Loading the dataset of wordnet mammal hypernyms
###
mammalHypernyms = Matrix(CSV.read("wordnet_mammal_hypernyms.tsv", DataFrame, delim='\t', header=false))
mammalStrings = unique([mammalHypernyms...])



###
# The mammals hypernyms form a directed acyclic graph (DAG),
# we want to remove redundant edges to make a tree
###
n = length(mammalStrings)
mammalDAG = DiGraph(n)
for j in 1:size(mammalHypernyms, 1)
    s1, s2 = mammalHypernyms[j, :]
    if s1 == s2
        # exclude loops
        continue
    end
    i1 = findfirst(isequal(s1), mammalStrings)
    i2 = findfirst(isequal(s2), mammalStrings)
    add_edge!(mammalDAG, i1, i2)
end


# Or use a manual approach if the graph is large
paths = floyd_warshall_shortest_paths(mammalDAG)

# Keep edges that are not implied by longer paths
edgesToKeep = []
for edge in edges(mammalDAG)
    i1, i2 = src(edge), dst(edge)
    hasIntermediateVertex = any(paths.dists[i1, j] < typemax(Int)
                                && paths.dists[j, i2] < typemax(Int)
                                && j != i1
                                && j != i2
                                for j in 1:n)
    if !hasIntermediateVertex
        push!(edgesToKeep, (i1, i2))
    end
end

# ###
# # Trying to visualize the graph in OSCAR (bad idea, too large)
# ###
# using Oscar
# mammalTree = Oscar.Graph{Undirected}(Graphs.nv(mammalDAG))
# for (i1,i2) in edgesToKeep
#     Oscar.add_edge!(mammalTree, i1, i2)
# end
# visualize(mammalTree)
