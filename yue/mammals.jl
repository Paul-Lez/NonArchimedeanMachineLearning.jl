using CSV, DataFrames
using Graphs
using Oscar

# Load wordnet_mammal_hypernyms.tsv
mammalHypernyms = Matrix(CSV.read("wordnet_mammal_hypernyms.tsv", DataFrame, delim='\t', header=false))

# remove loops in mammal graph
mammalHypernyms = [ collect(mammalHypernym) for mammalHypernym in eachrow(mammalHypernyms) if mammalHypernym[1] != mammalHypernym[2] ]
mammalHypernyms = permutedims(reduce(hcat,mammalHypernyms))

# create a vector of unique strings in dataset
mammalStrings = sort(unique([mammalHypernyms...]))


# construct an Oscar graph from the CSV data
mammalGraph = Oscar.Graph{Directed}(length(mammalStrings))
for mammalHypernym in eachrow(mammalHypernyms)
    i = findfirst(isequal(mammalHypernym[1]), mammalStrings)
    j = findfirst(isequal(mammalHypernym[2]), mammalStrings)
    if i!=j
        Oscar.add_edge!(mammalGraph, i, j)
    end
end

# todo: construct tree from it!

# foo1 = Polymake.graph.random_spanningtree(mammalGraph.pm_graph)
# foo2 = Polymake.graph.random_spanningtree(mammalGraph.pm_graph)

# G = Oscar.Graph{Directed}(3)
# Oscar.add_edge!(G, 1, 2)
# Oscar.add_edge!(G, 2, 3)
# Oscar.add_edge!(G, 1, 3)
# Polymake.graph.all_spanningtrees(G.pm_graph)

# G = Oscar.Graph{Undirected}(3)
# Oscar.add_edge!(G, 1, 2)
# Oscar.add_edge!(G, 2, 3)
# Oscar.add_edge!(G, 1, 3)
# Polymake.graph.all_spanningtrees(G.pm_graph)


# # construct a graph from the CSV data
# mammalGraph = Graphs.SimpleGraph(length(mammalStrings))
# for mammalHypernym in eachrow(mammalHypernyms)
#     i = findfirst(isequal(mammalHypernym[1]), mammalStrings)
#     j = findfirst(isequal(mammalHypernym[2]), mammalStrings)
#     @assert !isnothing(i) && !isnothing(j)
#     if i!=j
#         Graphs.add_edge!(mammalGraph, i, j)
#     end
# end

# # find all cycles in the graph
# mammalCycles = Graphs.cycle_basis(mammalGraph)

# mammal = mammalStrings[54]
# for mammalHypernym in eachrow(mammalHypernyms)
#     if mammalHypernym[1] == mammal || mammalHypernym[2] == mammal
#         println(mammalHypernym)
#     end
# end

# mammalCycle = mammalCycles[1]
# for mammalHypernym in eachrow(mammalHypernyms)
#     if length(findall(mammal->(mammal in mammalHypernym), mammalStrings[mammalCycle])) > 1
#         println(mammalHypernym)
#     end
# end
