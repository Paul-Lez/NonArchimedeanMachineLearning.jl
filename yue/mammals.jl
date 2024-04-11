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

###
# Removing redundant edges to make a tree
###


# assign each unique string a depth value
mammalStringsCopy = copy(mammalStrings)
mammalHypernymsCopy = copy(mammalHypernyms)
mammalStringsFiltered = []

# iterate over all directed edges as long as some remain
while !isempty(mammalHypernymsCopy)
    # identify the current ends of the directed graph
    mammalEnds = [ mammalString for mammalString in mammalStringsCopy if !(mammalString in mammalHypernymsCopy[:,2])]
    push!(mammalStringsFiltered, mammalEnds)

    # remove the current ends from the list of strings and edges
    mammalStringsCopy = mammalStringsCopy[findall(mammal->!(mammal in mammalEnds),mammalStringsCopy)]
    mammalHypernymsCopy = mammalHypernymsCopy[findall(mammal->!(mammal in mammalEnds),mammalHypernymsCopy[:,1]),:]
end
push!(mammalStringsFiltered, ["mammal.n.01"]) # add the root node




# # construct an Oscar graph from the CSV data for visualization # bad idea!
# mammalGraph = Oscar.Graph{Directed}(length(mammalStrings))
# for mammalHypernym in eachrow(mammalHypernyms)
#     i = findfirst(isequal(mammalHypernym[1]), mammalStrings)
#     j = findfirst(isequal(mammalHypernym[2]), mammalStrings)
#     if i!=j
#         Oscar.add_edge!(mammalGraph, i, j)
#     end
# end

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
