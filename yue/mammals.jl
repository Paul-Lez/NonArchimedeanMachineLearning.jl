using CSV, DataFrames
using Graphs


###
# Loading the dataset of wordnet mammal hypernyms
###
# read the tsv file and convert it to a Nx2 matrix (one row per hypernym)
mammalHypernyms = Matrix(CSV.read("wordnet_mammal_hypernyms.tsv", DataFrame, delim='\t', header=false))



###
# The mammals hypernyms form a directed acyclic graph (DAG),
# we want to remove redundant edges to make a tree
###

# remove all loops in the DAG
nonLoopIndices = findall(i->(mammalHypernyms[i,1]!=mammalHypernyms[i,2]), 1:size(mammalHypernyms,1))
mammalHypernyms = mammalHypernyms[nonLoopIndices,:]

# create a list of vertices of the DAG
mammalStrings = unique([mammalHypernyms...])


# filter the vertices of the DAG:
# - first are vertices with only outgoing edges
# - second are vertices the first vertices are connected to
# - third are the vertices the second vertices are connected to
# - etc.
mammalStringsCopy = copy(mammalStrings)
mammalHypernymsCopy = copy(mammalHypernyms)
mammalStringsFiltered = Vector{String}[]

# iterate over all directed edges as long as some remain
while !isempty(mammalHypernymsCopy)
    # identify the (current) vertices with only outgoing edges
    mammalEnds = [ mammalString for mammalString in mammalStringsCopy if !(mammalString in mammalHypernymsCopy[:,2])]
    push!(mammalStringsFiltered, mammalEnds)

    # remove them from the list of vertices
    mammalStringsCopy = mammalStringsCopy[findall(mammal->!(mammal in mammalEnds),mammalStringsCopy)]
    # and their edges from the list of edges
    mammalHypernymsCopy = mammalHypernymsCopy[findall(mammal->!(mammal in mammalEnds),mammalHypernymsCopy[:,1]),:]
end
@assert length(mammalStringsCopy) == 1 # there should be only one vertex left and it should be "mammals"
push!(mammalStringsFiltered, mammalStringsCopy)


# convert filtered list of vertices into a height vector
# - first vertices are height 1
# - second vertices are height 2
# - etc.
D = length(mammalStringsFiltered)
mammalStringsHeights = [ findfirst(i->(mammalString in mammalStringsFiltered[i]),1:D) for mammalString in mammalStrings ]


# assign each hyponym a length based on the height difference of its vertices
mammalHypernymsLengths = Int[]
for mammalHypernym in eachrow(mammalHypernyms)
    i = findfirst(isequal(mammalHypernym[1]), mammalStrings)
    j = findfirst(isequal(mammalHypernym[2]), mammalStrings)
    mammalHypernymHeight = abs(mammalStringsHeights[i]-mammalStringsHeights[j])
    push!(mammalHypernymsLengths, mammalHypernymHeight)
end
@assert min(mammalHypernymsLengths...) == 1  # the minimal length should be 1 (no edges between vertices of the same height)
@assert max(mammalHypernymsLengths...) == D-1 # the maximal length should be D-1 (all vertices are connected to the root)


# filter the edges based on their length
mammalHypernymsFiltered = [ [ mammalHypernyms[j,:] for j in findall(isequal(i), mammalHypernymsLengths)] for i in 1:D-1 ]


# construct mammal tree using only hyponyms that are not composition of shorter hyponyms
G = Graph(length(mammalStrings))
for (k, mammalHypernymsBatch) in enumerate(mammalHypernymsFiltered)
    for (l,mammalHypernym) in enumerate(mammalHypernymsBatch)
        println("Processing batch $k/$D, edge $l/$(length(mammalHypernymsBatch))")
        i = findfirst(isequal(mammalHypernym[1]), mammalStrings)
        j = findfirst(isequal(mammalHypernym[2]), mammalStrings)
        path = Graphs.a_star(G, i, j)
        if isempty(path) # empty means there is no path between i and j
            add_edge!(G, i, j)
        end
    end
end
@assert is_tree(G) # final test that G is indeed a tree


####
# # Trying to visualize the graph in OSCAR (bad idea, too large)
# ###
# using Oscar
# Goscar = Oscar.Graph{Undirected}(nv(G))
# for edge in edges(G)
#     Oscar.add_edge!(Goscar, src(edge), dst(edge))
# end
# visualize(Goscar)
