using CSV, DataFrames
using Graphs
# using Oscar

# Load wordnet_mammal_hypernyms.tsv
mammalHypernyms = Matrix(CSV.read("wordnet_mammal_hypernyms.tsv", DataFrame, delim='\t', header=false))

# remove loops in mammal graph
mammalHypernyms = [ collect(mammalHypernym) for mammalHypernym in eachrow(mammalHypernyms) if mammalHypernym[1] != mammalHypernym[2] ]
mammalHypernyms = permutedims(reduce(hcat,mammalHypernyms))

# create a vector of unique strings in dataset
mammalStrings = unique([mammalHypernyms...])

###
# Removing redundant edges to make a tree
###

# assign each unique string a depth value
mammalStringsCopy = copy(mammalStrings)
mammalHypernymsCopy = copy(mammalHypernyms)
mammalStringsFiltered = Vector{String}[]

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

# convert filtered list into a height vector
D = length(mammalStringsFiltered)
mammalStringsHeights = Int[ findfirst(i->(mammalString in mammalStringsFiltered[i]),1:D) for mammalString in mammalStrings ]


# assign each hyponym a length value
mammalHypernymsLengths = Int[]
for mammalHypernym in eachrow(mammalHypernyms)
    i = findfirst(isequal(mammalHypernym[1]), mammalStrings)
    j = findfirst(isequal(mammalHypernym[2]), mammalStrings)
    mammalHypernymHeight = abs(mammalStringsHeights[i]-mammalStringsHeights[j])
    @assert !iszero(mammalHypernymHeight)
    push!(mammalHypernymsLengths, mammalHypernymHeight)
end
@assert max(mammalHypernymsLengths...) == D-1

# group hyponyms based on their length
mammalHypernymsFiltered = [ [ mammalHypernyms[j,:] for j in findall(isequal(i), mammalHypernymsLengths)] for i in 1:D-1 ]


# construct mammal tree using only hyponyms that are composition of shorter hyponyms
G = Graph(length(mammalStrings))
for (k, mammalHypernymsBatch) in enumerate(mammalHypernymsFiltered)
    for (l,mammalHypernym) in enumerate(mammalHypernymsBatch)
        println("Processing batch $k/$D, edge $l/$(length(mammalHypernymsBatch))")
        i = findfirst(isequal(mammalHypernym[1]), mammalStrings)
        j = findfirst(isequal(mammalHypernym[2]), mammalStrings)
        path = Graphs.a_star(G, i, j)
        if isempty(path)
            add_edge!(G, i, j)
        end
    end
end
