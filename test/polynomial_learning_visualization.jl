include("../src/naml.jl")

p, prec = 2, 20
K = padic_field(p, precision=prec)

R, (x, a, b) =  K["x", "a", "b"]
g = PolydiscFunction([(x-a)*(x-b)])
f = AbstractModel(g, [true, false, false])

# Setting up data points through which we want to fit f
# (high valuation = small disk, low valuation = large disk)
p1 = ValuationPolydisc([K(p^0)], Vector{Int}([prec]))
p2 = ValuationPolydisc([K(p^1)], Vector{Int}([prec]))
p3 = ValuationPolydisc([K(p^2)], Vector{Int}([prec]))
# data = [(p1, 1), (p2, 0), (p3, 0)]
data = [(p2, 0), (p3, 0)]

# Create loss function
ell = MPE_loss_init(f, data, 2)

# Setting up initial parameters
model = Model(f, ValuationPolydisc([K(0),K(0)], [0,0]))

parameterSpaceRoot = model.param
parameterSpaceLayers = [[parameterSpaceRoot]]
parameterSpaceIndex = Dict{typeof(parameterSpaceRoot),Int}(parameterSpaceRoot=>1)
parameterSpaceLoss = Dict{typeof(parameterSpaceRoot),Float64}(parameterSpaceRoot=>ell.eval([model.param])[1])
parameterSpaceEdges = Vector{Int}[];
parameterSpaceEdgesDescending = Vector{Int}[];

for iteration in 1:4
    parameterSpaceBoundary = last(parameterSpaceLayers)
    parameterSpaceBeyond = []
    for boundaryNode in parameterSpaceBoundary
        boundaryNodeIndex = parameterSpaceIndex[boundaryNode]
        boundaryNodeLoss = parameterSpaceLoss[boundaryNode]
        println(boundaryNodeIndex)
        boundaryNodeChildren = children(boundaryNode)
        for beyondNode in boundaryNodeChildren
            beyondNodeLoss = ell.eval([beyondNode])[1]
            if haskey(parameterSpaceIndex,beyondNode)
                beyondNodeIndex = parameterSpaceIndex[beyondNode]
            else
                beyondNodeIndex = length(parameterSpaceIndex) + 1
                parameterSpaceIndex[beyondNode] = beyondNodeIndex
                parameterSpaceLoss[beyondNode] = beyondNodeLoss
            end
            push!(parameterSpaceEdges,[boundaryNodeIndex,beyondNodeIndex])
            if boundaryNodeLoss > beyondNodeLoss
                push!(parameterSpaceEdgesDescending,[boundaryNodeIndex,beyondNodeIndex])
            end
        end
        append!(parameterSpaceBeyond,boundaryNodeChildren)
    end
    push!(parameterSpaceLayers,unique(parameterSpaceBeyond))
end

G = Graph{Undirected}(length(parameterSpaceIndex))
for edge in parameterSpaceEdgesDescending
    add_edge!(G,edge[1],edge[2])
end

parameterSpaceIndexReversed = Dict(value => key for (key, value) in parameterSpaceIndex)

vertexLabels = String[]
for vertex in 1:n_vertices(G)
    push!(vertexLabels,string(vertex)*": "*string(round(parameterSpaceLoss[parameterSpaceIndexReversed[vertex]],digits=3)))
end

visualize(G,VertexLabels=vertexLabels)
