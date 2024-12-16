include("../src/naml.jl")

p, prec = 2, 20
K = padic_field(p, precision=prec)

R, (x, a, b) =  K["x", "a", "b"]
g = PolydiscFunction([(x-a)*(x-b)])
f = AbstractModel(g, [true, false, false])
ell = MPE_loss_init(1)

# Setting up data points through which we want to fit f
# (high valuation = small disk, low valuation = large disk)
p1 = ValuationPolydisc([K(p^0)], Vector{Int}([prec]))
p2 = ValuationPolydisc([K(p^1)], Vector{Int}([prec]))
p3 = ValuationPolydisc([K(p^2)], Vector{Int}([prec]))
# data = [(p1, 1), (p2, 0), (p3, 0)]
data = [(p2, 0), (p3, 0)]

# Setting up initial parameters
model = Model(f, ValuationPolydisc([K(0),K(0)], [0,0]))
greedy_optim = greedy_descent_init(data, model, ell, 1)


parameterSpaceRoot = model.param
parameterSpaceLayers = [[parameterSpaceRoot]]
parameterSpaceIndex = Dict{typeof(parameterSpaceRoot),Int}(parameterSpaceRoot=>1)
parameterSpaceLoss = Dict{typeof(parameterSpaceRoot),Float64}(parameterSpaceRoot=>ell.eval(model,data,model.param))
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
            beyondNodeLoss = ell.eval(model,data,beyondNode)
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
