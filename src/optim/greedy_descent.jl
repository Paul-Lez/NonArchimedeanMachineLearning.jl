include("basic.jl")

########### Greedy descent algorithm ###########

# In this section we implement greedy descent algorithms

# The optimiser function for greedy descent
function greedy_descent(data::Vector{Tuple{ValuationPolydisc{S, T}, U}}, model::Model{S, T}, loss, degree=1) where S where T where U
    below_nodes = children(model.param, degree)
    # In greedy descent, we look at the children of the 
    # current parameter point and take the child 
    # that minimises the loss
end   
    return argmin(param -> loss(model, data, param), below_nodes)

# This function outputs the OptimSetup object for greedy descent
function greedy_descent_init(data::Vector{Tuple{ValuationPolydisc{S, T}, U}}, model::Model{S, T}, loss, degree=1) where S where T where U
    return OptimSetup(data, model, loss, (dat, mod) -> greedy_descent(dat, mod, loss.eval, degree))
end

# TODO: Implement the 1D tree search algorithm