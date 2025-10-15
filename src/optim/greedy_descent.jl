########### Greedy descent algorithm ###########

# In this section we implement greedy descent algorithms

# The optimiser function for greedy descent
# The loss is already a closure over the data, so we don't need to pass data or model separately
function greedy_descent(loss::Loss, param::ValuationPolydisc{S, T}, degree=1) where S where T
    below_nodes = children(param, degree)
    # In greedy descent, we look at the children of the
    # current parameter point and take the child
    # that minimises the loss
    return argmin(p -> loss.eval(p), below_nodes)
end

# This function outputs the OptimSetup object for greedy descent
function greedy_descent_init(param::ValuationPolydisc{S, T}, loss::Loss, degree=1) where S where T
    return OptimSetup(loss, param, (l, p) -> greedy_descent(l, p, degree))
end

# TODO: Implement the 1D tree search algorithm
