########### Greedy descent algorithm ###########

# In this section we implement greedy descent algorithms

# The optimiser function for greedy descent
function greedy_descent(
    loss::Loss,
    param::ValuationPolydisc{S,T},
    next_branch::Int,
    settings::Tuple{Bool,Int}
) where S where T
    (strict, degree) = settings
    if strict
        below_nodes = children_along_branch(param, next_branch)
        next_branch = next_branch == dim(param) ? 1 : next_branch + 1
        # println("$(next_branch)")
    else
        below_nodes = children(param, degree)
    end
    # In greedy descent, we look at the children of the
    # current parameter point and take the child
    # that minimises the loss
    loss_values = loss.eval(below_nodes)
    return (below_nodes[argmin(loss_values)], next_branch)
end

# function strict_greedy_descent(loss::Loss, param::ValuationPolydisc{S,T}, degree=1) where S where T


# end

# This function outputs the OptimSetup object for greedy descent
function greedy_descent_init(
    param::ValuationPolydisc{S,T},
    loss::Loss,
    next_branch::Int,
    settings::Tuple{Bool,Int}
) where S where T
    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> greedy_descent(l, p, st, ctx),
        next_branch,
        settings
    )
end

# TODO: Implement the 1D tree search algorithm
