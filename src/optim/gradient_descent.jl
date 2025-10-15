########### Gradient descent optimiser #########

# In this section we implement the tools necessary for gradient descent, and the gradient descent algorithm

# gradient wrt the parameters of the model
function gradient_param(m::AbstractModel{S}, val::ValuationPolydisc{S, T},
    v::ValuationTangent{S, T}) where S where T
    # TODO: this doesn't allow arbitrary shapes for the variable of the model (i.e.
    # this only works if the parameters are the last variables.
    # Do we really need to have something more general?
    new_base = concatenate(val, v.point)
    new_direction = [val.center ; v.direction]
    new_v = ValuationTangent(new_base, new_direction, [zeros(T, dim(val)) ; v.magnitude])
    grad_indices = (dim(val)+1):(dim(val)+dim(v))
    ## CHANGE ME!
    return partial_gradient(m.fun, new_v, grad_indices)
end

# gradient wrt the data variable
function gradient_data(m::Model, data)
    return "implement me"
end

# This is the optimiser function used for the "Gradient Descent" OptimSetup structure.
# The loss is already a closure over the data, so we don't need to pass data separately
function gradient_descent(loss::Loss, param::ValuationPolydisc{S, T}, degree=1) where S where T
    # Compute the children of the point param
    below_nodes = children(param, degree)
    # Get the corresponding tangent vectors
    tangents = [ValuationTangent(param, lower_point.center, zeros(T, dim(param))) for lower_point in below_nodes]
    # In gradient descent, we look at the children of the current parameter point and take the child
    # that maximises the norm of the (downwards pointing) gradient
    val, ind = findmax([LinearAlgebra.norm(loss.grad(v)) for v in tangents])
    return below_nodes[ind]
end

# Helper function to initialise the "Gradient Descent" OptimSetup structure.
function gradient_descent_init(param::ValuationPolydisc{S, T}, loss::Loss, degree=1) where S where T
    return OptimSetup(loss, param, (l, p) -> gradient_descent(l, p, degree))
end
