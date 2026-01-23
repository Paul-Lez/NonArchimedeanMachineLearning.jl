########### Gradient descent optimiser #########

# In this section we implement the tools necessary for gradient descent, and the gradient descent algorithm

@doc raw"""
    gradient_param(m::AbstractModel{S}, val::ValuationPolydisc{S,T}, v::ValuationTangent{S,T}) where {S,T}

Compute the gradient of a model with respect to its parameters.

# Arguments
- `m::AbstractModel{S}`: The abstract model
- `val::ValuationPolydisc{S,T}`: Data variable values
- `v::ValuationTangent{S,T}`: Tangent vector in parameter space

# Returns
Gradient vector with respect to parameters

# Notes
Currently assumes parameters are the last variables. More general shapes may be needed.
"""
function gradient_param(
    m::AbstractModel{S},
    val::ValuationPolydisc{S,T},
    v::ValuationTangent{S,T}
) where {S, T}
    # TODO: this doesn't allow arbitrary shapes for the variable of the model (i.e.
    # this only works if the parameters are the last variables.
    # Do we really need to have something more general?    
    new_base = concatenate(val, v.point)
    new_direction = [val.center; v.direction]
    new_v = ValuationTangent(new_base, new_direction, [zeros(T, dim(val)); v.magnitude])
    grad_indices = (dim(val)+1):(dim(val)+dim(v))
    ## CHANGE ME!
    return partial_gradient(m.fun, new_v, grad_indices)
end

# TODO: Function below is generally less useful, but would be nice to have
# for completness.

@doc raw"""
    gradient_data(m::Model, data)

Compute the gradient of a model with respect to data variables.

# Arguments
- `m::Model`: The model
- `data`: Data values

# Returns
Gradient vector with respect to data

# Notes
Currently unimplemented - returns placeholder string.
"""
function gradient_data(m::Model, data)
    return "implement me"
end

@doc raw"""
    gradient_descent(loss::Loss, param::ValuationPolydisc{S,T}, state::U, degree::Int) where {S,T,U}

Perform one step of gradient descent optimization.

Computes children of the current parameter point and selects the child that maximizes
the gradient norm (steepest descent direction).

# Arguments
- `loss::Loss`: The loss function structure
- `param::ValuationPolydisc{S,T}`: Current parameter values
- `state::U`: Current optimizer state (unused in gradient descent)
- `degree::Int`: Degree for generating child polydiscs

# Returns
`Tuple{ValuationPolydisc{S,T}, U}`: New parameters and state (state unchanged)
"""
function gradient_descent(
    loss::Loss,
    param::ValuationPolydisc{S,T},
    state::U,
    degree::Int
) where {S, T, U}
    # Compute the children of the point param
    below_nodes = children(param, degree)
    # Get the corresponding tangent vectors
    tangents = [ValuationTangent(param, lower_point.center, zeros(T, dim(param))) for lower_point in below_nodes]
    # In gradient descent, we look at the children of the current parameter point and take the child
    # that maximises the norm of the (downwards pointing) gradient
    grad_values = loss.grad(tangents)
    val, ind = findmax([LinearAlgebra.norm(g) for g in grad_values])
    return below_nodes[ind], state
end

@doc raw"""
    gradient_descent_init(param::ValuationPolydisc{S,T}, loss::Loss, state::U, degree=1) where {S,T,U}

Initialize an optimization setup for gradient descent.

# Arguments
- `param::ValuationPolydisc{S,T}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `state::U`: Initial state (unused in gradient descent, but included for API consistency)
- `degree::Int=1`: Degree parameter for child generation (default: 1)

# Returns
`OptimSetup`: Configured optimization setup for gradient descent

# Notes
The state parameter does nothing in gradient descent but is included for consistency.
"""
function gradient_descent_init(
    param::ValuationPolydisc{S,T},
    loss::Loss,
    state::U,
    degree=1
) where {S, T, U}
    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> gradient_descent(l, p, st, ctx),
        state,
        degree
    )
end
