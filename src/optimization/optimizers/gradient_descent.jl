"""
Gradient-based descent routines for valuation-polydisc parameters.
"""

@doc raw"""
    gradient_param(m::AbstractModel{S}, val::ValuationPolydisc{S,T,N1}, v::ValuationTangent{S,T,N2}) where {S,T,N1,N2}

Compute the gradient of a model with respect to its parameters using the symbolic path.

# Arguments
- `m::AbstractModel{S}`: The abstract model
- `val::ValuationPolydisc{S,T,N1}`: Data variable values
- `v::ValuationTangent{S,T,N2}`: Tangent vector in parameter space

# Returns
Gradient vector with respect to parameters

# Notes
Currently assumes parameters are the last variables. More general shapes may be needed.
"""
function gradient_param(
        m::AbstractModel{S},
        val::ValuationPolydisc{S, T, N1},
        v::ValuationTangent{S, T, N2}
) where {S, T, N1, N2}
    new_base = concatenate(val, v.point)
    new_direction = concatenate(val, v.direction)
    new_v = ValuationTangent(new_base, new_direction, [zeros(T, dim(val)); v.magnitude])
    grad_indices = (dim(val) + 1):(dim(val) + dim(v))
    return partial_gradient(m.fun, new_v, grad_indices)
end

# VFP lifting: unwrap ValuedFieldPoint types and delegate
function gradient_param(
        m::AbstractModel{S},
        val::ValuationPolydisc{ValuedFieldPoint{P, Prec, S}, T, N1},
        v::ValuationTangent{ValuedFieldPoint{P, Prec, S}, T, N2}
) where {S, P, Prec, T, N1, N2}
    unwrapped_val = ValuationPolydisc{S, T, N1}(unwrap(val.center), val.radius)
    unwrapped_point = ValuationPolydisc{S, T, N2}(unwrap(v.point.center), v.point.radius)
    unwrapped_direction = ValuationPolydisc{S, T, N2}(unwrap(v.direction.center), v.direction.radius)
    unwrapped_v = ValuationTangent{S, T, N2}(unwrapped_point, unwrapped_direction, v.magnitude)
    return gradient_param(m, unwrapped_val, unwrapped_v)
end

@doc raw"""
    gradient_param(m::AbstractModel, fun_eval::PolydiscFunctionEvaluator, val::ValuationPolydisc, v::ValuationTangent)

Compute the gradient of a model with respect to its parameters using a typed evaluator.

# Arguments
- `m::AbstractModel`: The abstract model (used for dimension info)
- `fun_eval::PolydiscFunctionEvaluator`: Typed evaluator for the model function
- `val::ValuationPolydisc`: Data variable values
- `v::ValuationTangent`: Tangent vector in parameter space

# Returns
Gradient vector with respect to parameters
"""
function gradient_param(
        m::AbstractModel,
        fun_eval::PolydiscFunctionEvaluator,
        val::ValuationPolydisc,
        v::ValuationTangent
)
    new_base = concatenate(val, v.point)
    new_direction = concatenate(val, v.direction)
    T = eltype(v.magnitude)
    new_v = ValuationTangent(new_base, new_direction, [zeros(T, dim(val)); v.magnitude])
    grad_indices = (dim(val) + 1):(dim(val) + dim(v))
    return partial_gradient(fun_eval, new_v, grad_indices)
end

@doc raw"""
    GradientDescentConfig(; strict=false, degree=1, start_branch=1)

Configuration for gradient descent optimization.

# Fields
- `strict::Bool`: If true, descend one coordinate at a time; if false, descend all coordinates
- `degree::Int`: Number of children to explore per polydisc node (non-strict mode)
- `start_branch::Int`: Coordinate to descend along first (strict mode)

# Example
```julia
config = GradientDescentConfig(strict = false, degree = 1)
optim = gradient_descent_init(param, loss, config)
```
"""
struct GradientDescentConfig <: AbstractOptimConfig
    strict::Bool
    degree::Int
    start_branch::Int

    function GradientDescentConfig(;
            strict::Bool = false,
            degree::Int = 1,
            start_branch::Int = 1
    )
        @req degree >= 1 "degree must be positive"
        @req start_branch >= 1 "start_branch must be positive"
        new(strict, degree, start_branch)
    end
end

@doc raw"""
    gradient_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, next_branch::Int, settings::GradientDescentConfig) where {S,T,N}

Perform one step of gradient descent optimization.

Computes children of the current parameter point and selects the child that maximizes
the gradient norm (steepest descent direction).

# Arguments
- `loss::Loss`: The loss function structure
- `param::ValuationPolydisc{S,T,N}`: Current parameter values
- `next_branch::Int`: Index of the next branch to descend in strict mode
- `settings::GradientDescentConfig`: Configuration for gradient descent

# Returns
`Tuple{ValuationPolydisc{S,T,N}, Int, Bool}`: New parameters, next branch index,
and convergence status
"""
function gradient_descent(
        loss::Loss,
        param::ValuationPolydisc{S, T, N},
        next_branch::Int,
        settings::GradientDescentConfig
) where {S, T, N}
    # Compute the children of the point param
    below_nodes, next_branch = _descent_candidates(param, next_branch, settings)
    isempty(below_nodes) && return (param, next_branch, true)
    # Get the corresponding tangent vectors.
    # Evaluate gradient at each child (not at param): children have positive radius in one
    # coordinate, which makes the p-adic directional derivative non-trivial. Evaluating at
    # param (radius 0 everywhere) would give gradient 0 for all children.
    tangents = [ValuationTangent(param, lower_point, zeros(T, dim(lower_point)))
                for lower_point in below_nodes]
    # In gradient descent, we look at the children of the current parameter point and take the child
    # that maximises the norm of the (downwards pointing) gradient
    grad_values = loss.grad(tangents)
    ind = rand(findall(u -> u == minimum(grad_values), grad_values))
    return below_nodes[ind], next_branch, false
end

@doc raw"""
    gradient_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, settings::GradientDescentConfig=GradientDescentConfig()) where {S,T,N}

Initialize an optimization setup for gradient descent.

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `settings::GradientDescentConfig`: Configuration controlling descent behavior

# Returns
`OptimSetup`: Configured optimization setup for gradient descent

# Notes
`settings.start_branch` is used only when `strict` mode is enabled.
"""
function gradient_descent_init(
        param::ValuationPolydisc{S, T, N},
        loss::Loss,
        settings::GradientDescentConfig = GradientDescentConfig()
) where {S, T, N}
    @req settings.start_branch <= N "start_branch must be at most the dimension of the polydisc"
    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> gradient_descent(l, p, st, ctx),
        settings.start_branch,
        settings,
        false
    )
end
