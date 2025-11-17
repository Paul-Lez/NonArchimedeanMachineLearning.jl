## This file contains various "standard" loss functions for non-Archimedean
## optimisation

############ Loss functions ###################

# Note: The Loss struct is defined in basic.jl
# To specify a loss, one needs to provide a function to evaluate the loss
# and a function to evaluate the gradient of the loss wrt parameters
# (since we don't have any autodiff mechanism implemented yet!)
# Both functions should be closures that capture any necessary data.

#################################################

# Helper functions to construct standard loss functions

@doc raw"""
    MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}) where {S,T,U}

Initialize a Mean Squared Error (MSE) loss function.

Creates a `Loss` structure with evaluation and gradient functions for MSE loss:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n (f(x_i; \theta) - y_i)^2``

# Arguments
- `model::AbstractModel{S}`: The model to optimize
- `data::Vector{Tuple{ValuationPolydisc{S,T},U}}`: Training data as `(input, output)` pairs

# Returns
`Loss`: Loss structure with MSE evaluation and gradient functions

# Implementation Notes
The loss and gradient functions are closures that capture the training data and model.
Both functions support batch evaluation for multiple parameter values simultaneously.
"""
function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}) where S where T where U
    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return [1 / length(data) * sum([(evaluate(model, val, param) - out)^2 for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # The gradient is evaluated at each tangent's base point (v.point)
    function MSE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return [1 / length(data) * sum([2 * (evaluate(model, val, v.point) - out) * gradient_param(model, val, v) for (val, out) in data]) for v in vs]
    end
    return Loss(MSE_compute, MSE_grad)
end

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}, p::Int) where {S,T,U}

Initialize a Mean p-Power Error (MPE) loss function.

Creates a `Loss` structure using the ``\ell^p`` norm instead of ``\ell^2``:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(x_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model to optimize
- `data::Vector{Tuple{ValuationPolydisc{S,T},U}}`: Training data as `(input, output)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with MPE evaluation and gradient functions

# Implementation Notes
The loss and gradient functions are closures that capture the training data and model.
Both functions support batch evaluation for multiple parameter values simultaneously.
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}, p::Int) where S where T where U
    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return [1 / length(data) * sum([(evaluate(model, val, param) - out)^p for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # The gradient is evaluated at each tangent's base point (v.point)
    function MPE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return [1 / length(data) * sum([p * (evaluate(model, val, v.point) - out)^(p - 1) * gradient_param(model, val, v) for (val, out) in data]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end
