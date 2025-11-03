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

function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}) where S where T where U
    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return [1 / length(data) * sum([(eval_abs(model, val, param) - out)^2 for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # The gradient is evaluated at each tangent's base point (v.point)
    function MSE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return [1 / length(data) * sum([2 * (eval_abs(model, val, v.point) - out) * gradient_param(model, val, v) for (val, out) in data]) for v in vs]
    end
    return Loss(MSE_compute, MSE_grad)
end

function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}, p::Int) where S where T where U
    # MPE is the "Mean p-power error", i.e. same as the MSE but now we use the ℓᵖ norm instead of the ℓ² one.
    # Here we need finite p. For p = ∞, see the sup loss (TODO Paul: implement the sup loss)

    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return [1 / length(data) * sum([(eval_abs(model, val, param) - out)^p for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # The gradient is evaluated at each tangent's base point (v.point)
    function MPE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return [1 / length(data) * sum([p * (eval_abs(model, val, v.point) - out)^(p - 1) * gradient_param(model, val, v) for (val, out) in data]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end
