## This file contains various "standard" loss functions for non-Archimedean
## optimisation

############ Loss functions ###################

# We use a simple wrapper for these, the Loss
# structure. To specify this, one needs to
# provide a function to evaluate the loss
# and a function to evaluate the gradient
# of the loss wrt parameters (since we don't
# have any autodiff mechanism implemented yet!)

#################################################

struct Loss
    eval
    grad
end

# Helper functions to construct standard loss functions

function MSE_loss_init()
    # compute the value of the MSE
    function MSE_compute(model::Model{S, T}, data::Vector{Tuple{ValuationPolydisc{S, T}, U}},
        param::ValuationPolydisc{S, T}) where S where T where U
        return 1/length(data) * sum([(eval_abs(model.fun, val, param) - out)^2 for (val, out) in data])
    end
    # compute the gradient of the loss along a tangent direction v
    # (TODO Paul: Do we want to allow v to weight the sum?)
    function MSE_grad(model::Model{S, T}, data::Vector{Tuple{ValuationPolydisc{S, T}, U}}, v::ValuationTangent{S, T}) where S where T where U
        return 1/length(data) * sum([2*(eval_abs(model.fun, val, model.param) - out) * gradient_param(model.fun, val, v) for (val, out) in data])
    end
    return Loss(MSE_compute, MSE_grad)
end

function MPE_loss_init(p::Int)
    # MPE is the "Mean p-power error", i.e. same as the MSE but now we us the ℓᵖ norm instead of the ℓ² one.
    # Here we need finite p. For p = ∞, see the sup loss (TODO Paul: implement the sup loss)

    # compute the value of the MPE
    function MPE_compute(model::Model{S, T}, data::Vector{Tuple{ValuationPolydisc{S, T}, U}},
        param::ValuationPolydisc{S, T}) where S where T where U
        return 1/length(data) * sum([(eval_abs(model.fun, val, param) - out)^p for (val, out) in data])
    end
    # compute the gradient of the loss along a tangent direction v
    # (TODO Paul: Do we want to allow v to weight the sum?)
    function MPE_grad(model::Model{S, T}, data::Vector{Tuple{ValuationPolydisc{S, T}, U}}, v::ValuationTangent{S, T}) where S where T where U
        return 1/length(data) * sum([p*(eval_abs(model.fun, val, model.param) - out)^(p-1) * gradient_param(model.fun, val, v) for (val, out) in data])
    end
    return Loss(MPE_compute, MPE_grad)
end
