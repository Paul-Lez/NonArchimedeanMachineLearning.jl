## This file contains various "standard" loss functions for non-Archimedean optimisation.
##
## Note: The Loss struct is defined in basic.jl
## To specify a loss, one needs to provide a function to evaluate the loss
## and a function to evaluate the gradient of the loss wrt parameters
## (since we don't have any autodiff mechanism implemented yet!)
## Both functions should be closures that capture any necessary data.

#################################################
# Mean Squared Error (MSE) Loss Functions
#################################################

@doc raw"""
    MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T,N},U}}) where {S,T,N,U}

Initialize a Mean Squared Error (MSE) loss function with polydisc-valued inputs.

Creates a `Loss` structure with batch evaluation and gradient functions for MSE loss:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n (f(x_i; \theta) - y_i)^2``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{ValuationPolydisc{S,T,N},U}}`: Training data as `(input_polydisc, output_value)` pairs

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Notes
Both the evaluation and gradient functions operate on batches of parameters and tangent vectors,
allowing efficient parallel computation. Uses typed evaluators for improved performance.
"""
function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T,N},U}}) where {S, T, N, U}
    # Initialize TYPED batch evaluation for the model
    # Determine full dimension from model structure
    full_dim = length(model.param_info)
    model_eval = batch_evaluate_init(model, ValuationPolydisc{S,T,full_dim})

    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{<:ValuationPolydisc{S,T,N}})
        return [1 / length(data) * sum([(model_eval(val, param) - out)^2 for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # Uses the typed evaluator for gradient computation via gradient_param
    function MSE_grad(vs::Vector{<:ValuationTangent{S,T,N}})
        return [1 / length(data) * sum([2 * (model_eval(val, v.point) - out) * gradient_param(model, model_eval.fun_eval, val, v) for (val, out) in data]) for v in vs]
    end
    return Loss(MSE_compute, MSE_grad)
end

@doc raw"""
    MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{S,U}}) where {S,U}

Initialize a Mean Squared Error (MSE) loss function with field-valued inputs.

Creates a `Loss` structure for data where inputs are elements of the base field (not polydiscs).
This variant uses model specialization and batch evaluation for computational efficiency.

Computes: ``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n (f(x_i; \theta) - y_i)^2``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{S,U}}`: Training data as `(field_element_input, output_value)` pairs

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Implementation Notes
This overload specializes the model at each data point by substituting the field-valued inputs,
then uses typed evaluators for both evaluation and gradient computation.
"""
function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{S,U}}) where {S, U}
    # Specialize the model at each data point
    specialized_models = [specialise(model, [val]) for (val, _) in data]

    # Count parameters (false entries in param_info)
    param_dim = count(!model.param_info)

    # Initialize TYPED batch evaluation for each specialized model
    # Specialized models are just PolydiscFunctions with only parameters
    batch_evals = [batch_evaluate_init(specialized_models[i], ValuationPolydisc{S,Int,param_dim}) for i in eachindex(specialized_models)]

    # Convert outputs to Float64 (p-adic absolute values)
    out_values = [abs(out) for (_, out) in data]

    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{ValuationPolydisc{S,T,N}}) where {S, T, N}
        return 1 / length(data) * [sum([(batch_evals[i](param) - out_values[i])^2 for i in eachindex(data)]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # Uses typed evaluators for gradient computation
    function MSE_grad(vs::Vector{ValuationTangent{S,T,N}}) where {S, T, N}
        return 1 / length(data) * [sum([2 * (batch_evals[i](v.point) - out_values[i]) * directional_derivative(batch_evals[i], v) for i in eachindex(data)]) for v in vs]
    end
    return Loss(MSE_compute, MSE_grad)
end

@doc raw"""
    MSE_loss_init_new(model::AbstractModel{S}, data::Vector{Tuple{S,U}}) where {S,U}

Experimental MSE loss using a single composed function (slower, for profiling comparison).
"""
function MSE_loss_init_new(model::AbstractModel{S}, data::Vector{Tuple{S,U}}) where {S, U}
    # Specialize the model at each data point
    specialized_model = 1 / length(data) * sum([
        (specialise(model, [val]) - abs(out))^2 for (val, out) in data
    ])

    # Initialize batch evaluation for each specialized model
    batch_eval = batch_evaluate_init(specialized_model)

    # TODO: this is currently quite slow compared to the previous
    # implementation. Do some profiling!

    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{ValuationPolydisc{S,T,N}}) where {S, T, N}
        return map(batch_eval, params)
    end

    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    function MSE_grad(vs::Vector{ValuationTangent{S,T,N}}) where {S, T, N}
        return [directional_derivative(batch_eval, v) for v in vs]
    end
    return Loss(MSE_compute, MSE_grad)
end


#################################################
# Mean p-Power Error (MPE) Loss Functions
#################################################

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T,N},U}}, p::Int) where {S,T,N,U}

Initialize a Mean p-Power Error (MPE) loss function with polydisc-valued inputs.

Generalizes MSE by using the ``\ell^p`` norm instead of ``\ell^2``. Computes:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(x_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{ValuationPolydisc{S,T,N},U}}`: Training data as `(input_polydisc, output_value)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Notes
For MSE (Euclidean loss), use ``p = 2``. The gradient computation uses the power rule.
Uses typed evaluators for improved performance.
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T,N},U}}, p::Int) where {S, T, N, U}
    # Initialize TYPED batch evaluation for the model
    # Determine full dimension from model structure
    full_dim = length(model.param_info)
    model_eval = batch_evaluate_init(model, ValuationPolydisc{S,T,full_dim})

    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{<:ValuationPolydisc{S,T,N}})
        return [1 / length(data) * sum([(model_eval(val, param) - out)^p for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # Uses the typed evaluator for gradient computation via gradient_param
    function MPE_grad(vs::Vector{<:ValuationTangent{S,T,N}})
        return [1 / length(data) * sum([p * (model_eval(val, v.point) - out)^(p - 1) * gradient_param(model, model_eval.fun_eval, val, v) for (val, out) in data]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end

# TODO(Paul-Lez): vectorise computations!

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{S,U}}, p::Int) where {S,U}

Initialize a Mean p-Power Error (MPE) loss function with field-valued inputs.

Generalizes MSE using the ``\ell^p`` norm with field-valued (not polydisc-valued) inputs.
Uses model specialization and typed evaluators for computational efficiency.

Computes: ``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(x_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{S,U}}`: Training data as `(field_element_input, output_value)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Implementation Notes
This overload specializes the model at each data point, making it more efficient for
field-valued inputs than the polydisc variant. Uses typed evaluators for improved performance.
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{S,U}}, p::Int) where {S, U}
    # Specialize the model at each data point
    specialized_models = [specialise(model, [val]) for (val, _) in data]

    # Count parameters (false entries in param_info)
    param_dim = count(!model.param_info)

    # Initialize TYPED batch evaluation for each specialized model
    batch_evals = [batch_evaluate_init(specialized_models[i], ValuationPolydisc{S,Int,param_dim}) for i in eachindex(specialized_models)]

    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{ValuationPolydisc{S,T,N}}) where {S, T, N}
        return 1 / length(data) * [sum([(batch_evals[i](param) - out)^p for (i, (_, out)) in enumerate(data)]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # Uses typed evaluators for gradient computation
    function MPE_grad(vs::Vector{ValuationTangent{S,T,N}}) where {S, T, N}
        return 1 / length(data) * [sum([p * (batch_evals[i](v.point) - out)^(p - 1) * directional_derivative(batch_evals[i], v) for (i, (_, out)) in enumerate(data)]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{Vector{S},U}}, p::Int) where {S, U}

Initialize a Mean p-Power Error (MPE) loss function with vector-valued inputs.

Generalizes MPE for data where inputs are vectors of field elements (representing
multivariate data points). Uses model specialization and typed evaluators for efficiency.

Computes: ``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(\mathbf{x}_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{Vector{S},U}}`: Training data as `(vector_of_field_elements, output_value)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Implementation Notes
This overload handles multivariate data where each input is a vector of field elements.
The model is specialized at each data point vector and uses typed evaluators for improved performance.
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{Vector{S},U}}, p::Int) where {S, U}
    # For vector-valued data, specialize the model at each data point vector
    specialized_models = [specialise(model, val) for (val, _) in data]

    # Count parameters (false entries in param_info)
    param_dim = count(!model.param_info)

    # Initialize TYPED batch evaluation for each specialized model
    batch_evals = [batch_evaluate_init(specialized_models[i], ValuationPolydisc{S,Int,param_dim}) for i in eachindex(specialized_models)]

    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{ValuationPolydisc{S,T,N}}) where {S, T, N}
        return [1 / length(data) * sum([(batch_evals[i](param) - out)^p for (i, (_, out)) in enumerate(data)]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # Uses typed evaluators for gradient computation
    function MPE_grad(vs::Vector{ValuationTangent{S,T,N}}) where {S, T, N}
        return [1 / length(data) * sum([p * (batch_evals[i](v.point) - out)^(p - 1) * directional_derivative(batch_evals[i], v) for (i, (_, out)) in enumerate(data)]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end

#################################################
# Lifting Dispatch: model{S} + ValuedFieldPoint{P,Prec,S} data
#################################################
#
# When users define models with type S but data is auto-wrapped to ValuedFieldPoint{P,Prec,S},
# these methods eagerly lift the evaluator to ValuedFieldPoint at creation time.
# No runtime adapters needed — conversion happens once, not on every evaluation.

function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N},U}}, power::Int) where {S,P,Prec,T,N,U}
    full_dim = length(model.param_info)
    model_eval = batch_evaluate_init(model, ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,full_dim})

    function MPE_compute(params::Vector{<:ValuationPolydisc})
        return [1 / length(data) * sum([(model_eval(val, param) - out)^power for (val, out) in data]) for param in params]
    end

    function MPE_grad(vs::Vector{<:ValuationTangent})
        return [1 / length(data) * sum([power * (model_eval(val, v.point) - out)^(power - 1) * gradient_param(model, model_eval.fun_eval, val, v) for (val, out) in data]) for v in vs]
    end

    return Loss(MPE_compute, MPE_grad)
end

function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N},U}}) where {S,P,Prec,T,N,U}
    full_dim = length(model.param_info)
    model_eval = batch_evaluate_init(model, ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,full_dim})

    function MSE_compute(params::Vector{<:ValuationPolydisc})
        return [1 / length(data) * sum([(model_eval(val, param) - out)^2 for (val, out) in data]) for param in params]
    end

    function MSE_grad(vs::Vector{<:ValuationTangent})
        return [1 / length(data) * sum([2 * (model_eval(val, v.point) - out) * gradient_param(model, model_eval.fun_eval, val, v) for (val, out) in data]) for v in vs]
    end

    return Loss(MSE_compute, MSE_grad)
end
