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
    MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}) where {S,T,U}

Initialize a Mean Squared Error (MSE) loss function with polydisc-valued inputs.

Creates a `Loss` structure with batch evaluation and gradient functions for MSE loss:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n (f(x_i; \theta) - y_i)^2``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{ValuationPolydisc{S,T},U}}`: Training data as `(input_polydisc, output_value)` pairs

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Notes
Both the evaluation and gradient functions operate on batches of parameters and tangent vectors,
allowing efficient parallel computation. Uses batch evaluation for improved performance.
"""
function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}) where S where T where U
    # Initialize batch evaluation for the model
    model_batch_eval = batch_evaluate_init(model)

    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return [1 / length(data) * sum([(model_batch_eval(val, param) - out)^2 for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # The gradient is evaluated at each tangent's base point (v.point)
    function MSE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return [1 / length(data) * sum([2 * (model_batch_eval(val, v.point) - out) * gradient_param(model, val, v) for (val, out) in data]) for v in vs]
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
then uses batch evaluation for improved performance.
"""
function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{S,U}}) where S where U
    # Specialize the model at each data point
    specialized_models = [specialise(model, [val]) for (val, _) in data]

    # Initialize batch evaluation for each specialized model
    batch_evals = [batch_evaluate_init(specialized_models[i]) for i in eachindex(specialized_models)]

    # Convert outputs to Float64 (p-adic absolute values)
    out_values = [abs(out) for (_, out) in data]

    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return 1 / length(data) * [sum([(batch_evals[i](param) - out_values[i])^2 for i in eachindex(data)]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    function MSE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return 1 / length(data) * [sum([2 * (batch_evals[i](v.point) - out_values[i]) * directional_derivative(specialized_models[i], v) for i in eachindex(data)]) for v in vs]
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
then uses batch evaluation for improved performance.
"""
function MSE_loss_init_new(model::AbstractModel{S}, data::Vector{Tuple{S,U}}) where S where U
    # Specialize the model at each data point
    specialized_model = 1 / length(data) * sum([
        (specialise(model, [val]) - abs(out))^2 for (val, out) in data
    ])

    # Initialize batch evaluation for each specialized model
    batch_eval = batch_evaluate_init(specialized_model)

    # TODO: this is currently quite slow compared to the previous 
    # implementation. Do some profiling!

    # Create a closure that computes the MSE for a batch of parameter values
    function MSE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return map(batch_eval, params)
    end

    # TODO Paul: do same as above for the gradient computation!

    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    function MSE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return 1 / length(data) * [sum([2 * (batch_evals[i](v.point) - out) * directional_derivative(specialized_models[i], v) for (i, (_, out)) in enumerate(data)]) for v in vs]
    end
    return Loss(MSE_compute, MSE_grad)
end


#################################################
# Mean p-Power Error (MPE) Loss Functions
#################################################

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}, p::Int) where {S,T,U}

Initialize a Mean p-Power Error (MPE) loss function with polydisc-valued inputs.

Generalizes MSE by using the ``\ell^p`` norm instead of ``\ell^2``. Computes:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(x_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{ValuationPolydisc{S,T},U}}`: Training data as `(input_polydisc, output_value)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Notes
For MSE (Euclidean loss), use ``p = 2``. The gradient computation uses the power rule.
Uses batch evaluation for improved performance.
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}, p::Int) where S where T where U
    # Initialize batch evaluation for the model
    model_batch_eval = batch_evaluate_init(model)

    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return [1 / length(data) * sum([(model_batch_eval(val, param) - out)^p for (val, out) in data]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    # The gradient is evaluated at each tangent's base point (v.point)
    function MPE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return [1 / length(data) * sum([p * (model_batch_eval(val, v.point) - out)^(p - 1) * gradient_param(model, val, v) for (val, out) in data]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end

# TODO(Paul-Lez): vectorise computations!

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{S,U}}, p::Int) where {S,U}

Initialize a Mean p-Power Error (MPE) loss function with field-valued inputs.

Generalizes MSE using the ``\ell^p`` norm with field-valued (not polydisc-valued) inputs.
Uses model specialization and batch evaluation for computational efficiency.

Computes: ``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(x_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{S,U}}`: Training data as `(field_element_input, output_value)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Implementation Notes
This overload specializes the model at each data point, making it more efficient for
field-valued inputs than the polydisc variant. Uses batch evaluation for improved performance.
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{S,U}}, p::Int) where S where U
    # Specialize the model at each data point
    specialized_models = [specialise(model, [val]) for (val, _) in data]

    # Initialize batch evaluation for each specialized model
    batch_evals = [batch_evaluate_init(specialized_models[i]) for i in eachindex(specialized_models)]

    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return 1 / length(data) * [sum([(batch_evals[i](param) - out)^p for (i, (_, out)) in enumerate(data)]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    function MPE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return 1 / length(data) * [sum([p * (batch_evals[i](v.point) - out)^(p - 1) * directional_derivative(specialized_models[i], v) for (i, (_, out)) in enumerate(data)]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{Vector{S},U}}, p::Int) where {S,U}

Initialize a Mean p-Power Error (MPE) loss function with vector-valued inputs.

Generalizes MPE for data where inputs are vectors of field elements (representing
multivariate data points). Uses model specialization and batch evaluation for efficiency.

Computes: ``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(\mathbf{x}_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model structure specifying the function and parameter layout
- `data::Vector{Tuple{Vector{S},U}}`: Training data as `(vector_of_field_elements, output_value)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with closures for batch evaluation and gradient computation

# Implementation Notes
This overload handles multivariate data where each input is a vector of field elements.
The model is specialized at each data point vector and uses batch evaluation for improved performance.
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{Vector{S},U}}, p::Int) where S where U
    # For vector-valued data, specialize the model at each data point vector
    specialized_models = [specialise(model, val) for (val, _) in data]

    # Initialize batch evaluation for each specialized model
    batch_evals = [batch_evaluate_init(specialized_models[i]) for i in eachindex(specialized_models)]

    # Create a closure that computes the MPE for a batch of parameter values
    function MPE_compute(params::Vector{ValuationPolydisc{S,T}}) where S where T
        return [1 / length(data) * sum([(batch_evals[i](param) - out)^p for (i, (_, out)) in enumerate(data)]) for param in params]
    end
    # Create a closure that computes the gradient of the loss along a batch of tangent directions
    function MPE_grad(vs::Vector{ValuationTangent{S,T}}) where S where T
        return [1 / length(data) * sum([p * (batch_evals[i](v.point) - out)^(p - 1) * directional_derivative(specialized_models[i], v) for (i, (_, out)) in enumerate(data)]) for v in vs]
    end
    return Loss(MPE_compute, MPE_grad)
end
