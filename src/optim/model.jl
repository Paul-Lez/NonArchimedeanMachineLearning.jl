@doc raw"""
    AbstractModel{S}

A model structure that captures the underlying function and parameter/variable mapping.

Represents a model without specified parameter values. Identifies which variables in the
function are data variables vs. parameters.

# Fields
- `fun::PolydiscFunction{S}`: The underlying function (sum of absolute polynomials)
- `param_info`: Binary vector indicating which variables are data (true/1) vs. parameters (false/0)

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)

# Example
If the function is ``f(x_1, \theta_1, x_2, \theta_2)`` with data variables ``x_1, x_2`` and
parameters ``\theta_1, \theta_2``, then `param_info = [1, 0, 1, 0]`.
"""
struct AbstractModel{S}
    fun::PolydiscFunction{S}
    # The data of which variables are parameters
    # E.g. if the function is f(x_1, θ_1, x_2, θ_2) then param_info = [1, 0, 1, 0]
    param_info
end

@doc raw"""
    Model{S, T}

A complete model with specified parameter values.

Combines an abstract model (function + parameter mapping) with concrete parameter values.
The structure is mutable to allow parameter updates during optimization.

# Fields
- `fun::AbstractModel{S}`: The abstract model (function and parameter info)
- `param::ValuationPolydisc{S, T}`: The current parameter values

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)
- `T`: The type for radius/valuation values
"""
mutable struct Model{S, T}
    fun::AbstractModel{S}
    # the values of the parameters
    param::ValuationPolydisc{S,T}
end

@doc raw"""
    update_weights!(m::Model, param)

Update the parameter values of a model in place.

# Arguments
- `m::Model`: The model to update
- `param`: New parameter values

# Notes
Mutates the model structure directly since `Model` is mutable.
"""
function update_weights!(m::Model, param)
    m.param = param
end

@doc raw"""
    var_indices(m::AbstractModel)

Get the indices of data variables in an abstract model.

# Arguments
- `m::AbstractModel`: The abstract model

# Returns
`Vector{Int}`: Indices where `param_info` is true (data variables)

# Example
For `param_info = [true, true, false, false]`, returns `[1, 2]`
"""
function var_indices(m::AbstractModel)
    return findall(x -> x, m.param_info)
end

@doc raw"""
    param_indices(m::AbstractModel)

Get the indices of parameters in an abstract model.

# Arguments
- `m::AbstractModel`: The abstract model

# Returns
`Vector{Int}`: Indices where `param_info` is false (parameters)

# Example
For `param_info = [true, true, false, false]`, returns `[3, 4]`
"""
function param_indices(m::AbstractModel)
    return findall(x -> !x, m.param_info)
end

@doc raw"""
    getkeys(m::AbstractModel)

Map each model variable to its position within data variables or parameters.

For each variable in the model, returns its index within either the data variables or
the parameters, depending on its type.

# Arguments
- `m::AbstractModel`: The abstract model

# Returns
`Vector{Int}`: Array ``[a_1, \ldots, a_n]`` where ``a_i`` is the index of the ``i``-th
variable within its category (data or parameter)

# Example
For ``f(x, \theta, y, z, \phi)`` with parameters ``\theta, \phi``, returns `[1, 1, 2, 3, 2]`
since ``x`` is the 1st data variable, ``\theta`` is the 1st parameter, ``y`` is the 2nd data
variable, ``z`` is the 3rd data variable, and ``\phi`` is the 2nd parameter.
"""
function getkeys(m::AbstractModel)
    vars = var_indices(m)
    param = param_indices(m)
    return [m.param_info[i] ? findfirst(item -> item == i, vars) : findfirst(item -> item == i, param) for i in Base.eachindex(m.param_info)]
end

@doc raw"""
    set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S, T}, param::ValuationPolydisc{S, T}) where {S,T}

Construct a point for evaluation by interleaving data and parameter values.

Given data variable values and parameter values, constructs a point in the full model space
that can be evaluated using polynomial evaluation mechanisms.

# Arguments
- `m::AbstractModel{S}`: The abstract model
- `val::ValuationPolydisc{S, T}`: Data variable values
- `param::ValuationPolydisc{S, T}`: Parameter values

# Returns
`ValuationPolydisc{S, T}`: Point with data and parameters interleaved according to `param_info`

# Example
For model ``f(x, \theta, y, \phi)`` with data ``(x, y) = (1, 2)`` and parameters
``(\theta, \phi) = (3, 4)``, returns the point ``(1, 3, 2, 4)``.
"""
function set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S, T}, param::ValuationPolydisc{S, T}) where S where T
    keys = getkeys(m)
    abstract_model_variable_radius = Vector{T}([m.param_info[i] ? val.radius[keys[i]] : param.radius[keys[i]] for i in Base.eachindex(m.param_info)])
    abstract_model_variable_center = Vector{S}([m.param_info[i] ? val.center[keys[i]] : param.center[keys[i]] for i in Base.eachindex(m.param_info)])
    #println(length(abstract_model_variable_center))
    return ValuationPolydisc{S,T}(abstract_model_variable_center, abstract_model_variable_radius)
end

@doc raw"""
    set_model_variable(m::Model{S, T}, val::ValuationPolydisc{S, T}) where {S,T}

Construct an evaluation point using a model's current parameters and given data.

Convenience wrapper around `set_abstract_model_variable` that uses the model's stored
parameter values.

# Arguments
- `m::Model{S, T}`: The model (with stored parameters)
- `val::ValuationPolydisc{S, T}`: Data variable values

# Returns
`ValuationPolydisc{S, T}`: Point with data and model parameters interleaved for evaluation
"""
function set_model_variable(m::Model{S, T}, val::ValuationPolydisc{S, T}) where S where T
    return set_abstract_model_variable(m.fun, val, m.param)
end

# Output the PolydiscFunction obtained by specialising the non-parameter varibles of `m` to the values in `val
function specialise_abstract_model_parameter(m::AbstractModel{S}, val::Vector{S})::PolydiscFunction{S} where S
    # Dispatch to type-specific implementations
    if isa(m.fun, AbsolutePolynomialSum)
        return specialise_absolute_polynomial_sum(m.fun, m.param_info, val)
    elseif isa(m.fun, LinearAbsolutePolynomialSum)
        return specialise_linear_absolute_polynomial_sum(m.fun, m.param_info, val)
    else
        error("specialise_abstract_model_parameter only implemented for AbsolutePolynomialSum and LinearAbsolutePolynomialSum")
    end
end

# Specialise an AbsolutePolynomialSum by substituting data variables
function specialise_absolute_polynomial_sum(f::AbsolutePolynomialSum{S}, param_info, val::Vector{S})::AbsolutePolynomialSum{S} where S
    # Get the original polynomial ring and variables
    R = f.polys[1].parent
    x = gens(R)

    # Get indices of data variables and parameters
    var_inds = findall(x -> x, param_info)
    param_inds = findall(x -> !x, param_info)

    # Create a new polynomial ring with only parameter variables
    param_var_names = [string(x[i]) for i in param_inds]
    new_R, new_params = polynomial_ring(Nemo.base_ring(f.polys[1]), param_var_names)

    # Build substitution list:
    # - For data variables: use provided values
    # - For parameters: use the new ring generators
    substitution = Vector(undef, length(x))
    for i in eachindex(x)
        if param_info[i]
            # This is a data variable - substitute with the provided value
            var_pos = findfirst(item -> item == i, var_inds)
            substitution[i] = val[var_pos]
        else
            # This is a parameter - use the new ring generator
            param_pos = findfirst(item -> item == i, param_inds)
            substitution[i] = new_params[param_pos]
        end
    end

    # println("The substitution is $substitution")
    h = hom(R, new_R, substitution)
    # Apply substitution to each polynomial in the original ring
    # new_polys = [AbstractAlgebra.evaluate(poly, substitution) for poly in f.polys]
    new_polys = [h(poly) for poly in f.polys]
    return AbsolutePolynomialSum(new_polys)
end

# Specialise a LinearAbsolutePolynomialSum by substituting data variables
function specialise_linear_absolute_polynomial_sum(f::LinearAbsolutePolynomialSum{S}, param_info, val::Vector{S})::LinearAbsolutePolynomialSum{S} where S
    # Specialize each linear polynomial
    new_polys = [specialise_linear_polynomial_data(poly, param_info, val) for poly in f.polys]
    return LinearAbsolutePolynomialSum(new_polys)
end

# Helper function to specialize a LinearPolynomial with param_info and data values
function specialise_linear_polynomial_data(poly::LinearPolynomial{S}, param_info, val::Vector{S})::LinearPolynomial{S} where S
    # For a linear polynomial: a_1 * T_1 + ... + a_n * T_n + b
    # Coefficients are indexed by variable position (1-based)
    # Specialization: keep only coefficients for parameters, and add contributions from data variables to constant term

    # Start with the original constant term
    new_constant = poly.constant

    # Build new coefficients for parameters only
    new_coefficients = S[]

    # For each coefficient in the original polynomial
    for i in eachindex(poly.coefficients)
        if param_info[i]
            # This is a data variable - add its contribution to the constant term
            var_inds = findall(x -> x, param_info)
            var_pos = findfirst(item -> item == i, var_inds)
            new_constant = new_constant + poly.coefficients[i] * val[var_pos]
        else
            # This is a parameter - keep its coefficient
            push!(new_coefficients, poly.coefficients[i])
        end
    end

    return LinearPolynomial(new_coefficients, new_constant)
end

# Combine specializations of an abstract model at multiple data points into a single polydisc function
# by appending the polynomial vectors from each specialization
function specialise_abstract_model_data(m::AbstractModel{S}, data::Vector{S})::PolydiscFunction{S} where S
    # Specialize the model at each data point
    specialized_funcs = [specialise_abstract_model_parameter(m, [val]) for val in data]

    # Combine by appending polynomials
    if isa(specialized_funcs[1], AbsolutePolynomialSum)
        combined_polys = reduce(vcat, [f.polys for f in specialized_funcs])
        return AbsolutePolynomialSum(combined_polys)
    elseif isa(specialized_funcs[1], LinearAbsolutePolynomialSum)
        combined_polys = reduce(vcat, [f.polys for f in specialized_funcs])
        return LinearAbsolutePolynomialSum(combined_polys)
    else
        error("specialise_abstract_model_data only implemented for AbsolutePolynomialSum and LinearAbsolutePolynomialSum")
    end
end

@doc raw"""
    evaluate(m::AbstractModel, val, param)

Evaluate an abstract model at given data and parameter values.

# Arguments
- `m::AbstractModel`: The abstract model
- `val`: Data variable values
- `param`: Parameter values

# Returns
`Float64`: The model evaluation result

# Notes
Current implementation is specific to absolute polynomial sums. Will need updates for
more general model functions.
"""
function evaluate(m::AbstractModel, val, param)
    var = set_abstract_model_variable(m, val, param)
    return evaluate(m.fun, var)
end

@doc raw"""
    evaluate(m::Model, val)

Evaluate a model at given data using the model's stored parameters.

# Arguments
- `m::Model`: The model (with stored parameters)
- `val`: Data variable values

# Returns
`Float64`: The model evaluation result
"""
function evaluate(m::Model, val)
    return evaluate(m.fun, val, m.param)
end