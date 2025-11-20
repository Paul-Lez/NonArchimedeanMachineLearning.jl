@doc raw"""
    AbstractModel{S}

A model structure that captures the underlying function and parameter/variable mapping.

Represents a model without specified parameter values. Encodes which variables in the
function are data variables (inputs) versus parameters (to be optimized). This separation
allows the loss and optimization machinery to work correctly.

# Fields
- `fun::PolydiscFunction{S}`: The underlying function (e.g., sum of absolute polynomials)
- `param_info`: Boolean vector where `param_info[i] = true` means variable ``i`` is a data variable,
  and `param_info[i] = false` means variable ``i`` is a parameter

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers like `PadicFieldElem`)

# Example
For a function ``f(x_1, \theta_1, x_2, \theta_2)`` with data variables ``x_1, x_2`` and
parameters ``\theta_1, \theta_2``, use `param_info = [true, false, true, false]`.

# Notes
Variables are ordered as they appear in the polynomial ring. Use `var_indices` and
`param_indices` to extract the positions of data variables and parameters respectively.
"""
struct AbstractModel{S}
    fun::PolydiscFunction{S}
    # Boolean vector indicating variable type:
    # true = data variable, false = parameter
    # E.g. for f(x, θ, y, φ) with param_info = [true, false, true, false]:
    # x and y are data, θ and φ are parameters
    param_info
end

@doc raw"""
    Model{S,T}

A complete model with specified parameter values.

Combines an abstract model (function and parameter mapping) with concrete current parameter
values. The structure is mutable to allow parameters to be updated during optimization.

# Fields
- `fun::AbstractModel{S}`: The abstract model encoding the function structure and parameter layout
- `param::ValuationPolydisc{S,T}`: The current parameter values in polydisc space

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)
- `T`: The type for radius/valuation values (typically `Int`)

# Example
```julia
# Create a model for f(x, θ) where x is data and θ is parameter
f = AbstractModel(polynomial_function, [true, false])
params = ValuationPolydisc([K(1), K(2)], [0, 0])  # θ = (1, 2)
model = Model(f, params)
```

# Notes
Models are mutable so that optimization algorithms can update `param` in place.
Use `update_weights!` to modify parameter values.
"""
mutable struct Model{S,T}
    fun::AbstractModel{S}
    # Current parameter values
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
    set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S,T}, param::ValuationPolydisc{S,T}) where {S,T}

Construct a polydisc by interleaving data and parameter values according to the model layout.

Given data variable values and parameter values, constructs a polydisc point in the full
model variable space that can be evaluated using polynomial evaluation mechanisms.

# Arguments
- `m::AbstractModel{S}`: The abstract model defining the variable layout via `param_info`
- `val::ValuationPolydisc{S,T}`: The data variable values (polydisc in data space)
- `param::ValuationPolydisc{S,T}`: The parameter values (polydisc in parameter space)

# Returns
`ValuationPolydisc{S,T}`: A polydisc point with all variables interleaved in model order

# Example
For model ``f(x, \theta, y, \phi)`` with `param_info = [true, false, true, false]`:
- Data inputs: `(x, y) = (1, 2)` → `val = ValuationPolydisc([K(1), K(2)], [r_x, r_y])`
- Parameters: `(\theta, \phi) = (3, 4)` → `param = ValuationPolydisc([K(3), K(4)], [r_θ, r_φ])`
- Returns: `ValuationPolydisc([K(1), K(3), K(2), K(4)], [r_x, r_θ, r_y, r_φ])`

# Notes
The returned polydisc has the same dimension as the original polynomial ring and can be
directly passed to polynomial evaluation functions.
"""
function set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S,T}, param::ValuationPolydisc{S,T}) where S where T
    keys = getkeys(m)
    abstract_model_variable_radius = Vector{T}([m.param_info[i] ? val.radius[keys[i]] : param.radius[keys[i]] for i in Base.eachindex(m.param_info)])
    abstract_model_variable_center = Vector{S}([m.param_info[i] ? val.center[keys[i]] : param.center[keys[i]] for i in Base.eachindex(m.param_info)])
    return ValuationPolydisc{S,T}(abstract_model_variable_center, abstract_model_variable_radius)
end

@doc raw"""
    set_model_variable(m::Model{S,T}, val::ValuationPolydisc{S,T}) where {S,T}

Construct an evaluation point using a model's stored parameter values and given data.

Convenience wrapper around `set_abstract_model_variable` that uses the model's current
parameter values rather than requiring them as an argument.

# Arguments
- `m::Model{S,T}`: The model (containing stored parameters)
- `val::ValuationPolydisc{S,T}`: The data variable values

# Returns
`ValuationPolydisc{S,T}`: A polydisc point with all variables interleaved in model order

# See Also
- `set_abstract_model_variable`: The underlying function with explicit parameters
"""
function set_model_variable(m::Model{S,T}, val::ValuationPolydisc{S,T}) where S where T
    return set_abstract_model_variable(m.fun, val, m.param)
end

@doc raw"""
    specialise_abstract_model_parameter(m::AbstractModel{S}, val::Vector{S})::PolydiscFunction{S} where S

Specialize a model by substituting data variable values.

Removes data variables from the model by substituting given values, returning a function
that depends only on parameters. This is used for computational efficiency when the data
is fixed.

# Arguments
- `m::AbstractModel{S}`: The abstract model
- `val::Vector{S}`: Values to substitute for data variables (length = number of data variables)

# Returns
`PolydiscFunction{S}`: A specialized polydisc function depending only on parameters

# Implementation Notes
Dispatches to type-specific implementations based on the function type
(`AbsolutePolynomialSum` or `LinearAbsolutePolynomialSum`).

# Errors
Raises an error if the model function type is not supported.

# Example
For a model with data variables x, y and parameters θ, φ:
- Original: ``f(x, \theta, y, \phi)``
- After specialization with `val = [x_0, y_0]`: ``f_specialized(\theta, \phi)``
"""
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

@doc raw"""
    specialise_absolute_polynomial_sum(f::AbsolutePolynomialSum{S}, param_info, val::Vector{S})::AbsolutePolynomialSum{S} where S

Specialize an AbsolutePolynomialSum by substituting data variable values.

Creates a new polynomial ring with only parameter variables and applies polynomial ring
homomorphisms to substitute data values into each polynomial.

# Arguments
- `f::AbsolutePolynomialSum{S}`: The polynomial sum to specialize
- `param_info`: Boolean vector indicating variable types
- `val::Vector{S}`: Values to substitute for data variables

# Returns
`AbsolutePolynomialSum{S}`: A polynomial sum depending only on parameters

# Implementation Details
Uses ring homomorphisms to correctly substitute values while maintaining the polynomial
structure in the new parameter-only ring.
"""
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
    # TODO: It's probably more efficient to create the ring hom only once for
    # large absolute polynomial sums!
    h = hom(R, new_R, substitution)
    new_polys = [h(poly) for poly in f.polys]
    return AbsolutePolynomialSum(new_polys)
end

@doc raw"""
    specialise_linear_absolute_polynomial_sum(f::LinearAbsolutePolynomialSum{S}, param_info, val::Vector{S})::LinearAbsolutePolynomialSum{S} where S

Specialize a LinearAbsolutePolynomialSum by substituting data variable values.

Applies specialization to each linear polynomial in the sum.

# Arguments
- `f::LinearAbsolutePolynomialSum{S}`: The linear polynomial sum to specialize
- `param_info`: Boolean vector indicating variable types
- `val::Vector{S}`: Values to substitute for data variables

# Returns
`LinearAbsolutePolynomialSum{S}`: A linear polynomial sum depending only on parameters
"""
function specialise_linear_absolute_polynomial_sum(
    f::LinearAbsolutePolynomialSum{S},
    param_info,
    val::Vector{S}
)::LinearAbsolutePolynomialSum{S} where S
    new_polys = [specialise_linear_polynomial_data(poly, param_info, val) for poly in f.polys]
    return LinearAbsolutePolynomialSum(new_polys)
end

@doc raw"""
    specialise_linear_polynomial_data(poly::LinearPolynomial{S}, param_info, val::Vector{S})::LinearPolynomial{S} where S

Specialize a single LinearPolynomial by substituting data variable values.

For a linear polynomial ``a_1 T_1 + \cdots + a_n T_n + b``, removes data variables by:
- Computing their contribution to the constant term
- Keeping only the coefficients for parameters

# Arguments
- `poly::LinearPolynomial{S}`: The linear polynomial to specialize
- `param_info`: Boolean vector indicating variable types
- `val::Vector{S}`: Values to substitute for data variables

# Returns
`LinearPolynomial{S}`: A linear polynomial depending only on parameters

# Example
For ``3x + 2\theta + 5y + 1`` with `x, y` as data and `\theta` as parameter:
- Substitute `x=1, y=2`: ``3(1) + 2\theta + 5(2) + 1 = 2\theta + 14``
- Returns: `LinearPolynomial([2], 14)`
"""
function specialise_linear_polynomial_data(
    poly::LinearPolynomial{S},
    param_info,
    val::Vector{S}
)::LinearPolynomial{S} where S
    # For a linear polynomial: a_1 * T_1 + ... + a_n * T_n + b
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

@doc raw"""
    specialise_abstract_model_data(m::AbstractModel{S}, data::Vector{S})::PolydiscFunction{S} where S

Specialize a model at multiple data points and combine the results.

Specializes the model at each data point separately, then combines all specialized
polynomials into a single function. This is useful for constructing batch loss functions.

# Arguments
- `m::AbstractModel{S}`: The abstract model
- `data::Vector{S}`: Data points at which to specialize the model

# Returns
`PolydiscFunction{S}`: A combined polydisc function with all specializations

# Implementation Notes
Concatenates polynomial vectors from each specialization. Supports
`AbsolutePolynomialSum` and `LinearAbsolutePolynomialSum` types.
"""
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

Constructs the full variable point by interleaving data and parameters, then evaluates
the underlying function.

# Arguments
- `m::AbstractModel`: The abstract model
- `val`: Data variable values (typically a `ValuationPolydisc`)
- `param`: Parameter values (typically a `ValuationPolydisc`)

# Returns
The evaluated function value (typically a `Float64` for absolute polynomial sums)

# Notes
Current implementation is specific to absolute polynomial sums. Will need updates
for more general polydisc functions.
"""
function evaluate(m::AbstractModel, val, param)
    var = set_abstract_model_variable(m, val, param)
    return evaluate(m.fun, var)
end

@doc raw"""
    evaluate(m::Model, val)

Evaluate a model at given data using the model's stored parameters.

Convenience wrapper that uses the model's current parameter values.

# Arguments
- `m::Model`: The model (containing stored parameters)
- `val`: Data variable values (typically a `ValuationPolydisc`)

# Returns
The evaluated function value (typically a `Float64` for absolute polynomial sums)

# See Also
- `evaluate(::AbstractModel, val, param)`: The underlying function with explicit parameters
"""
function evaluate(m::Model, val)
    return evaluate(m.fun, val, m.param)
end

# @doc raw"""
#     batch_evaluate_init(m::AbstractModel{S})

# Initialize a batch evaluation function for an abstract model.

# Returns a closure that evaluates the model at given data and parameter values.
# The returned function accepts two arguments: data and parameters (as vectors of polydiscs).

# # Arguments
# - `m::AbstractModel{S}`: The abstract model

# # Returns
# `Function`: A closure `(data::ValuationPolydisc, param::ValuationPolydisc) -> Float64`
# that can be applied to evaluate the model

# # Notes
# This function creates an evaluation closure that is optimized for batch operations.
# It interleaves data and parameter values according to the model's variable layout.
# """
# function batch_evaluate_init(m::AbstractModel{S}) where S
#     # Get the batch evaluation function for the underlying polydisc function
#     batch_fun_eval = batch_evaluate_init(m.fun)

#     # Return a closure that takes data and param values
#     function model_eval(val::ValuationPolydisc{S,T}, param::ValuationPolydisc{S,T}) where T
#         # Interleave the data and parameter values according to the model layout
#         full_var = set_abstract_model_variable(m, val, param)
#         # Evaluate the underlying function at the interleaved point
#         return batch_fun_eval(full_var)
#     end

#     return model_eval
# end

# @doc raw"""
#     batch_evaluate_init(m::Model{S,T})

# Initialize a batch evaluation function for a model with stored parameters.

# Returns a closure that evaluates the model at given data values using the model's
# stored parameter values. The returned function accepts a single argument: data (polydisc).

# # Arguments
# - `m::Model{S,T}`: The model (containing stored parameters)

# # Returns
# `Function`: A closure `(data::ValuationPolydisc) -> Float64` that evaluates the model
# at the given data using the stored parameters

# # Notes
# This is a convenience wrapper around `batch_evaluate_init(::AbstractModel)` that captures
# the model's current parameters in the closure.
# """
# function batch_evaluate_init(m::Model{S,T}) where S where T
#     # Get the batch evaluation function for the abstract model
#     abstract_batch_eval = batch_evaluate_init(m.fun)

#     # Capture the parameters in a closure
#     param = m.param

#     # Return a closure that takes only data values
#     function model_eval_with_params(val::ValuationPolydisc{S,T}) where T
#         return abstract_batch_eval(val, param)
#     end

#     return model_eval_with_params
# end