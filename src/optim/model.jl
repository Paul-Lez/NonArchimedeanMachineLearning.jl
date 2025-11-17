# This struct captures the underlying function of a model, plus which variables
# are parameters, but does not specify any values for the parameters.
struct AbstractModel{S}
    fun::PolydiscFunction{S}
    # The data of which variables are parameters
    # E.g. if the function is f(x_1, θ_1, x_2, θ_2) then param_info = [1, 0, 1, 0]
    param_info
end

# This structure captures a model, i.e. the underlying function, the
# data of which variables are parameters, and values for each parameter
mutable struct Model{S,T}
    fun::AbstractModel{S}
    # the values of the parameters
    param::ValuationPolydisc{S,T}
end

# Updates the weights of a model. Notice that this changes
# the structure directly since Model is a mutable structure.
function update_weights!(m::Model, param)
    m.param = param
end

# returns a vector with the indices of the variables of an abtract model
# e.g for param_info = [true, true, false, false] this would be [1, 2]
function var_indices(m::AbstractModel)
    return findall(x -> x, m.param_info)
end

# returns a vector with the indices of the parameters of an abtract model
# e.g for param_info = [true, true, false, false] this would be [3, 4]
function param_indices(m::AbstractModel)
    return findall(x -> !x, m.param_info)
end

# Given an abstract model in n variables (data variables + param), returns an array [a_1, ..., a_n]
# where a_i is the index such that the i-th variable of the model is the a_i-th parameter
# or the a_i-th data variable.
# E.g. if the model is f(x, θ, y, z, ϕ) where the parameterss are θ, ϕ then the output is
# [1, 1, 2, 3, 2] since x is the first data variable, θ is the first parameter, etc.
function getkeys(m::AbstractModel)
    vars = var_indices(m)
    param = param_indices(m)
    return [m.param_info[i] ? findfirst(item -> item == i, vars) : findfirst(item -> item == i, param) for i in Base.eachindex(m.param_info)]
end

# given a value for the parameters and for the data, this function outputs a point x that can be evaluated
# using the evaluation mechanisms for absolute polynomials.
# E.g. if the model is f(x, θ, y, ϕ) where the parameters is (θ, ϕ) are we are given (x, y) = (1, 2), (θ, ϕ) = (3, 4)
# then the function will output (1, 3, 2, 4).
function set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S,T}, param::ValuationPolydisc{S,T}) where S where T
    keys = getkeys(m)
    abstract_model_variable_radius = Vector{T}([m.param_info[i] ? val.radius[keys[i]] : param.radius[keys[i]] for i in Base.eachindex(m.param_info)])
    abstract_model_variable_center = Vector{S}([m.param_info[i] ? val.center[keys[i]] : param.center[keys[i]] for i in Base.eachindex(m.param_info)])
    #println(length(abstract_model_variable_center))
    return ValuationPolydisc{S,T}(abstract_model_variable_center, abstract_model_variable_radius)
end

# given a value for the parameters and for the data, this function outputs a point x that can be evaluated
# using the evaluation mechanisms for absolute polynomials.
# E.g. if the model is f(x, θ, y, ϕ) where the parameters is (θ, ϕ) are we are given (x, y) = (1, 2), (θ, ϕ) = (3, 4)
# then the function will output (1, 3, 2, 4).
function set_model_variable(m::Model{S,T}, val::ValuationPolydisc{S,T}) where S where T
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

# Evaluate the abstract model (this is currently an implentation that is specific to
# absolute polynomial sums and will need to be updated when we move to more general
# functions for our models)
function evaluate(m::AbstractModel, val, param)
    var = set_abstract_model_variable(m, val, param)
    return evaluate(m.fun, var)
end

# Evaluate model `m` at a choice of input `val`
function evaluate(m::Model, val)
    return evaluate(m.fun, val, m.param)
end