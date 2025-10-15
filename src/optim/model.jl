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
mutable struct Model{S, T}
    fun::AbstractModel{S}
    # the values of the parameters
    param::ValuationPolydisc{S, T}
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
function set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S, T}, param::ValuationPolydisc{S, T}) where S where T
    keys = getkeys(m)
    abstract_model_variable_radius = Vector{T}([m.param_info[i] ? val.radius[keys[i]] : param.radius[keys[i]] for i in Base.eachindex(m.param_info)])
    abstract_model_variable_center = Vector{S}([m.param_info[i] ? val.center[keys[i]] : param.center[keys[i]] for i in Base.eachindex(m.param_info)])
    #println(length(abstract_model_variable_center))
    return ValuationPolydisc{S, T}(abstract_model_variable_center, abstract_model_variable_radius)
end 

# given a value for the parameters and for the data, this function outputs a point x that can be evaluated
# using the evaluation mechanisms for absolute polynomials.
# E.g. if the model is f(x, θ, y, ϕ) where the parameters is (θ, ϕ) are we are given (x, y) = (1, 2), (θ, ϕ) = (3, 4)
# then the function will output (1, 3, 2, 4).
function set_model_variable(m::Model{S, T}, val::ValuationPolydisc{S, T}) where S where T
    return set_abstract_model_variable(m.fun, val, m.param)
end

# # This function is deprecated.
# # TODO Paul: remove this
# function specialise_abstract_model_data(m::AbstractModel{S}, val::ValuationPolydisc{S, T}) where S where T
#     keys = getkeys(m)
#     R = parent(model.fun)
#     x = gens(R)
#     abstract_model_variable_radius = Vector{T}([m.param_info[i] ? val.radius[keys[i]] : param.radius[keys[i]] for i in Base.eachindex(m.param_info)])
#     abstract_model_variable_center = Vector{S}([m.param_info[i] ? val.center[keys[i]] : param.center[keys[i]] for i in Base.eachindex(m.param_info)])

# end

# Evaluate the abstract model (this is currently an implentation that is specific to
# absolute polynomial sums and will need to be updated when we move to more general
# functions for our models)
function eval_abs(m::AbstractModel, val, param)
    var = set_abstract_model_variable(m, val, param)
    return eval_abs(m.fun, var)
end

# Evaluate model `m` at a choice of input `val`
function eval_abs(m::Model, val)
    return eval_abs(m.fun, val, m.param)
end