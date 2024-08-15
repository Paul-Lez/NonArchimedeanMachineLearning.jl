## This file will contain the code for doing gradient descent on polydisc space 
include("functions.jl")

###### THIS SECTIONS DEFINES THE DATA TYPES NEEDED FOR DOING OPTIMISATION AND PROVIDES SOME BASIC API #####

struct AbstractModel{S}
    fun::PolydiscFunction{S}
    # The data of which variables are parameters
    # E.g. if the function is f(x_1, θ_1, x_2, θ_2) then param_info = [1, 0, 1, 0]
    param_info 
end 

mutable struct Model{S, T}
    fun::AbstractModel{S}
    # the values of the parameters
    param::ValuationPolydisc{S, T}
end 

# OptimSetup is the structure that capture everything we need to do 
# Optimisation. Notice that we are imposing the data to be chosen and 
# a choice of a parameter. However these can vary along the way since 
# the structure is mutable. 
mutable struct OptimSetup{S, T}
    # The data we're using for the optimisation process
    data::Vector{ValuationPolydisc{S, T}}
    # The model. We're imposing that the parameter is chosen when constructing the structure, 
    # but this parameter will change when the optimisation starts
    model::Model{S, T}
    # The loss function. This should have type (data, model, param) -> scalar 
    # TODO Paul : is this really the type we want? 
    loss 
    # an optimiser is a function that takes in the data and the model and loss (plus eventually other parameters, e.g. learning rate)
    # and outputs a new choice of parameters for the model
    optimiser 
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
    @show vars = var_indices(m)
    @show param = param_indices(m)
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

# Evaluate the abstract model (this is currently an implentation that is specific to 
# absolute polynomial sums and will need to be updated when we move to more general 
# functions for our models)
function eval_abs(m::AbstractModel, val, param)
    var = set_abstract_model_variable(m, val, param)
    return eval_abs(m.fun, var)
end 

# Evaluate the loss with respect to the current choice of parameters. 
function eval_loss(optim::OptimSetup, param)
    return optim.loss(optim.data, param)
end 

# Make a step in the direction given by the optimiser function of the OptimSetup structure.
# This directly changes the Model field of the OptimSetup structure (since OptimSetup is mutable)
function step!(optim_setup::OptimSetup)
    new_weights = optim_setup.optimiser(optim_setup.data, optim_setup.model)
    update_weights!(optim_setup.model, new_weights)
end 

########### Greedy descent algorithm ###########

# In this section we implement greedy descent algorithms

# The optimiser function for greedy descent
function greedy_descent(data::Vector{ValuationPolydisc{S, T}}, model::Model{S, T}, loss, degree=1) where S where T
    below_nodes = children(model.param, degree)
    # In greedy descent, we look at the children of the 
    # current parameter point and take the child 
    # that minimises the loss
    val, ind = findmin([loss(data, model, param) for param in below_nodes]) 
    return below_nodes[ind]
end   

# This function outputs the OptimSetup object for greedy descent
function greedy_descent_init(data::Vector{ValuationPolydisc{S, T}}, model::Model{S, T}, loss, degree=1) where S where T
    return OptimSetup(data, model, loss, (dat, mod) -> greedy_descent(dat, mod, loss, degree))
end

########### Gradient descent algorithm #########
# In this section we implement the tools necessary for gradient descent, and the gradient descent algorithm

function eval(m::Model, input)
    # instantiate the variable (i.e. input value + parameter value) we're evalutating at
    return "implement me "
end 

# gradient wrt the parameters of the model
function gradient_param(m::Model, data)
    return "implement me"
end 

# gradient wrt the data variable
function gradient_data(m::Model, data)
    return "implement me"
end 