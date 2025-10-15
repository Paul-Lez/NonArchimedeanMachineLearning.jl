###### THIS SECTIONS DEFINES THE DATA TYPES NEEDED FOR DOING OPTIMISATION AND PROVIDES SOME BASIC API #####

# Loss structure wraps an evaluation function and a gradient function
# Both should be closures that capture any necessary data
struct Loss
    eval
    grad
end

# Refactor Note: the goal of the refactor is to remove the dependence on data when optimising.
# Basically we should be able to just minimise an arbitrary function without having data
# entering the picture. Users should create loss functions that are closures over the data.

# OptimSetup is the structure that captures everything we need to do
# optimisation. The loss function should already have data baked in as a closure.
# The parameter can vary during optimisation since the structure is mutable.
mutable struct OptimSetup{S, T}
    # The loss function (should be a closure over any data)
    # loss.eval should have type (param) -> scalar
    # loss.grad should have type (tangent_vector) -> scalar
    loss::Loss
    # The current parameter value
    param::ValuationPolydisc{S, T}
    # An optimiser is a function that takes in the loss and param
    # (plus eventually other parameters, e.g. learning rate)
    # and outputs a new choice of parameters
    optimiser
end



# Evaluate the loss with respect to the current choice of parameters.
function eval_loss(optim::OptimSetup)
    return optim.loss.eval(optim.param)
end

# Update the parameter in the OptimSetup structure
function update_param!(optim::OptimSetup, param)
    optim.param = param
end

# Make a step in the direction given by the optimiser function of the OptimSetup structure.
# This directly changes the param field of the OptimSetup structure (since OptimSetup is mutable)
function step!(optim_setup::OptimSetup)
    new_param = optim_setup.optimiser(optim_setup.loss, optim_setup.param)
    update_param!(optim_setup, new_param)
end
