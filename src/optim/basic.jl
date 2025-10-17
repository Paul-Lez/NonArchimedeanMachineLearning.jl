###### THIS SECTIONS DEFINES THE DATA TYPES NEEDED FOR DOING OPTIMISATION AND PROVIDES SOME BASIC API #####

# Loss structure wraps an evaluation function and a gradient function
# Both should be closures that capture any necessary data
struct Loss
    eval::Function
    grad::Function
end

# TODO: possible refactor:
# We can bundle the value of param in the state, 
# by assuming that the state type always has a method
# "get_param!" available.  

# OptimSetup is the structure that captures everything we need to do
# optimisation. The loss function should already have data baked in as a closure.
# The parameter can vary during optimisation since the structure is mutable.
mutable struct OptimSetup{S,T,U,V}
    # The loss function (should be a closure over any data)
    # loss.eval should have type (param) -> scalar
    # loss.grad should have type (tangent_vector) -> scalar
    loss::Loss
    # The current parameter value
    param::ValuationPolydisc{S,T}
    # An optimiser is a function that takes in the loss and param
    # (plus eventually other parameters, e.g. learning rate)
    # and outputs a new choice of parameters
    optimiser::Function
    # The state is an optional field that records the state of the optimisation
    # process, e.g. previous steps that were made, etc.
    # This is useful since some optimisation methods may depend on the state. 
    state::U
    # The context type. This records things like settings for the optimiser, etc
    context::V
end

# Evaluate the loss with respect to the current choice of parameters.
function eval_loss(optim::OptimSetup)
    return optim.loss.eval(optim.param)
end

# Update the parameter in the OptimSetup structure
function update_param!(
    optim::OptimSetup{S,T,U,V},
    param::ValuationPolydisc{S,T}
) where S where T where U where V
    optim.param = param
end

function update_state!(optim::OptimSetup{S,T,U,V}, state::U) where S where T where U where V
    optim.state = state
end

# Make a step in the direction given by the optimiser function of the OptimSetup structure.
# This directly changes the param field of the OptimSetup structure (since OptimSetup is mutable)
function step!(optim_setup::OptimSetup)
    new_param, new_state = optim_setup.optimiser(
        optim_setup.loss,
        optim_setup.param,
        optim_setup.state,
        optim_setup.context
    )
    update_param!(optim_setup, new_param)
    update_state!(optim_setup, new_state)
end
