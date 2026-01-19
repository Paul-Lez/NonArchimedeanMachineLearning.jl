###### THIS SECTIONS DEFINES THE DATA TYPES NEEDED FOR DOING OPTIMISATION AND PROVIDES SOME BASIC API #####

@doc raw"""
    Loss

A loss function structure for optimization.

Wraps both an evaluation function and a gradient function. Both functions should be
closures that capture any necessary data (e.g., training data).

# Fields
- `eval::Function`: Function to evaluate the loss, signature: `(param) -> scalar`
- `grad::Function`: Function to compute gradient, signature: `(tangent_vector) -> scalar`
"""
struct Loss
    eval::Function
    grad::Function
end

# TODO: possible refactor:
# We can bundle the value of param in the state, 
# by assuming that the state type always has a method
# "get_param!" available.  

@doc raw"""
    OptimSetup{S,T,U,V}

Complete optimization setup containing loss, parameters, optimizer, and state.

Mutable structure that captures everything needed for optimization. The loss function
should have data baked in as a closure.

# Fields
- `loss::Loss`: Loss function (closure over data) with `eval: (param) -> scalar` and `grad: (tangent) -> scalar`
- `param::ValuationPolydisc{S,T}`: Current parameter values (mutable during optimization)
- `optimiser::Function`: Optimizer function `(loss, param, state, context) -> (new_param, new_state)`
- `state::U`: Optimization state (e.g., previous steps, momentum, etc.)
- `context::V`: Optimizer settings (e.g., learning rate, degree, etc.)

# Type Parameters
- `S`: Coefficient type (typically p-adic numbers)
- `T`: Radius/valuation type
- `U`: State type
- `V`: Context type
"""
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

@doc raw"""
    eval_loss(optim::OptimSetup)

Evaluate the loss function at the current parameter values.

# Arguments
- `optim::OptimSetup`: The optimization setup

# Returns
Scalar value of the loss at the current parameters
"""
function eval_loss(optim::OptimSetup)
    return optim.loss.eval([optim.param])[1]
end

@doc raw"""
    update_param!(optim::OptimSetup{S,T,U,V}, param::ValuationPolydisc{S,T}) where {S,T,U,V}

Update the parameter values in the optimization setup.

# Arguments
- `optim::OptimSetup{S,T,U,V}`: The optimization setup
- `param::ValuationPolydisc{S,T}`: New parameter values

# Notes
Mutates the optimization setup in place.
"""
function update_param!(
    optim::OptimSetup{S,T,U,V},
    param::ValuationPolydisc{S,T}
) where {S, T, U, V}
    optim.param = param
end

@doc raw"""
    update_state!(optim::OptimSetup{S,T,U,V}, state::U) where {S,T,U,V}

Update the optimizer state in the optimization setup.

# Arguments
- `optim::OptimSetup{S,T,U,V}`: The optimization setup
- `state::U`: New state value

# Notes
Mutates the optimization setup in place.
"""
function update_state!(optim::OptimSetup{S,T,U,V}, state::U) where {S, T, U, V}
    optim.state = state
end

@doc raw"""
    step!(optim_setup::OptimSetup)

Perform one optimization step.

Calls the optimizer function to compute new parameters and state, then updates the
optimization setup accordingly.

# Arguments
- `optim_setup::OptimSetup`: The optimization setup

# Notes
Mutates the optimization setup by updating both parameters and state.
"""
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
