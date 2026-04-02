##################################################
# Value Transforms
#
# Functions that transform raw loss values into reward signals
# for tree search optimizers (MCTS, DAG-MCTS, etc.).
#
# All transforms map loss → value where higher value = better.
##################################################

@doc raw"""
    sigmoid_transform(; scale=1.0, center=0.0)

Create a sigmoid value transform: `loss → 1 / (1 + exp(scale * (loss - center)))`.

Low loss maps to high value (~1), high loss maps to low value (~0).
The `center` parameter shifts the midpoint (where output = 0.5).
The `scale` parameter controls the steepness of the transition.

# Keyword Arguments
- `scale::Float64=1.0`: Steepness of the sigmoid. Larger = sharper transition.
- `center::Float64=0.0`: Loss value at which the output is 0.5.
"""
function sigmoid_transform(; scale::Float64=1.0, center::Float64=0.0)
    return loss -> 1.0 / (1.0 + exp(scale * (loss - center)))
end

@doc raw"""
    tanh_transform(; scale=1.0, center=0.0)

Create a tanh value transform: `loss → (1 - tanh(scale * (loss - center))) / 2`.

Rescaled to [0, 1] range. Low loss maps to high value (~1), high loss maps to low value (~0).

# Keyword Arguments
- `scale::Float64=1.0`: Steepness of the transition.
- `center::Float64=0.0`: Loss value at which the output is 0.5.
"""
function tanh_transform(; scale::Float64=1.0, center::Float64=0.0)
    return loss -> (1.0 - tanh(scale * (loss - center))) / 2.0
end

@doc raw"""
    negation_transform()

Create the simple negation transform: `loss → -loss`.

This is the classic transform used in early MCTS implementations.
Unbounded output range.
"""
function negation_transform()
    return loss -> -loss
end

@doc raw"""
    inverse_transform(; epsilon=1e-10)

Create the inverse transform: `loss → 1 / (loss + epsilon)`.

# Keyword Arguments
- `epsilon::Float64=1e-10`: Small constant to avoid division by zero.
"""
function inverse_transform(; epsilon::Float64=1e-10)
    return loss -> 1.0 / (loss + epsilon)
end

const DEFAULT_VALUE_TRANSFORM = sigmoid_transform()
