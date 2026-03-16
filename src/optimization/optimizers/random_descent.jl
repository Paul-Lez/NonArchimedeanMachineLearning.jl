########### Random descent algorithm (Baseline) ###########

# IMPORTANT: This optimizer exists ONLY for experimental/baseline comparison purposes.
# It demonstrates worst-case random behavior to show that structured algorithms
# (like greedy descent or MCTS) perform significantly better than random exploration.
#
# This optimizer randomly selects a child at each step without considering loss values.
# Expected to perform poorly - serves as a lower bound for algorithm performance.

@doc raw"""
    random_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, state::Int, settings::Tuple{Bool,Int}) where {S,T,N}

Perform one step of random descent (baseline optimizer).

**BASELINE ONLY**: Randomly selects a child without evaluating loss. Used to demonstrate
that structured optimization algorithms outperform random exploration.

# Arguments
- `loss::Loss`: The loss function structure (not used in selection)
- `param::ValuationPolydisc{S,T,N}`: Current parameter values
- `state::Int`: Unused state parameter (for compatibility)
- `settings::Tuple{Bool,Int}`: `(strict, degree)` where `strict` enables single-coordinate descent

# Returns
`Tuple{ValuationPolydisc{S,T,N}, Int}`: Randomly selected child and state
"""
function random_descent(
    loss::Loss,
    param::ValuationPolydisc{S,T,N},
    state::Int,
    settings::Tuple{Bool,Int}
) where {S, T, N}
    (strict, degree) = settings
    if strict
        # In strict mode, cycle through branches
        next_branch = state
        below_nodes = children_along_branch(param, next_branch)
        state = next_branch == dim(param) ? 1 : next_branch + 1
    else
        # In non-strict mode, generate all children
        below_nodes = children(param, degree)
    end

    isempty(below_nodes) && return (param, state, true)

    # RANDOM SELECTION: Pick a random child without considering loss
    # This is the key difference from greedy descent
    random_index = rand(1:length(below_nodes))

    return (below_nodes[random_index], state, false)
end

@doc raw"""
    random_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, state::Int, settings::Tuple{Bool,Int}) where {S,T,N}

Initialize an optimization setup for random descent.

**BASELINE ONLY**: This optimizer is used for baseline comparison to demonstrate
the effectiveness of structured optimization algorithms.

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `state::Int`: Starting state (typically 1)
- `settings::Tuple{Bool,Int}`: `(strict, degree)` controlling descent behavior

# Returns
`OptimSetup`: Configured optimization setup for random descent

# Example
```julia
# Create baseline optimizer for comparison
random_optim = random_descent_init(param, loss, 1, (false, 1))
greedy_optim = greedy_descent_init(param, loss, 1, (false, 1))

# Compare performance
for i in 1:20
    step!(random_optim)
    step!(greedy_optim)
end

println("Random: ", eval_loss(random_optim))
println("Greedy: ", eval_loss(greedy_optim))  # Should be much better
```
"""
function random_descent_init(
    param::ValuationPolydisc{S,T,N},
    loss::Loss,
    state::Int,
    settings::Tuple{Bool,Int}
) where {S, T, N}
    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> random_descent(l, p, st, ctx),
        state,
        settings,
        false
    )
end
