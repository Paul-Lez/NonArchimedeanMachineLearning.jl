"""
Baseline random descent optimizer for experimental comparison against structured
search methods.

This optimizer deliberately ignores loss values when selecting children, so it
serves as a lower-bound baseline rather than a recommended optimization method.
"""

@doc raw"""
    RandomDescentConfig(; strict=false, degree=1, start_branch=1)

Configuration for random descent (baseline optimizer).

# Fields
- `strict::Bool`: If true, descend one coordinate at a time; if false, descend all coordinates
- `degree::Int`: Number of children to explore per polydisc node (non-strict mode)
- `start_branch::Int`: Coordinate to descend along first (strict mode)
"""
struct RandomDescentConfig <: AbstractOptimConfig
    strict::Bool
    degree::Int
    start_branch::Int

    function RandomDescentConfig(;
            strict::Bool = false,
            degree::Int = 1,
            start_branch::Int = 1
    )
        @req degree >= 1 "degree must be positive"
        @req start_branch >= 1 "start_branch must be positive"
        new(strict, degree, start_branch)
    end
end

@doc raw"""
    random_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, next_branch::Int, settings::RandomDescentConfig) where {S,T,N}

Perform one step of random descent (baseline optimizer).

**BASELINE ONLY**: Randomly selects a child without evaluating loss. Used to demonstrate
that structured optimization algorithms outperform random exploration.

# Arguments
- `loss::Loss`: The loss function structure (not used in selection)
- `param::ValuationPolydisc{S,T,N}`: Current parameter values
- `next_branch::Int`: Index of the next branch to descend in strict mode
- `settings::RandomDescentConfig`: Configuration for random descent

# Returns
`Tuple{ValuationPolydisc{S,T,N}, Int, Bool}`: Randomly selected child, next branch
index, and convergence status
"""
function random_descent(
        loss::Loss,
        param::ValuationPolydisc{S, T, N},
        next_branch::Int,
        settings::RandomDescentConfig
) where {S, T, N}
    below_nodes, next_branch = _descent_candidates(param, next_branch, settings)
    isempty(below_nodes) && return (param, next_branch, true)

    # RANDOM SELECTION: Pick a random child without considering loss
    # This is the key difference from greedy descent
    random_index = rand(1:length(below_nodes))

    return (below_nodes[random_index], next_branch, false)
end

@doc raw"""
    random_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, settings::RandomDescentConfig=RandomDescentConfig()) where {S,T,N}

Initialize an optimization setup for random descent.

**BASELINE ONLY**: This optimizer is used for baseline comparison to demonstrate
the effectiveness of structured optimization algorithms.

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `settings::RandomDescentConfig`: Configuration controlling descent behavior

# Returns
`OptimSetup`: Configured optimization setup for random descent

# Example
```julia
# Create baseline optimizer for comparison
random_optim = random_descent_init(param, loss, RandomDescentConfig())
greedy_optim = greedy_descent_init(param, loss, GreedyDescentConfig())

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
        param::ValuationPolydisc{S, T, N},
        loss::Loss,
        settings::RandomDescentConfig = RandomDescentConfig()
) where {S, T, N}
    @req settings.start_branch <= N "start_branch must be at most the dimension of the polydisc"
    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> random_descent(l, p, st, ctx),
        settings.start_branch,
        settings,
        false
    )
end
