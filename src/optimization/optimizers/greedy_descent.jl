"""
Greedy local descent optimizer over the children of the current polydisc.
"""

@doc raw"""
    GreedyDescentConfig(; strict=false, degree=1, start_branch=1)

Configuration for greedy descent optimization.

# Fields
- `strict::Bool`: If true, descend one coordinate at a time; if false, descend all coordinates
- `degree::Int`: Number of children to explore per polydisc node (non-strict mode)
- `start_branch::Int`: Coordinate to descend along first (strict mode)

# Example
```julia
config = GreedyDescentConfig(strict = false, degree = 1)
optim = greedy_descent_init(param, loss, config)
```
"""
struct GreedyDescentConfig <: AbstractOptimConfig
    strict::Bool
    degree::Int
    start_branch::Int

    function GreedyDescentConfig(;
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
    greedy_descent(loss::Loss, param::ValuationPolydisc{S,T,N}, next_branch::Int, settings::GreedyDescentConfig) where {S,T,N}

Perform one step of greedy descent optimization.

Computes children of the current parameter and selects the child that minimizes the loss.
Can operate in strict mode (one coordinate at a time) or full mode (all coordinates).

# Arguments
- `loss::Loss`: The loss function structure
- `param::ValuationPolydisc{S,T,N}`: Current parameter values
- `next_branch::Int`: Index of next branch to descend (in strict mode)
- `settings::GreedyDescentConfig`: Configuration for greedy descent

# Returns
`Tuple{ValuationPolydisc{S,T,N}, Int, Bool}`: New parameters, next branch index,
and convergence status
"""
function greedy_descent(
        loss::Loss,
        param::ValuationPolydisc{S, T, N},
        next_branch::Int,
        settings::GreedyDescentConfig
) where {S, T, N}
    below_nodes, next_branch = _descent_candidates(param, next_branch, settings)
    isempty(below_nodes) && return (param, next_branch, true)
    # In greedy descent, we look at the children of the
    # current parameter point and take the child
    # that minimises the loss
    loss_values = loss.eval(below_nodes)
    # Pick a *random* minimum amond the possible ones
    min = rand(findall(u -> u == minimum(loss_values), loss_values))
    return (below_nodes[min], next_branch, false)
end

@doc raw"""
    greedy_descent_init(param::ValuationPolydisc{S,T,N}, loss::Loss, settings::GreedyDescentConfig=GreedyDescentConfig()) where {S,T,N}

Initialize an optimization setup for greedy descent.

# Arguments
- `param::ValuationPolydisc{S,T,N}`: Initial parameter values
- `loss::Loss`: The loss function structure
- `settings::GreedyDescentConfig`: Configuration controlling descent behavior

# Returns
`OptimSetup`: Configured optimization setup for greedy descent
"""
function greedy_descent_init(
        param::ValuationPolydisc{S, T, N},
        loss::Loss,
        settings::GreedyDescentConfig = GreedyDescentConfig()
) where {S, T, N}
    @req settings.start_branch <= N "start_branch must be at most the dimension of the polydisc"
    return OptimSetup(
        loss,
        param,
        (l, p, st, ctx) -> greedy_descent(l, p, st, ctx),
        settings.start_branch,
        settings,
        false
    )
end

# TODO: Implement the 1D tree search algorithm
