# This file provides tools for computing the Frechet mean
# of a set X = {p1, ..., pm} of n-polydiscs.

@doc raw"""
    frechet_mean(X::Vector{Vector{PadicFieldElem}})

Compute the Fréchet mean of a collection of p-adic vectors.

Computes the Fréchet mean with respect to the ``\ell^1``-metric on ``\mathcal{B}^n`` by
minimizing ``\sum_{i} d(x, x_i)`` coordinatewise.

# Arguments
- `X::Vector{Vector{PadicFieldElem}}`: Collection of p-adic vectors

# Returns
`Vector{PadicFieldElem}`: The Fréchet mean vector

# Algorithm
For each coordinate, selects the sample point that minimizes the sum of distances to
all other points in that coordinate.
"""
function frechet_mean(X::Vector{Vector{PadicFieldElem}})
    A = transpose(hcat(X...))
    mean = Vector{PadicFieldElem}()
    for j in Base.axes(A, 2)
        loss = (x::PadicFieldElem) -> sum([padic_abs(x - A[i, j]) for i in Base.axes(A, 1)])
        # surely there's a cleaner way of doing this.
        min, i = findmin(loss, [A[i, j] for i in Base.axes(A, 1)])
        push!(mean, A[i, j])
    end
    return mean
end

@doc raw"""
    frechet_mean(X::Vector{ValuationPolydisc{S, T}}, prec) where {S,T}

Compute the Fréchet mean of a collection of polydiscs.

Uses greedy descent optimization to minimize the sum of distances to all polydiscs.
Starts from the join of all polydiscs and refines for a specified number of steps.

# Arguments
- `X::Vector{ValuationPolydisc{S, T}}`: Collection of polydiscs
- `prec`: Number of optimization steps (precision/iterations)

# Returns
`ValuationPolydisc{S, T}`: The approximate Fréchet mean polydisc

# Implementation Notes
Uses a workaround by constructing a dummy model to leverage the optimization library.
This should be refactored for a cleaner implementation.
"""
function frechet_mean(X::Vector{ValuationPolydisc{S, T}}, prec) where S where T
    mean_point = Vector{S}()
    mean_radius = Vector{T}()
    # Define the Frechet loss for batches
    function loss_eval(params::Vector{ValuationPolydisc{S, T}}) where S where T
        return [sum([dist(x, param) for x in X]) for param in params]
    end
    loss = Loss(loss_eval, x -> ones(length(x)))
    K = Base.parent(X[1].center[1])
    R, (x, ) = polynomial_ring(K, ["x"])
    starting_point = X[1]
    for i in 2:length(X)
        starting_point = join(starting_point, X[i])
    end
    optim = greedy_descent_init(starting_point, loss, 1, (false, 1))
    for i in 1:prec
        step!(optim)
    end
    return optim.param
end
