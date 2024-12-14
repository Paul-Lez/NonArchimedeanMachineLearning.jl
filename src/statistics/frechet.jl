# This file provides tools for computing the Frechet mean
# of a set X = {p1, ..., pm} of n-polydiscs.

# Returns the Frechet mean of X wrt the ℓ¹-metric on ℬⁿ.
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

# Computes the Frechet Mean of a set of discs.
# Implementation note: the optimisation library is constructed in
# terms of parametrised models, so here we end up having to contruct a bogus model
# simply to be able to use the descent tools to minimise the Frechet loss.
# This should be refactored eventually.
function frechet_mean(X::Vector{ValuationPolydisc{S, T}}, prec) where S where T
    mean_point = Vector{S}()
    mean_radius = Vector{T}()
    # Define the Frechet loss
    function loss_eval(abs_model, data, param)
        return sum([dist(x, param) for x in X])
    end
    loss = Loss(loss_eval, x -> 1)
    K = Base.parent(X[1].center[1])
    R, (x, ) = polynomial_ring(K, ["x"])
    abstract_model = AbstractModel(PolydiscFunction([x]), [false])
    starting_point = X[1]
    for i in 2:length(X)
        starting_point = join(starting_point, X[i])
    end
    model = Model(abstract_model, starting_point)
    optim = greedy_descent_init([(x, 1) for x in X], model, loss)
    for i in 1:prec
        step!(optim)
    end
    return optim.model.param
end
