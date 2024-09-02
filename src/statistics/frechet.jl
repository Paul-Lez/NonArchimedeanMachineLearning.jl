include("../basic/functions.jl")

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

# TODO: Implement a frecher_mean function for vectors of polydiscs
# This will probably require greedy descent.