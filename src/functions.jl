include("tangent_vector.jl")

## This file will contain the basic of functions on the polydisc space and their calculus

# Evaluate the valuation of a polynomial at a point p
function evaluate_abs(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S, T}) where S where T
    t = gens(f.parent)
    vec = [t[i] - p.center[i] for i in eachindex(p.center)]
    g = AbstractAlgebra.evaluate(f, vec) 
    # coeff seems to return a constant polynomial so need to move this to padics by "taking the constant term" before taking absolute value
    max, _ = findmin([padic_abs(Nemo.coeff(g, v)) * prod(p.radius .^ v)  for v in Nemo.exponent_vectors(g)])
    return max
end

## The functions we're interested in are the composition of a vector of MV polynomials with a differentiable function on the Euclidean space.

## For now we specialise to absolute polynomial sums. We should later modify this type to get the general differentiable case.
## TODO Paul: type constraints? i.e. S, T...
struct PolydiscFunction{S}
    polys::Vector{AbstractAlgebra.Generic.MPoly{S}}
end 

# At the moment we work with multiple differential operators: the directional derivative along a tangent vector, and the gradient at a point.

# This will need to be reimplented when we move to the general case. This is the hard thing to implement.
function directional_derivative(f::PolydiscFunction, v) 
    T = base_ring(f.parent)
    _, x = polynomial_ring(T, "x")
    # 
    p = [j != i ? point(v)[j] : x for j in eachindex(v.tangents)]
    g = AbstractAlgebra.evaluate(f, p)
    return dir_deriv(g, point(v).disks[i], BerkovichPoint(direction(v)[i], 0))
end

# Compute symbolical gradient of rational function f at evaluated at the base point of v, in the direction of v
function grad(f, P::ValuationPolydisc{S, T}, Q::Vector{S}) where S<:Oscar.scalar_types where T<:Oscar.scalar_types
    return [directional_derivative(f, basis_vector(P, Q, i)) for i in Base.eachindex(Q)]
end

function eval_abs(fun::PolydiscFunction{S}, var::ValuationPolydisc{S, T}) where S where T
    return sum([evaluate_abs(f, var) for f in fun.polys])
end 