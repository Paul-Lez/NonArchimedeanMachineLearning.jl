## This file contains the basic of functions on the polydisc space and their calculus

@doc raw"""
An abstract class for the functions on polydisc spaces that we're interested in.
"""
abstract type PolydiscFunction{S} end

## The functions we're interested in are the composition of a vector of MV polynomials with a differentiable function on the Euclidean space.

## For now we specialise to absolute polynomial sums. We should later modify this type to get the general differentiable case.
@doc raw"""
A structure representing absolute polynomial sums.

Note: we may later want to generalise to allow polydiscs as coefficients.
"""
struct AbsolutePolynomialSum{S} <: PolydiscFunction{S}
    polys::Vector{AbstractAlgebra.Generic.MPoly{S}}
end

struct LinearPolynomial{S}
    # The array of coefficient of the variables
    coefficients::Vector{S}
    constant::S
end

struct LinearAbsolutePolynomialSum{S} <: PolydiscFunction{S}
    polys::Vector{LinearPolynomial{S}}
end

function parent(F::PolydiscFunction{S}) where S
    return parent(F)
end

function parent(F::AbsolutePolynomialSum{S}) where S
    return F.polys[1].parent
end

@doc raw"""
Evaluate a polydisc function at a polydisc.
"""
function evaluate(f::PolydiscFunction{S}, p::ValuationPolydisc{S,T}) where S where T
    return evaluate(f, p)
end

@doc raw"""
Evaluate the directional derivative of a polydisc function at a tangent vector v.
"""
function directional_derivative(f::PolydiscFunction{S}, v::ValuationTangent{S,T}) where S where T
    return directional_derivative(f, v)
end

# Evaluate the valuation of a polynomial at a point p
function evaluate_abs(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S,T}) where S where T
    t = gens(f.parent)
    # Is this the right thing to compute?
    vec = [t[i] + p.center[i] for i in eachindex(p.center)]
    g = AbstractAlgebra.evaluate(f, vec)
    # TODO Paul: check this
    max, _ = findmax([padic_abs(Nemo.coeff(g, v)) * (Float64(prime(p))^(-sum(p.radius .* v))) for v in Nemo.exponent_vectors(g)])
    return max
end

# Evaluate an absolute polynomial sum at a polydisc
function evaluate(fun::AbsolutePolynomialSum{S}, var::ValuationPolydisc{S,T}) where S where T
    return sum([evaluate_abs(f, var) for f in fun.polys])
end

function directional_derivative(fun::AbsolutePolynomialSum{S}, v::ValuationTangent{S,T}) where S where T
    return sum([directional_derivative(f, v) for f in fun.polys])
end

## TODO Claude
function evaluate(f::LinearAbsolutePolynomialSum{S}, p::ValuationPolydisc{S,T}) where S where T
    # Implementation: if f = a_1 T_1 + ... + a_n T _n + b then
    # the output is for a single f is max(abs(a_1) * r_1, ..., abs(a_n) * r_n, abs(b + a_1 * c_1 + ... + a_n * c_n))
    # where r is the radius of the polydisc and c the center.
    # When there is more than one f in the array then the output is the sum of the output for an individual f. This means that
    # one can paralllize the computation.
    return sum([evaluate(poly, p) for poly in f.polys])
end

function evaluate(poly::LinearPolynomial{S}, p::ValuationPolydisc{S,T}) where S where T
    # For a linear polynomial: a_1 * T_1 + ... + a_n * T_n + b
    # Compute max(|a_1| * r_1, ..., |a_n| * r_n, |b + a_1 * c_1 + ... + a_n * c_n|)

    # Evaluate the constant term plus the dot product of coefficients with center
    constant_term = poly.constant + sum(poly.coefficients[i] * p.center[i] for i in eachindex(poly.coefficients))

    # Compute absolute values of all terms
    abs_values = [padic_abs(poly.coefficients[i]) * (Float64(prime(p))^(-p.radius[i])) for i in eachindex(poly.coefficients)]
    push!(abs_values, padic_abs(constant_term))

    # Return the maximum
    return maximum(abs_values)
end

# At the moment we work with multiple differential operators: the directional derivative along a tangent vector, and the gradient at a point.


# Returns the directional exponent of an absolute polynomial |f| in the direction of a tangent vector v, i.e.
# the exponent vector n such that when we move in the direction of v, |f| is given by a monomial of exponent n.
# This is not uniquely defined, so the output of this function is an array of exponents.
function directional_exponent(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where S where T
    t = gens(f.parent)
    g = AbstractAlgebra.evaluate(f, t + v.direction)
    abs_terms = [padic_abs(Nemo.coeff(g, n)) * prod(v.point.radius .^ n) for n in Nemo.exponent_vectors(g)]
    # Find all exponents at which the max is attained
    max_exponents = findall(a -> a == maximum(abs_terms), abs_terms)
    # In principle this if clause isn't necessary (the "else" part works for all possible cases)
    # However I think this makes things faster.
    # TODO Paul: do some benchmarking to see if that's the case.
    if length(max_exponents) == 1
        return max_exponents
    else
        # Find minimal exponents at which the max is attained. These
        # are the directional exponents.
        return findall(a -> sum(a) == minimum([sum(n) for n in max_exponents]), max_exponents)
    end
end

# This will need to be reimplented when we move to the general case. This is the hard thing to implement.
# We use the fact that if locally in the direction of v, |f| = a_n r^n for some multi-index n, then we have
# d_v |f| = - |n| * |a_n| * r^n where r is the radius of the basepoint of v.
function directional_derivative(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where S where T
    # Recover the variables of the polynomial ring we're working over
    x = gens(f.parent)
    # Compute the expansion of f around the direction a of the tangent vector v, i.e.
    # The coefficients a_n such that f = ∑_n a_n (T-a)^n. We do this by computing the
    # expansion around 0 of the polynomial g(T) = f(T+a).
    #p = [j != i ? point(v)[j] : x for j in eachindex(v.tangents)]
    g = AbstractAlgebra.evaluate(f, x + v.direction)
    # Next we need to compute the directional exponent of f along v
    n = first(directional_exponent(f, v))
    # Use the formula to get d_v
    d_v = -sum(n) * padic_abs(coeff(g, n)) * (Float64(prime(v.point))^(-sum(v.point.radius .* n))) # prod(v.point.radius .^ n)
    return d_v
end

function directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{S,T}) where S where T
    return sum([directional_derivative(f, v) for f in fun.polys])
end

# Compute symbolical gradient of an absolute polynomial f evaluated at a polydisc P for the local direction given by
# point Q.
function grad(f, v::ValuationTangent{S,T}) where S where T
    return [directional_derivative(f, basis_vector(P, Q, i)) for i in Base.eachindex(Q)]
end

function partial_gradient(f, v::ValuationTangent{S,T}, gradient_indices) where S where T
    return [directional_derivative(f, basis_vector(v, i)) for i in gradient_indices]
end
