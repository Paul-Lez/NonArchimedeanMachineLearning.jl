## This file contains the basic of functions on the polydisc space and their calculus

@doc raw"""
    PolydiscFunction{S}

Abstract base type for functions on polydisc spaces.

Represents any function that can be evaluated on polydisc points and whose directional
derivatives can be computed.

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)
"""
abstract type PolydiscFunction{S} end

## The functions we're interested in are the composition of a vector of MV polynomials with a differentiable function on the Euclidean space.

## For now we specialise to absolute polynomial sums. We should later modify this type to get the general differentiable case.
@doc raw"""
    AbsolutePolynomialSum{S}

A structure representing absolute polynomial sums.

Represents a function as a sum of multivariate polynomials, where absolute values are
taken to define the evaluation at arbitrary polydiscs.

# Fields
- `polys::Vector{AbstractAlgebra.Generic.MPoly{S}}`: Vector of multivariate polynomials

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)

# Note
We may later want to generalise to allow polydiscs as coefficients.
"""
struct AbsolutePolynomialSum{S} <: PolydiscFunction{S}
    polys::Vector{AbstractAlgebra.Generic.MPoly{S}}
end

@doc raw"""
    LinearPolynomial{S}

A structure representing a linear polynomial.

Encodes a polynomial of the form ``a_1 T_1 + \cdots + a_n T_n + b`` where ``a_i`` are
coefficients and ``b`` is a constant term.

# Fields
- `coefficients::Vector{S}`: Array of coefficients for each variable
- `constant::S`: The constant term

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)
"""
struct LinearPolynomial{S}
    # The array of coefficient of the variables
    coefficients::Vector{S}
    constant::S
end

# TODO: add macros (or custom notations) to parse things more easily.

struct LinearRationalFunction{S} <: PolydiscFunction{S}
    num::LinearPolynomial{S}
    den::LinearPolynomial{S}
end

struct Add{S} <: PolydiscFunction{S}
    left::PolydiscFunction{S}
    right::PolydiscFunction{S}
end

struct Mul{S} <: PolydiscFunction{S}
    left::PolydiscFunction{S}
    right::PolydiscFunction{S}
end

struct Sub{S} <: PolydiscFunction{S}
    left::PolydiscFunction{S}
    right::PolydiscFunction{S}
end

struct Div{S} <: PolydiscFunction{S}
    top::PolydiscFunction{S}
    bottom::PolydiscFunction{S}
end

# Scalar multiplication
struct SMul{S} <: PolydiscFunction{S}
    left::Number
    right::PolydiscFunction{S}
end

# Composition of a polydisc function with a real function
struct Comp{S} <: PolydiscFunction{S}
    left::Function
    right::PolydiscFunction{S}
end

struct Constant{S} <: PolydiscFunction{S}
    value::Number
end

Base.:+(a::PolydiscFunction{S}, b::PolydiscFunction{S}) where S = Add(a, b)
Base.:-(a::PolydiscFunction{S}, b::PolydiscFunction{S}) where S = Sub(a, b)
Base.:*(a::PolydiscFunction{S}, b::PolydiscFunction{S}) where S = Mul(a, b)
Base.:/(a::PolydiscFunction{S}, b::PolydiscFunction{S}) where S = Div(a, b)
Base.:*(a::Number, b::PolydiscFunction{S}) where S = SMul(a, b)

# Scalar operations
Base.:-(a::PolydiscFunction{S}, b::Number) where S = Sub(a, Constant{S}(b))
Base.:-(a::Number, b::PolydiscFunction{S}) where S = Sub(Constant{S}(a), b)
Base.:+(a::PolydiscFunction{S}, b::Number) where S = Add(a, Constant{S}(b))
Base.:+(a::Number, b::PolydiscFunction{S}) where S = Add(Constant{S}(a), b)
Base.:*(a::PolydiscFunction{S}, b::Number) where S = SMul(b, a)
Base.:/(a::PolydiscFunction{S}, b::Number) where S = SMul(1 / b, a)

# TODO: implement this in a smarter way!
function Base.:^(a::PolydiscFunction{S}, b::Int) where S
    if b == 0
        return 1
    elseif b > 0
        return a * (a^(b - 1))
    elseif b < 0
        return 1 / (a^b)
    end
end

comp(f::Function, g::PolydiscFunction{S}) where S = Comp(f, g)
# function smul(a::Number, b::PolydiscFunction{S})

@doc raw"""
    LinearAbsolutePolynomialSum{S}

A structure representing a sum of linear polynomials.

Composed of multiple linear polynomials, allowing for efficient evaluation and
gradient computation over polydisc spaces.

# Fields
- `polys::Vector{LinearPolynomial{S}}`: Vector of linear polynomials

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)
"""
struct LinearAbsolutePolynomialSum{S} <: PolydiscFunction{S}
    polys::Vector{LinearPolynomial{S}}
end

struct LinearRationalFunctionSum{S} <: PolydiscFunction{S}
    rats::Vector{LinearRationalFunction{S}}
end

@doc raw"""
    parent(F::PolydiscFunction{S}) where S

Get the polynomial ring of a polydisc function.

# Arguments
- `F::PolydiscFunction{S}`: The polydisc function

# Returns
The parent polynomial ring
"""
function parent(F::PolydiscFunction{S}) where S
    return parent(F)
end

# TODO(Paul-Lez): I think this doesn't make sense?
@doc raw"""
    parent(F::AbsolutePolynomialSum{S}) where S

Get the polynomial ring of an absolute polynomial sum.

# Arguments
- `F::AbsolutePolynomialSum{S}`: The absolute polynomial sum

# Returns
`Ring`: The parent ring of the first polynomial
"""
function parent(F::AbsolutePolynomialSum{S}) where S
    return F.polys[1].parent
end

@doc raw"""
    evaluate(f::PolydiscFunction{S}, p::ValuationPolydisc{S,T}) where S where T

Evaluate a polydisc function at a polydisc.

# Arguments
- `f::PolydiscFunction{S}`: The polydisc function
- `p::ValuationPolydisc{S,T}`: The evaluation point

# Returns
The function value at the point
"""
function evaluate(f::PolydiscFunction{S}, p::ValuationPolydisc{S,T}) where S where T
    return evaluate(f, p)
end

@doc raw"""
    directional_derivative(f::PolydiscFunction{S}, v::ValuationTangent{S,T}) where S where T

Evaluate the directional derivative of a polydisc function at a tangent vector.

# Arguments
- `f::PolydiscFunction{S}`: The polydisc function
- `v::ValuationTangent{S,T}`: The tangent vector direction

# Returns
`Float64`: The directional derivative in the direction of `v`
"""
function directional_derivative(f::PolydiscFunction{S}, v::ValuationTangent{S,T}) where S where T
    return directional_derivative(f, v)
end


# TODO(Paul-Lez): there are various optimisations to be done here:
# Profiling suggests that:
# 1) There are some type instabilities when this function is called by the optimisation part of the library
# 2) Too much time is spent allocating memory, i.e AbstractAlgebra.evaluate is suboptimal here. In particular we
#   may want to preallocate memory when computing the expansion at a given point
@doc raw"""
    evaluate_abs(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S,T}) where S where T

Evaluate the absolute value of a multivariate polynomial at a polydisc.

Computes the p-adic absolute value by expanding the polynomial around the center
and finding the maximum absolute value term weighted by the radius.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: A multivariate polynomial
- `p::ValuationPolydisc{S,T}`: The evaluation point (polydisc)

# Returns
`Float64`: The absolute value of the polynomial at the polydisc
"""
function evaluate_abs(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S,T}) where S where T
    t = gens(f.parent)
    vec = [t[i] + p.center[i] for i in eachindex(p.center)]
    g = AbstractAlgebra.evaluate(f, vec)
    max, _ = findmax([abs(Nemo.coeff(g, v)) * (Float64(prime(p))^(-sum(p.radius .* v))) for v in Nemo.exponent_vectors(g)])
    return max
end

function evaluate(fun::Add{S}, var::ValuationPolydisc{S,T}) where S where T
    return evaluate(fun.left, var) + evaluate(fun.right, var)
end

function evaluate(fun::Mul{S}, var::ValuationPolydisc{S,T}) where S where T
    return evaluate(fun.left, var) * evaluate(fun.right, var)
end

function evaluate(fun::Sub{S}, var::ValuationPolydisc{S,T}) where S where T
    return evaluate(fun.left, var) - evaluate(fun.right, var)
end

function evaluate(fun::Div{S}, var::ValuationPolydisc{S,T}) where S where T
    return evaluate(fun.top, var) / evaluate(fun.bottom, var)
end

function evaluate(fun::SMul{S}, var::ValuationPolydisc{S,T}) where S where T
    return fun.left * evaluate(fun.right, var)
end

function evaluate(fun::Comp{S}, var::ValuationPolydisc{S,T}) where S where T
    return fun.left(evaluate(fun.right, var))
end

function evaluate(c::Constant{S}, var::ValuationPolydisc{S,T}) where S where T
    return c.value
end

@doc raw"""
    evaluate(fun::AbsolutePolynomialSum{S}, var::ValuationPolydisc{S,T}) where S where T

Evaluate an absolute polynomial sum at a polydisc.

Computes the sum of absolute values of each polynomial in the sum evaluated at the point.

# Arguments
- `fun::AbsolutePolynomialSum{S}`: The polynomial sum
- `var::ValuationPolydisc{S,T}`: The evaluation point

# Returns
`Float64`: The sum of polynomial evaluations
"""
function evaluate(fun::AbsolutePolynomialSum{S}, var::ValuationPolydisc{S,T}) where S where T
    return sum([evaluate_abs(f, var) for f in fun.polys])
end

function evaluate(f::LinearRationalFunction{S}, var::ValuationPolydisc{S,T}) where S where T
    return evaluate(f.num, var) / evaluate(f.den, var)
end

function evaluate(f::LinearRationalFunctionSum{S}, var::ValuationPolydisc{S,T}) where S where T
    return sum([evaluate(fun, var) for fun in f.rats])
end

@doc raw"""
    directional_derivative(fun::AbsolutePolynomialSum{S}, v::ValuationTangent{S,T}) where S where T

Compute the directional derivative of a polynomial sum along a tangent direction.

# Arguments
- `fun::AbsolutePolynomialSum{S}`: The polynomial sum
- `v::ValuationTangent{S,T}`: The tangent direction

# Returns
`Float64`: The directional derivative in direction `v`
"""
function directional_derivative(fun::AbsolutePolynomialSum{S}, v::ValuationTangent{S,T}) where S where T
    return sum([directional_derivative(f, v) for f in fun.polys])
end

function directional_derivative(fun::Add{S}, v::ValuationTangent{S,T}) where S where T
    return directional_derivative(fun.left, v) + directional_derivative(fun.right, v)
end

function directional_derivative(fun::Sub{S}, v::ValuationTangent{S,T}) where S where T
    return directional_derivative(fun.left, v) - directional_derivative(fun.right, v)
end

function directional_derivative(fun::Mul{S}, v::ValuationTangent{S,T}) where S where T
    # Product rule: (f*g)' = f'*g + f*g'
    return directional_derivative(fun.left, v) * evaluate(fun.right, v.point) + evaluate(fun.left, v.point) * directional_derivative(fun.right, v)
end

function directional_derivative(fun::Div{S}, v::ValuationTangent{S,T}) where S where T
    # Quotient rule: (f/g)' = (f'*g - f*g') / g²
    f = fun.top
    g = fun.bottom
    f_val = evaluate(f, v.point)
    g_val = evaluate(g, v.point)
    f_deriv = directional_derivative(f, v)
    g_deriv = directional_derivative(g, v)
    return (f_deriv * g_val - f_val * g_deriv) / (g_val^2)
end

function directional_derivative(fun::SMul{S}, v::ValuationTangent{S,T}) where S where T
    return fun.left * directional_derivative(fun.right, v)
end

function directional_derivative(c::Constant{S}, v::ValuationTangent{S,T}) where S where T
    # Constant functions have zero derivative
    return 0.0
end

@doc raw"""
    evaluate(f::LinearAbsolutePolynomialSum{S}, p::ValuationPolydisc{S,T}) where S where T

Evaluate a sum of linear polynomials at a polydisc.

For each linear polynomial ``a_1 T_1 + \cdots + a_n T_n + b``, computes
``\max(|a_1| r_1, \ldots, |a_n| r_n, |b + a_1 c_1 + \cdots + a_n c_n|)``
where ``r`` is the radius and ``c`` the center of the polydisc.

# Arguments
- `f::LinearAbsolutePolynomialSum{S}`: The sum of linear polynomials
- `p::ValuationPolydisc{S,T}`: The evaluation point

# Returns
`Float64`: The sum of evaluations across all linear polynomials
"""
function evaluate(f::LinearAbsolutePolynomialSum{S}, p::ValuationPolydisc{S,T}) where S where T
    return sum([evaluate(poly, p) for poly in f.polys])
end

@doc raw"""
    evaluate(poly::LinearPolynomial{S}, p::ValuationPolydisc{S,T}) where S where T

Evaluate a single linear polynomial at a polydisc.

For a linear polynomial ``a_1 T_1 + \cdots + a_n T_n + b``, computes
``\max(|a_1| r_1, \ldots, |a_n| r_n, |b + a_1 c_1 + \cdots + a_n c_n|)``

# Arguments
- `poly::LinearPolynomial{S}`: The linear polynomial
- `p::ValuationPolydisc{S,T}`: The evaluation point

# Returns
`Float64`: The maximum absolute value term
"""
function evaluate(poly::LinearPolynomial{S}, p::ValuationPolydisc{S,T}) where S where T
    # Evaluate the constant term plus the dot product of coefficients with center
    constant_term = poly.constant + sum(poly.coefficients[i] * p.center[i] for i in eachindex(poly.coefficients))
    # TODO(Paul-Lez): it's probably more efficient to do this in terms of valuation?
    # Compute absolute values of all terms
    abs_values = [abs(poly.coefficients[i]) * (Float64(prime(p))^(-p.radius[i])) for i in eachindex(poly.coefficients)]
    push!(abs_values, abs(constant_term))
    # Return the maximum
    return maximum(abs_values)
end

function batch_evaluate_init(f::PolydiscFunction{S})::Function where S
    return batch_evaluate_init(f)
end

function batch_evaluate_init(poly::LinearPolynomial{S})::Function where S
    abs_poly_coeffs = map(valuation, poly.coefficients)
    num_coeffs = length(poly.coefficients)
    function eval(p::ValuationPolydisc{S,T}) where T
        # Only use the first num_coeffs coordinates (in case the polydisc is higher dimensional)
        constant_term = poly.constant + sum(poly.coefficients[i] * p.center[i] for i in 1:num_coeffs)
        # Compute valuations of all terms
        abs_values = [abs_poly_coeffs[i] + p.radius[i] for i in 1:num_coeffs]
        push!(abs_values, valuation(constant_term))
        # Compute the absolute value
        return Float64(prime(p))^minimum(abs_values)
    end
    return eval
end

function batch_evaluate_init(f::Add{S})::Function where S
    left_eval = batch_evaluate_init(f.left)
    right_eval = batch_evaluate_init(f.right)
    return p -> left_eval(p) + right_eval(p)
end

function batch_evaluate_init(f::Mul{S})::Function where S
    left_eval = batch_evaluate_init(f.left)
    right_eval = batch_evaluate_init(f.right)
    return p -> left_eval(p) * right_eval(p)
end

function batch_evaluate_init(f::Sub{S})::Function where S
    left_eval = batch_evaluate_init(f.left)
    right_eval = batch_evaluate_init(f.right)
    return p -> left_eval(p) - right_eval(p)
end

function batch_evaluate_init(f::Div{S})::Function where S
    left_eval = batch_evaluate_init(f.left)
    right_eval = batch_evaluate_init(f.right)
    return p -> left_eval(p) / right_eval(p)
end

function batch_evaluate_init(f::SMul{S})::Function where S
    right_eval = batch_evaluate_init(f.right)
    return p -> f.left * right_eval(p)
end

function batch_evaluate_init(f::Comp{S})::Function where S
    right_eval = batch_evaluate_init(f.right)
    function eval(p::ValuationPolydisc{S,T}) where T
        return f.left(right_eval(p))
    end
    return eval
end

function batch_evaluate_init(f::Constant{S})::Function where S
    function eval(p::ValuationPolydisc{S,T}) where T
        return f.value
    end
    return eval
end

function batch_evaluate_init(f::LinearAbsolutePolynomialSum{S})::Function where S
    # Get the array of functions
    evaluation_functions::Array{Function} = map(batch_evaluate_init, f.polys)
    # Return the lambda function that sends `p`` to the sum of the evaluations of each element of `evaluation_functions` at `p`
    function eval(p::ValuationPolydisc{S,T}) where T
        return sum(
            map(f -> f(p), evaluation_functions))
    end
    return eval
end

function batch_evaluate_init(f::AbstractAlgebra.Generic.MPoly{S})::Function where S
    function eval(p::ValuationPolydisc{S,T}) where T
        return evaluate_abs(f, p)
    end
    return eval
end

function batch_evaluate_init(f::AbsolutePolynomialSum{S})::Function where S
    # Get the array of functions
    evaluation_functions = map(batch_evaluate_init, f.polys)
    # Return the lambda function that sends `p`` to the sum of the evaluations of each element of `evaluation_functions` at `p`
    function eval(p::ValuationPolydisc{S,T}) where T
        return sum(map(f -> f(p), evaluation_functions))
    end
    return eval
end

# At the moment we work with multiple differential operators: the directional derivative along a tangent vector, and the gradient at a point.

@doc raw"""
    directional_exponent(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where S where T

Find the exponent vector(s) along which a polynomial achieves its maximum absolute value.

For a polynomial ``f`` and tangent vector ``v``, finds all exponent vectors ``n`` such that
locally in the direction of ``v``, ``|f| = a_n r^n`` for some coefficient ``a_n``. Among
all maximum exponents, returns the minimal ones in terms of sum of components.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: A multivariate polynomial
- `v::ValuationTangent{S,T}`: The tangent vector defining the direction

# Returns
`Vector`: Array of exponent vectors (as tuples) where the maximum is attained in a minimal fashion
"""
function directional_exponent(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where S where T
    t = gens(f.parent)
    g = AbstractAlgebra.evaluate(f, t + v.direction)
    abs_terms = [abs(Nemo.coeff(g, n)) * prod(v.point.radius .^ n) for n in Nemo.exponent_vectors(g)]
    # Find all exponents at which the max is attained
    max_exponents = findall(a -> a == maximum(abs_terms), abs_terms)
    # In principle this if clause isn't necessary (the "else" part works for all possible cases)
    # However I think this makes things faster.
    if length(max_exponents) == 1
        return max_exponents
    else
        # Find minimal exponents at which the max is attained. These
        # are the directional exponents.
        return findall(a -> sum(a) == minimum([sum(n) for n in max_exponents]), max_exponents)
    end
end

@doc raw"""
    directional_derivative(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where S where T

Compute the directional derivative of a multivariate polynomial along a tangent direction.

Uses the formula: if locally ``|f| = a_n r^n`` for exponent ``n``, then
``d_v |f| = -|n| |a_n| r^n`` where ``r`` is the radius of the basepoint.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: The polynomial
- `v::ValuationTangent{S,T}`: The tangent direction

# Returns
`Float64`: The directional derivative
"""
function directional_derivative(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where S where T
    # Recover the variables of the polynomial ring we're working over
    x = gens(f.parent)
    # Compute the expansion of f around the direction a of the tangent vector v, i.e.
    # The coefficients a_n such that f = ∑_n a_n (T-a)^n. We do this by computing the
    # expansion around 0 of the polynomial g(T) = f(T+a).
    g = AbstractAlgebra.evaluate(f, x + v.direction)
    # Next we need to compute the directional exponent of f along v
    n = first(directional_exponent(f, v))
    # Use the formula to get d_v
    d_v = -sum(n) * abs(coeff(g, n)) * (Float64(prime(v.point))^(-sum(v.point.radius .* n)))
    return d_v
end

@doc raw"""
    directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{S,T}) where S where T

Compute the directional derivative of a polydisc function (sum of polynomials).

# Arguments
- `fun::PolydiscFunction{S}`: The polydisc function
- `v::ValuationTangent{S,T}`: The tangent direction

# Returns
`Float64`: Sum of directional derivatives across all polynomials
"""
function directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{S,T}) where S where T
    return sum([directional_derivative(f, v) for f in fun.polys])
end

@doc raw"""
    grad(f, v::ValuationTangent{S,T}) where S where T

Compute the gradient of a polynomial by evaluating directional derivatives along all coordinates.

# Arguments
- `f`: The polynomial or function
- `v::ValuationTangent{S,T}`: A reference tangent vector defining the space

# Returns
`Vector`: Gradient components, one for each coordinate direction
"""
function grad(f, v::ValuationTangent{S,T}) where S where T
    return [directional_derivative(f, basis_vector(v, i)) for i in Base.eachindex(v.magnitude)]
end

@doc raw"""
    partial_gradient(f, v::ValuationTangent{S,T}, gradient_indices) where S where T

Compute partial derivatives along specified coordinate directions.

# Arguments
- `f`: The polynomial or function
- `v::ValuationTangent{S,T}`: A reference tangent vector
- `gradient_indices`: Indices of coordinates for which to compute derivatives

# Returns
`Vector`: Directional derivatives for the specified coordinates
"""
function partial_gradient(f, v::ValuationTangent{S,T}, gradient_indices) where S where T
    return [directional_derivative(f, basis_vector(v, i)) for i in gradient_indices]
end
