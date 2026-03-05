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

# DifferentiableFunction: a real-valued function paired with its derivative
struct DifferentiableFunction
    f::Function    # Float64 -> Float64
    df::Function   # Float64 -> Float64 (derivative)
end
(d::DifferentiableFunction)(x) = d.f(x)

# Composition of a polydisc function with a differentiable real function
struct Comp{S} <: PolydiscFunction{S}
    left::DifferentiableFunction
    right::PolydiscFunction{S}
end

struct Constant{S} <: PolydiscFunction{S}
    value::Number
end

struct Lambda{S} <: PolydiscFunction{S}
    func::Function
    derivative::Union{Function, Nothing}
end
Lambda{S}(func) where S = Lambda{S}(func, nothing)

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
Base.:-(a::PolydiscFunction{S}) where S = SMul(-1, a)

Base.zero(::Type{PolydiscFunction{S}}) where S = Constant{S}(0)
Base.zero(::PolydiscFunction{S}) where S = Constant{S}(0)


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

comp(f::DifferentiableFunction, g::PolydiscFunction{S}) where S = Comp(f, g)
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
    evaluate(f::PolydiscFunction{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}

Evaluate a polydisc function at a polydisc.

# Arguments
- `f::PolydiscFunction{S}`: The polydisc function
- `p::ValuationPolydisc{S,T,N}`: The evaluation point

# Returns
The function value at the point
"""
function evaluate(f::PolydiscFunction{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}
    return evaluate(f, p)
end

@doc raw"""
    directional_derivative(f::PolydiscFunction{S}, v::ValuationTangent{S,T}) where {S, T}

Evaluate the directional derivative of a polydisc function at a tangent vector.

# Arguments
- `f::PolydiscFunction{S}`: The polydisc function
- `v::ValuationTangent{S,T}`: The tangent vector direction

# Returns
`Float64`: The directional derivative in the direction of `v`
#
# Note: The actual implementation for PolydiscFunction is defined later in this file
# after the specific implementations for AbsolutePolynomialSum and other subtypes.
"""
# Removed duplicate stub definition - see line ~562 for actual implementation


# TODO(Paul-Lez): there are various optimisations to be done here:
# Profiling suggests that:
# 1) There are some type instabilities when this function is called by the optimisation part of the library
# 2) Too much time is spent allocating memory, i.e AbstractAlgebra.evaluate is suboptimal here. In particular we
#   may want to preallocate memory when computing the expansion at a given point
@doc raw"""
    evaluate(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}

Evaluate the absolute value of a multivariate polynomial at a polydisc.

Computes the p-adic absolute value by expanding the polynomial around the center
and finding the maximum absolute value term weighted by the radius.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: A multivariate polynomial
- `p::ValuationPolydisc{S,T,N}`: The evaluation point (polydisc)

# Returns
`Float64`: The absolute value of the polynomial at the polydisc
"""
function evaluate(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}
    t = gens(f.parent)
    vec = [t[i] + p.center[i] for i in eachindex(p.center)]
    g = AbstractAlgebra.evaluate(f, vec)
    monomials = [abs(Nemo.coeff(g, v)) * (Float64(prime(p))^(-sum(p.radius .* v))) for v in Nemo.exponent_vectors(g)]
    max, _ = findmax(monomials)
    return max
end

function evaluate(fun::Add{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return evaluate(fun.left, var) + evaluate(fun.right, var)
end

function evaluate(fun::Mul{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return evaluate(fun.left, var) * evaluate(fun.right, var)
end

function evaluate(fun::Sub{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return evaluate(fun.left, var) - evaluate(fun.right, var)
end

function evaluate(fun::Div{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return evaluate(fun.top, var) / evaluate(fun.bottom, var)
end

function evaluate(fun::SMul{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return fun.left * evaluate(fun.right, var)
end

function evaluate(fun::Comp{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return fun.left(evaluate(fun.right, var))
end

function evaluate(c::Constant{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return c.value
end

function evaluate(l::Lambda{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return l.func(var)
end

@doc raw"""
    evaluate(fun::AbsolutePolynomialSum{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}

Evaluate an absolute polynomial sum at a polydisc.

Computes the sum of absolute values of each polynomial in the sum evaluated at the point.

# Arguments
- `fun::AbsolutePolynomialSum{S}`: The polynomial sum
- `var::ValuationPolydisc{S,T,N}`: The evaluation point

# Returns
`Float64`: The sum of polynomial evaluations
"""
function evaluate(fun::AbsolutePolynomialSum{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return sum([evaluate(f, var) for f in fun.polys])
end

function evaluate(f::LinearRationalFunction{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return evaluate(f.num, var) / evaluate(f.den, var)
end

function evaluate(f::LinearRationalFunctionSum{S}, var::ValuationPolydisc{S,T,N}) where {S, T, N}
    return sum([evaluate(fun, var) for fun in f.rats])
end

@doc raw"""
    directional_derivative(fun::AbsolutePolynomialSum{S}, v::ValuationTangent{S,T}) where {S, T}

Compute the directional derivative of a polynomial sum along a tangent direction.

# Arguments
- `fun::AbsolutePolynomialSum{S}`: The polynomial sum
- `v::ValuationTangent{S,T}`: The tangent direction

# Returns
`Float64`: The directional derivative in direction `v`
"""
function directional_derivative(fun::AbsolutePolynomialSum{S}, v::ValuationTangent{S,T}) where {S, T}
    return sum([directional_derivative(f, v) for f in fun.polys])
end

function directional_derivative(fun::Add{S}, v::ValuationTangent{S,T}) where {S, T}
    return directional_derivative(fun.left, v) + directional_derivative(fun.right, v)
end

function directional_derivative(fun::Sub{S}, v::ValuationTangent{S,T}) where {S, T}
    return directional_derivative(fun.left, v) - directional_derivative(fun.right, v)
end

function directional_derivative(fun::Mul{S}, v::ValuationTangent{S,T}) where {S, T}
    # Product rule: (f*g)' = f'*g + f*g'
    return directional_derivative(fun.left, v) * evaluate(fun.right, v.point) + evaluate(fun.left, v.point) * directional_derivative(fun.right, v)
end

function directional_derivative(fun::Div{S}, v::ValuationTangent{S,T}) where {S, T}
    # Quotient rule: (f/g)' = (f'*g - f*g') / g²
    f = fun.top
    g = fun.bottom
    f_val = evaluate(f, v.point)
    g_val = evaluate(g, v.point)
    f_deriv = directional_derivative(f, v)
    g_deriv = directional_derivative(g, v)
    return (f_deriv * g_val - f_val * g_deriv) / (g_val^2)
end

function directional_derivative(fun::SMul{S}, v::ValuationTangent{S,T}) where {S, T}
    return fun.left * directional_derivative(fun.right, v)
end

function directional_derivative(c::Constant{S}, v::ValuationTangent{S,T}) where {S, T}
    # Constant functions have zero derivative
    return 0.0
end

function directional_derivative(fun::Comp{S}, v::ValuationTangent{S,T}) where {S, T}
    # Chain rule: (f ∘ g)' = f'(g(x)) * g'(x)
    inner_val = evaluate(fun.right, v.point)
    inner_deriv = directional_derivative(fun.right, v)
    return fun.left.df(inner_val) * inner_deriv
end

function directional_derivative(l::Lambda{S}, v::ValuationTangent{S,T}) where {S, T}
    l.derivative === nothing && error("Lambda function has no derivative. Provide a derivative function at construction time.")
    return l.derivative(v)
end

@doc raw"""
    evaluate(f::LinearAbsolutePolynomialSum{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}

Evaluate a sum of linear polynomials at a polydisc.

For each linear polynomial ``a_1 T_1 + \cdots + a_n T_n + b``, computes
``\max(|a_1| r_1, \ldots, |a_n| r_n, |b + a_1 c_1 + \cdots + a_n c_n|)``
where ``r`` is the radius and ``c`` the center of the polydisc.

# Arguments
- `f::LinearAbsolutePolynomialSum{S}`: The sum of linear polynomials
- `p::ValuationPolydisc{S,T,N}`: The evaluation point

# Returns
`Float64`: The sum of evaluations across all linear polynomials
"""
function evaluate(f::LinearAbsolutePolynomialSum{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}
    return sum([evaluate(poly, p) for poly in f.polys])
end

function directional_derivative(fun::LinearAbsolutePolynomialSum{S}, v::ValuationTangent{S,T,N}) where {S, T, N}
    return sum(directional_derivative(poly, v) for poly in fun.polys)
end

@doc raw"""
    evaluate(poly::LinearPolynomial{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}

Evaluate a single linear polynomial at a polydisc.

For a linear polynomial ``a_1 T_1 + \cdots + a_n T_n + b``, computes
``\max(|a_1| r_1, \ldots, |a_n| r_n, |b + a_1 c_1 + \cdots + a_n c_n|)``

# Arguments
- `poly::LinearPolynomial{S}`: The linear polynomial
- `p::ValuationPolydisc{S,T,N}`: The evaluation point

# Returns
`Float64`: The maximum absolute value term
"""
function evaluate(poly::LinearPolynomial{S}, p::ValuationPolydisc{S,T,N}) where {S, T, N}
    # Evaluate the constant term plus the dot product of coefficients with center
    constant_term = poly.constant + sum([poly.coefficients[i] * p.center[i] for i in eachindex(poly.coefficients)])
    # TODO(Paul-Lez): it's probably more efficient to do this in terms of valuation?
    # Compute absolute values of all terms
    abs_values = [abs(poly.coefficients[i]) * (Float64(prime(p))^(-p.radius[i])) for i in eachindex(poly.coefficients)]
    push!(abs_values, abs(constant_term))
    # Return the maximum
    return maximum(abs_values)
end

#=============================================================================
 Typed Evaluators - Refactor 2: Replace closures with callable structs
=============================================================================#

@doc raw"""
    PolydiscFunctionEvaluator{S,T,N}

Abstract base type for typed function evaluators.

Separates mathematical function definition (PolydiscFunction) from efficient
computation (PolydiscFunctionEvaluator). Encodes full type information S, T, N
at compile time for optimization.

# Type Parameters
- `S`: Coefficient type (e.g., ValuedFieldPoint{P,Prec,PadicFieldElem})
- `T`: Radius type (typically Int)
- `N`: Dimension of polydisc space

# Design Philosophy
- **PolydiscFunction{S}**: "What is the function?" (mathematical definition)
- **PolydiscFunctionEvaluator{S,T,N}**: "How do we efficiently evaluate it?" (computation)

# Usage
Evaluators are callable structs created via `batch_evaluate_init`:
```julia
f = LinearPolynomial([K(1), K(2)], K(0))
eval = batch_evaluate_init(f, ValuationPolydisc{S,T,N})
result = eval(polydisc)  # Fully typed, no closures
```
"""
abstract type PolydiscFunctionEvaluator{S,T,N} end

# --- LinearPolynomial Evaluator ---
@doc raw"""
    LinearPolynomialEvaluator{S,T,N}

Typed evaluator for LinearPolynomial functions.

Precomputes coefficient valuations for efficient evaluation.
"""
struct LinearPolynomialEvaluator{S,T,N} <: PolydiscFunctionEvaluator{S,T,N}
    coefficients::NTuple{N,S}
    coeff_valuations::NTuple{N,Int}  # Precomputed valuations
    constant::S
end

function (eval::LinearPolynomialEvaluator{S,T,N})(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    # Compute constant term
    constant_term = eval.constant + sum(eval.coefficients[i] * p.center[i] for i in 1:N)

    # Find minimum valuation
    min_val = valuation(constant_term)
    for i in 1:N
        v = eval.coeff_valuations[i] + p.radius[i]
        v < min_val && (min_val = v)
    end

    # Compute absolute value
    return Float64(prime(p))^(-min_val)
end

# --- Constant Evaluator ---
@doc raw"""
    ConstantEvaluator{S,T,N}

Typed evaluator for Constant functions.
"""
struct ConstantEvaluator{S,T,N} <: PolydiscFunctionEvaluator{S,T,N}
    value::Float64
end

function (eval::ConstantEvaluator{S,T,N})(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.value
end

# --- Binary Operation Evaluators ---
struct AddEvaluator{S,T,N,L<:PolydiscFunctionEvaluator{S,T,N},R<:PolydiscFunctionEvaluator{S,T,N}} <: PolydiscFunctionEvaluator{S,T,N}
    left::L
    right::R
end

function (eval::AddEvaluator)(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.left(p) + eval.right(p)
end

struct SubEvaluator{S,T,N,L<:PolydiscFunctionEvaluator{S,T,N},R<:PolydiscFunctionEvaluator{S,T,N}} <: PolydiscFunctionEvaluator{S,T,N}
    left::L
    right::R
end

function (eval::SubEvaluator)(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.left(p) - eval.right(p)
end

struct MulEvaluator{S,T,N,L<:PolydiscFunctionEvaluator{S,T,N},R<:PolydiscFunctionEvaluator{S,T,N}} <: PolydiscFunctionEvaluator{S,T,N}
    left::L
    right::R
end

function (eval::MulEvaluator)(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.left(p) * eval.right(p)
end

struct DivEvaluator{S,T,N,L<:PolydiscFunctionEvaluator{S,T,N},R<:PolydiscFunctionEvaluator{S,T,N}} <: PolydiscFunctionEvaluator{S,T,N}
    top::L
    bottom::R
end

function (eval::DivEvaluator)(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.top(p) / eval.bottom(p)
end

# --- Scalar Multiplication Evaluator ---
struct SMulEvaluator{S,T,N,R<:PolydiscFunctionEvaluator{S,T,N}} <: PolydiscFunctionEvaluator{S,T,N}
    scalar::Float64
    right::R
end

function (eval::SMulEvaluator)(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.scalar * eval.right(p)
end

# --- Composition with Differentiable Real Function Evaluator ---
struct CompEvaluator{S,T,N,F<:DifferentiableFunction,R<:PolydiscFunctionEvaluator{S,T,N}} <: PolydiscFunctionEvaluator{S,T,N}
    outer::F
    inner::R
end

function (eval::CompEvaluator)(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.outer(eval.inner(p))
end

# --- Sum of Homogeneous Evaluators ---
struct SumEvaluator{S,T,N,E<:PolydiscFunctionEvaluator{S,T,N}} <: PolydiscFunctionEvaluator{S,T,N}
    evaluators::Vector{E}
end

function (eval::SumEvaluator)(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return sum(e(p) for e in eval.evaluators)
end

# --- Lambda Evaluator (wraps arbitrary functions with optional derivative) ---
struct LambdaEvaluator{S,T,N} <: PolydiscFunctionEvaluator{S,T,N}
    func::Function
    derivative::Union{Function, Nothing}
end
LambdaEvaluator{S,T,N}(func) where {S,T,N} = LambdaEvaluator{S,T,N}(func, nothing)

function (eval::LambdaEvaluator{S,T,N})(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return eval.func(p)
end

# --- MPoly Evaluator (wraps raw polynomial evaluation) ---
struct MPolyEvaluator{S,T,N,P<:AbstractAlgebra.Generic.MPoly} <: PolydiscFunctionEvaluator{S,T,N}
    poly::P
end

function (eval::MPolyEvaluator{S,T,N,P})(p::ValuationPolydisc{S,T,N}) where {S,T,N,P}
    return evaluate(eval.poly, p)
end

#=============================================================================
 New batch_evaluate_init Interface - Takes Type, Returns Typed Evaluator
=============================================================================#

@doc raw"""
    batch_evaluate_init(f::PolydiscFunction{S}, ::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}

Create a typed evaluator for efficient batch evaluation.

**NEW INTERFACE**: Takes a polydisc type parameter and returns a fully-typed callable struct
instead of an untyped closure. This enables compile-time specialization.

# Arguments
- `f::PolydiscFunction{S}`: The function to evaluate
- `::Type{ValuationPolydisc{S,T,N}}`: The polydisc type (determines T and N at compile time)

# Returns
`PolydiscFunctionEvaluator{S,T,N}`: A typed, callable evaluator struct

# Example
```julia
f = LinearPolynomial([K(1), K(2)], K(0))
eval = batch_evaluate_init(f, ValuationPolydisc{ValuedFieldPoint{2,20,PadicFieldElem},Int,2})
result = eval(some_polydisc)  # Fully specialized
```

# Design Benefits
- **Type stability**: S, T, N known at compile time
- **No closures**: Evaluators are concrete structs, not function objects
- **Inlining**: Julia can inline evaluator calls
- **Specialization**: Full method specialization on all type parameters
"""
function batch_evaluate_init(f::PolydiscFunction{S}, ::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    error("batch_evaluate_init not implemented for $(typeof(f)) with type ValuationPolydisc{$S,$T,$N}")
end

# Convenience: infer type from example polydisc
function batch_evaluate_init(f::PolydiscFunction{S}, p::ValuationPolydisc{S,T,N}) where {S,T,N}
    return batch_evaluate_init(f, ValuationPolydisc{S,T,N})
end

# --- Typed evaluator implementations for each function type ---

function batch_evaluate_init(poly::LinearPolynomial{S}, ::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    @assert length(poly.coefficients) == N "LinearPolynomial has $(length(poly.coefficients)) coefficients but polydisc has dimension $N"
    coefficients = ntuple(i -> poly.coefficients[i], N)
    coeff_valuations = ntuple(i -> Int(valuation(poly.coefficients[i])), N)
    return LinearPolynomialEvaluator{S,T,N}(coefficients, coeff_valuations, poly.constant)
end

function batch_evaluate_init(c::Constant{S}, ::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    return ConstantEvaluator{S,T,N}(Float64(c.value))
end

function batch_evaluate_init(f::Add{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    left = batch_evaluate_init(f.left, P)
    right = batch_evaluate_init(f.right, P)
    return AddEvaluator{S,T,N,typeof(left),typeof(right)}(left, right)
end

function batch_evaluate_init(f::Sub{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    left = batch_evaluate_init(f.left, P)
    right = batch_evaluate_init(f.right, P)
    return SubEvaluator{S,T,N,typeof(left),typeof(right)}(left, right)
end

function batch_evaluate_init(f::Mul{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    left = batch_evaluate_init(f.left, P)
    right = batch_evaluate_init(f.right, P)
    return MulEvaluator{S,T,N,typeof(left),typeof(right)}(left, right)
end

function batch_evaluate_init(f::Div{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    top = batch_evaluate_init(f.top, P)
    bottom = batch_evaluate_init(f.bottom, P)
    return DivEvaluator{S,T,N,typeof(top),typeof(bottom)}(top, bottom)
end

function batch_evaluate_init(f::SMul{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    right = batch_evaluate_init(f.right, P)
    return SMulEvaluator{S,T,N,typeof(right)}(Float64(f.left), right)
end

function batch_evaluate_init(f::Comp{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    inner = batch_evaluate_init(f.right, P)
    return CompEvaluator{S,T,N,typeof(f.left),typeof(inner)}(f.left, inner)
end

function batch_evaluate_init(l::Lambda{S}, ::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    return LambdaEvaluator{S,T,N}(l.func, l.derivative)
end

function batch_evaluate_init(f::LinearAbsolutePolynomialSum{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    evaluators = [batch_evaluate_init(poly, P) for poly in f.polys]
    E = eltype(evaluators)
    return SumEvaluator{S,T,N,E}(evaluators)
end

function batch_evaluate_init(poly::AbstractAlgebra.Generic.MPoly{S}, ::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    return MPolyEvaluator{S,T,N,typeof(poly)}(poly)
end

function batch_evaluate_init(f::AbsolutePolynomialSum{S}, P::Type{ValuationPolydisc{S,T,N}}) where {S,T,N}
    evaluators = [batch_evaluate_init(poly, P) for poly in f.polys]
    E = eltype(evaluators)
    return SumEvaluator{S,T,N,E}(evaluators)
end

#=============================================================================
 Legacy batch_evaluate_init Interface - Backwards Compatibility
=============================================================================#

@doc raw"""
    batch_evaluate_init(f::PolydiscFunction{S})::Function where S

Legacy interface returning untyped closures. Retained for backwards compatibility.

**DEPRECATED**: Use the typed interface `batch_evaluate_init(f, ::Type{ValuationPolydisc{S,T,N}})` instead.

This interface returns untyped closures which prevent compile-time specialization on T and N.
"""
function batch_evaluate_init(f::PolydiscFunction{S})::Function where S
    return batch_evaluate_init(f)
end

function batch_evaluate_init(poly::LinearPolynomial{S})::Function where S
    abs_poly_coeffs = map(valuation, poly.coefficients)
    num_coeffs = length(poly.coefficients)
    function eval(p::ValuationPolydisc{S,T,N}) where {T,N}
        # Only use the first num_coeffs coordinates (in case the polydisc is higher dimensional)
        constant_term = poly.constant + sum(poly.coefficients[i] * p.center[i] for i in 1:num_coeffs)
        # Compute valuations of all terms
        abs_values = [abs_poly_coeffs[i] + p.radius[i] for i in 1:num_coeffs]
        push!(abs_values, valuation(constant_term))
        # Compute the absolute value
        return Float64(prime(p))^(- minimum(abs_values))
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
    top_eval = batch_evaluate_init(f.top)
    bottom_eval = batch_evaluate_init(f.bottom)
    return p -> top_eval(p) / bottom_eval(p)
end

function batch_evaluate_init(f::SMul{S})::Function where S
    right_eval = batch_evaluate_init(f.right)
    return p -> f.left * right_eval(p)
end

function batch_evaluate_init(f::Comp{S})::Function where S
    right_eval = batch_evaluate_init(f.right)
    function eval(p::ValuationPolydisc{S,T,N}) where {T,N}
        return f.left(right_eval(p))
    end
    return eval
end

function batch_evaluate_init(f::Constant{S})::Function where S
    function eval(p::ValuationPolydisc{S,T,N}) where {T,N}
        return f.value
    end
    return eval
end

function batch_evaluate_init(l::Lambda{S})::Function where S
    return l.func
end

function batch_evaluate_init(f::LinearAbsolutePolynomialSum{S})::Function where S
    # Get the array of functions
    evaluation_functions::Array{Function} = map(batch_evaluate_init, f.polys)
    # Return the lambda function that sends `p`` to the sum of the evaluations of each element of `evaluation_functions` at `p`
    function eval(p::ValuationPolydisc{S,T,N}) where {T,N}
        return sum(
            map(f -> f(p), evaluation_functions))
    end
    return eval
end

function batch_evaluate_init(f::AbstractAlgebra.Generic.MPoly{S})::Function where S
    function eval(p::ValuationPolydisc{S,T,N}) where {T,N}
        return evaluate(f, p)
    end
    return eval
end

function batch_evaluate_init(f::AbsolutePolynomialSum{S})::Function where S
    # Get the array of functions
    evaluation_functions = map(batch_evaluate_init, f.polys)
    # Return the lambda function that sends `p`` to the sum of the evaluations of each element of `evaluation_functions` at `p`
    function eval(p::ValuationPolydisc{S,T,N}) where {T,N}
        return sum(map(f -> f(p), evaluation_functions))
    end
    return eval
end

# At the moment we work with multiple differential operators: the directional derivative along a tangent vector, and the gradient at a point.

@doc raw"""
    directional_exponent(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where {S, T}

Find the exponent vector(s) along which a polynomial achieves its maximum absolute value.

For a polynomial ``f`` and tangent vector ``v``, finds all exponent vectors ``n`` such that
locally in the direction of ``v``, ``|f| = a_n r^n`` for some coefficient ``a_n``.

The maximum of ``|a_n| \cdot p^{-\langle r, n \rangle}`` is equivalent to minimizing the
integer quantity ``v(a_n) + \langle r, n \rangle`` over non-zero terms.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: A multivariate polynomial
- `v::ValuationTangent{S,T}`: The tangent vector defining the direction

# Returns
`Vector`: The exponent vector with minimum valuation weight, breaking ties by minimizing `dot(n, v.magnitude)`
"""
function directional_exponent(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where {S, T}
    t = gens(f.parent)
    g = AbstractAlgebra.evaluate(f, t + v.direction)
    exp_vecs = collect(Nemo.exponent_vectors(g))
    # Compute v(a_n) + ⟨radius, n⟩ for each term. Sparse polynomial representation
    # guarantees all stored coefficients are nonzero, so no filtering is needed.
    # Minimizing this integer quantity is equivalent to maximizing |a_n| * p^{-⟨r,n⟩}.
    val_weights = [valuation(Nemo.coeff(g, i)) + sum(v.point.radius .* exp_vecs[i]) for i in eachindex(exp_vecs)]
    min_weight = minimum(val_weights)
    ties =  [exp_vecs[j] for j in findall(==(min_weight), val_weights)]
    return reduce((n, m) -> dot(n, v.magnitude) < dot(m, v.magnitude) ? n : m, ties)
end

@doc raw"""
    directional_derivative(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where {S, T}

Compute the directional derivative of a multivariate polynomial along a tangent direction.

Uses the formula: if locally ``|f| = a_n r^n`` for exponent ``n``, then
``d_v |f| = -|n| |a_n| r^n`` where ``r`` is the radius of the basepoint.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: The polynomial
- `v::ValuationTangent{S,T}`: The tangent direction

# Returns
`Float64`: The directional derivative
"""
function directional_derivative(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S,T}) where {S, T}
    # Recover the variables of the polynomial ring we're working over
    x = gens(f.parent)
    # Compute the expansion of f around the direction a of the tangent vector v, i.e.
    # The coefficients a_n such that f = ∑_n a_n (T-a)^n. We do this by computing the
    # expansion around 0 of the polynomial g(T) = f(T+a).
    g = AbstractAlgebra.evaluate(f, x + v.direction)
    # Next we need to compute the directional exponent of f along v
    n = directional_exponent(f, v)
    # Use the formula to get d_v
    d_v = -sum(n) * abs(coeff(g, n)) * (Float64(prime(v.point))^(-sum(v.point.radius .* n)))
    return d_v
end

@doc raw"""
    directional_derivative(poly::LinearPolynomial{S}, v::ValuationTangent{S,T,N}) where {S, T, N}

Compute the directional derivative of a linear polynomial along a tangent direction.

For `poly = Σᵢ aᵢTᵢ + b`, expanding around `v.direction` gives a constant term
`c₀ = b + Σᵢ aᵢdᵢ` and linear terms with coefficients `aᵢ`. The winning term is
found by minimizing the valuation weight `v(aₙ) + ⟨r, n⟩`, breaking ties via
`dot(n, v.magnitude)`. Since all non-constant terms have degree 1, the derivative is:
- `0` if the constant term wins (degree 0)
- `-p^{-best_val}` if a linear term wins

# Arguments
- `poly::LinearPolynomial{S}`: The linear polynomial
- `v::ValuationTangent{S,T,N}`: The tangent direction

# Returns
`Float64`: The directional derivative
"""
function directional_derivative(poly::LinearPolynomial{S}, v::ValuationTangent{S,T,N}) where {S, T, N}
    r = v.point.radius
    # Expand poly(T + direction): constant term c₀ = b + Σᵢ aᵢdᵢ, linear terms unchanged.
    c₀ = poly.constant + sum(poly.coefficients[i] * v.direction[i] for i in eachindex(poly.coefficients))
    # Initialise with the constant term (exponent 0: val_weight = v(c₀), mag_weight = 0).
    best_val = iszero(c₀) ? typemax(Int) : valuation(c₀)
    best_mag = Base.zero(T)
    linear_wins = false
    # Check each linear term (exponent eᵢ: val_weight = v(aᵢ) + rᵢ, mag_weight = magnitude[i]).
    for i in eachindex(poly.coefficients)
        aᵢ = poly.coefficients[i]
        iszero(aᵢ) && continue
        w   = valuation(aᵢ) + r[i]
        mag = v.magnitude[i]
        if w < best_val || (w == best_val && mag < best_mag)
            best_val  = w
            best_mag  = mag
            linear_wins = true
        end
    end
    # Constant term winning means degree 0, so derivative is 0.
    linear_wins || return 0.0
    return -Float64(prime(v.point))^(-best_val)
end

@doc raw"""
    directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{S,T}) where {S, T}

Compute the directional derivative of a polydisc function (sum of polynomials).

# Arguments
- `fun::PolydiscFunction{S}`: The polydisc function
- `v::ValuationTangent{S,T}`: The tangent direction

# Returns
`Float64`: Sum of directional derivatives across all polynomials
"""
function directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{S,T}) where {S, T}
    return sum([directional_derivative(f, v) for f in fun.polys])
end

@doc raw"""
    grad(f, v::ValuationTangent{S,T}) where {S, T}

Compute the gradient of a polynomial by evaluating directional derivatives along all coordinates.

# Arguments
- `f`: The polynomial or function
- `v::ValuationTangent{S,T}`: A reference tangent vector defining the space

# Returns
`Vector`: Gradient components, one for each coordinate direction
"""
function grad(f, v::ValuationTangent{S,T}) where {S, T}
    return [directional_derivative(f, basis_vector(v, i)) for i in Base.eachindex(v.magnitude)]
end

@doc raw"""
    partial_gradient(f, v::ValuationTangent{S,T}, gradient_indices) where {S, T}

Compute partial derivatives along specified coordinate directions.

# Arguments
- `f`: The polynomial or function
- `v::ValuationTangent{S,T}`: A reference tangent vector
- `gradient_indices`: Indices of coordinates for which to compute derivatives

# Returns
`Vector`: Directional derivatives for the specified coordinates
"""
function partial_gradient(f, v::ValuationTangent{S,T}, gradient_indices) where {S, T}
    return [directional_derivative(f, basis_vector(v, i)) for i in gradient_indices]
end

#=============================================================================
 Lifting Adapters for ValuedFieldPoint Integration

 When a PolydiscFunction{S} is used with a ValuationPolydisc{ValuedFieldPoint{P,Prec,S}},
 these adapters bridge the type gap. They work for any coefficient type S, not just
 PadicFieldElem, provided that S has the right methods (valuation, abs, etc).
=============================================================================#

@doc raw"""
    evaluate(fun::PolydiscFunction{S}, var::ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}) where {S,P,Prec,T,N}

Lifting adapter: evaluate a function with coefficients of type `S` on a polydisc with `ValuedFieldPoint{P,Prec,S}` coordinates.
Unwraps the polydisc and delegates to the standard evaluate method.
"""
function evaluate(fun::PolydiscFunction{S}, var::ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}) where {S,P,Prec,T,N}
    unwrapped_polydisc = ValuationPolydisc{S,T,N}(var.center |> unwrap, var.radius)
    return evaluate(fun, unwrapped_polydisc)
end

function batch_evaluate_init(poly::LinearPolynomial{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    new_coeffs = [VFP(c) for c in poly.coefficients]
    new_const = VFP(poly.constant)
    new_poly = LinearPolynomial(new_coeffs, new_const)
    return batch_evaluate_init(new_poly, ValuationPolydisc{VFP,T,N})
end

function batch_evaluate_init(f::LinearAbsolutePolynomialSum{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    new_polys = Vector{LinearPolynomial{VFP}}()
    for poly in f.polys
        new_coeffs = [VFP(c) for c in poly.coefficients]
        new_const = VFP(poly.constant)
        push!(new_polys, LinearPolynomial(new_coeffs, new_const))
    end
    new_f = LinearAbsolutePolynomialSum(new_polys)
    return batch_evaluate_init(new_f, ValuationPolydisc{VFP,T,N})
end

function batch_evaluate_init(f::Constant{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    return ConstantEvaluator{ValuedFieldPoint{P,Prec,S},T,N}(Float64(f.value))
end

function batch_evaluate_init(poly::AbstractAlgebra.Generic.MPoly{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    function wrapped_eval(p::ValuationPolydisc)
        unwrapped_polydisc = ValuationPolydisc{S,T,N}(p.center |> unwrap, p.radius)
        return evaluate(poly, unwrapped_polydisc)
    end
    function wrapped_deriv(v::ValuationTangent)
        unwrapped_point = ValuationPolydisc{S,T,N}(v.point.center |> unwrap, v.point.radius)
        unwrapped_direction = collect(unwrap(v.direction))
        unwrapped_tangent = ValuationTangent{S,T,N}(unwrapped_point, unwrapped_direction, v.magnitude)
        return directional_derivative(poly, unwrapped_tangent)
    end
    return LambdaEvaluator{VFP,T,N}(wrapped_eval, wrapped_deriv)
end

function batch_evaluate_init(f::AbsolutePolynomialSum{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    evaluators = [batch_evaluate_init(poly, ValuationPolydisc{VFP,T,N}) for poly in f.polys]
    E = eltype(evaluators)
    return SumEvaluator{VFP,T,N,E}(evaluators)
end

function batch_evaluate_init(f::Add{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    PT = ValuationPolydisc{VFP,T,N}
    left = batch_evaluate_init(f.left, PT)
    right = batch_evaluate_init(f.right, PT)
    return AddEvaluator{VFP,T,N,typeof(left),typeof(right)}(left, right)
end

function batch_evaluate_init(f::Sub{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    PT = ValuationPolydisc{VFP,T,N}
    left = batch_evaluate_init(f.left, PT)
    right = batch_evaluate_init(f.right, PT)
    return SubEvaluator{VFP,T,N,typeof(left),typeof(right)}(left, right)
end

function batch_evaluate_init(f::Mul{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    PT = ValuationPolydisc{VFP,T,N}
    left = batch_evaluate_init(f.left, PT)
    right = batch_evaluate_init(f.right, PT)
    return MulEvaluator{VFP,T,N,typeof(left),typeof(right)}(left, right)
end

function batch_evaluate_init(f::Div{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    PT = ValuationPolydisc{VFP,T,N}
    top = batch_evaluate_init(f.top, PT)
    bottom = batch_evaluate_init(f.bottom, PT)
    return DivEvaluator{VFP,T,N,typeof(top),typeof(bottom)}(top, bottom)
end

function batch_evaluate_init(f::SMul{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    PT = ValuationPolydisc{VFP,T,N}
    right = batch_evaluate_init(f.right, PT)
    return SMulEvaluator{VFP,T,N,typeof(right)}(Float64(f.left), right)
end

function batch_evaluate_init(f::Comp{S}, ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    PT = ValuationPolydisc{VFP,T,N}
    inner = batch_evaluate_init(f.right, PT)
    return CompEvaluator{VFP,T,N,typeof(f.left),typeof(inner)}(f.left, inner)
end

function directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{ValuedFieldPoint{P,Prec,S},T,N}) where {S,P,Prec,T,N}
    unwrapped_point = ValuationPolydisc{S,T,N}(v.point.center |> unwrap, v.point.radius)
    unwrapped_direction = collect(unwrap(v.direction))
    unwrapped_tangent = ValuationTangent{S,T,N}(unwrapped_point, unwrapped_direction, v.magnitude)
    return directional_derivative(fun, unwrapped_tangent)
end

function directional_derivative(poly::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{ValuedFieldPoint{P,Prec,S},T,N}) where {S,P,Prec,T,N}
    unwrapped_point = ValuationPolydisc{S,T,N}(v.point.center |> unwrap, v.point.radius)
    unwrapped_direction = collect(unwrap(v.direction))
    unwrapped_tangent = ValuationTangent{S,T,N}(unwrapped_point, unwrapped_direction, v.magnitude)
    return directional_derivative(poly, unwrapped_tangent)
end

#=============================================================================
 Directional Derivatives for Typed Evaluators
=============================================================================#

function directional_derivative(eval::ConstantEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    return 0.0
end

function directional_derivative(eval::AddEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    return directional_derivative(eval.left, v) + directional_derivative(eval.right, v)
end

function directional_derivative(eval::SubEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    return directional_derivative(eval.left, v) - directional_derivative(eval.right, v)
end

function directional_derivative(eval::MulEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    # Product rule: (f*g)' = f'*g(p) + f(p)*g'
    return directional_derivative(eval.left, v) * eval.right(v.point) +
           eval.left(v.point) * directional_derivative(eval.right, v)
end

function directional_derivative(eval::DivEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    # Quotient rule: (f/g)' = (f'g - fg') / g²
    f_val = eval.top(v.point)
    g_val = eval.bottom(v.point)
    f_deriv = directional_derivative(eval.top, v)
    g_deriv = directional_derivative(eval.bottom, v)
    return (f_deriv * g_val - f_val * g_deriv) / (g_val^2)
end

function directional_derivative(eval::SMulEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    return eval.scalar * directional_derivative(eval.right, v)
end

function directional_derivative(eval::SumEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    return sum(directional_derivative(e, v) for e in eval.evaluators)
end

function directional_derivative(eval::MPolyEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    return directional_derivative(eval.poly, v)
end

function directional_derivative(eval::CompEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    # Chain rule: (f ∘ g)' = f'(g(x)) * g'(x)
    inner_val = eval.inner(v.point)
    inner_deriv = directional_derivative(eval.inner, v)
    return eval.outer.df(inner_val) * inner_deriv
end

function directional_derivative(eval::LambdaEvaluator{S,T,N}, v::ValuationTangent{S,T,N}) where {S,T,N}
    eval.derivative === nothing && error("LambdaEvaluator has no derivative. Provide a derivative function at construction time.")
    return eval.derivative(v)
end