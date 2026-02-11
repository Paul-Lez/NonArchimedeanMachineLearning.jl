################### VALUED FIELD POINT WRAPPER ###################

# This file provides a wrapper type that encodes field parameters
# (prime, precision) at the type level for compile-time optimization.
#
# The motivation is that ValuationPolydisc{S,T,N} where S is a raw
# PadicFieldElem doesn't know the prime at compile time. Hot-path
# functions like children() call prime(p) at runtime (~100K calls
# in some runs). By wrapping the element, we make P available to
# the compiler.
#
# Usage:
#   K = PadicField(2, 20)
#   x = ValuedFieldPoint(K(1))  # Automatically extracts P=2, Prec=20
#   prime(x)           # Returns 2 (compile-time constant)
#   precision(x)       # Returns 20 (compile-time constant)
#
# The wrapper delegates arithmetic to the underlying Oscar type.

####################################################################

## TODO: make this into a proper ring in the sense of AbstractAlgebra.jl.

@doc raw"""
    ValuedFieldPoint{P,Prec,S}

A wrapper around valued field elements that encodes the prime `P` and
precision `Prec` at the type level for compile-time optimization.

# Type Parameters
- `P`: The prime of the underlying p-adic field (compile-time constant)
- `Prec`: The precision of the underlying p-adic field (compile-time constant)
- `S`: The type of the underlying field element (e.g., `PadicFieldElem`)

# Fields
- `elem::S`: The underlying field element

# Example
```julia
K = PadicField(2, 20)
x = ValuedFieldPoint(K(5))
prime(x)       # Returns 2
precision(x)   # Returns 20
x + x          # Works like normal arithmetic
```

# Notes
With this wrapper, `ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}` knows
`P` at compile time, enabling loop optimizations in hot-path functions.
"""
struct ValuedFieldPoint{P,Prec,S} <: RingElem
    elem::S
end

#=============================================================================
 Type-Level Accessors (Compile-Time)
=============================================================================#

@doc raw"""
    prime(::Type{ValuedFieldPoint{P,Prec,S}}) where {P,Prec,S}
    prime(::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}

Return the prime of the underlying p-adic field.

This is a compile-time constant available via the type parameter `P`.
"""
@inline prime(::Type{ValuedFieldPoint{P,Prec,S}}) where {P,Prec,S} = P
@inline prime(::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} = P

@doc raw"""
    precision(::Type{ValuedFieldPoint{P,Prec,S}}) where {P,Prec,S}
    precision(::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}

Return the precision of the underlying p-adic field.

This is a compile-time constant available via the type parameter `Prec`.
"""
@inline Base.precision(::Type{ValuedFieldPoint{P,Prec,S}}) where {P,Prec,S} = Prec
@inline Base.precision(::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} = Prec

#=============================================================================
 Constructors
=============================================================================#

@doc raw"""
    ValuedFieldPoint(x::PadicFieldElem)

Construct a `ValuedFieldPoint` from a `PadicFieldElem`, automatically extracting
the prime and precision from the parent field.

# Example
```julia
K = PadicField(2, 20)
x = ValuedFieldPoint(K(5))  # ValuedFieldPoint{2,20,PadicFieldElem}(...)
```
"""
function ValuedFieldPoint(x::PadicFieldElem)
    K = Base.parent(x)
    P = Int(Nemo.prime(K))
    Prec = Int(Oscar.precision(K))
    return ValuedFieldPoint{P,Prec,PadicFieldElem}(x)
end

@doc raw"""
    ValuedFieldPoint{P,Prec}(x::S) where {P,Prec,S}

Construct a `ValuedFieldPoint` with explicit type parameters.
Useful when you already know P and Prec.
"""
function ValuedFieldPoint{P,Prec}(x::S) where {P,Prec,S}
    return ValuedFieldPoint{P,Prec,S}(x)
end

#=============================================================================
 Unwrapping (Oscar Interop)
=============================================================================#

@doc raw"""
    unwrap(x::ValuedFieldPoint)

Extract the underlying field element from a `ValuedFieldPoint`.

# Example
```julia
K = PadicField(2, 20)
x = ValuedFieldPoint(K(5))
unwrap(x)  # Returns the PadicFieldElem
```
"""
unwrap(x::ValuedFieldPoint) = x.elem

@doc raw"""
    unwrap(v::NTuple{N,<:ValuedFieldPoint}) where N

Unwrap a tuple of `ValuedFieldPoint`s into a tuple of underlying elements.
"""
unwrap(v::NTuple{N,<:ValuedFieldPoint}) where N = ntuple(i -> v[i].elem, N)

@doc raw"""
    unwrap(v::Vector{<:ValuedFieldPoint})

Unwrap a vector of `ValuedFieldPoint`s into a vector of underlying elements.
"""
unwrap(v::Vector{<:ValuedFieldPoint}) = [x.elem for x in v]

#=============================================================================
 Parent Field Access
=============================================================================#

@doc raw"""
    Base.parent(x::ValuedFieldPoint)

Return the parent field of the underlying element.
"""
Base.parent(x::ValuedFieldPoint) = Base.parent(x.elem)

#=============================================================================
 Arithmetic Operations (Delegate to Oscar)
=============================================================================#

Base.:+(a::ValuedFieldPoint{P,Prec,S}, b::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(a.elem + b.elem)

Base.:-(a::ValuedFieldPoint{P,Prec,S}, b::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(a.elem - b.elem)

Base.:-(a::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(-a.elem)

Base.:*(a::ValuedFieldPoint{P,Prec,S}, b::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(a.elem * b.elem)

# Division using Oscar's divexact (exact division in p-adic fields)
Base.:/(a::ValuedFieldPoint{P,Prec,S}, b::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(divexact(a.elem, b.elem))

Base.://(a::ValuedFieldPoint{P,Prec,S}, b::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(divexact(a.elem, b.elem))

#=============================================================================
 Mixed-Type Operations (PadicFieldElem ↔ ValuedFieldPoint)
=============================================================================#

# These are essential for Oscar's polynomial evaluation: when evaluating
# MPoly{PadicFieldElem} at ValuedFieldPoint values, coefficients (PadicFieldElem)
# must interact with substituted values (ValuedFieldPoint).

Base.:+(a::ValuedFieldPoint{P,Prec,PadicFieldElem}, b::PadicFieldElem) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(a.elem + b)

Base.:+(a::PadicFieldElem, b::ValuedFieldPoint{P,Prec,PadicFieldElem}) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(a + b.elem)

Base.:-(a::ValuedFieldPoint{P,Prec,PadicFieldElem}, b::PadicFieldElem) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(a.elem - b)

Base.:-(a::PadicFieldElem, b::ValuedFieldPoint{P,Prec,PadicFieldElem}) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(a - b.elem)

Base.:*(a::ValuedFieldPoint{P,Prec,PadicFieldElem}, b::PadicFieldElem) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(a.elem * b)

Base.:*(a::PadicFieldElem, b::ValuedFieldPoint{P,Prec,PadicFieldElem}) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(a * b.elem)

Base.:/(a::ValuedFieldPoint{P,Prec,PadicFieldElem}, b::PadicFieldElem) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(divexact(a.elem, b))

Base.:/(a::PadicFieldElem, b::ValuedFieldPoint{P,Prec,PadicFieldElem}) where {P,Prec} =
    ValuedFieldPoint{P,Prec,PadicFieldElem}(divexact(a, b.elem))

#=============================================================================
 Scalar Operations (Integer ↔ ValuedFieldPoint)
=============================================================================#

Base.:*(a::Integer, b::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(a * b.elem)

Base.:*(a::ValuedFieldPoint{P,Prec,S}, b::Integer) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(a.elem * b)

Base.:^(a::ValuedFieldPoint{P,Prec,S}, n::Integer) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(a.elem^n)

#=============================================================================
 Comparison
=============================================================================#

Base.:(==)(a::ValuedFieldPoint{P,Prec,S}, b::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    a.elem == b.elem

# Allow comparison with unwrapped elements
Base.:(==)(a::ValuedFieldPoint{P,Prec,PadicFieldElem}, b::PadicFieldElem) where {P,Prec} =
    a.elem == b

Base.:(==)(a::PadicFieldElem, b::ValuedFieldPoint{P,Prec,PadicFieldElem}) where {P,Prec} =
    a == b.elem

Base.hash(x::ValuedFieldPoint, h::UInt) = hash(x.elem, h)

Base.iszero(x::ValuedFieldPoint) = iszero(x.elem)
Base.isone(x::ValuedFieldPoint) = isone(x.elem)

#=============================================================================
 Valuation and Absolute Value
=============================================================================#

@doc raw"""
    valuation(x::ValuedFieldPoint)

Compute the valuation of a wrapped valued field element.
Delegates to the underlying element's valuation.
"""
valuation(x::ValuedFieldPoint) = valuation(x.elem)

@doc raw"""
    Base.abs(x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}

Compute the absolute value of a wrapped valued field element.

Uses the type-level prime `P` for efficiency: `|x|_p = P^(-v(x))`
"""
Base.abs(x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} = Float64(P)^(-valuation(x.elem))

#=============================================================================
 Zero and One
=============================================================================#

@doc raw"""
    Base.zero(x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}

Return the zero element in the same field as `x`.
"""
Base.zero(x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(Base.parent(x.elem)(0))

@doc raw"""
    Base.one(x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}

Return the one element in the same field as `x`.
"""
Base.one(x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(Base.parent(x.elem)(1))

@doc raw"""
    Oscar.zero(::Type{ValuedFieldPoint{P,Prec,S}}, K) where {P,Prec,S}

Return the zero element of the valued field `K` wrapped as a `ValuedFieldPoint`.
Note: requires the field K to be passed since type alone doesn't contain field reference.
"""
Oscar.zero(::Type{ValuedFieldPoint{P,Prec,S}}, K) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(zero(K))

@doc raw"""
    Oscar.one(::Type{ValuedFieldPoint{P,Prec,S}}, K) where {P,Prec,S}

Return the one element of the valued field `K` wrapped as a `ValuedFieldPoint`.
Note: requires the field K to be passed since type alone doesn't contain field reference.
"""
Oscar.one(::Type{ValuedFieldPoint{P,Prec,S}}, K) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(one(K))

#=============================================================================
 Display
=============================================================================#

function Base.show(io::IO, x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}
    print(io, "ValuedFieldPoint{", P, ",", Prec, "}(", x.elem, ")")
end

#=============================================================================
 Conversion and Promotion
=============================================================================#

# Allow implicit conversion from the underlying type
Base.convert(::Type{ValuedFieldPoint{P,Prec,S}}, x::S) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}(x)

# Promote operations between wrapped and unwrapped
Base.promote_rule(::Type{ValuedFieldPoint{P,Prec,S}}, ::Type{S}) where {P,Prec,S} =
    ValuedFieldPoint{P,Prec,S}

#=============================================================================
 Lifting Functions
=============================================================================#

@doc raw"""
    lift(v::Vector{S}) where S<:PadicFieldElem

Lift a vector of p-adic field elements to `ValuedFieldPoint` wrappers.
All elements must come from the same field.

# Example
```julia
K = PadicField(2, 20)
v = [K(1), K(2), K(3)]
lifted = lift(v)  # Vector{ValuedFieldPoint{2,20,PadicFieldElem}}
```
"""
function lift(v::Vector{S}) where S<:PadicFieldElem
    isempty(v) && error("Cannot lift empty vector - need field information")
    K = Base.parent(first(v))
    P = Int(Nemo.prime(K))
    Prec = Int(Oscar.precision(K))
    return [ValuedFieldPoint{P,Prec,S}(x) for x in v]
end

@doc raw"""
    lift(t::NTuple{N,S}) where {N,S<:PadicFieldElem}

Lift a tuple of p-adic field elements to `ValuedFieldPoint` wrappers.
"""
function lift(t::NTuple{N,S}) where {N,S<:PadicFieldElem}
    N == 0 && error("Cannot lift empty tuple - need field information")
    K = Base.parent(first(t))
    P = Int(Nemo.prime(K))
    Prec = Int(Oscar.precision(K))
    return ntuple(i -> ValuedFieldPoint{P,Prec,S}(t[i]), N)
end

@doc raw"""
    lift(R::ZZRing, x::ValuedFieldPoint{P,Prec,S})

Lift a `ValuedFieldPoint` to the integer ring ZZ by delegating to the underlying element.
This is used primarily by `canonical_center` for polydisc hashing.

# Example
```julia
K = PadicField(2, 20)
x = ValuedFieldPoint{2,20,PadicFieldElem}(K(5))
lifted = lift(ZZ, x)  # ZZRingElem
```
"""
function lift(R::ZZRing, x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}
    return Oscar.lift(R, x.elem)
end

@doc raw"""
    lift(R::ZZRing, x::S) where S<:PadicFieldElem

Lift a plain p-adic field element to the integer ring ZZ.
This handles polydiscs with unwrapped PadicFieldElem centers.

# Example
```julia
K = PadicField(2, 20)
lifted = lift(ZZ, K(5))  # ZZRingElem
```
"""
function lift(R::ZZRing, x::S) where S<:PadicFieldElem
    return Oscar.lift(R, x)
end

@doc raw"""
    lift(x::ValuedFieldPoint{P,Prec,S})

Scalar lift for `ValuedFieldPoint` (delegates to underlying element).
Provided for consistency with Oscar's lift interface.
"""
function lift(x::ValuedFieldPoint{P,Prec,S}) where {P,Prec,S}
    return Oscar.lift(x.elem)
end

