## This file will contain the code for tangent vectors
include("polydisc.jl")

@doc raw"""
    ValuationTangent{S, T}

A tangent vector at a point in polydisc space.

Represents an element of the tangent space at a polydisc point. Unlike normalized tangent vectors,
this allows the tangent to have non-unit magnitude for each coordinate.

# Fields
- `point::ValuationPolydisc{S, T}`: The basepoint of the tangent vector
- `direction::Vector{S}`: The direction vector in p-adic space
- `magnitude::Vector{T}`: The magnitude of the tangent vector in each coordinate

# Type Parameters
- `S`: The type of the coordinate values (typically p-adic numbers)
- `T`: The type of the magnitude values (typically integers or rationals)
"""
struct ValuationTangent{S, T} #where T<:Oscar.scalar_types
    point::ValuationPolydisc{S, T}
    direction::Vector{S}
    # Possibly we should allow the magnitude to have a different type?
    # not sure if that would ever come up
    magnitude::Vector{T}
end

@doc raw"""
    dim(v::ValuationTangent)

Return the dimension of the tangent space.

# Arguments
- `v::ValuationTangent`: The tangent vector

# Returns
`Int`: The dimension (equal to the dimension of the basepoint polydisc)
"""
function dim(v::ValuationTangent)
    return dim(v.point)
end

@doc raw"""
    zero(P::ValuationPolydisc{S, T}, Q::Vector{S}) where {S,T}

Create the zero tangent vector at a polydisc pointing in a specified direction.

# Arguments
- `P::ValuationPolydisc{S, T}`: The basepoint polydisc
- `Q::Vector{S}`: The direction vector

# Returns
`ValuationTangent{S, T}`: Zero tangent vector at `P` in direction of segment `[P, Q]`
"""
function zero(P::ValuationPolydisc{S, T}, Q::Vector{S}) where {S, T}
    return ValuationTangent(P, Q, [Oscar.zero(T) for i in eachindex(Q)])
end

@doc raw"""
    zero(v::ValuationTangent{S, T}) where {S,T}

Create the zero tangent vector with the same basepoint and direction as a given tangent vector.

# Arguments
- `v::ValuationTangent{S, T}`: The reference tangent vector

# Returns
`ValuationTangent{S, T}`: Zero tangent vector with same basepoint and direction as `v`
"""
function zero(v::ValuationTangent{S, T}) where {S, T}
    return zero(v.point, v.direction)
end 

@doc raw"""
    basis_vector(P::ValuationPolydisc{S, T}, Q::Vector{S}, i) where {S,T}

Create the i-th standard basis tangent vector.

# Arguments
- `P::ValuationPolydisc{S, T}`: The basepoint polydisc
- `Q::Vector{S}`: The direction vector
- `i`: The coordinate index for the basis vector

# Returns
`ValuationTangent{S, T}`: Standard basis vector with magnitude 1 in coordinate `i` and 0 elsewhere
"""
function basis_vector(P::ValuationPolydisc{S, T}, Q::Vector{S}, i) where {S, T}
    return ValuationTangent(P, Q, [j == i ? Oscar.one(T) : Oscar.zero(T) for j in Base.eachindex(Q)])
end 

@doc raw"""
    basis_vector(v::ValuationTangent{S, T}, i) where {S,T}

Create the i-th standard basis vector with the same basepoint and direction as a given tangent vector.

# Arguments
- `v::ValuationTangent{S, T}`: The reference tangent vector
- `i`: The coordinate index for the basis vector

# Returns
`ValuationTangent{S, T}`: Standard basis vector with same basepoint and direction as `v`
"""
function basis_vector(v::ValuationTangent{S, T}, i) where {S, T}
    return basis_vector(v.point, v.direction, i)
end 

@doc raw"""
    Base.:+(P::ValuationTangent{S, T}, Q::ValuationTangent{S, T}) where {S,T}

Add two tangent vectors at the same basepoint and direction.

# Arguments
- `P::ValuationTangent{S, T}`: First tangent vector
- `Q::ValuationTangent{S, T}`: Second tangent vector (must have same basepoint and direction)

# Returns
`ValuationTangent{S, T}`: Sum of the two tangent vectors (magnitudes are added componentwise)
"""
function Base.:+(P::ValuationTangent{S, T}, Q::ValuationTangent{S, T}) where {S, T}
    return ValuationTangent(P.point, P.direction, P.magnitude + Q.magnitude)
end

@doc raw"""
    ValuationTangent(P::ValuationPolydisc{S, T}, Q::ValuationPolydisc{S, T}) where {S,T}

Create a tangent vector corresponding to the straight line from polydisc P to polydisc Q.

# Arguments
- `P::ValuationPolydisc{S, T}`: The starting polydisc (basepoint)
- `Q::ValuationPolydisc{S, T}`: The target polydisc

# Returns
`ValuationTangent{S, T}`: Tangent vector from `P` toward `Q`

# Notes
Currently returns the zero tangent - full implementation is TODO.
"""
function ValuationTangent(P::ValuationPolydisc{S, T}, Q::ValuationPolydisc{S, T}) where {S, T}
    # TODO: implement this!
    return zero(P, Q.center)
end 
