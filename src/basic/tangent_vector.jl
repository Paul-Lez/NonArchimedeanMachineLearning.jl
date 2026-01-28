## This file will contain the code for tangent vectors
## Note: polydisc.jl is included by the main NAML module before this file

@doc raw"""
    ValuationTangent{S, T, N}

A tangent vector at a point in polydisc space.

Represents an element of the tangent space at a polydisc point. Unlike normalized tangent vectors,
this allows the tangent to have non-unit magnitude for each coordinate.

# Fields
- `point::ValuationPolydisc{S, T, N}`: The basepoint of the tangent vector
- `direction::Vector{S}`: The direction vector in p-adic space
- `magnitude::Vector{T}`: The magnitude of the tangent vector in each coordinate

# Type Parameters
- `S`: The type of the coordinate values (typically p-adic numbers)
- `T`: The type of the magnitude values (typically integers or rationals)
- `N`: The dimension of the polydisc space
"""
struct ValuationTangent{S, T, N} #where T<:Oscar.scalar_types
    point::ValuationPolydisc{S, T, N}
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
    zero(P::ValuationPolydisc{S, T, N}, Q::Vector{S}) where {S,T,N}

Create the zero tangent vector at a polydisc pointing in a specified direction.

# Arguments
- `P::ValuationPolydisc{S, T, N}`: The basepoint polydisc
- `Q::Vector{S}`: The direction vector

# Returns
`ValuationTangent{S, T, N}`: Zero tangent vector at `P` in direction of segment `[P, Q]`
"""
function zero(P::ValuationPolydisc{S, T, N}, Q::Vector{S}) where {S, T, N}
    return ValuationTangent(P, Q, [Oscar.zero(T) for i in eachindex(Q)])
end

@doc raw"""
    zero(v::ValuationTangent{S, T, N}) where {S,T,N}

Create the zero tangent vector with the same basepoint and direction as a given tangent vector.

# Arguments
- `v::ValuationTangent{S, T, N}`: The reference tangent vector

# Returns
`ValuationTangent{S, T, N}`: Zero tangent vector with same basepoint and direction as `v`
"""
function zero(v::ValuationTangent{S, T, N}) where {S, T, N}
    return zero(v.point, v.direction)
end

@doc raw"""
    basis_vector(P::ValuationPolydisc{S, T, N}, Q::Vector{S}, i) where {S,T,N}

Create the i-th standard basis tangent vector.

# Arguments
- `P::ValuationPolydisc{S, T, N}`: The basepoint polydisc
- `Q::Vector{S}`: The direction vector
- `i`: The coordinate index for the basis vector

# Returns
`ValuationTangent{S, T, N}`: Standard basis vector with magnitude 1 in coordinate `i` and 0 elsewhere
"""
function basis_vector(P::ValuationPolydisc{S, T, N}, Q::Vector{S}, i) where {S, T, N}
    return ValuationTangent(P, Q, [j == i ? Oscar.one(T) : Oscar.zero(T) for j in Base.eachindex(Q)])
end

@doc raw"""
    basis_vector(v::ValuationTangent{S, T, N}, i) where {S,T,N}

Create the i-th standard basis vector with the same basepoint and direction as a given tangent vector.

# Arguments
- `v::ValuationTangent{S, T, N}`: The reference tangent vector
- `i`: The coordinate index for the basis vector

# Returns
`ValuationTangent{S, T, N}`: Standard basis vector with same basepoint and direction as `v`
"""
function basis_vector(v::ValuationTangent{S, T, N}, i) where {S, T, N}
    return basis_vector(v.point, v.direction, i)
end

@doc raw"""
    Base.:+(P::ValuationTangent{S, T, N}, Q::ValuationTangent{S, T, N}) where {S,T,N}

Add two tangent vectors at the same basepoint and direction.

# Arguments
- `P::ValuationTangent{S, T, N}`: First tangent vector
- `Q::ValuationTangent{S, T, N}`: Second tangent vector (must have same basepoint and direction)

# Returns
`ValuationTangent{S, T, N}`: Sum of the two tangent vectors (magnitudes are added componentwise)
"""
function Base.:+(P::ValuationTangent{S, T, N}, Q::ValuationTangent{S, T, N}) where {S, T, N}
    return ValuationTangent(P.point, P.direction, P.magnitude + Q.magnitude)
end

@doc raw"""
    ValuationTangent(P::ValuationPolydisc{S, T, N}, Q::ValuationPolydisc{S, T, N}) where {S,T,N}

Create a tangent vector corresponding to the straight line from polydisc P to polydisc Q.

# Arguments
- `P::ValuationPolydisc{S, T, N}`: The starting polydisc (basepoint)
- `Q::ValuationPolydisc{S, T, N}`: The target polydisc

# Returns
`ValuationTangent{S, T, N}`: Tangent vector from `P` toward `Q`

# Notes
Currently returns the zero tangent - full implementation is TODO.
"""
function ValuationTangent(P::ValuationPolydisc{S, T, N}, Q::ValuationPolydisc{S, T, N}) where {S, T, N}
    # TODO: implement this!
    return zero(P, collect(Q.center))
end 
