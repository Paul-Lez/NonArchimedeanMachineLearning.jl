## This file will contain the code for tangent vectors
include("polydisc.jl")

# BerkovichTagent represents an element of the tangent space at a Berkovich point. Here we allow the tangent to have a magnitude instead of normalising it to 1.
struct ValuationTangent{S, T} #where T<:Oscar.scalar_types
    point::ValuationPolydisc{S, T}
    direction::Vector{S}
    # Possibly we should allow the magnitude to have a different type?
    # not sure if that would ever come up 
    magnitude::Vector{T}
end

# Returns the dimension of the tangent vector `v`
function dim(v::ValuationTangent)
    return dim(v.point)
end

# the zero tangent vector at a Berkovich PolyDisk P, in the direction of segment [P, Q]
function zero(P::ValuationPolydisc{S, T}, Q::Vector{S}) where S where T
    return ValuationTangent(P, Q, [Oscar.zero(T) for i in eachindex(Q)])
end

# Outputs the zero tangent vector with same basepoint and direction as `v`
function zero(v::ValuationTangent{S, T}) where S where T
    return zero(v.point, v.direction)
end 

# Returns the i-th basis vector with basepoint `P` and direction `Q`
function basis_vector(P::ValuationPolydisc{S, T}, Q::Vector{S}, i) where S where T
    return ValuationTangent(P, Q, [j == i ? Oscar.one(T) : Oscar.zero(T) for j in Base.eachindex(Q)])
end 

# Returns the i-th standard basis vector with same basepoint and direction as `v`
function basis_vector(v::ValuationTangent{S, T}, i) where S where T
    return basis_vector(v.point, v.direction, i)
end 

## TODO: add code for other operations
function Base.:+(P::ValuationTangent{S, T}, Q::ValuationTangent{S, T}) where S where T
    return ValuationTangent(P.point, P.direction, P.magnitude + Q.magnitude)
end

# Tangent corresponding to straight line from P to Q
function ValuationTangent(P::ValuationPolydisc{S, T}, Q::ValuationPolydisc{S, T}) where S where T
    # TODO: implement this!
    return zero(P, Q.center)
end 
