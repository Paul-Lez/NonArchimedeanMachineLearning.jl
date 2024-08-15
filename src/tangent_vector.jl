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

function Base.length(v::ValuationTangent)
    return length(v.point)
end 

# the zero tangent vector at a Berkovich PolyDisk P, in the direction of segment [P, Q]
function zero(P::ValuationPolydisc{S, T}, Q::Vector{S}) where S<:Oscar.scalar_types where T <:Oscar.scalar_types
    return ValuationTangent(P, Q, [Oscar.zero(T) for i in eachindex(Q)])
end

function zero(v::ValuationTangent{S, <:Oscar.scalar_types}) where S<:Oscar.scalar_types
    return zero(v.point, v.direction)
end 

function basis_vector(P::ValuationPolydisc{S, T}, Q::Vector{S}, i) where S<:Oscar.scalar_types where T<:Oscar.scalar_types
    return ValuationTangent(P, Q, [j == i ? Oscar.one(T) : Oscar.zero(T) for j in Base.eachindex(Q)])
end 

function basis_vector(v::ValuationTangent{S, <:Oscar.scalar_types}, i) where S<:Oscar.scalar_types
    return basis_vector(v.point, v.direction, i)
end 

## TODO: add code for operations
function Base.:+(P::ValuationTangent{S, T}, Q::ValuationTangent{S, T}) where S<:Oscar.scalar_types where T <:Oscar.scalar_types
    return ValuationTangent(P.point, P.direction, P.magnitude + Q.magnitude)
end
