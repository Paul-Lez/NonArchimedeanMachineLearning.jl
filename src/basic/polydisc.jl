################### POLYDISCS ###################

# This file sets up the basic structures and
# API to work with polydiscs over the p-adics
# Our convention is that the radius is always
# measured wrt the valuation rather than the
# absolute value.

##################################################

@doc raw"""
    ValuationPolydisc{S,T,N}

A polydisc over a p-adic field, where the radius is measured with respect to valuation.

Represents the product of p-adic discs ``B(c_1, r_1) \times \cdots \times B(c_n, r_n)``
where each disc ``B(c_i, r_i) = \{x : v(x - c_i) \geq r_i\}`` with ``v`` the p-adic valuation.

# Fields
- `center::NTuple{N,S}`: The center point of the polydisc in p-adic space
- `radius::NTuple{N,T}`: The radius vector measured in valuation coordinates

# Type Parameters
- `S`: The type of the center coordinates (typically p-adic numbers)
- `T`: The type of the radius values (typically integers or rationals)
- `N`: The dimension of the polydisc
"""
struct ValuationPolydisc{S,T,N}
    center::NTuple{N,S}
    # For valued points, the radius is measured with respect to the valuation
    radius::NTuple{N,T}
end

# Convenience constructor that accepts vectors and converts to tuples
function ValuationPolydisc(center::Vector{S}, radius::Vector{T}) where {S,T}
    N = length(center)
    @assert length(radius) == N "center and radius must have the same length"
    return ValuationPolydisc{S,T,N}(tuple(center...), tuple(radius...))
end

@doc raw"""
    AbsPolydisc{S,T}

A polydisc over a p-adic field, where the radius is measured with respect to the norm.

Represents the product of p-adic discs where each disc is defined using the p-adic absolute value
rather than valuation.

Note: we generally prefer the valuation format.

# Fields
- `center::Vector{S}`: The center point of the polydisc in p-adic space
- `radius::Vector{T}`: The radius vector measured in norm coordinates

# Type Parameters
- `S`: The type of the center coordinates (typically p-adic numbers)
- `T`: The type of the radius values (typically real numbers)
"""
struct AbsPolydisc{S,T}
    center::Vector{S}
    # For normed points, the radius is measured with respect to the norm
    radius::Vector{T}
end

@doc raw"""
    Base.eachindex(p::ValuationPolydisc)

Return an iterator over the valid indices of the polydisc coordinates.

# Arguments
- `p::ValuationPolydisc`: The polydisc

# Returns
Iterator over indices `1:dim(p)`
"""
function Base.eachindex(p::ValuationPolydisc)
    return Base.eachindex(p.center)
end

@doc raw"""
    center(p::ValuationPolydisc)

Extract the center point of a polydisc.

# Arguments
- `p::ValuationPolydisc`: The polydisc

# Returns
`Vector{S}`: The center coordinates of the polydisc
"""
function center(p::ValuationPolydisc)
    return p.center
end

@doc raw"""
    radius(p::ValuationPolydisc)

Extract the radius vector of a polydisc.

# Arguments
- `p::ValuationPolydisc`: The polydisc

# Returns
`Vector{T}`: The radius values (in valuation coordinates) for each dimension
"""
function radius(p::ValuationPolydisc)
    return p.radius
end

@doc raw"""
    base_ring(p::ValuationPolydisc)

Get the underlying p-adic field of the polydisc.

# Arguments
- `p::ValuationPolydisc`: The polydisc

# Returns
The parent ring (p-adic field) of the center coordinates
"""
function base_ring(p::ValuationPolydisc)
    return Base.parent(first(center(p)))
end

import Base.show
@doc raw"""
    Base.show(io::IO, p::ValuationPolydisc)

Display a polydisc in human-readable format.

# Arguments
- `io::IO`: The output stream
- `p::ValuationPolydisc`: The polydisc to display

Prints the base ring, center, and radius information.
"""
function Base.show(io::IO, p::ValuationPolydisc)
    print(io, "Polydisc over $(base_ring(p)) with center $(center(p)) and radius $(radius(p))")
end

import Base.==
@doc raw"""
    Base.:(==)(p::ValuationPolydisc, q::ValuationPolydisc)

Check equality of two polydiscs in Berkovich space.

Two polydiscs D(a, r) and D(b, s) represent the same Berkovich point if and only if:
1. They have the same radius: r = s
2. Their centers are "close enough": v(a - b) >= r

This is the correct condition for Berkovich equality in non-Archimedean geometry.

# Arguments
- `p::ValuationPolydisc`: First polydisc
- `q::ValuationPolydisc`: Second polydisc

# Returns
`Bool`: `true` if the polydiscs represent the same Berkovich point, `false` otherwise

# Mathematical Note
The condition v(a-b) >= r (not > r) is crucial: if v(a-b) = r exactly, then
D(a, r) and D(b, r) are the same disc by the ultrametric property.
"""
function Base.:(==)(p::ValuationPolydisc{S,T,N}, q::ValuationPolydisc{S,T,N}) where {S,T,N}
    # Check if radii are equal
    if radius(p) != radius(q)
        return false
    end

    # Check if centers represent the same Berkovich point
    # Two centers with the same radius r are equal if v(a-b) >= r
    return all(valuation.(center(p) .- center(q)) .>= radius(p))
end

@doc raw"""
    Base.isequal(p::ValuationPolydisc, q::ValuationPolydisc)

Check equality of two polydiscs for use in hash-based collections (Dict, Set).

This must be consistent with `hash`: if `isequal(p, q)` then `hash(p) == hash(q)`.
We use the same definition as `==` to ensure Dict lookups work correctly.

# Note
Julia's Dict uses `isequal` (not `==`) for key comparison. Without this definition,
Dict would use `===` (identity), which is too strict for our purposes.
"""
function Base.isequal(p::ValuationPolydisc{S,T,N}, q::ValuationPolydisc{S,T,N}) where {S,T,N}
    return p == q
end

@doc raw"""
    canonical_center(p::ValuationPolydisc)

Compute a canonical representation of the polydisc center for hashing purposes.

For each coordinate i, computes `center[i] mod p^radius[i]` as an integer tuple.
This ensures that polydiscs representing the same region (same radius and
centers differing by elements with valuation > radius) have the same canonical form.

# Arguments
- `p::ValuationPolydisc`: The polydisc

# Returns
`NTuple{N, BigInt}`: Canonical integer representation of the center
"""
function canonical_center(p::ValuationPolydisc{S,T,N}) where {S,T,N}
    pr = Int(prime(p))
    return ntuple(N) do i
        # Get the center as a rational and extract the p-adic digits up to radius[i]
        # The canonical form is center mod p^radius[i]
        c = p.center[i]
        r = p.radius[i]
        # Compute p^r as the modulus
        modulus = BigInt(pr)^r
        # Lift the p-adic number to an integer (truncated to precision)
        # For p-adic numbers, we can use lift to get the representative integer
        lifted = BigInt(lift(ZZ, c))
        # Take mod to get canonical form
        mod(lifted, modulus)
    end
end

@doc raw"""
    Base.hash(m::ValuationPolydisc, h::UInt)

Compute a hash value for a polydisc for use in hash-based collections.

Uses a canonical representation based on radius and center mod p^radius to ensure
that equivalent polydiscs (same radius, centers differing by high-valuation elements)
hash to the same value. This is essential for transposition table lookups in DAG-MCTS.

# Arguments
- `m::ValuationPolydisc`: The polydisc to hash
- `h::UInt`: The initial hash value

# Returns
`UInt`: The hash value
"""
function Base.hash(m::ValuationPolydisc{S,T,N}, h::UInt) where {S,T,N}
    # Hash based on radius and canonical center representation
    # Two polydiscs are equal iff they have same radius and centers differ
    # by elements with valuation > radius, so canonical_center captures this
    return hash((m.radius, canonical_center(m)), h)
end


# Some of the code (e.g the function below) might be nicer if we can use some unifying type
# E.g. Polydisk

@doc raw"""
    prime(p::ValuationPolydisc)

Return the prime number of the p-adic field underlying the polydisc.

# Arguments
- `p::ValuationPolydisc`: The polydisc

# Returns
Integer prime ``p`` such that the base ring is ``\mathbb{Q}_p`` or an extension thereof
"""
function prime(p::ValuationPolydisc)
    return Nemo.prime(p.center[1].parent)
end

function base_field(p::ValuationPolydisc)
    return p.center[1].parent
end

# @doc raw"""
#     residue_size(p::ValuationPolydisc)

# Get the size of the residue field of the underlying p-adic field.

# # Arguments
# - `p::ValuationPolydisc`: The polydisc

# # Returns
# The size of the residue field (currently unimplemented)
# """
# function residue_size(p::ValuationPolydisc)
#     return
# end

@doc raw"""
    dim(p::ValuationPolydisc)

Return the dimension of the polydisc space.

The dimension is the number of coordinate discs in the product, i.e., the length of the
center vector.

# Arguments
- `p::ValuationPolydisc`: The polydisc

# Returns
`Int`: The dimension of the polydisc space
"""
function dim(p::ValuationPolydisc)
    return length(p.center)
end

@doc raw"""
    join(b1::ValuationPolydisc{S,T}, b2::ValuationPolydisc{S,T}) where {S,T}

Compute the join (smallest common ancestor) of two polydiscs in the Bruhat-Tits tree.

The join is the largest polydisc containing both input polydiscs. For each coordinate ``i``,
the radius is ``\min(r_1^i, v(c_1^i - c_2^i), r_2^i)`` where ``r_j^i`` is the ``i``-th radius
of polydisc ``j`` and ``c_j^i`` is the ``i``-th center coordinate.

# Arguments
- `b1::ValuationPolydisc{S,T}`: First polydisc
- `b2::ValuationPolydisc{S,T}`: Second polydisc

# Returns
`ValuationPolydisc{S,T}`: The join polydisc with center from `b1` and computed radii
"""
function join(b1::ValuationPolydisc{S,T,N}, b2::ValuationPolydisc{S,T,N}) where {S, T, N}
    r = tuple([min(b1.radius[i], valuation(b1.center[i] - b2.center[i]), b2.radius[i]) for i in Base.eachindex(b1)]...)
    # check correctness (max vs min)
    return ValuationPolydisc{S,T,N}(b1.center, r)
end

@doc raw"""
    dist(b1::ValuationPolydisc{S,T}, b2::ValuationPolydisc{S,T}) where {S,T}

Compute the distance between two polydiscs in the polydisc space.

The distance is computed as the sum over all coordinates of the symmetric difference
between the discs, measured using the formula:
``d = \sum_i (p^{-r_j^i} - p^{-r_1^i} + p^{-r_j^i} - p^{-r_2^i})``
where ``r_j^i`` is the radius of the join in coordinate ``i``.

# Arguments
- `b1::ValuationPolydisc{S,T}`: First polydisc
- `b2::ValuationPolydisc{S,T}`: Second polydisc

# Returns
`Float64`: The distance between the two polydiscs
"""
function dist(b1::ValuationPolydisc{S,T,N}, b2::ValuationPolydisc{S,T,N}) where {S, T, N}
    p = prime(b1)
    b = Float64(p)
    j = join(b1, b2)
    return sum([b^(-j.radius[i]) - b^(-b1.radius[i]) + b^(-j.radius[i]) - b^(-b2.radius[i]) for i in Base.eachindex(b1)])

end

# TODO: this should also work when the valuation is negative...

@doc raw"""
    children(p::ValuationPolydisc{S,T}, degree=1) where {S,T}

Generate all child polydiscs obtained by refining the polydisc in a specified number of coordinates.

Computes all polydiscs obtained by increasing the valuation radius by 1 in exactly `degree`
coordinates and adjusting centers according to residue classes. For `degree=1`, produces
``p \cdot n`` children where ``n = \dim(p)`` is the dimension and ``p`` is the prime.

# Arguments
- `p::ValuationPolydisc{S,T}`: The parent polydisc
- `degree::Int=1`: Number of coordinates to refine simultaneously (must satisfy `degree ≤ dim(p)`)

# Returns
`Vector{ValuationPolydisc{S,T}}`: All child polydiscs (of length ``\binom{n}{degree} \cdot p^{degree}``)

# Notes
- Currently only works for ``\mathbb{Q}_p`` (not general extensions)
- Enumerates residue classes as `0:prime(p)-1`
"""
function children(p::ValuationPolydisc{S,T,N}, degree=1) where {S, T, N}
    @req dim(p) >= degree "degree exceeding dimension of polydisc"
    output = Vector{ValuationPolydisc{S,T,N}}()
    # The point p has prime(p)^degree children.
    sizehint!(output, Int(prime(p))^degree)
    K = base_field(p)
    prime_as_padic = K(prime(p))
    # iterate over all possible lists that have precisely degree times the value 1 and 0 everywhere else
    for coordinatesToShrink in AbstractAlgebra.combinations(dim(p), degree)
        # a "unit shrink" along a radius is the same as increasing the valuation
        # measure of the radius by 1
        new_radius = ntuple(N) do i
            if i in coordinatesToShrink
                p.radius[i] + 1
            else
                p.radius[i]
            end
        end

        # We can shrink along various centers so we need to be sure to include them all
        for radius_changes in Iterators.product([0:Int(prime(p))-1 for i in coordinatesToShrink]...)
            new_center = ntuple(N) do i
                idx_in_shrink = findfirst(==(i), coordinatesToShrink)
                if idx_in_shrink !== nothing
                    p.center[i] + radius_changes[idx_in_shrink] * (prime_as_padic ^ p.radius[i])
                else
                    p.center[i]
                end
            end
            push!(output, ValuationPolydisc{S,T,N}(new_center, new_radius))
        end
    end
    return output
end

# TODO: this should also work when the valuation is negative...

@doc raw"""
    children_along_branch(p::ValuationPolydisc{S,T}, branch_index::Int) where {S,T}

Generate all child polydiscs obtained by refining along a single coordinate branch.

Produces ``p`` children (where ``p`` is the prime) by increasing the radius in coordinate
`branch_index` by 1 and varying the center over all residue classes in that coordinate.

# Arguments
- `p::ValuationPolydisc{S,T}`: The parent polydisc
- `branch_index::Int`: The coordinate index to refine (must satisfy `1 ≤ branch_index ≤ dim(p)`)

# Returns
`Vector{ValuationPolydisc{S,T}}`: All child polydiscs along this branch (of length `prime(p)`)
"""
function children_along_branch(
    p::ValuationPolydisc{S,T,N},
    branch_index::Int
) where {S, T, N}
    # @req dim(p)>=degree "degree exceeding dimension of polydisc"
    output = Vector{ValuationPolydisc{S,T,N}}()
    # The point p has prime(p) children below branch i
    sizehint!(output, Int(prime(p)))
    # iterate over all possible lists that have precisely degree times the value 1 and 0 everywhere else
    # for coordinatesToShrink in 0::(Int(prime(p))-1)
    # a "unit shrink" along a radius is the same as increasing the valuation
    # measure of the radius by 1
    new_radius = ntuple(i -> i == branch_index ? p.radius[i] + 1 : p.radius[i], N)
    K = base_field(p)
    # TODO: make less hacky
    prime_as_padic = K(prime(p))
    # We can shrink along various centers so we need to be sure to include them all
    for residue_class in 0:Int(prime(p))-1
        new_center = ntuple(N) do i
            if i == branch_index
                p.center[i] + residue_class * (prime_as_padic ^ p.radius[branch_index])
            else
                p.center[i]
            end
        end
        push!(output, ValuationPolydisc{S,T,N}(new_center, new_radius))
    end
    return output
end


@doc raw"""
    concatenate(p::ValuationPolydisc{S,T}, q::ValuationPolydisc{S,T}) where {S,T}

Concatenate two polydiscs to form a higher-dimensional polydisc.

If ``p = B(a, r)`` is an ``n``-dimensional polydisc and ``q = B(a', r')`` is an ``m``-dimensional
polydisc, returns the ``(n+m)``-dimensional polydisc ``B((a, a'), (r, r'))``.

# Arguments
- `p::ValuationPolydisc{S,T}`: First polydisc
- `q::ValuationPolydisc{S,T}`: Second polydisc

# Returns
`ValuationPolydisc{S,T}`: The concatenated polydisc with `dim(p) + dim(q)` dimensions
"""
function concatenate(p::ValuationPolydisc{S,T,N1}, q::ValuationPolydisc{S,T,N2}) where {S, T, N1, N2}
    new_center = (p.center..., q.center...)
    new_radius = (p.radius..., q.radius...)
    return ValuationPolydisc{S,T,N1+N2}(new_center, new_radius)
end

# function aggregate(p::ValuationPolydisc{S, T}, q::ValuationPolydisc{S, T}, p_coords::Vector{Bool}) where {S, T}
#     new_center = Vector{T}
# end

function subdisc(p::ValuationPolydisc{S,T,N}, idx::Array{Int}) where {S, T, N}
    new_center = [p.center[i] for i in idx]
    new_radius = [p.radius[i] for i in idx]
    return ValuationPolydisc(new_center, new_radius)
end

function components(p::ValuationPolydisc{S,T,N}) where {S, T, N}
    # There will be a less cursed implementation for this:)
    return [subdisc(p, [i]) for i in 1:length(p.center)]
end

function Base.:<=(p::ValuationPolydisc{S,T,N}, q::ValuationPolydisc{S,T,N})::Bool where {S, T, N}
    return all(p.radius .>= q.radius) && all(valuation.(p.center .- q.center) .>= q.radius)
end