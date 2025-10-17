################### POLYDISCS ###################

# This file sets up the basic structures and
# API to work with polydiscs over the p-adics
# Our convention is that the radius is always
# measured wrt the valuation rather than the
# absolute value.

##################################################

struct ValuationPolydisc{S,T}
    center::Vector{S}
    # For valued points, the radius is measured with respect to the valuation
    radius::Vector{T}
end

# Polydisk with radius coordinate given in terms of valuation.
struct AbsPolydisc{S,T}
    center::Vector{S}
    # For normed points, the radius is measured with respect to the norm
    radius::Vector{T}
end

function Base.eachindex(p::ValuationPolydisc)
    return Base.eachindex(p.center)
end

function center(p::ValuationPolydisc)
    return p.center
end

function radius(p::ValuationPolydisc)
    return p.radius
end

function base_ring(p::ValuationPolydisc)
    return Base.parent(first(center(p)))
end

import Base.show
function Base.show(io::IO, p::ValuationPolydisc)
    print(io, "Polydisc over $(base_ring(p)) with center $(center(p)) and radius $(radius(p))")
end

import Base.==
function Base.:(==)(p::ValuationPolydisc, q::ValuationPolydisc)
    # check whether the radii coincide
    # and if yes, check whether coordinate-wise difference of centers
    # has lower valuation than radii
    return radius(p) == radius(q) && all(valuation.(center(p) .- center(q)) .> radius(p))
end

function Base.hash(m::ValuationPolydisc, h::UInt)
    return hash(h)
end


# Some of the code (e.g the function below) might be nicer if we can use some unifying type
# E.g. Polydisk

@doc raw"""prime(p) returns the prime number of the padic field of p as an integer

    Parameters
    ----------
    p: BerkovichPoint
        A point of type I, II or III

    Output
    ----------
    p: ZZRing
        The prime number of the padic field of p
"""
function prime(p::ValuationPolydisc)
    return Nemo.prime(p.center[1].parent)
end

## This currently only works for unramified extensions of the p-adic numbers at the moment
function residue_size(p::ValuationPolydisc)
    return
end

# Returns the dimension of the space in which the polydisc `p` lies,
# i.e. the dimension of the center of `p`
function dim(p::ValuationPolydisc)
    return length(p.center)
end

# Returns the p-adic valuation of a p-adic number `a`
function padic_abs(a::padic)
    v = valuation(a)
    return Float64(Nemo.prime(a.parent))^(-v)
end

@doc raw"""join(b1, b2) returns the join of b1 and b2

    Parameters
    ----------
    b1, b2 : ValuationPolydisk{S, T}
"""
function join(b1::ValuationPolydisc{S,T}, b2::ValuationPolydisc{S,T}) where S where T
    r = [min(b1.radius[i], valuation(b1.center[i] - b2.center[i]), b2.radius[i]) for i in Base.eachindex(b1)]
    # check correctness (max vs min)
    return ValuationPolydisc(b1.center, r)
end

# Returns the distance between two valuation polydiscs
function dist(b1::ValuationPolydisc{S,T}, b2::ValuationPolydisc{S,T}) where S where T
    p = prime(b1)
    b = Float64(p)
    j = join(b1, b2)
    return sum([b^(-j.radius[i]) - b^(-b1.radius[i]) + b^(-j.radius[i]) - b^(-b2.radius[i]) for i in Base.eachindex(b1)])

end

# TODO Paul: make variable names more evocative

# Returns the list of the children of a point p in the polydisc space,
# i.e. the polydiscs obtained by making "one" step down in one or more
# of the radii. The number of radii allowed to shrink is controlled by
# the parameter degree.
# WARNING: this will need to chanage to work for fields that aren't the p-adic numbers
# since we're enumerating residue classes as 0:prime(p)-1
# TODO Paul: do we want some type safety mechanism?
function children(p::ValuationPolydisc{S,T}, degree=1) where S where T
    @req dim(p) >= degree "degree exceeding dimension of polydisc"
    output = Vector{ValuationPolydisc{S,T}}()
    # The point p has prime(p)^degree children.
    sizehint!(output, Int(prime(p))^degree)
    # iterate over all possible lists that have precisely degree times the value 1 and 0 everywhere else
    for coordinatesToShrink in AbstractAlgebra.combinations(dim(p), degree)
        # a "unit shrink" along a radius is the same as increasing the valuation
        # measure of the radius by 1
        new_radius = copy(p.radius)
        new_radius[coordinatesToShrink] .+= 1
        # We can shrink along various centers so we need to be sure to include them all
        for radius_changes in Iterators.product([0:Int(prime(p))-1 for i in coordinatesToShrink]...)
            new_center = copy(p.center)
            new_center[coordinatesToShrink] .+= radius_changes .* (prime(p) .^ p.radius[coordinatesToShrink])
            push!(output, ValuationPolydisc(new_center, new_radius))
        end
    end
    return output
end

function children_along_branch(
    p::ValuationPolydisc{S,T},
    branch_index::Int
) where S where T
    # @req dim(p)>=degree "degree exceeding dimension of polydisc"
    output = Vector{ValuationPolydisc{S,T}}()
    # The point p has prime(p) children below branch i
    sizehint!(output, Int(prime(p)))
    # iterate over all possible lists that have precisely degree times the value 1 and 0 everywhere else
    # for coordinatesToShrink in 0::(Int(prime(p))-1)
    # a "unit shrink" along a radius is the same as increasing the valuation
    # measure of the radius by 1
    new_radius = copy(p.radius)
    new_radius[branch_index] += 1
    # We can shrink along various centers so we need to be sure to include them all
    for residue_class in 0:Int(prime(p))-1
        new_center = copy(p.center)
        new_center[branch_index] += residue_class * (prime(p)^p.radius[branch_index])
        push!(output, ValuationPolydisc(new_center, new_radius))
    end
    return output
end


# Concatenate two polydiscs. I.e. if `p = B(a, r)` and `q = B(a', r')` then this returns
# the polydisc `B((a, a'), (r, r'))`.
function concatenate(p::ValuationPolydisc{S,T}, q::ValuationPolydisc{S,T}) where S where T
    new_center = [p.center; q.center]
    new_radius = [p.radius; q.radius]
    return ValuationPolydisc(new_center, new_radius)
end

# function aggregate(p::ValuationPolydisc{S, T}, q::ValuationPolydisc{S, T}, p_coords::Vector{Bool}) where S where T
#     new_center = Vector{T}
# end
