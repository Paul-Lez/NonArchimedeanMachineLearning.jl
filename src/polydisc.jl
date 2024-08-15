using Oscar
using Combinatorics

# TODO(Paul-Lez): add constraints on the type of T, and check those on S
# In theory at this level of abstraction, T could actually be any metric space
struct ValuationPolydisc{S, T}
    center::Vector{S}
    # For valued points, the radius is measured with respect to the valuation 
    radius::Vector{T}
end

# Polydisk with radius coordinate given in terms of valuation. 
struct AbsPolydisc{S, T}
    center::Vector{S}
    # For normed points, the radius is measured with respect to the norm 
    radius::Vector{T}
end

function Base.eachindex(p::ValuationPolydisc)
    return Base.eachindex(p.center)
end 

# Some of the code (e.g the function below) might be nicer if we can use some unifying type 
# E.g. Polydisk

function prime(p::ValuationPolydisc) 
    """prime(p) returns the prime number of the padic field of p as an integer

    Parameters
    ----------
    p: BerkovichPoint
        A point of type I, II or III

    Output
    ----------
    p: ZZRing
        The prime number of the padic field of p
    """
    return Nemo.prime(parent(p.center[1]))
end

## This currently only works for unramified extensions of the p-adic numbers at the moment
function residue_size(p::ValuationPolydisc) 
    return 
end 

function dim(p::ValuationPolydisc) 
    return length(p.center)
end 

function padic_abs(a::padic)
    v = valuation(a)
    return Float64(Nemo.prime(a.parent))^(-v)
end

function join(b1::ValuationPolydisc{S, T}, b2::ValuationPolydisc{S, T}) where S where T
    """join(b1, b2) returns the join of b1 and b2
    
    Parameters
    ----------
    b1, b2 : ValuationPolydisk{S, T}
    """
    r = [min(b1.radius[i], valuation(b1.center[i] - b2.center[i]), b2.radius[i]) for i in Base.eachindex(b1)]
    # check correctness (max vs min)
    return ValuationPolydisc(b1.center, r)
end


# Returns the list of the children of a point p in the polydisc space, 
# i.e. the polydiscs obtained by making "one" step down in one or more 
# of the radii. The number of radii allowed to shrink is controlled by 
# the parameter degree.
# WARNING: this will need to chanage to work for fields that aren't the p-adic numbers
# since we're enumerating residue classes as 0:prime(p)-1
# TODO Paul: do we want some type safety mechanism?
function children(p::ValuationPolydisc{S, T}, degree=1) where S where T
    output = Vector{ValuationPolydisc{S, T}}()
    # The point p has prime(p)^degree children.
    sizehint!(output, Int(prime(p))^degree)
    # iterate over all possible lists that have precisely degree times the value 1 and 0 everywhere else
    for shrink in Combinatorics.permutations([ones(T, degree) ; zeros(T, dim(p) - degree)])
        # a "unit shrink" along a radius is the same as increasing the valuation 
        # measure of the radius by 1 
        new_radius = p.radius + shrink
        # We can shrink along various centers so we need to be sure to include them all
        # TODO Paul: explain the line below!
        for radius_changes in Iterators.product([shrink[i] == 1 ? (0:Int(prime(p))-1) : (0:0) for i in Base.eachindex(p)]...)
            # TODO Paul: figure out what's happening with S
            R = parent(p.center[1])
            new_center = p.radius + R.(([(Int(i)) for i in radius_changes])) 
            push!(output, ValuationPolydisc(new_center, new_radius))
        end 
    end 
    return output
end 


# function join(b1::ValuationDisc{S, T}, b2::ValuationDisc{S, T}) where S<:Oscar.scalar_types where T<:Oscar.scalar_types
#     return "implement me"
# end 

# function valuation_coeff(f::PolyRingElem{T}, i) where T <: RingElement
#     return valuation(coeff(f, i))
# end

# # TODO(Paul-Lez): here we probably want some constraints on the coefficient ring 
# function val_eval(f::PolyRingElem, p::ValuationDisk{S, T}) where T <: RingElement T<:Oscar.scalar_types
#     """abs_eval(f, p) returns the evaluation of the valuation of a 
#        polynomial f at the point p

#     Parameters
#     ----------
#     f : PolynomialPolynomialRing(Qp(p,10), 't') 
#     p : ValuationDisk{S, T}"""
    
#     #we use the formula v(f) = min (v(b_m) + bm) where f = b_n (t-a)^n + ... + b_0 and b is the valuation radius.
#     m = typemax(Float64)
#     # consider the polynomial h = f(t+a) since the the m-th coefficient of h is the same as the m-th coefficient of g (expanded around t = a)
#     t = gen(f.parent)
#     h = compose(f, t + a)
#     e = - log(r, p) 
#     # compute the min. 
#     for i in 0:degree(f)
#         m = min(m, valuation_coeff(h, i) + i*e)
#     end
#     return m
# end