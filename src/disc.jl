using Oscar

# TODO(Paul-Lez): add constraints on the type of T, and check those on S
# In theory at this level of abstraction, T could actually be any metric space
struct ValuationDisc{S, T} where T<:Oscar.scalar_types 
    center::S
    # For valued points, the radius is measured with respect to the valuation 
    radius::T
end

# Polydisk with radius coordinate given in terms of valuation. 
struct AbsDisc{S, T} where T<:Oscar.scalar_types 
    center::S
    # For normed points, the radius is measured with respect to the norm 
    radius::T
end

# Some of the code (e.g the function below) might be nicer if we can use some unifying type 
# E.g. Polydisk

function prime(p::ValuationDisc{S, T}) where T<:Oscar.scalar_types 
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
    return Nemo.prime(S)
end

function padic_abs(a::padic)
    v = valuation(a)
    return Float64(Nemo.prime(a.parent))^(-v)
end

function join(b1::ValuationDisc{S, T}, b2::ValuationDisc{S, T}) where S<:Oscar.scalar_types
    """join(b1, b2) returns the join of b1 and b2
    
    Parameters
    ----------
    b1, b2 : ValuationPolydisk{S, T}
    """
    v = valuation(b1.center - b2.center)
    # check correctness (max vs min)
    return ValuationDisc(b1.center, min(b1.radius, v, b2.radius))
end


function join(b1::AbsDisc{S, T}, b2::AbsDisc{S, T}) where S<:Oscar.scalar_types
    return "implement me"
end 

function hyperbolic_distance(b1::ValuationDisc{S, T}, b2::ValuationDisc{S, T})  where S<:Oscar.scalar_types
    """ hyperbolic_distance(b1, b2) returns the hyperbolic distance between two points of type I or II
    
    Parameters
    ----------
    b1, b2 : ValuationDisc{S, T}
    """
    j = join(b1, b2)
    return (j.radius - b1.radius) + (j.radius - b2.radius)
end

function valuation_coeff(f::PolyRingElem{T}, i) where T <: RingElement
    return valuation(coeff(f, i))
end

# TODO(Paul-Lez): here we probably want some constraints on the coefficient ring 
function val_eval(f::PolyRingElem, p::ValuationDisk{S, T}) where T <: RingElement
    """abs_eval(f, p) returns the evaluation of the valuation of a 
       polynomial f at the point p

    Parameters
    ----------
    f : PolynomialPolynomialRing(Qp(p,10), 't') 
    p : ValuationDisk{S, T}"""
    
    #we use the formula v(f) = min (v(b_m) + bm) where f = b_n (t-a)^n + ... + b_0 and b is the valuation radius.
    m = typemax(Float64)
    # consider the polynomial h = f(t+a) since the the m-th coefficient of h is the same as the m-th coefficient of g (expanded around t = a)
    t = gen(f.parent)
    h = compose(f, t + a)
    e = - log(r, p) 
    # compute the min. 
    for i in 0:degree(f)
        m = min(m, valuation_coeff(h, i) + i*e)
    end
    return m
end