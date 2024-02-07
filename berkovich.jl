using Oscar

struct BerkovichPoint
    center::padic
    radius::Float64
end

function prime(p::BerkovichPoint)
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
    return Nemo.prime(p.center.parent)
end

# For now, we will only consider points of type 1 or 2 (since radius is a float, and not e.g. a symbolic real number)
function type(p::BerkovichPoint)
    if p.radius == 0
        return 1
    else
        return 2
    end
end

function padic_abs(a::padic)
    v = valuation(a)
    return Float64(Nemo.prime(a.parent))^(-v)
end

function join(b1::BerkovichPoint, b2::BerkovichPoint)
    """join(b1, b2) returns the join of b1 and b2
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I or II
    """
    return BerkovichPoint(b1.center, padic_abs(b1.center - b2.center))
end

function big_metric(b1::BerkovichPoint, b2::BerkovichPoint)
    """big_metric(b1, b2) returns the distance between two points of type I or II
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I or II
    """
    j = join(b1, b2)
    p = Float64(prime(b1))
    d1 = log(p, j.radius) - log(p, b1.radius)
    d2 = log(p, j.radius) - log(p, b2.radius)
    return d1+d2
end

function berkovichValuation(f::PolyRingElem{T}, a, r) where T <: RingElement
    """berkovichValuation(f, a, r, p) returns the valuation of polynomial f over Q_p at a point of type II or III corresponding to the ball B(a,r)

    Parameters
    ----------
    f : PolynomialPolynomialRing(Qp(p,10), 't') 
    a : Qp(p,10)
        The center of the ball
    r : QQ 
        The radius of the ball
    p : int
        the residue characteristic"""
    
    #we use the formula |f| = max (r^m |b_m|) where f = b_n (t-a)^n + ... + b_0 
    if r == 0
        return padic_abs(f(a))
    else
        m = typemax(Float)
        # consider the polynomial h = f(t+a) since the the m-th coefficient of h is the same as the m-th coefficient of g (expanded around t = a)
        h = compose(f, a + T)
        e = log(r, 1/a.parent().prime()) 
        # the max taken wrt absolute values corresponds to the min wrt valuations
        for i in 0:degree(f)
            m = min(m, h.valuation_of_coefficient(i) + i*e)
        end
        # convert result back to absolute value
        return Float(prime(a.parent))^(-m)
    end
end

function berkovichVal(f::PolyRingElem{T}, b::BerkovichPoint) where T <: RingElement
    #= berkovichVal(f, b) returns the valuation of polynomial f over Q_p at a point b

    Parameters
    ----------
    f : PolynomialPolynomialRing(Qp(p,10), 't') 
    b : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    """=#
    return berkovichValuation(f, b.center(), b.radius()) 
end
   
function path_helper_lt(b1, b2, t)
    """ path_helper_lt(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed (starting at t = -dist(b1, b2) and ending at t = 0). This assumes that b1 < b2.
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    p = prime(b1)
    r = big_metric(b1, b2)
    # if t is less than r, stay at b1
    if t <= -r
        return b1
    # if t is greater than 0 then we are already at b2
    elseif t >= 0
        return b2
    else
        # otherwise return the ball with the same center as b1 and with radius b2.radius()*p^t
        return B(b1.center(), Float64(p)^Float64(t)*b2.radius)
    end
end

function path_helper_gt(b1, b2, t)
    """ path_helper_gt(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed (starting at t = 0 and ending at t = dist(b1, b2)). This assumes that b2 < b1.
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    p = prime(b1)
    r = big_metric(b1, b2)
    # if t is less than 0, stay at b1
    if t <= 0
        return b1
    elseif t >= r
        # if t is greater than r then we are already at b2
        return b2
    else
        # otherwise return the ball with the same center as b1 and with radius b1.radius()*p^(-t)
        return B(b2.center(), b1.radius*Float64(p)^(-t))
    end
end   

function path(b1, b2, t)
    """ path(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed 
    (starting at t = - dist(b1, b2.join(b1)) and ending at t = dist(b2.join(b1, b1)). 
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    b3 = join(b1, b2)
    r1 = big_metric(b1, b3)
    r2 = big_metric(b3, b2)
    # if t is less than -r1 then stay at p1 
    if t <= -r1
        return b1
    # if t is between -r1 and 0 then go along the path [b1, b3]
    elseif -r1 <= t && t <= 0
        return path_helper_lt(b1, b3, t)
    # if t is between 0 and r2 then go along the path [b3, b2]
    elseif 0 <= t && t <= r2
        return path_helper_gt(b3, b2, t)
    else
    # is t is greater than r2 then stay at p2
        return b2
    end
end

#######################################################
# A Few tests

Q = PadicField(3, 10)
s1 = BerkovichPoint(Q(3), 0.0)
s2 = BerkovichPoint(Q(2), 0.5)
R, x = polynomial_ring(Q, "x")
f = x+1

println(berkovichValuation(f, s1.center, s1.radius))
println(path(s1, s2, 0))

println(path(s1, s2, 7))
     
# TODO: translate the rest of the code below to Julia  
    
#=
function abs_path_values(f, b1, b2, t1, t2, num)
    """ abs_path_values(f, b1, b2, t1, t2, num) returns two arrays: np.linspace(t1, t2, num) and the 
    image of np.linspace(t1, t2, num) by the map sending t to the valuation of f at the point attained at 
    time t on the path [p1, p2]
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t1 t2 : any real numbers. 
        The bounds on the sample times
    num : any positive integer
        The number of time samples

    Output
    ---------
    l = np.linspace(t1, t2, num)
    im : np.ndarray of shape (num)
    """
    l = np.linspace(t1, t2, num)
    im = np.zeros(num)
    for i in range(0, num)
        im[i] = berkovichVal(f, path(b1, b2, RealField(50)(l[i])))
    end
    return l, im
    

function abs_path_values_sum(s, b1, b2, t1, t2, num)
    """ abs_path_values_sum(s, b1, b2, t1, t2, num) returns two arrays: np.linspace(t1, t2, num) and the 
    image of np.linspace(t1, t2, num) by the map sending t to the valuation of f = \sum f_i at the point attained at 
    time t on the path [p1, p2] where s = [f_1, ..., f_n]
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t1 t2 : any real numbers. 
        The bounds on the sample times
    num : any positive integer
        The number of time samples

    Output
    ---------
    l = np.linspace(t1, t2, num)
    im : np.ndarray of shape (num)
    """
    a = np.zeros(num)
    for f in s
        l, im = abs_path_values(f, b1, b2, t1, t2, num)
        a += im
    end
    return l, a


#######################################################

function dir_deriv(f, b1, b2)
    """ dir_deriv(f, b1, b2) returns the directional derivative of the function |f(x)|_p at the point b1 in the direction b2.

    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(p)
        Points of type I, II or III.
    f : PolynomialPolynomialRing(Qp(p,10), 't')
        A polynomial in one variable over the p-adics, Q_p
    """
    p = b1.prime()
    if b1.gt(b2) == False and b1.lt(b2) == False
        # b1 and b2 are not ordered so replace b2 by the join
        b2 = b2.join(b1)
    end
    if b1 == b2
        return False
        ############# maybe raise an error?
    end
    a1 = b1.center()
    a2 = b2.center()
    r1 = b1.radius()
    r2 = b2.radius()
    if r1 < r2
        # take a1 to be the common centre
        h = f(t = t + a1)
        m = float('inf')
        e = log(r1, 1/p)
        for i in range(0, f.degree(t) + 1)
            m = min(m, h.valuation_of_coefficient(i) + i*e)
        end
        # this computes minimum of valuations, we also need to store when this minimum is obtained
        l = []
        for i in range(0, f.degree(t) + 1)
            if h.valuation_of_coefficient(i) + i*e == m
                l.append(i)
            end
        end
        # the directional derivative is in direction of increasing radius, so |f| = |c_n|*r^n locally above b1 for n the largest element of the list l
        n = l[-1]
        # derivative is then n*|c_n|*r^n = n*p**(-m)
        # if we want the derivative with respect to the radius metric then this is n*|c_n|*r^{n-1}
        return n*(p**(-m))
    end
    if r1 > r2
        # take a2 to be the common centre
        h = f(t = t + a2)
        m = float('inf')
        e = log(r1, 1/p)
        for i in range(0, f.degree(t) + 1)
            m = min(m, h.valuation_of_coefficient(i) + i*e)
        end
        # this computes minimum of valuations, we also need to store when this minimum is obtained
        l = []
        for i in range(0, f.degree(t) + 1)
            if h.valuation_of_coefficient(i) + i*e == m
                l.append(i)
            end 
        end 
        # the directional derivative is in direction of decreasing radius, so |f| = |c_n|*r^n locally above b1 for n the smallest element of the list l
        n = l[0]
        # derivative is then -n*|c_n|*r^n = -n*p**(-m)
        # if we want the derivative with respect to the radius metric then this is n*|c_n|*r^{n-1}
        return -n*(p**(-m))   
    end

 =#


