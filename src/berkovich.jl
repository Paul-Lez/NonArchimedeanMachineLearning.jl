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

function valuation_coeff(f::PolyRingElem{T}, i) where T <: RingElement
    return valuation(coeff(f, i))
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
        m = typemax(Float64)
        # consider the polynomial h = f(t+a) since the the m-th coefficient of h is the same as the m-th coefficient of g (expanded around t = a)
        t = gen(f.parent)
        h = compose(f, t + a)
        p = Float64(Nemo.prime(a.parent))
        e = - log(r, p) 
        # the max taken wrt absolute values corresponds to the min wrt valuations
        for i in 0:degree(f)
            m = min(m, valuation_coeff(h, i) + i*e)
        end
        # convert result back to absolute value
        return Float64(Nemo.prime(a.parent))^(-m)
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

function abs_path_values(f, b1, b2, t1, t2, num)
    """ abs_path_values(f, b1, b2, t1, t2, num) returns two arrays: linspace(t1, t2, num) and the 
    image of linspace(t1, t2, num) by the map sending t to the valuation of f at the point attained at 
    time t on the path [p1, p2]
    
    Parameters
    ----------
    b1, b2 : BerkovichPoint
        A point of type I, II or III
    t1 t2 : Real 
        The bounds on the sample times
    num : Int
        The number of time samples

    Output
    ---------
    l = linspace(t1, t2, num)
    im : array of shape (num)
    """
    l = linspace(t1, t2, num)
    im = zeros(num)
    for i in 1:num
        im[i] = berkovichVal(f, path(b1, b2, RealField(50)(l[i])))
    end
    return l, im
end
    

#= function abs_path_values_sum(s, b1, b2, t1, t2, num)
    """ abs_path_values_sum(s, b1, b2, t1, t2, num) returns two arrays: linspace(t1, t2, num) and the 
    image of linspace(t1, t2, num) by the map sending t to the valuation of f = \sum f_i at the point attained at 
    time t on the path [p1, p2] where s = [f_1, ..., f_n]
    
    Parameters
    ----------
    s : Array of PolyRingElem{T}
    b1, b2 : BerkovichPoint
        A point of type I, II or III
    t1, t2 : Real 
        The bounds on the sample times
    num : Int
        The number of time samples

    Output
    ---------
    l = linspace(t1, t2, num)
    a : array of shape (num)
    """
    a = zeros(num)
    for f in s
        l, im = abs_path_values(f, b1, b2, t1, t2, num)
        a .+= im
    end
    return l, a
end =#

function gt(b1::BerkovichPoint, b2::BerkovichPoint)
    """gt(b1, b2) returns true if the Berkovich point b1 is greater than b2 in the sense that the ball b2 is strictly contained in the ball b1.

    Parameters
    ----------
    b1, b2 : BerkovichPoint
    """
    a1, a2 = b1.center, b2.center
    r1, r2 = b1.radius, b2.radius

    return padic_abs(a1 - a2) < r1 && r2 < r1
end

function lt(b1::BerkovichPoint, b2::BerkovichPoint)
    """gt(b1, b2) returns true if the Berkovich point b1 is lesse than b2 in the sense that the ball b1 is strictly contained in the ball b2.

    Parameters
    ----------
    b1, b2 : BerkovichPoint
    """
    a1, a2 = b1.center, b2.center
    r1, r2 = b1.radius, b2.radius

    return padic_abs(a1 - a2) < r2 && r1 < r2
end



function dir_deriv(f, b1::BerkovichPoint, b2::BerkovichPoint)
    """ dir_deriv(f, b1, b2) returns the directional derivative of the function |f(x)|_p at the point b1 in the direction b2.

    Parameters
    ----------
    b1, b2 : BerkovichPoint
        Points of type I, II or III.
    f : PolyRingElem{T}
        A polynomial in one variable over the p-adics, Q_p
    """
    p = prime(b1)
    T = gen(f.parent)
    if !gt(b1, b2)  && !lt(b1, b2) 
        # b1 and b2 are not ordered so replace b2 by the join
        b2 = join(b1, b2)
    end
    if b1 == b2
        return false
        ############# maybe raise an error?
    end

    a1, a2 = b1.center, b2.center
    r1, r2 = b1.radius, b2.radius

    if r1 < r2
        # take a1 to be the common centre
        h = compose(f, T + a1)
        m = typemax(Float64)
        e = log(1/(Float64(p)), r1)

        for i in 0:degree(f)
            m = min(m, valuation_coeff(h, i) + i*e)
        end
        # this computes the minimum of valuations, we also need to store when this minimum is obtained
        l = Int[]
        for i in 0:degree(f)
            if valuation_coeff(h, i) + i*e == m
                push!(l, i)
            end
        end
        # the directional derivative is in direction of increasing radius, so |f| = |c_n|*r^n locally above b1 for n the largest element of the list l
        n = last(l)
        # derivative is then n*|c_n|*r^n = n*p**(-m)
        # if we want the derivative with respect to the radius metric then this is n*|c_n|*r^{n-1}
        return n * (Float64(p)^(-m))
    end
    if r1 > r2
        # take a2 to be the common centre
        h = compose(f, T + a2)
        m = typemax(Float64)
        e = log(1/(Float64(p)), r1)
        for i in 0:degree(f)
            m = min(m, valuation_coeff(h, i) + i*e)
        end
        # this computes minimum of valuations, we also need to store when this minimum is obtained
        l = Int[]
        for i in 0:degree(f)
            if valuation_coeff(h, i) + i*e == m
                push!(l, i)
            end
        end
        # the directional derivative is in direction of decreasing radius, so |f| = |c_n|*r^n locally above b1 for n the smallest element of the list l
        n = first(l)
        # derivative is then -n*|c_n|*r^n = -n*p**(-m)
        # if we want the derivative with respect to the radius metric then this is n*|c_n|*r^{n-1}
        return -n * (Float64(p)^(-m))   
    end
end 


#todo: make this implementation smarter. This should also test whether the point has type II or III.
function enumerate_centers(P::BerkovichPoint)
    """enumerate_centers(P) returns a list of elements of the padic field such that represent the branches below P
    """
    # If P is of type I, then the only center is P itself (as a K-point)
    if P.radius == 0
        return [P.center]
    else
        p = Nemo.prime(P.center.parent)
        # This isn't very elegant, but it works for now. We find the integer m such that the radius of P is p^(-m). This needs to be an integer so we can raise p to the power of it.
        m = Int64(floor(log(Float64(p), P.radius)))
        #otherwise, if the radius is p^m then we take the centers to be the points of the form P.center + p^(-m)*i for i in 0, 1, ..., p^m
        a = P.center
        A = [a]
        for i in 1:(p-1)
            Q = a.parent
            a = a + Q(p)^(-m)
            push!(A, a)
        end
        return A
    end
end

function max_radius(F, p)
    """max_radius(F, p) returns the maximum of the absolute values of the coefficients of the polynomials in F
    """
    m = typemax(Float64)
    for f in F
        for i in 0:degree(f)
            m = min(m, valuation_coeff(f, i))
        end
    end
    m = Float64(p)^(-m)
    return m
end

#TODO: check this function. Also it would be useful to have it output extra information e.g. the regime of f at each point, etc. 
function enumerate_change_points(f::PolyRingElem{R}, P::BerkovichPoint, regime=degree(f)) where R <: RingElement
    """enumerate_change_points(F, P) returns the list of points along the path from the center of P to P for which the valuation of the polynomial f changes degree (as a function of the radius). The regime parameter allows the user to specify that we are only interested in the r+1 first coefficients of f - this is useful since when the radius is small, the higher degree coefficients are not important as they won't change the valuation, and we can use this to speed up calculations.
    """
    # r is the index we are currently considering.
    r = regime 
    # Initialise the list of change points.
    changes = []
    changes_regime = Dict()
    p = Nemo.prime(P.center.parent)
    # Find the valuation corresponding to the radius of P
    val = -Int64(floor(log(Float64(p), P.radius)))
    loop_count = 0
    new_r = r
    T = gen(f.parent)
    # consider the polynomial h = f(t+a) since the the m-th coefficient of h is the same as the m-th coefficient of g (expanded around t = a)
    f = compose(f, T+P.center)
    has_jumps = false
    while r != 0
        loop_count += 1
        new_val = val
        jump = false
        for j in (r-1):-1:0
            # if the coefficient is 0 we can ignore it and move on to the next interation of the loop.
            if coeff(f, j) != 0
                #= otherwise we compute at which radius a jump from coeff r to coeff j would occur (i.e. for which radius s do we have 
                |coeff(j)| => |coeff(r)|*p^(r-j).  =#
                temp_val = (valuation_coeff(f, r) - valuation_coeff(f, j))/(r-j)
                # if this radius is smaller than the current radius and larger than the current jump radius we update it.
                if -temp_val >= val && (!jump || -temp_val <= new_val)
                    jump = true
                    new_val = -temp_val
                    # Update the regime of f at that point.
                    new_r = j
                end
            end 
        end
        # if r has changed then we must have found a change point above so we can add it to the list
        if r != new_r
            Q = BerkovichPoint(P.center, Float64(p)^(-new_val))
            push!(changes, Q)
            changes_regime[(Q, f)] = new_r
            r = new_r
            val = new_val
            has_jumps = true
        # otherwise the modulus of f is constant for the rest of the branch so we can stop.
        else 
            break
        end
    end 
    return changes, changes_regime, has_jumps
end

# Naive implementation of the tree search algorithm. Still need to implements some of the dependencies plus add some of the 
# branch cutting mechanisms.
function min_find(F, N, R) #= ::Vector{PolyRingElem{T}} =# #= where T <: RingElement =#
    """min_find(F, N, R) returns a set type II points of lying above the roots of the polynomials in F.  
    N is the number of iterations we allow and R is the p-adic field (R should really be removed since we should be able to access it from the polynomials in F). 
    """
    p = Nemo.prime(R)
    # Initialise set of type II points  
    B = [BerkovichPoint(R(0), max_radius(F, p))]
    BR = Dict()
    for i in 1:N
        # Initialise new set of type II points
        C = []
        CR = Dict()
        for b in B
            #println(B)
            # Enumerate the centers of the branches below p
            L = enumerate_centers(b)
            for f in F
                for a in L
                    if i == 1 || BR[(b, f)] != 0 
                        # consider the polynomial h = f(t+a) since the the m-th coefficient of h is the same as the m-th coefficient of g (expanded around t = a)
                        T = gen(f.parent)
                        #h = compose(f, a + T)
                        #add all change points to the new set of type II points
                        cp, cpr, has_jumps = enumerate_change_points(f, BerkovichPoint(a, b.radius), BR[b])
                        if has_jumps
                            append!(C, cp)
                            merge!(CR, cpr)
                        else 
                            println("No jumps - fixme")
                            print("Poly = ", f, "  a = ", a, "  b = ", b)
                        end
                    else 
                        
                    end
                end
            end
        end
        # Update the set of type II points
        B = C
    end 
    return B
end


################### UNIT TESTS #################### 


# Define a few objects
Q = PadicField(3, 10)
s1 = BerkovichPoint(Q(3), 0.0)
s2 = BerkovichPoint(Q(3), 0.5)
R, x = polynomial_ring(Q, "x")
f = (x-3)*(x-12)*(x-30)*(x-84)
g = (x-1)
println(dir_deriv(f, s2, s1))
println(dir_deriv(g, s2, s1))
#println(compose(f, f))

#######################################################
#= 
# Compute some paths
println(berkovichValuation(f, s1.center, s2.radius))
println(path(s1, s2, 0))
println(path(s1, s2, 7))    

# List the centers of the branches below s
println(enumerate_centers(BerkovichPoint(Q(3), 1)))
println(enumerate_centers(BerkovichPoint(Q(3), 0)))  =#

#println(enumerate_change_points(f, BerkovichPoint(Q(3), 3)))
#println(enumerate_change_points(g, BerkovichPoint(Q(3), 0.5)))

# Find the minimum of a polynomial
#println(max_radius([f], 3))
#println(min_find([g], 10, Q))
