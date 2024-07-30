import Pkg
include("valuation_disk.jl")
Pkg.add("Nemo")

struct ValuationPolydisk{S, T} where T<:Oscar.scalar_types 
    disks::Vector{BerkovichPoint{S, T}}
end

function BerkovichPolyDisk(center::Vector{S}, radius::Vector{T}) where T<:Oscar.scalar_types 
    if length(center) != length(radius)
        error("Center and radius vectors should have the same length")
    end
    disks = []
    for i in 1:length(center)
        push!(disks, BerkovichPoint(center[i], radius[i]))
    end
    return BerkovichPolyDisk(disks)
end

function radius(b::ValuationPolydisk{S, T}) where T<:Oscar.scalar_types
    return b.radius
end

function radius(d::ValuationPolydisk{S, T}) where T<:Oscar.scalar_types
    return [radius(b) for b in d.disks]
end

function center(b::ValuationPolydisk{S, T}) where T<:Oscar.scalar_types
    return b.center
end 

function center(d::ValuationPolydisk{S, T}) where T<:Oscar.scalar_types
    return [center(b) for b in d.disks]::Vector{S}
end

# BerkovichTagent represents an element of the tangent space at a Berkovich point. Here we allow the tangent to have a magnitude instead of normalising it to 1.
struct BerkovichTangent{S, T} where T<:Oscar.scalar_types
    point::BerkovichPoint{S, T}
    direction::S
    # Possibly we should allow the magnitude to have a different type?
    # not sure if that would ever come up 
    magnitude::T
end

struct BerkovichPolyTangent{S, T} where T<:Oscar.scalar_types
    tangents::Vector{BerkovichTangent{S, T}}
end

function evaluate(f::MPolyRingElem, p::ValuationPolydisk{S, T}) where T<:Oscar.scalar_types
    t = gens(f.parent)
    vec = [t[i] - center(p)[i] for i in eachindex(center(p))]
    g = AbstractAlgebra.evaluate(f, vec) 
    # coeff seems to return a constant polynomial so need to move this to padics by "taking the constant term" before taking absolute value
    max, _ = findmin([padic_abs(constant_coefficient(coeff(g, t))) * prod(radius(p) .^ v)  for v in exponent_vectors(g)])
    return max
end

function BerkovichPolyTangent(point, direction, magnitude)
    if length(point.disks) != length(direction) | length(direction) != length(magnitude) | length(magnitude) != length(point.disks)
        println("lengths of input arrays are incompatible")
    else
        arr = []
        for i in eachindex(point.disks)
            push!(arr, BerkovichTangent(point.disks[i], direction[i], magnitude[i]))
        end 
        return BerkovichPolyTangent(arr)
    end 
end 

function Base.length(v::BerkovichPolyTangent)
    return length(v.tangents)
end 

function point(v::BerkovichPolyTangent)
    return BerkovichPolyDisk([p.point for p in v.tangents])
end

function direction(v::BerkovichPolyTangent)
    return [p.direction for p in v.tangents]
end 

function magnitude(v::BerkovichPolyTangent)
    return [p.magnitude for p in v.tangents]
end 

# the zero tangent vector at a Berkovich PolyDisk P, in the direction of segment [P, Q]
function BerkovichPolyTangent_zero(P::BerkovichPolyDisk, Q::Vector{padic})
    return BerkovichPolyTangent([BerkovichTangent(P.disks[i], Q[i], 0) for i in eachindex(P.disks)])
end

function BerkovichPolyTangent_zero(v::BerkovichPolyTangent)
    return BerkovichPolyTangent_zero(point(v), direction(v))
end 

#addition of two tangent vectors. Warning: this is only well behaved when they have the same base and direction...
function Base.:+(P::BerkovichPolyTangent, Q::BerkovichPolyTangent)
    return BerkovichPolyTangent(point(P), direction(P), magnitude(P) + magnitude(Q))
end

function extend(P::BerkovichPolyDisk, p::BerkovichPoint)
    return BerkovichPolyDisk([P.disks, p])
end 


function extend(v::BerkovichPolyTangent, point::BerkovichPoint, dir::padic, mag::Float64)
    return BerkovichPolyTangent(extend(point(v), point), [direction(v), dir], [magnitude(v), mag])
end

function change_center(P::BerkovichPolyDisk, a::padic, i)
    disks = P.disks
    disks[i] = BerkovichPoint(a, disks[i].radius)
    return BerkovichPolyDisk(disks)
end 

function prime(b::BerkovichPoint)
    center = b.center
    return convert(Int64, Nemo.prime(center.parent))
    # we want p as an integer not ZZRingElt
end

# Define global variable
prec = 20 # all p adic fields should be precise up to power 20

# first define the rational function appearing in the loss function

function num(t0,t1,t2,x)
    """ num(t0,t1,t2,x) returns the supremum of cy^2+by+a on {(a,b,c,y) in t0*t1*t2*x (a 4-Berkovich point)}
        more specifically -log_p of this supremum, which is the minimal valuation instead 

    Parameters
    ----------
    t0, t1, t2, x : BerkovichPoint 
    """
    p = prime(t0)
    K = PadicField(p, prec)
    R, (a, b, c, y) = PolynomialRing(K, ["a", "b", "c", "y"])
    f = c*(y^2) + b*y + a
    c0, c1, c2, c3 = t0.center, t1.center, t2.center, x.center
    r = [t0.radius, t1.radius, t2.radius, x.radius]
    # Translate f by centres
    g = f(a+c0, b+c1, c+c2, y+c3)
    expo = collect(Nemo.exponent_vectors(g))
    coeff = collect(Nemo.coefficients(g))
    # these are the exponents and coefficients of the terms in the polynomial getting
    l = length(expo)
    m = typemax(Int64)
    for i in 1:l
        m = min(m, valuation(coeff[i]) + dot(expo[i], r))
        # we break g into monomial terms which are products of powers of a,b,c,y and a coefficient
        # the minimial valuation of this monomial on t0*t1*t2*x is the valuation of the coeff + 
        # the sum of the radii (which are really valuations not the absolute value - the radius should be p^{-valuation}),
        # where the radii are weighted by the exponents of a,b,c,y in the monomial, and this is given by the dot product.
    end
    return m
end


############## TEST #############

# K = PadicField(2,prec)
# t0 = BerkovichPoint(K(5), 6)
# t1 = BerkovichPoint(K(3), 1)
# t2 = BerkovichPoint(K(2), 2)
# x = BerkovichPoint(K(6), 3)
# print(num(t0,t1,t2,x))


#################################

function loss(t0,t1,t2,s0,s1,s2,X,Y)
    """ loss(t0,t1,t2,s0,s1,s2,X,Y) returns the value of our choice of loss function with respect to training subsets X and Y
        and the parameters t0,t1,t2,s0,s1,s2 

    Parameters
    ----------
    t0, t1, t2, s0, s1, s2 : BerkovichPoint
    X, Y : List(BerkovichPoint)
    """
    p = prime(t0)
    Xsum = 0
    for x in X
        rat = -num(t0,t1,t2,x) + num(s0,s1,s2,x)
        abs = exp(rat * log(p))
        # computing p^rat, but need to use exp to get a floating point number instead of ring element which can't be exponentiated
        Xsum += 1 - exp(-abs)
    end
    Ysum = 0
    for y in Y
        rat = -num(t0,t1,t2,y) + num(s0,s1,s2,y)
        abs = exp(rat * log(p))
        Ysum += exp(-abs)
    end
    return Xsum + Ysum
end


# define the descent method for 1 step only changing one radius
# we will ask radii to be nonincreasing so we can have a (maybe probabilistic) termination criterion

function greedy_descent_step(t0,t1,t2,s0,s1,s2,X,Y,alpha)
    """ greedy_descent(t0,t1,t2,s0,s1,s2,X,Y,alpha) changes a parameter t0,t1,t2,s0,s1,s2 determined by alpha
        in such a way to minimse the loss function along X and Y. alpha is an integer between 1 and 6.
        Returns updated list [t0, t1, t2, s0, s1, s2]     
        
        Parameters
        ----------
        t0, t1, t2, s0, s1, s2 : BerkovichPoint
        alpha : Int64
        X, Y : List(BerkovichPoint)
    """
    choice = [t0, t1, t2, s0, s1, s2]
    t = choice[alpha]
    # this is the parameter we will change        
    r = t.radius
    c = t.center
    p = prime(t)
    # now we want to update t. Create a list of length p+1 where each element we have the BerkovichPoints t0,...,s2 with t updated
    list = [choice]
    for i in 1:p
        newradius = r + 1
        newcenter = c + (i-1)*p^r
        my_copy = copy(choice)
        my_copy[alpha] = BerkovichPoint(newcenter, newradius)
        # update t
        push!(list, my_copy)
    end
    # create list of loss values
    losslist = []
    for i in 1:p+1
        param = list[i]
        push!(losslist, loss(param[1], param[2], param[3], param[4], param[5], param[6], X, Y))
    end
    j = argmin(losslist)
    return list[j]
end

# other option is choose the best of (p+1)^6 options when we change all parameters simultaneously

# easier to define function which updates parameters separately

function update_param(t,i)
    """update_param(t,i) updates the Berkovich point t by choosing a new center and radius according to i
       If i = 0 we make no changes, and if 1 <= i <= p then we increase the radius valuation by 1 and shift the center in p ways
       We should have 0 <= i <= p 

       Parameters
       ----------
       t : BerkovichPoint
       i : Int64

    """
    if i == 0
        return t
    end
    r = t.radius
    c = t.center
    p = prime(t)
    if i in 1:p
        newradius = r + 1
        newcenter = c + (i-1)*p^r
        return BerkovichPoint(newcenter, newradius)
    end
    error("i is out of range")
end

function simul_descent_step(t0,t1,t2,s0,s1,s2,X,Y)
    """ simul_descent_step(t0,t1,t2,s0,s1,s2,X,Y) changes all parameters by one step
        in such a way to minimse the loss function along X and Y. 
        Returns updated list [t0, t1, t2, s0, s1, s2]     
        
        Parameters
        ----------
        t0, t1, t2, s0, s1, s2 : BerkovichPoint
        X, Y : List(BerkovichPoint)

    """ 
    p = prime(t0)
    # Create list length (p+1)^6 of all possible adjacent choices of parameters.
    list = []
    for (i1, i2, i3, i4, i5, i6) in Iterators.product(0:p, 0:p, 0:p, 0:p, 0:p, 0:p)
        push!(list, [update_param(t0,i1), update_param(t1,i2), update_param(t2,i3), update_param(s0,i4), update_param(s1,i5), update_param(s2,i6),])
    end        
    # create list of loss values
    losslist = []
    for i in 1:(p+1)^6
        param = list[i]
        push!(losslist, loss(param[1], param[2], param[3], param[4], param[5], param[6], X, Y))
    end
    j = argmin(losslist)
    return list[j]
end



# write algorithm for simul_descent using simul_descent_step
# termination is when one radii valuation is greater than some bound N, or when simul_descent_step returns the same parameters 

function simul_descent(X, Y, N)
    """ simul_descent(X, Y, N) returns parameters t0, t1, t2, s0, s1, s2 defining a candidate function f which distinguishes sets X and Y
        The algorithm terminates when the descent step no longer improves our loss function, or the radius of one of the parameters reaches N 

    Parameters
    ----------
    X, Y : List(BerkovichPoint)
    N : Int64
    """

    if length(X) == 0
        error("X should not be empty")
    end
    K = parent(X[1].center)
    # this is the p adic field
    t0, t1, t2, s0, s1, s2 = BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0)
    # initialised parameters to the ball at 0 radius 1 (valuation 0)
    r = 0
    while r < N
        new = simul_descent_step(t0, t1, t2, s0, s1, s2, X, Y)
        T0, T1, T2, S0, S1, S2 = new[1], new[2], new[3], new[4], new[5], new[6]
        if [T0, T1, T2, S0, S1, S2] == [t0, t1, t2, s0, s1, s2]
            return [t0, t1, t2, s0, s1, s2]
        end
        # no improvement was found from simul_descent_step
        rad = [T0.radius, T1.radius, T2.radius, S0.radius, S1.radius, S2.radius]
        r = maximum(rad)
        # if this is at least N then break the while loop
        t0, t1, t2, s0, s1, s2 = T0, T1, T2, S0, S1, S2
        # update parameters from descent step
    end
    return [t0, t1, t2, s0, s1, s2]
end

function abs_directional_derivative(f::MPolyRingElem, v, i) 
    T = base_ring(f.parent)
    _, x = polynomial_ring(T, "x")
    p = [j != i ? point(v)[j] : x for j in eachindex(v.tangents)]
    g = AbstractAlgebra.evaluate(f, p)
    return dir_deriv(g, point(v).disks[i], BerkovichPoint(direction(v)[i], 0))
end 


# Compute symbolical gradient of rational function f at evaluated at the base point of v, in the direction of v
function abs_grad_sym(f, v) 
    return BerkovichPolyTangent(point(v), direction(v), [abs_directional_derivative(f, v, i) for i in eachindex(v.tangents)])
end 

# IMPLEMENT ME!
function abs_sum_grad(F, v)
    """ abs_sum_grad(F, v) returns the gradient of the sum of absolute values of rational functions in vector F in the direction of tangent vector v.

    Parameters
    ----------
    F : Vector of rational functions
    a : Tangent vector 
    """
    grad = BerkovichPolyTangent_zero(v)
    for f in F
        grad = grad + abs_grad_sym(f, v)
    end
    return grad
end

#  Implement me! Gradient descent step but allows one to control which directions are used for computing the gradient.
function gradient_step(F, V, alpha)
    """ gradient_step(F, V, alpha) returns a Berkovich point corresponding to taking a step in the direction of the smallest gradient of the sum of absolute values of rational functions in F in the direction of tangent vectors in V. Alpha represents a learning rate which can be modified to control the size of the step.

    Parameters
    ----------
    F : Vector of rational functions
    V : Vector of tangent vector
    alpha : Float64
    """
    # compute the gradient at each tangent vector in V and the norm of the gradient
    grads = [abs_sum_grad(F, v) for v in V]
    norms = [norm(magnitude(grad)) for grad in grads]
    # pick v that gives the greatest gradient
    _, j = findmax(norms)
    v = grads[j]
    # pick center a in the direction of v
    a = point(v)
    r = radius(a)
    # make a step in the direction of v with along the gradient (scaled by alpha)
    for i in eachindex(r)
        r[i] = max(0, r[i] + alpha * magnitude(v)[i])
    end
    return BerkovichPolyDisk(center(a), r)
end

function enumerate_axial_centers(P) 
    centers = []
    for i in eachindex(center(P))
        ith_axial_centers = enumerate_centers(P.disks[i])
        append!(centers, [change_center(P, c, i) for c in ith_axial_centers])
    end   
    return centers
end 

function enumerate_axial_directions(P)
    return [BerkovichPolyTangent(P, center(axial_centers), 1) for axial_centers in enumerate_axial_centers(P)]
end

function enumerate_all_directions(P)
    centers = []
    n = length(P.disks)
    if n > 1
        one_dim_centers = enumerate_centers(P.disks[n])
        temp_centers = enumerate_all_directions(BerkovichPolyDisk(P.disks[1:n-1]))
        for c in one_dim_centers
            for t in temp_centers
                append!(centers, extend(t, P.disks[n], c, 1))
            end
        end
    else 
        return enumerate_centers(P.disks[1])
    end         
end 


# Implement me! Gradient descent algorithm that uses a subset of all (normalized) tangent vectors: those that correspond to shrinking only one radius at a time. 
function restrained_gradient_descent(F, N, P, alpha)
    """ restrained_gradient_descent(F, N, P) returns a BerkovichPolyDisk corresponding to the parameters that are obtained after running N iterations of gradient descent starting at point P. At each step, we only do computations for the n*p axial directions"""
    for _ in 1:N
        P = gradient_step(F, enumerate_axial_directions(P), alpha)
    end 
    return P
end 
        
# Implement me! Gradient descent algorithm that all descending tangent vectors at any given step.  
function greedy_gradient_descent(F, N, P)
    """ greedy_gradient_descent(F, N, P) returns a BerkovichPolyDisk corresponding to the parameters that are obtained after running N iterations of gradient descent starting at point P. """
    for _ in 1:N
        P = gradient_step(F, enumerate_axial_directions(P), alpha)
    end 
    return P
end 

# ########## TEST ############

# K = PadicField(3,prec)
# # Take set X to all be 0 mod 9 (so radius at least 2)
# X1 = BerkovichPoint(K(0), 4)
# X2 = BerkovichPoint(K(9), 3)
# X3 = BerkovichPoint(K(18), 6)
# X4 = BerkovichPoint(K(0), 5)
# X5 = BerkovichPoint(K(27), 2)
# X = [X1, X2, X3, X4, X5]

# # And Y to all be 1 mod 9
# Y1 = BerkovichPoint(K(1), 3)
# Y2 = BerkovichPoint(K(10), 4)
# Y3 = BerkovichPoint(K(10), 5)
# Y4 = BerkovichPoint(K(19), 2)
# Y = [Y1, Y2, Y3, Y4]


# param = simul_descent(X,Y,5)

# x = BerkovichPoint(K(95), 3)
# t0, t1, t2, s0, s1, s2 = param[1], param[2], param[3], param[4], param[5], param[6]
# rat = -num(t0,t1,t2,x) + num(s0,s1,s2,x)
# abs = exp(rat * log(3))
# print(abs)
# ############################

function test_function(t0,t1,t2,s0,s1,s2,x)
    """ test_function(t0,t1,t2,s0,s1,s2,x) evaluates the function defined by parameters t0,...,s2 at the point X

    Parameters
    ----------
    t0, t1, t2, s0, s1, s2, x : BerkovichPoint

    """
    rat = -num(t0,t1,t2,x) + num(s0,s1,s2,x)
    p = prime(t0)
    return exp(rat * log(p))
end

# write a similar algorithm for greedy descent


function greedy_descent(X, Y, N, M)
    """ greedy_descent(X, Y, N) returns parameters t0, t1, t2, s0, s1, s2 defining a candidate function f which distinguishes sets X and Y
        The algorithm terminates when the descent step no longer improves our loss function after M attempts, or the radius of one of the parameters reaches N 

    Parameters
    ----------
    X, Y : List(BerkovichPoint)
    N, M : Int64
    """

    if length(X) == 0
        error("X should not be empty")
    end
    K = parent(X[1].center)
    # this is the p adic field
    t0, t1, t2, s0, s1, s2 = BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0), BerkovichPoint(K(0), 0)
    # initialised parameters to the ball at 0 radius 1 (valuation 0)
    r = 0
    count = 0
    # count the number of consecutive times greedy descent doesn't improve loss
    while r < N
        alpha = rand(1:6)
        # randomly choose one parameter to optimise
        new = greedy_descent_step(t0, t1, t2, s0, s1, s2, X, Y, alpha)
        T0, T1, T2, S0, S1, S2 = new[1], new[2], new[3], new[4], new[5], new[6]
        if [T0, T1, T2, S0, S1, S2] == [t0, t1, t2, s0, s1, s2]
            count += 1
            if count == M
                return [t0, t1, t2, s0, s1, s2]
            end
        else
        # if greedy descent changed a parameter, check if radius now exceeds N
        rad = [T0.radius, T1.radius, T2.radius, S0.radius, S1.radius, S2.radius]
        r = maximum(rad)
        # if this is at least N then break the while loop
        t0, t1, t2, s0, s1, s2 = T0, T1, T2, S0, S1, S2
        # update parameters from descent step
        count = 0
        # reset count if greedy_descent_step changed a parameter
        end
    end
    return [t0, t1, t2, s0, s1, s2]
end

Q = PadicField(3, 10)
s1 = BerkovichPolyDisk([Q(3)], [0.0])
s2 = BerkovichPolyDisk([Q(3)], [9*9*81.0])
R, (x, ) = polynomial_ring(Q, ["x"])
f = (x-Q(3))*(x-Q(12))*(x-Q(30))*(x-Q(84))
g = (x-Q(1))

for i in 1:3
    local s3 = restrained_gradient_descent([f, g], i , s2, 0.001 / Float64(i))
    println(evaluate(f, s3)) 
    #local s4 = restrained_gradient_descent([f, g], i , s2, 0.001 / Float64(i))
    #println(evaluate(f, s4))
end 

#println(evaluate(f, s2))
#println(evaluate(f, s3))
