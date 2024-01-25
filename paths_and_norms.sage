import numpy as np
import matplotlib.pyplot as plt
from sage.all import * 

def berkovichValuation(f, a, r): 
    """ berkovichValuation(f, a, r, p) returns the valuation of polynomial f over Q_p at a point of type II or III corresponding to the ball B(a,r)

    Parameters
    ----------
    f : PolynomialPolynomialRing(Qp(p,10), 't') 
    a : Qp(p,10)
        The center of the ball
    r : QQ 
        The radius of the ball
    p : int
        the residue characteristic
    """ 
    #we use the formula |f| = max (r^m |b_m|) where f = b_n (t-a)^n + ... + b_0 
    if r == 0: 
        return f(a).abs()
    else:
        m = float('inf')
        # consider the polynomial h = g(t+a) since the the m-th coefficient of h is the same as the m-th coefficient of g (expanded around t = a)
        h = f (t = t + a)
        e = log(r, 1/a.parent().prime()) 
        # the max taken wrt absolute values corresponds to the min wrt valuations
        for i in range(0, f.degree(t)+1):
            m = min(m, h.valuation_of_coefficient(i) + i*e)
        # convert result back to absolute value
        return a.parent().prime()**(-m)

def berkovichVal(f, b):
    """ berkovichVal(f, b) returns the valuation of polynomial f over Q_p at a point b

    Parameters
    ----------
    f : PolynomialPolynomialRing(Qp(p,10), 't') 
    b : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    """
    return berkovichValuation(f, b.center(), b.radius())

   
def path_helper_lt(b1, b2, t):
    """ path_helper_lt(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed (starting at t = -dist(b1, b2) and ending at t = 0). This assumes that b1 < b2.
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    p = b1.prime()
    r = b1.big_metric(b2)
    # if t is less than r, stay at b1
    if t <= -r:
        return b1
    # if t is greater than 0 then we are already at b2
    elif t >= 0:
        return b2
    else:
        # otherwise return the ball with the same center as b1 and with radius b2.radius()*p^t
        return B(b1.center(), p**(t+b2.power()))
        
    
def path_helper_gt(b1, b2, t):
    """ path_helper_gt(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed (starting at t = 0 and ending at t = dist(b1, b2)). This assumes that b2 < b1.
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    p = b1.prime()
    r = b1.big_metric(b2)
    # if t is less than 0, stay at b1
    if t <= 0:
        return b1
    elif t >= r:
        # if t is greater than r then we are already at b2
        return b2
    else :
        # otherwise return the ball with the same center as b1 and with radius b1.radius()*p^(-t)
        return B(b2.center(), p**(b1.power() - t))
    

def path(b1, b2, t):
    """ path(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed 
    (starting at t = - dist(b1, b2.join(b1)) and ending at t = dist(b2.join(b1, b1)). 
    
    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    b3 = b1.join(b2)
    r1 = b1.big_metric(b3)
    r2 = b3.big_metric(b2)
    # if t is less than -r1 then stay at p1 
    if t <= -r1:
        return b1
    # if t is between -r1 and 0 then go along the path [b1, b3]
    elif -r1 <= t and t <= 0:
        return path_helper_lt(b1, b3, t)
    # if t is between 0 and r2 then go along the path [b3, b2]
    elif 0 <= t and t <= r2:
        return path_helper_gt(b3, b2, t)
    else:
    # is t is greater than r2 then stay at p2
        return b2
    
def abs_path_values(f, b1, b2, t1, t2, num):
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
    for i in range(0, num):
        im[i] = berkovichVal(f, path(b1, b2, RealField(50)(l[i])))
    return l, im

def abs_path_values_sum(s, b1, b2, t1, t2, num):
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
    for f in s:
        l, im = abs_path_values(f, b1, b2, t1, t2, num)
        a += im
    return l, a


###########################################################
################ Some tests ###############################

## f_t(x) is the polynomial in t given by (x - t)(x-2t)(x-4t)
def f_t(x): 
    return (x - t)*(x-2*t)*(x-4*t)

B = Berkovich_Cp_Affine(2)
K = Qp(2,20)
R.<t> = K[]

Q1 = B(8)
Q2 = B(1)
Q3 = Q1.join(Q2)

print(berkovichVal(f_t(1), Q1))
print(berkovichVal(f_t(2), Q1))
print(berkovichVal(f_t(4), Q1))
print(berkovichVal(f_t(4), Q2))

x, y = abs_path_values_sum([f_t(1), f_t(2), f_t(4)], Q1, Q2, -10, 10, 2000)
plt.plot(x, y)
plt.show()
