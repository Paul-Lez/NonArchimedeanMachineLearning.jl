import numpy as np

"""B = Berkovich_Cp_Affine(3)
K = Qp(3,10)
R.<t> = K[]
#R.<t> = PolynomialRing(QQ, 't')
#f = t+1
g = t^2+2*t"""

def berkovichValuation(f, a, r, p): 
    """ berkovichValuation(f, a, r, p) returns the valuation of polynomial f over Q_p at a point of type II or III corresponding to the ball B(a,r)

    Parameters
    ----------
    f : PolynomialPolynomialRing(Qp(p,10), 't') 
    a : Qp(3,10)
        The center of the ball
    r : QQ 
        The radius of the ball
    p : int
        the residue characteristic
    """ 
   
    m = float('inf')
    h = g (t = r*t + a)
    for i in range(0, f.degree(t)+1):
        l = h.valuation_of_coefficient(i)
        m = min(m, l)
    return p**(-1*m)

def berkovichVal(f, b):
    """ berkovichVal(f, b) returns the valuation of polynomial f over Q_p at a point b

    Parameters
    ----------
    f : PolynomialPolynomialRing(Qp(p,10), 't') 
    b : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    """
    return berkovichValuation(f, b.center(), b.radius(), b.prime())

   
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
    if t <= -r:
        return b1
    elif t >= 0:
        return b2
    else :
        return B(b1.center(), p**(t+b2.power()))
        
    
def path_helper_gt(b1, b2, t):
    """ path_helper_gt(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed (starting at t = 0 and ending at t = dist(b1, b2) ). This assumes that b2 < b1.
    

    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    p = b1.prime()
    r = b1.big_metric(b2)
    if t <= 0:
        return b1
    elif t >= r:
        print("here")
        return b2
    else :
        return B(b1.center(), p**(b1.power() - t))
    

def path(b1, b2, t):
    """ path(b1, b2, t) returns the value attained at time t when traversing path [b1, b2] at unit speed (starting at t = - dist(b1, b2.join(b1)) and ending at t = dist(b2.join(b1, b1)). 
    

    Parameters
    ----------
    b1, b2 : Berkovich_Cp_Affine(3)
        A point of type I, II or III
    t : any real number
    """
    p = b1.prime()
    b3 = b1.join(b2)
    r1 = b1.big_metric(b3)
    r2 = b3.big_metric(b2)
    if t <= -r1:
        return b1
    elif -r1 <= t and t <= 0:
        return path_helper_lt(b1, b3, t)
    elif 0 <= t and t <= r2:
        return path_helper_gt(b3, b2, t)
    else:
        return b2
    
berkovichValuation(g, 1, 1, 3)

Q1 = B(2, 1)
Q2 = B(3)
Q3 = Q1.join(Q2)
print(Q3.big_metric(Q2))

path(Q1, Q2, 78)
