include("../src/functions.jl")

# set up some random synthetic data
prec = 20
K = PadicField(3, prec)
a1 = [K(1), K(2)]
r1 = Vector{Int}([1, 2])
a2 = [K(2), K(2)]
r2 = Vector{Int}([2, 2])
p1 = ValuationPolydisc(a1, r1)
p2 = ValuationPolydisc(a2, r2)
data = [(p1, 1), (p2, 1)]
R, (x, y) = polynomial_ring(K, ["x", "y"])#, "θ", "ϕ"])

f = 3 * x#+3*θ+ϕ^15
g = 2 * y#+81*θ^89-2*ϕ^13
# Take the model to be the absolute polynomial sum f(x, y, θ, ϕ) = |3x+3θ+ϕ^15| + |2y+81θ+2ϕ|
fun = AbsolutePolynomialSum([f, g])

v = basis_vector(p1, p1.center, 1)

@show directional_derivative(f, v)



