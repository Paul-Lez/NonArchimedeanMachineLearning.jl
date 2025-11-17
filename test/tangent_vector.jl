# Test file for tangent vector operations.
#
# This file demonstrates and tests tangent vector creation, operations,
# and manipulations in polydisc space.

include("../src/basic/tangent_vector.jl")

prec = 20
K = PadicField(3, prec)

a1 = [K(1), K(2)]
r1 = [1.0, 2.0]

p1 = ValuationPolydisc(a1, r1)

# Create tangent vectors
v1 = ValuationTangent(p1, a1, [1.0, 1.0])
v2 = zero(v1)  # Zero vector in same space
v3 = basis_vector(v1, 1)  # Basis vector

# Test: Tangent vector addition
@show v1 + v2
@show v1 + v1
@show v2 + v3