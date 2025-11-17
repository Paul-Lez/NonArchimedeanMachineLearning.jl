# Test file for Fréchet mean computation.
#
# This file tests Fréchet mean computation for both p-adic vectors
# and polydiscs in non-Archimedean spaces.

include("../src/statistics/frechet.jl")

prec = 20
K = PadicField(3,prec)

# Create test polydiscs
p1 = ValuationPolydisc([K(1)], [0])
p2 = ValuationPolydisc([K(2)], [0])
p3 = ValuationPolydisc([K(29)], [0])

# Test 1: Fréchet mean of p-adic vectors
frechet_mean([[K(1), K(2)], [K(2), K(5)], [K(29), K(32)]])

# Test 2: Fréchet mean of polydiscs
frechet_mean([p1, p2, p3], 20)