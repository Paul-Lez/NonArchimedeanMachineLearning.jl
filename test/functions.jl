# Test file for polynomial evaluation and derivatives.
#
# This file tests polynomial evaluation and directional derivative computation
# on polydisc space, including absolute polynomial sums.

using Test
using Oscar
using NAML

@testset "Polynomial Functions" begin
    # Set up synthetic data
    prec = 20
    K = PadicField(3, prec)
    a1 = [K(1), K(2)]
    r1 = Vector{Int}([1, 2])
    a2 = [K(2), K(2)]
    r2 = Vector{Int}([2, 2])
    p1 = ValuationPolydisc(a1, r1)
    p2 = ValuationPolydisc(a2, r2)
    data = [(p1, 1), (p2, 1)]

    # Create polynomial ring
    R, (x, y) = polynomial_ring(K, ["x", "y"])

    @testset "Polynomial Creation" begin
        # Define polynomials
        f = 3 * x
        g = 2 * y
        # Absolute polynomial sum: |f| + |g| = |3x| + |2y|
        fun = AbsolutePolynomialSum([f, g])

        @test fun isa AbsolutePolynomialSum
        @test length(fun.polys) == 2
    end

    @testset "Directional Derivative" begin
        f = 3 * x
        # Create tangent vector
        v = NAML.basis_vector(p1, p1.center, 1)

        # Test: Directional derivative
        dd = directional_derivative(f, v)
        @test dd isa Number || dd isa Vector  # Result should be numeric
    end
end



