# Test file for tangent vector operations.
#
# This file demonstrates and tests tangent vector creation, operations,
# and manipulations in polydisc space.

using Test
using Oscar
using NAML

@testset "Tangent Vector Operations" begin
    prec = 20
    K = PadicField(3, prec)

    a1 = [K(1), K(2)]
    r1 = [1.0, 2.0]

    p1 = ValuationPolydisc(a1, r1)

    @testset "Tangent Vector Creation" begin
        # Create tangent vectors
        v1 = ValuationTangent(p1, a1, [1.0, 1.0])
        @test v1 isa ValuationTangent
        @test v1.point == p1
        @test v1.direction == a1
        @test v1.magnitude == [1.0, 1.0]

        v2 = NAML.zero(v1)  # Zero vector in same space
        @test v2 isa ValuationTangent
        @test v2.magnitude == [0.0, 0.0]

        v3 = NAML.basis_vector(v1, 1)  # Basis vector
        @test v3 isa ValuationTangent
    end

    @testset "Tangent Vector Addition" begin
        v1 = ValuationTangent(p1, a1, [1.0, 1.0])
        v2 = NAML.zero(v1)
        v3 = NAML.basis_vector(v1, 1)

        # Test: Tangent vector addition
        result1 = v1 + v2
        @test result1 isa ValuationTangent
        @test result1.magnitude == v1.magnitude  # Adding zero doesn't change

        result2 = v1 + v1
        @test result2 isa ValuationTangent
        @test result2.magnitude == [2.0, 2.0]  # Doubled

        result3 = v2 + v3
        @test result3 isa ValuationTangent
    end
end