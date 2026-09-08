# Test file for tangent vector operations.
#
# This file demonstrates and tests tangent vector creation, operations,
# and manipulations in polydisc space.

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Tangent Vector Operations" begin
    prec = 20
    K = PadicField(3, prec)

    a1 = [K(1), K(2)]
    r1 = [1.0, 2.0]

    p1 = ValuationPolydisc(a1, r1)
    # Direction polydisc: same center as p1, radius = p1.radius .+ magnitude [1.0, 1.0]
    dir1 = ValuationPolydisc(a1, [2.0, 3.0])

    @testset "Tangent Vector Creation" begin
        # Create tangent vectors
        v1 = ValuationTangent(p1, dir1, [1.0, 1.0])
        v2 = NonArchimedeanMachineLearning.zero(v1)  # Zero vector in same space
        @test v2.magnitude == [0.0, 0.0]

        v3 = NonArchimedeanMachineLearning.basis_vector(v1, 1)
        @test v3.magnitude == [1.0, 0.0]
    end

    @testset "Tangent Vector Addition" begin
        v1 = ValuationTangent(p1, dir1, [1.0, 1.0])
        v2 = NonArchimedeanMachineLearning.zero(v1)

        # Test: Tangent vector addition
        result1 = v1 + v2
        @test result1.magnitude == v1.magnitude  # Adding zero doesn't change

        result2 = v1 + v1
        @test result2.magnitude == [2.0, 2.0]  # Doubled
    end
end
