# Test file for Fréchet mean computation.
#
# This file tests Fréchet mean computation for both p-adic vectors
# and polydiscs in non-Archimedean spaces.

using Test
using Oscar
using NAML

@testset "Fréchet Mean" begin
    prec = 20
    K = PadicField(3, prec)

    # Create test polydiscs
    p1 = ValuationPolydisc([K(1)], [0])
    p2 = ValuationPolydisc([K(2)], [0])
    p3 = ValuationPolydisc([K(29)], [0])

    @testset "Fréchet Mean of p-adic Vectors" begin
        # Test 1: Fréchet mean of p-adic vectors
        result = frechet_mean([[K(1), K(2)], [K(2), K(5)], [K(29), K(32)]])
        @test result isa Vector
        @test length(result) == 2
        @test all(x -> x isa PadicFieldElem, result)
    end

    @testset "Fréchet Mean of Polydiscs" begin
        # Test 2: Fréchet mean of polydiscs
        result = frechet_mean([p1, p2, p3], 20)
        @test result isa ValuationPolydisc
        @test length(result.center) == 1
    end
end