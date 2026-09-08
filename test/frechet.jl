# Test file for Fréchet mean computation.
#
# This file tests Fréchet mean computation for both p-adic vectors
# and polydiscs in non-Archimedean spaces.

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Fréchet Mean" begin
    prec = 20
    K = PadicField(3, prec)

    # Create test polydiscs
    p1 = ValuationPolydisc([K(1)], [2])
    p2 = ValuationPolydisc([K(2)], [2])
    p3 = ValuationPolydisc([K(29)], [2])

    @testset "Fréchet Mean of p-adic Vectors" begin
        # Test 1: Fréchet mean of p-adic vectors
        result = frechet_mean([[K(1), K(2)], [K(2), K(5)], [K(29), K(32)]])
        @test result == [K(2), K(5)]
    end

    @testset "Fréchet Mean of Polydiscs" begin
        # Test 2: Fréchet mean of polydiscs
        # One refinement selects the residue class containing two of the three samples.
        result = frechet_mean([p1, p2, p3], 1)
        @test result == ValuationPolydisc([K(2)], [1])
    end
end
