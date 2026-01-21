# Test file for basic polydisc operations.
#
# This file demonstrates and tests the fundamental polydisc operations
# including creation, iteration, joining, and generating children.

using Test
using Oscar
using NAML

@testset "Polydisc Operations" begin
    prec = 20
    K = PadicField(3, prec)

    a1 = [K(1), K(2)]
    r1 = [1, 2]
    a2 = [K(2), K(2)]
    r2 = [2, 2]

    p1 = ValuationPolydisc(a1, r1)
    p2 = ValuationPolydisc(a2, r2)

    @testset "Polydisc Creation" begin
        @test length(p1.center) == 2
        @test length(p1.radius) == 2
        @test p1.center[1] == K(1)
        @test p1.center[2] == K(2)
        @test p1.radius[1] == 1
        @test p1.radius[2] == 2
    end

    @testset "Polydisc Iteration" begin
        # Test: Iterate over polydisc indices
        centers = []
        radii = []
        for i in Base.eachindex(p1)
            push!(centers, p1.center[i])
            push!(radii, p1.radius[i])
        end
        @test length(centers) == 2
        @test length(radii) == 2
        @test centers == [K(1), K(2)]
        @test radii == [1, 2]
    end

    @testset "Polydisc Join" begin
        # Test: Join (smallest common ancestor) of two polydiscs
        j = NAML.join(p1, p2)  # Using qualified name to avoid conflict with Base.join
        @test j isa ValuationPolydisc
        @test length(j.center) == 2
    end

    @testset "Polydisc Children" begin
        # Test: Generate children of a polydisc
        L = PadicField(2, prec)
        gauss = ValuationPolydisc([L(0)], [2])
        ch = children(gauss)
        @test ch isa Vector
        @test length(ch) > 0
        # Each child should be a ValuationPolydisc
        @test all(c -> c isa ValuationPolydisc, ch)
    end
end




