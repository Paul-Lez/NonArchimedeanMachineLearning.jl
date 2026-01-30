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

    @testset "Polydisc Accessors" begin
        # Test: center, radius, dim, prime accessors
        @test collect(NAML.center(p1)) == a1
        @test collect(NAML.radius(p1)) == r1
        @test NAML.dim(p1) == 2
        @test NAML.prime(p1) == 3
    end

    @testset "Polydisc Distance" begin
        # Test: distance between polydiscs
        d = NAML.dist(p1, p2)
        @test d isa Float64
        @test d >= 0
        # Distance to self should be 0
        @test NAML.dist(p1, p1) == 0.0
    end

    @testset "Polydisc Concatenate" begin
        # Test: concatenate two polydiscs
        L = PadicField(2, prec)
        q1 = ValuationPolydisc([L(1)], [1])
        q2 = ValuationPolydisc([L(2)], [2])
        q_concat = NAML.concatenate(q1, q2)
        @test NAML.dim(q_concat) == 2
        @test length(NAML.center(q_concat)) == 2
        @test length(NAML.radius(q_concat)) == 2
        @test NAML.center(q_concat)[1] == L(1)
        @test NAML.center(q_concat)[2] == L(2)
        @test NAML.radius(q_concat)[1] == 1
        @test NAML.radius(q_concat)[2] == 2
    end

    @testset "Polydisc Children Along Branch" begin
        # Test: generate children along a specific branch
        L = PadicField(2, prec)
        gauss = ValuationPolydisc([L(0), L(0)], [2, 2])
        ch_branch = children_along_branch(gauss, 1)
        @test ch_branch isa Vector
        @test length(ch_branch) == 2  # NAML.prime(gauss) = 2
        # All children should have increased radius in first coordinate
        @test all(c -> NAML.radius(c)[1] == 3, ch_branch)
        # Second coordinate radius should be unchanged
        @test all(c -> NAML.radius(c)[2] == 2, ch_branch)
    end

    @testset "Polydisc Equality" begin
        # Test: equality operator
        p_same = ValuationPolydisc([K(1), K(2)], [1, 2])
        @test p1 == p_same
        # Different radius should not be equal
        p_diff_radius = ValuationPolydisc([K(1), K(2)], [2, 2])
        @test !(p1 == p_diff_radius)
    end

    @testset "Polydisc Containment" begin
        # Test: <= operator (containment)
        # Smaller polydisc (larger radii values) contained in larger one
        p_small = ValuationPolydisc([K(1), K(2)], [2, 3])
        p_large = ValuationPolydisc([K(1), K(2)], [1, 2])
        @test p_small <= p_large
        @test !(p_large <= p_small)
    end

    @testset "Polydisc Subdisc and Components" begin
        # Test: extract subdisc
        p_sub = NAML.subdisc(p1, [1])
        @test NAML.dim(p_sub) == 1
        @test NAML.center(p_sub)[1] == K(1)
        @test NAML.radius(p_sub)[1] == 1

        # Test: get components
        comps = NAML.components(p1)
        @test length(comps) == 2
        @test all(c -> NAML.dim(c) == 1, comps)
        @test NAML.center(comps[1])[1] == K(1)
        @test NAML.center(comps[2])[1] == K(2)
    end

    @testset "Polydisc Hash and Canonical Center" begin
        # Test canonical_center function
        L = PadicField(2, prec)

        # Note: Polydisc equality uses STRICT inequality: v(center_diff) > radius
        # So for radius=2, we need v(diff) > 2, meaning diff divisible by 2^3=8

        # Create polydiscs with the same canonical form
        # Two polydiscs that are equal should have the same canonical_center
        p_a = ValuationPolydisc([L(1)], [2])
        p_b = ValuationPolydisc([L(1 + 8)], [2])  # v(8) = 3 > 2, so these are equal

        @test canonical_center(p_a) == canonical_center(p_b)
        @test p_a == p_b  # They should be equal polydiscs
        @test hash(p_a) == hash(p_b)  # Equal polydiscs must have equal hashes

        # Create polydiscs with different canonical forms
        p_c = ValuationPolydisc([L(3)], [2])  # 3 mod 4 = 3, different from 1 mod 4 = 1
        p_d = ValuationPolydisc([L(2)], [2])  # v(2-3) = v(-1) = 0, so different from p_c

        @test canonical_center(p_a) != canonical_center(p_c)  # 1 mod 4 != 3 mod 4
        @test !(p_a == p_c)
        # Different polydiscs should (almost always) have different hashes
        @test hash(p_a) != hash(p_c)

        # Test hash works correctly for Dict usage
        dict = Dict{ValuationPolydisc, Int}()
        dict[p_a] = 1
        dict[p_c] = 2

        # p_b should map to same entry as p_a (they're equal)
        @test haskey(dict, p_b)
        @test dict[p_b] == 1

        # Verify dict has correct number of entries (p_a and p_c are different)
        @test length(dict) == 2

        # Test with different radii (should be different hash)
        p_e = ValuationPolydisc([L(1)], [3])
        @test hash(p_a) != hash(p_e)

        # Test multidimensional polydiscs - need v(diff) > radius for equality
        p_2d_a = ValuationPolydisc([L(1), L(2)], [2, 3])
        p_2d_b = ValuationPolydisc([L(1 + 8), L(2 + 16)], [2, 3])  # v(8)=3>2, v(16)=4>3
        @test canonical_center(p_2d_a) == canonical_center(p_2d_b)
        @test hash(p_2d_a) == hash(p_2d_b)

        # Test transposition scenario: different paths to same polydisc
        # Start from a parent and go to children
        parent = ValuationPolydisc([L(0), L(0)], [0, 0])
        ch = children(parent, 2)  # degree 2 children

        # Verify children have valid hashes for Dict usage
        child_dict = Dict{ValuationPolydisc, Int}()
        for (i, c) in enumerate(ch)
            child_dict[c] = i
        end
        @test length(child_dict) == length(ch)  # All children should be unique
    end
end




