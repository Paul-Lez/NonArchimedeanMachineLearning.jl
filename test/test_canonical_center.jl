using Test
using NonArchimedeanMachineLearning
using Oscar  # For PadicField

@testset "canonical_center with negative radius" begin
    # Set up p-adic field
    prec = 30
    p = 2
    K = PadicField(p, prec)


    # Test 1: Basic functionality with positive radius
    @testset "Positive radius cases" begin
        p1 = ValuationPolydisc([K(4)], [2])
        p2 = ValuationPolydisc([K(4 + 4)], [2])  # Same because 4 = p^2, diff is p^2

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)


        # These should be equal by Berkovich equality
        @test p1 == p2
        @test c1 == c2
        @test hash(p1) == hash(p2)
    end

    # Test 2: Zero radius
    @testset "Zero radius cases" begin
        p1 = ValuationPolydisc([K(5)], [0])
        p2 = ValuationPolydisc([K(7)], [0])
        p3 = ValuationPolydisc([K(5)], [0])

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)
        c3 = canonical_center(p3)


        # With radius 0, polydiscs are equal if v(c1-c2) >= 0
        # v(5-7) = v(-2) = 1 >= 0, so they ARE equal
        @test p1 == p2
        @test c1 == c2
        @test hash(p1) == hash(p2)

        # Same center should be equal
        @test p1 == p3
        @test c1 == c3
        @test hash(p1) == hash(p3)

        # Test with truly different centers (v(c1-c2) < 0)
        # Create K(1/2) which has negative valuation
        # But we can't easily create negative valuation in standard p-adics
        # So instead test that same r=0 gives same canonical for all non-negative val
        p4 = ValuationPolydisc([K(15)], [0])
        c4 = canonical_center(p4)
        @test p1 == p4  # All have v(c1-c2) >= 0
        @test c1 == c4
    end

    # Test 3: Negative radius (the key new test!)
    @testset "Negative radius cases" begin

        # With negative radius, the disc is actually quite large
        # v(a - b) >= r is easier to satisfy when r is negative
        p1 = ValuationPolydisc([K(0)], [-1])
        p2 = ValuationPolydisc([K(1)], [-1])  # v(0-1) = 0 >= -1, so should be equal
        p3 = ValuationPolydisc([K(3)], [-1])  # v(0-3) = 0 >= -1, so should be equal
        p4 = ValuationPolydisc([K(4)], [-1])  # v(0-4) = 2 >= -1, so should be equal

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)
        c3 = canonical_center(p3)
        c4 = canonical_center(p4)



        # These should all be equal with radius -1
        @test p1 == p2
        @test p1 == p3
        @test p1 == p4

        # And should have same canonical center
        @test c1 == c2
        @test c1 == c3
        @test c1 == c4

        # And same hash
        @test hash(p1) == hash(p2)
        @test hash(p1) == hash(p3)
        @test hash(p1) == hash(p4)
    end

    # Test 4: More negative radius cases
    @testset "Various negative radii" begin

        p1 = ValuationPolydisc([K(0)], [-2])
        p2 = ValuationPolydisc([K(8)], [-2])  # v(0-8) = 3 >= -2

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)


        @test p1 == p2
        @test c1 == c2
        @test hash(p1) == hash(p2)
    end

    # Test 5: Multi-dimensional with mixed radii
    @testset "Multi-dimensional with negative radius" begin

        p1 = ValuationPolydisc([K(0), K(1)], [-1, 2])
        p2 = ValuationPolydisc([K(1), K(1)], [-1, 2])  # First coord should match due to r=-1
        p3 = ValuationPolydisc([K(0), K(1 + 4)], [-1, 2])  # Second coord should match due to r=2
        p4 = ValuationPolydisc([K(1), K(1 + 4)], [-1, 2])  # Both should match

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)
        c3 = canonical_center(p3)
        c4 = canonical_center(p4)



        @test p1 == p2
        @test p1 == p3
        @test p1 == p4

        @test c1 == c2
        @test c1 == c3
        @test c1 == c4

        @test hash(p1) == hash(p2)
        @test hash(p1) == hash(p3)
        @test hash(p1) == hash(p4)
    end

    # Test 6: Hash collision detection - ensure distinct polydiscs have distinct hashes
    @testset "No spurious hash collisions" begin

        # Create many polydiscs with different radii
        test_polydiscs = [ValuationPolydisc([K(0)], [i]) for i in -3:5]

        hashes = [hash(p) for p in test_polydiscs]


        # All should have different hashes since they have different radii
        @test length(unique(hashes)) == length(hashes)

        # Test with different centers at same radius
        centers = [K(i) for i in 0:7]
        radii = [-2, -1, 0, 1, 2]
        polydisc_type = typeof(ValuationPolydisc([K(0)], [0]))

        test_cases = polydisc_type[]
        for r in radii
            for c in centers
                push!(test_cases, ValuationPolydisc([c], [r]))
            end
        end


        # Group by hash
        hash_groups = Dict{UInt, Vector{polydisc_type}}()
        for p in test_cases
            h = hash(p)
            if !haskey(hash_groups, h)
                hash_groups[h] = polydisc_type[]
            end
            push!(hash_groups[h], p)
        end


        # Check that all polydiscs in the same hash group are actually equal
        all_valid = true
        for (h, group) in hash_groups
            for i in 1:length(group)
                for j in (i + 1):length(group)
                    if !(group[i] == group[j])
                        all_valid = false
                    end
                end
            end
        end

        @test all_valid

        # Also verify no equal polydiscs have different hashes
        for i in 1:length(test_cases)
            for j in (i + 1):length(test_cases)
                if test_cases[i] == test_cases[j]
                    @test hash(test_cases[i]) == hash(test_cases[j])
                end
            end
        end
    end

    # Test 7: Dict usage with negative radius
    @testset "Dict/Set usage with negative radius" begin

        p1 = ValuationPolydisc([K(0)], [-1])
        p2 = ValuationPolydisc([K(1)], [-1])  # Should be equal to p1
        p3 = ValuationPolydisc([K(0)], [1])   # Different radius
        d = Dict{typeof(p1), String}()

        d[p1] = "first"
        d[p3] = "third"


        # Accessing with p2 (equal to p1) should get "first"
        @test haskey(d, p2)
        @test d[p2] == "first"

        # Setting with p2 should overwrite p1's value
        d[p2] = "second"
        @test d[p1] == "second"
        @test length(d) == 2  # Still only 2 entries


        # Test with Set
        s = Set([p1, p2, p3])
        @test length(s) == 2  # p1 and p2 are the same

        @test p1 in s
        @test p2 in s
        @test p3 in s
    end

    # Test 8: Edge cases
    @testset "Edge cases" begin

        # Very negative radius
        p1 = ValuationPolydisc([K(0)], [-10])
        p2 = ValuationPolydisc([K(1023)], [-10])

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)


        @test p1 == p2
        @test c1 == c2
        @test hash(p1) == hash(p2)

        # Mixed very positive and very negative
        p3 = ValuationPolydisc([K(0), K(0)], [-5, 10])
        p4 = ValuationPolydisc([K(31), K(1024)], [-5, 10])  # v(0-31)=0, v(0-1024)=10


        @test p3 == p4
        @test canonical_center(p3) == canonical_center(p4)
        @test hash(p3) == hash(p4)
    end

    # Test 9: Comprehensive branch coverage for canonical_center
    @testset "Branch coverage: c.v >= r" begin

        # Case 1: c.v > r (strictly greater)
        p1 = ValuationPolydisc([K(4)], [1])  # v(4)=2, r=1, so 2 > 1
        c1 = canonical_center(p1)
        @test c1 == (0,)

        # Case 2: c.v == r (exactly equal)
        p2 = ValuationPolydisc([K(4)], [2])  # v(4)=2, r=2, so 2 == 2
        c2 = canonical_center(p2)
        @test c2 == (0,)

        # Case 3: c is zero (v=30 for our precision)
        p3 = ValuationPolydisc([K(0)], [5])  # v(0)=30, r=5, so 30 > 5
        c3 = canonical_center(p3)
        @test c3 == (0,)

        # Case 4: Large valuation vs small radius
        p4 = ValuationPolydisc([K(64)], [2])  # v(64)=6, r=2, so 6 > 2
        c4 = canonical_center(p4)
        @test c4 == (0,)

        # Verify these are all Berkovich-equal
        @test p1 != p2  # Different radii
        @test ValuationPolydisc([K(0)], [1]) == ValuationPolydisc([K(4)], [1])
        @test canonical_center(ValuationPolydisc([K(0)], [1])) ==
              canonical_center(ValuationPolydisc([K(4)], [1]))
    end

    @testset "Branch coverage: c.v < r" begin

        # Case 1: c is a unit (v=0), various radii
        p1 = ValuationPolydisc([K(1)], [1])  # v(1)=0, r=1, so 0 < 1
        c1 = canonical_center(p1)
        @test c1 == (1,)  # 1 mod 2^1 = 1

        p2 = ValuationPolydisc([K(1)], [2])  # v(1)=0, r=2
        c2 = canonical_center(p2)
        @test c2 == (1,)  # 1 mod 2^2 = 1

        p3 = ValuationPolydisc([K(3)], [2])  # v(3)=0, r=2
        c3 = canonical_center(p3)
        @test c3 == (3,)  # 3 mod 2^2 = 3

        # Case 2: c has small valuation, larger radius
        p4 = ValuationPolydisc([K(2)], [2])  # v(2)=1, r=2, so 1 < 2
        c4 = canonical_center(p4)
        @test c4 == (2,)  # 2 mod 2^2 = 2

        p5 = ValuationPolydisc([K(2)], [3])  # v(2)=1, r=3, so 1 < 3
        c5 = canonical_center(p5)
        @test c5 == (2,)  # 2 mod 2^3 = 2

        p6 = ValuationPolydisc([K(6)], [3])  # v(6)=1 (since 6=2*3), r=3
        c6 = canonical_center(p6)
        @test c6 == (6,)  # 6 mod 2^3 = 6

        # Case 3: Verify formula works correctly
        # For c = p^v * u, canonical should be (p^v * (u mod p^(r-v)))
        p7 = ValuationPolydisc([K(10)], [4])  # 10 = 2*5, v=1
        c7 = canonical_center(p7)
        # 10 = 2^1 * 5, so canonical = 2^1 * (5 mod 2^3) = 2 * 5 = 10
        @test c7 == (10,)

        # Verify these create different hashes when they should
        @test p1 != p4  # Different centers, same radius
        @test hash(p1) != hash(p4)
    end

    @testset "Negative radius comprehensive" begin

        # All polydiscs with same negative radius should be equal
        test_centers = [K(0), K(1), K(2), K(3), K(7), K(15), K(32), K(100)]

        for r in [-1, -2, -5, -10]
            polydiscs = [ValuationPolydisc([c], [r]) for c in test_centers]
            canonicals = [canonical_center(pd) for pd in polydiscs]

            # All should have same canonical center
            @test all(can == (0,) for can in canonicals)

            # All should be pairwise equal
            for i in 1:length(polydiscs)
                for j in (i + 1):length(polydiscs)
                    @test polydiscs[i] == polydiscs[j]
                    @test hash(polydiscs[i]) == hash(polydiscs[j])
                end
            end
        end
    end

    @testset "Edge cases and boundary conditions" begin

        # Boundary: c.v exactly equals r
        for r in [0, 1, 2, 3, 5]
            c_val = K(p^r)  # Create element with valuation exactly r (p=2 from line 9)
            pd = ValuationPolydisc([c_val], [r])
            can = canonical_center(pd)
            @test can == (0,)  # Should hit c.v >= r branch
        end

        # Large radius
        p1 = ValuationPolydisc([K(1)], [10])
        c1 = canonical_center(p1)
        @test c1 == (1,)

        # Multi-dimensional: mix of both branches
        p2 = ValuationPolydisc([K(1), K(4), K(2)], [2, 2, 3])
        # v(1)=0 < 2 (formula), v(4)=2 >= 2 (zero), v(2)=1 < 3 (formula)
        c2 = canonical_center(p2)
        @test c2 == (1, 0, 2)

        # Multi-dimensional with negative radius in one coordinate
        p3 = ValuationPolydisc([K(1), K(5)], [-1, 2])
        c3 = canonical_center(p3)
        @test c3 == (0, 1)  # First coord: any v >= -1 gives 0, second: 5 mod 4 = 1
    end

end
