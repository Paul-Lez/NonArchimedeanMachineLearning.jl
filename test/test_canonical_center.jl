using Test
include("../src/NAML.jl")
using .NAML
using Oscar  # For PadicField

@testset "canonical_center with negative radius" begin
    # Set up p-adic field
    prec = 30
    p = 2
    K = PadicField(p, prec)

    println("\n=== Testing canonical_center functionality ===\n")

    # Test 1: Basic functionality with positive radius
    @testset "Positive radius cases" begin
        println("Test 1: Positive radius")
        p1 = ValuationPolydisc([K(4)], [2])
        p2 = ValuationPolydisc([K(4 + 4)], [2])  # Same because 4 = p^2, diff is p^2

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)

        println("  p1 center: $(NAML.center(p1)[1]), radius: $(NAML.radius(p1)[1])")
        println("  p2 center: $(NAML.center(p2)[1]), radius: $(NAML.radius(p2)[1])")
        println("  canonical_center(p1): $c1")
        println("  canonical_center(p2): $c2")
        println("  Are they equal? $(p1 == p2)")
        println("  Same canonical center? $(c1 == c2)")

        # These should be equal by Berkovich equality
        @test p1 == p2
        @test c1 == c2
        @test hash(p1) == hash(p2)
    end

    # Test 2: Zero radius
    @testset "Zero radius cases" begin
        println("\nTest 2: Zero radius")
        p1 = ValuationPolydisc([K(5)], [0])
        p2 = ValuationPolydisc([K(7)], [0])
        p3 = ValuationPolydisc([K(5)], [0])

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)
        c3 = canonical_center(p3)

        println("  p1 center: $(NAML.center(p1)[1]), canonical: $c1")
        println("  p2 center: $(NAML.center(p2)[1]), canonical: $c2")
        println("  p3 center: $(NAML.center(p3)[1]), canonical: $c3")
        println("  valuation(p1 - p2) = $(NAML.valuation(NAML.center(p1)[1] - NAML.center(p2)[1]))")

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
        println("  p4 center: $(NAML.center(p4)[1]), canonical: $c4")
        @test p1 == p4  # All have v(c1-c2) >= 0
        @test c1 == c4
    end

    # Test 3: Negative radius (the key new test!)
    @testset "Negative radius cases" begin
        println("\nTest 3: Negative radius")

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

        println("  p1: center=$(NAML.center(p1)[1]), radius=-1, canonical=$c1")
        println("  p2: center=$(NAML.center(p2)[1]), radius=-1, canonical=$c2")
        println("  p3: center=$(NAML.center(p3)[1]), radius=-1, canonical=$c3")
        println("  p4: center=$(NAML.center(p4)[1]), radius=-1, canonical=$c4")

        println("  p1 == p2? $(p1 == p2)")
        println("  p1 == p3? $(p1 == p3)")
        println("  p1 == p4? $(p1 == p4)")

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
        println("\nTest 4: Various negative radii")

        p1 = ValuationPolydisc([K(0)], [-2])
        p2 = ValuationPolydisc([K(8)], [-2])  # v(0-8) = 3 >= -2

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)

        println("  p1: center=$(NAML.center(p1)[1]), radius=-2, canonical=$c1")
        println("  p2: center=$(NAML.center(p2)[1]), radius=-2, canonical=$c2")
        println("  p1 == p2? $(p1 == p2)")

        @test p1 == p2
        @test c1 == c2
        @test hash(p1) == hash(p2)
    end

    # Test 5: Multi-dimensional with mixed radii
    @testset "Multi-dimensional with negative radius" begin
        println("\nTest 5: Multi-dimensional with mixed radii")

        p1 = ValuationPolydisc([K(0), K(1)], [-1, 2])
        p2 = ValuationPolydisc([K(1), K(1)], [-1, 2])  # First coord should match due to r=-1
        p3 = ValuationPolydisc([K(0), K(1 + 4)], [-1, 2])  # Second coord should match due to r=2
        p4 = ValuationPolydisc([K(1), K(1 + 4)], [-1, 2])  # Both should match

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)
        c3 = canonical_center(p3)
        c4 = canonical_center(p4)

        println("  p1: canonical=$c1")
        println("  p2: canonical=$c2")
        println("  p3: canonical=$c3")
        println("  p4: canonical=$c4")

        println("  p1 == p2? $(p1 == p2)")
        println("  p1 == p3? $(p1 == p3)")
        println("  p1 == p4? $(p1 == p4)")

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
        println("\nTest 6: Hash collision detection")

        # Create many polydiscs with different radii
        test_polydiscs = [
            ValuationPolydisc([K(0)], [i]) for i in -3:5
        ]

        hashes = [hash(p) for p in test_polydiscs]

        println("  Testing $(length(test_polydiscs)) polydiscs with radii from -3 to 5")
        println("  Unique hashes: $(length(unique(hashes)))")

        # All should have different hashes since they have different radii
        @test length(unique(hashes)) == length(hashes)

        # Test with different centers at same radius
        centers = [K(i) for i in 0:7]
        radii = [-2, -1, 0, 1, 2]

        test_cases = []
        for r in radii
            for c in centers
                push!(test_cases, ValuationPolydisc([c], [r]))
            end
        end

        println("  Testing $(length(test_cases)) polydiscs ($(length(centers)) centers × $(length(radii)) radii)")

        # Group by hash
        hash_groups = Dict{UInt, Vector{ValuationPolydisc{PadicFieldElem, Int, 1}}}()
        for p in test_cases
            h = hash(p)
            if !haskey(hash_groups, h)
                hash_groups[h] = []
            end
            push!(hash_groups[h], p)
        end

        println("  Number of unique hash values: $(length(hash_groups))")

        # Check that all polydiscs in the same hash group are actually equal
        all_valid = true
        for (h, group) in hash_groups
            for i in 1:length(group)
                for j in i+1:length(group)
                    if !(group[i] == group[j])
                        println("  ERROR: Hash collision between non-equal polydiscs!")
                        println("    p1: center=$(NAML.center(group[i])[1]), radius=$(NAML.radius(group[i])[1])")
                        println("    p2: center=$(NAML.center(group[j])[1]), radius=$(NAML.radius(group[j])[1])")
                        println("    canonical_center(p1): $(canonical_center(group[i]))")
                        println("    canonical_center(p2): $(canonical_center(group[j]))")
                        println("    hash: $h")
                        println("    p1.center[1].N = $(group[i].center[1].N)")
                        println("    p1.center[1].u = $(group[i].center[1].u)")
                        println("    p2.center[1].N = $(group[j].center[1].N)")
                        println("    p2.center[1].u = $(group[j].center[1].u)")
                        println("    valuation(p1.center - p2.center) = $(valuation(group[i].center[1] - group[j].center[1]))")
                        all_valid = false
                    end
                end
            end
        end

        @test all_valid

        # Also verify no equal polydiscs have different hashes
        for i in 1:length(test_cases)
            for j in i+1:length(test_cases)
                if test_cases[i] == test_cases[j]
                    @test hash(test_cases[i]) == hash(test_cases[j])
                end
            end
        end
    end

    # Test 7: Dict usage with negative radius
    @testset "Dict/Set usage with negative radius" begin
        println("\nTest 7: Dict and Set usage")

        # Test that polydiscs can be used as Dict keys correctly
        d = Dict{ValuationPolydisc{PadicFieldElem, Int, 1}, String}()

        p1 = ValuationPolydisc([K(0)], [-1])
        p2 = ValuationPolydisc([K(1)], [-1])  # Should be equal to p1
        p3 = ValuationPolydisc([K(0)], [1])   # Different radius

        d[p1] = "first"
        d[p3] = "third"

        println("  Added p1 (r=-1) and p3 (r=1) to dict")
        println("  Dict size: $(length(d))")

        # Accessing with p2 (equal to p1) should get "first"
        @test haskey(d, p2)
        @test d[p2] == "first"

        # Setting with p2 should overwrite p1's value
        d[p2] = "second"
        @test d[p1] == "second"
        @test length(d) == 2  # Still only 2 entries

        println("  After updating via p2 (equal to p1): $(d[p1])")
        println("  Dict size: $(length(d))")

        # Test with Set
        s = Set([p1, p2, p3])
        println("  Set with p1, p2 (equal to p1), p3: size=$(length(s))")
        @test length(s) == 2  # p1 and p2 are the same

        @test p1 in s
        @test p2 in s
        @test p3 in s
    end

    # Test 8: Edge cases
    @testset "Edge cases" begin
        println("\nTest 8: Edge cases")

        # Very negative radius
        p1 = ValuationPolydisc([K(0)], [-10])
        p2 = ValuationPolydisc([K(1023)], [-10])

        c1 = canonical_center(p1)
        c2 = canonical_center(p2)

        println("  Very negative radius (-10):")
        println("    p1: canonical=$c1")
        println("    p2: canonical=$c2")
        println("    Equal? $(p1 == p2)")

        @test p1 == p2
        @test c1 == c2
        @test hash(p1) == hash(p2)

        # Mixed very positive and very negative
        p3 = ValuationPolydisc([K(0), K(0)], [-5, 10])
        p4 = ValuationPolydisc([K(31), K(1024)], [-5, 10])  # v(0-31)=0, v(0-1024)=10

        println("  Mixed radii (-5, 10):")
        println("    p3 == p4? $(p3 == p4)")

        @test p3 == p4
        @test canonical_center(p3) == canonical_center(p4)
        @test hash(p3) == hash(p4)
    end

    # Test 9: Comprehensive branch coverage for canonical_center
    @testset "Branch coverage: c.v >= r" begin
        println("\nTest 9: Branch coverage for c.v >= r (returns 0)")

        # Case 1: c.v > r (strictly greater)
        p1 = ValuationPolydisc([K(4)], [1])  # v(4)=2, r=1, so 2 > 1
        c1 = canonical_center(p1)
        println("  K(4) with r=1: v=2 > r=1, canonical=$c1")
        @test c1 == (0,)

        # Case 2: c.v == r (exactly equal)
        p2 = ValuationPolydisc([K(4)], [2])  # v(4)=2, r=2, so 2 == 2
        c2 = canonical_center(p2)
        println("  K(4) with r=2: v=2 == r=2, canonical=$c2")
        @test c2 == (0,)

        # Case 3: c is zero (v=30 for our precision)
        p3 = ValuationPolydisc([K(0)], [5])  # v(0)=30, r=5, so 30 > 5
        c3 = canonical_center(p3)
        println("  K(0) with r=5: v=30 > r=5, canonical=$c3")
        @test c3 == (0,)

        # Case 4: Large valuation vs small radius
        p4 = ValuationPolydisc([K(64)], [2])  # v(64)=6, r=2, so 6 > 2
        c4 = canonical_center(p4)
        println("  K(64) with r=2: v=6 > r=2, canonical=$c4")
        @test c4 == (0,)

        # Verify these are all Berkovich-equal
        @test p1 != p2  # Different radii
        @test ValuationPolydisc([K(0)], [1]) == ValuationPolydisc([K(4)], [1])
        @test canonical_center(ValuationPolydisc([K(0)], [1])) == canonical_center(ValuationPolydisc([K(4)], [1]))
    end

    @testset "Branch coverage: c.v < r" begin
        println("\nTest 10: Branch coverage for c.v < r (computes formula)")

        # Case 1: c is a unit (v=0), various radii
        p1 = ValuationPolydisc([K(1)], [1])  # v(1)=0, r=1, so 0 < 1
        c1 = canonical_center(p1)
        println("  K(1) with r=1: v=0 < r=1, canonical=$c1")
        @test c1 == (1,)  # 1 mod 2^1 = 1

        p2 = ValuationPolydisc([K(1)], [2])  # v(1)=0, r=2
        c2 = canonical_center(p2)
        println("  K(1) with r=2: v=0 < r=2, canonical=$c2")
        @test c2 == (1,)  # 1 mod 2^2 = 1

        p3 = ValuationPolydisc([K(3)], [2])  # v(3)=0, r=2
        c3 = canonical_center(p3)
        println("  K(3) with r=2: v=0 < r=2, canonical=$c3")
        @test c3 == (3,)  # 3 mod 2^2 = 3

        # Case 2: c has small valuation, larger radius
        p4 = ValuationPolydisc([K(2)], [2])  # v(2)=1, r=2, so 1 < 2
        c4 = canonical_center(p4)
        println("  K(2) with r=2: v=1 < r=2, canonical=$c4")
        @test c4 == (2,)  # 2 mod 2^2 = 2

        p5 = ValuationPolydisc([K(2)], [3])  # v(2)=1, r=3, so 1 < 3
        c5 = canonical_center(p5)
        println("  K(2) with r=3: v=1 < r=3, canonical=$c5")
        @test c5 == (2,)  # 2 mod 2^3 = 2

        p6 = ValuationPolydisc([K(6)], [3])  # v(6)=1 (since 6=2*3), r=3
        c6 = canonical_center(p6)
        println("  K(6) with r=3: v=1 < r=3, canonical=$c6")
        @test c6 == (6,)  # 6 mod 2^3 = 6

        # Case 3: Verify formula works correctly
        # For c = p^v * u, canonical should be (p^v * (u mod p^(r-v)))
        p7 = ValuationPolydisc([K(10)], [4])  # 10 = 2*5, v=1
        c7 = canonical_center(p7)
        println("  K(10) with r=4: v=1 < r=4, canonical=$c7")
        # 10 = 2^1 * 5, so canonical = 2^1 * (5 mod 2^3) = 2 * 5 = 10
        @test c7 == (10,)

        # Verify these create different hashes when they should
        @test p1 != p4  # Different centers, same radius
        @test hash(p1) != hash(p4)
    end

    @testset "Negative radius comprehensive" begin
        println("\nTest 11: Comprehensive negative radius tests")

        # All polydiscs with same negative radius should be equal
        test_centers = [K(0), K(1), K(2), K(3), K(7), K(15), K(32), K(100)]

        for r in [-1, -2, -5, -10]
            println("  Testing radius $r:")
            polydiscs = [ValuationPolydisc([c], [r]) for c in test_centers]
            canonicals = [canonical_center(pd) for pd in polydiscs]

            # All should have same canonical center
            @test all(can == (0,) for can in canonicals)

            # All should be pairwise equal
            for i in 1:length(polydiscs)
                for j in i+1:length(polydiscs)
                    @test polydiscs[i] == polydiscs[j]
                    @test hash(polydiscs[i]) == hash(polydiscs[j])
                end
            end
            println("    ✓ All $(length(test_centers)) centers equal with r=$r")
        end
    end

    @testset "Edge cases and boundary conditions" begin
        println("\nTest 12: Edge cases")

        # Boundary: c.v exactly equals r
        for r in [0, 1, 2, 3, 5]
            c_val = K(p^r)  # Create element with valuation exactly r (p=2 from line 9)
            pd = ValuationPolydisc([c_val], [r])
            can = canonical_center(pd)
            println("  v(c)=$r, r=$r: canonical=$can")
            @test can == (0,)  # Should hit c.v >= r branch
        end

        # Large radius
        p1 = ValuationPolydisc([K(1)], [10])
        c1 = canonical_center(p1)
        println("  K(1) with r=10: canonical=$c1")
        @test c1 == (1,)

        # Multi-dimensional: mix of both branches
        p2 = ValuationPolydisc([K(1), K(4), K(2)], [2, 2, 3])
        # v(1)=0 < 2 (formula), v(4)=2 >= 2 (zero), v(2)=1 < 3 (formula)
        c2 = canonical_center(p2)
        println("  Mixed: canonical=$c2")
        @test c2 == (1, 0, 2)

        # Multi-dimensional with negative radius in one coordinate
        p3 = ValuationPolydisc([K(1), K(5)], [-1, 2])
        c3 = canonical_center(p3)
        println("  Mixed with negative: canonical=$c3")
        @test c3 == (0, 1)  # First coord: any v >= -1 gives 0, second: 5 mod 4 = 1
    end

    println("\n=== All tests completed successfully! ===\n")
end
