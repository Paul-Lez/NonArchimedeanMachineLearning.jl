using Test
using NonArchimedeanMachineLearning
using Oscar

@testset "Canonical centers and hashing" begin
    K = PadicField(2, 30)

    @testset "Reduction modulo the radius" begin
        # Above, at, and below the valuation boundary; zero and non-unit centers.
        for (c, r, expected) in [(4, 1, 0), (4, 2, 0), (0, 5, 0),
                (1, 2, 1), (5, 2, 1), (6, 2, 2), (10, 4, 10)]
            @test canonical_center(ValuationPolydisc([K(c)], [r])) == (expected,)
        end
        mixed = ValuationPolydisc([K(1), K(4), K(2)], [2, 2, 3])
        @test canonical_center(mixed) == (1, 0, 2)
    end

    @testset "Equivalent centers have equal hashes" begin
        # Positive, zero, negative, and mixed radii, including equality at v(diff) = r.
        for (cs, shifted, rs) in [
                ([4], [8], [2]),
                ([1], [9], [2]),
                ([5], [7], [0]),
                ([0], [1], [-1]),
                ([0], [1023], [-10]),
                ([0, 1], [1, 5], [-1, 2]),
                ([0, 0], [31, 1024], [-5, 10])]
            a = ValuationPolydisc(K.(cs), rs)
            b = ValuationPolydisc(K.(shifted), rs)
            @test a == b
            @test canonical_center(a) == canonical_center(b)
            @test hash(a) == hash(b)
        end
        @test canonical_center(ValuationPolydisc([K(1), K(5)], [-1, 2])) == (0, 1)
    end

    @testset "Dict and Set respect disc equivalence" begin
        for r in (-1, 2)
            a = ValuationPolydisc([K(1)], [r])
            equivalent = ValuationPolydisc([K(9)], [r])
            different_radius = ValuationPolydisc([K(1)], [r + 1])
            d = Dict(a => "first", different_radius => "other")

            @test d[equivalent] == "first"
            d[equivalent] = "replacement"
            @test d[a] == "replacement"
            @test d[different_radius] == "other"
            @test length(d) == 2
            @test length(Set([a, equivalent, different_radius])) == 2
        end

        # Distinct residue classes must remain distinct keys; hash collisions are allowed.
        a = ValuationPolydisc([K(1)], [2])
        b = ValuationPolydisc([K(3)], [2])
        @test a != b
        @test canonical_center(a) != canonical_center(b)
        d = Dict(a => 1, b => 2)
        @test d[a] == 1
        @test d[b] == 2
    end
end
