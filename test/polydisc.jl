# Test file for basic polydisc operations.
#
# This file demonstrates and tests the fundamental polydisc operations
# including creation, iteration, joining, and generating children.

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Polydisc Operations" begin
    prec = 20
    K = PadicField(3, prec)

    a1 = [K(1), K(2)]
    r1 = [1, 2]
    a2 = [K(2), K(2)]
    r2 = [2, 2]

    p1 = ValuationPolydisc(a1, r1)
    p2 = ValuationPolydisc(a2, r2)

    @testset "Polydisc Iteration" begin
        # Test: Iterate over polydisc indices
        centers = []
        radii = []
        for i in Base.eachindex(p1)
            push!(centers, p1.center[i])
            push!(radii, p1.radius[i])
        end
        @test centers == [K(1), K(2)]
        @test radii == [1, 2]
    end

    @testset "Polydisc Join" begin
        # Test: Join (smallest common ancestor) of two polydiscs
        j = NonArchimedeanMachineLearning.join(p1, p2)  # Using qualified name to avoid conflict with Base.join
        @test j == ValuationPolydisc([K(1), K(2)], [0, 2])
    end

    @testset "Polydisc Children" begin
        # Test: Generate children of a polydisc
        L = PadicField(2, prec)
        gauss = ValuationPolydisc([L(0)], [2])
        ch = children(gauss)
        @test Set(ch) == Set([ValuationPolydisc([L(0)], [3]),
            ValuationPolydisc([L(4)], [3])])
    end

    @testset "Polydisc Accessors" begin
        # Test: center, radius, dim, prime accessors
        @test collect(NonArchimedeanMachineLearning.center(p1)) == a1
        @test collect(NonArchimedeanMachineLearning.radius(p1)) == r1
        @test NonArchimedeanMachineLearning.dim(p1) == 2
        @test NonArchimedeanMachineLearning.prime(p1) == 3
    end

    @testset "Polydisc Distance" begin
        # Test: distance between polydiscs
        d = NonArchimedeanMachineLearning.dist(p1, p2)
        @test d ≈ 14 / 9
        # Distance to self should be 0
        @test NonArchimedeanMachineLearning.dist(p1, p1) == 0.0
    end

    @testset "Polydisc Concatenate" begin
        # Test: concatenate two polydiscs
        L = PadicField(2, prec)
        q1 = ValuationPolydisc([L(1)], [1])
        q2 = ValuationPolydisc([L(2)], [2])
        q_concat = NonArchimedeanMachineLearning.concatenate(q1, q2)
        @test NonArchimedeanMachineLearning.dim(q_concat) == 2
        @test NonArchimedeanMachineLearning.center(q_concat)[1] == L(1)
        @test NonArchimedeanMachineLearning.center(q_concat)[2] == L(2)
        @test NonArchimedeanMachineLearning.radius(q_concat)[1] == 1
        @test NonArchimedeanMachineLearning.radius(q_concat)[2] == 2
    end

    @testset "Polydisc Children Along Branch" begin
        # Test: generate children along a specific branch
        L = PadicField(2, prec)
        gauss = ValuationPolydisc([L(0), L(0)], [2, 2])
        ch_branch = children_along_branch(gauss, 1)
        ch_branch_from_subset = children_along_branches(gauss, [1])
        @test ch_branch == ch_branch_from_subset
        @test length(ch_branch) == 2  # NonArchimedeanMachineLearning.prime(gauss) = 2
        # All children should have increased radius in first coordinate
        @test all(c -> NonArchimedeanMachineLearning.radius(c)[1] == 3, ch_branch)
        # Second coordinate radius should be unchanged
        @test all(c -> NonArchimedeanMachineLearning.radius(c)[2] == 2, ch_branch)
    end

    @testset "Polydisc Children Along Branches" begin
        L = PadicField(2, prec)
        parent = ValuationPolydisc{PadicFieldElem, Int, 3}((L(0), L(0), L(0)), (1, 1, 1))

        # Refine a fixed pair of coordinates and enumerate all residue classes there.
        ch_branches = children_along_branches(parent, [1, 3])
        @test length(ch_branches) == 4
        @test all(c -> c.radius == (2, 1, 2), ch_branches)
        @test length(Set(children(parent, 2))) == 12

        expected_centers = [(L(a), L(0), L(b)) for a in (0, 2) for b in (0, 2)]
        @test all(ec -> any(c -> c.center == ec, ch_branches), expected_centers)

        # The vector constructor auto-wraps centers, exercising the ValuedFieldPoint path.
        wrapped_parent = ValuationPolydisc([L(0), L(0), L(0)], [1, 1, 1])
        wrapped_children = children_along_branches(wrapped_parent, (2, 3))
        @test length(wrapped_children) == 4
        @test all(c -> c.radius == (1, 2, 2), wrapped_children)

        expected_wrapped_centers = [(L(0), L(a), L(b)) for a in (0, 2) for b in (0, 2)]
        @test all(ec -> any(c -> NonArchimedeanMachineLearning.unwrap(c.center) == ec,
                wrapped_children), expected_wrapped_centers)

        # By default saturated requested coordinates are skipped, so the other
        # requested coordinate can still be refined.
        terminal = ValuationPolydisc{PadicFieldElem, Int, 3}((L(0), L(0), L(0)), (1, prec, 1))
        partial_children = children_along_branches(terminal, [1, 2])
        @test length(partial_children) == 2
        @test all(c -> c.radius == (2, prec, 1), partial_children)
        # Opting out of skipping saturated coordinates restores all-or-nothing behavior.
        @test isempty(children_along_branches(terminal, [1, 2]; skip_saturated = false))
        @test isempty(children_along_branches(terminal, [2]))

        # Saturated-coordinate skipping also works for wrapped centers.
        wrapped_terminal = ValuationPolydisc([L(0), L(0), L(0)], [1, prec, 1])
        wrapped_partial_children = children_along_branches(wrapped_terminal, [2, 3])
        @test length(wrapped_partial_children) == 2
        @test all(c -> c.radius == (1, prec, 2), wrapped_partial_children)
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
        p_sub = NonArchimedeanMachineLearning.subdisc(p1, [1])
        @test NonArchimedeanMachineLearning.dim(p_sub) == 1
        @test NonArchimedeanMachineLearning.center(p_sub)[1] == K(1)
        @test NonArchimedeanMachineLearning.radius(p_sub)[1] == 1

        # Test: get components
        comps = NonArchimedeanMachineLearning.components(p1)
        @test length(comps) == 2
        @test all(c -> NonArchimedeanMachineLearning.dim(c) == 1, comps)
        @test NonArchimedeanMachineLearning.center(comps[1])[1] == K(1)
        @test NonArchimedeanMachineLearning.center(comps[2])[1] == K(2)
    end

end
