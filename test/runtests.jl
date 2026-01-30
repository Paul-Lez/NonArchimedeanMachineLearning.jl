# Main test runner for NAML package
#
# This file is the entry point for the test suite. It can be run with:
#   julia --project test/runtests.jl
# or via Pkg.test():
#   using Pkg; Pkg.test("NAML")

using Test
using NAML

@testset "NAML.jl" begin
    @testset "Basic Structures" begin
        include("polydisc.jl")
        include("tangent_vector.jl")
        include("functions.jl")
    end

    @testset "Statistics" begin
        include("frechet.jl")
    end

    @testset "Optimization" begin
        include("gradient_descent.jl")
        include("polynomial_learning.jl")
    end

    @testset "Tree Search Algorithms" begin
        include("dag_mcts.jl")
    end
end

println("\n✓ All tests completed successfully!")
