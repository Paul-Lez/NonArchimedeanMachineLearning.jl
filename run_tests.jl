# Temporary test runner to properly load NAML module
include("src/NAML.jl")

# Make NAML available as both .NAML and NAML for the tests
const NAML = Main.NAML

using Test

@testset "NAML.jl" begin
    @testset "Basic Structures" begin
        include("test/polydisc.jl")
        include("test/tangent_vector.jl")
        include("test/functions.jl")
    end

    @testset "Statistics" begin
        include("test/frechet.jl")
    end

    @testset "Optimization" begin
        include("test/gradient_descent.jl")
        include("test/polynomial_learning.jl")
    end
end

println("\n✓ All tests completed successfully!")
