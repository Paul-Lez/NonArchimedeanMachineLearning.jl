# Test file for ValuedFieldPoint wrapper type.
#
# This file tests the ValuedFieldPoint wrapper that encodes the prime
# and precision at the type level for compile-time optimization.

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "ValuedFieldPoint Operations" begin
    prec = 20
    K = PadicField(2, prec)

    @testset "Construction and Accessors" begin
        # Test: Construct from PadicFieldElem
        x = ValuedFieldPoint(K(5))
        @test x isa ValuedFieldPoint{2,20,PadicFieldElem}

        # Test: prime is a compile-time constant
        @test NonArchimedeanMachineLearning.prime(x) == 2
        @test NonArchimedeanMachineLearning.prime(typeof(x)) == 2

        # Test: precision is a compile-time constant
        @test precision(x) == 20
        @test precision(typeof(x)) == 20

        # Test: unwrap returns underlying element (use NonArchimedeanMachineLearning.unwrap to avoid conflict with Oscar.unwrap)
        @test NonArchimedeanMachineLearning.unwrap(x) == K(5)
        @test NonArchimedeanMachineLearning.unwrap(x) isa PadicFieldElem

        # Test: parent field
        @test parent(x) === K
    end

    @testset "Arithmetic Operations" begin
        x = ValuedFieldPoint(K(3))
        y = ValuedFieldPoint(K(7))

        # Addition
        z = x + y
        @test z isa ValuedFieldPoint{2,20,PadicFieldElem}
        @test NonArchimedeanMachineLearning.unwrap(z) == K(10)

        # Subtraction
        z = y - x
        @test NonArchimedeanMachineLearning.unwrap(z) == K(4)

        # Unary minus
        z = -x
        @test NonArchimedeanMachineLearning.unwrap(z) == -K(3)

        # Multiplication
        z = x * y
        @test NonArchimedeanMachineLearning.unwrap(z) == K(21)

        # Scalar multiplication
        z = 3 * x
        @test NonArchimedeanMachineLearning.unwrap(z) == K(9)
        z = x * 3
        @test NonArchimedeanMachineLearning.unwrap(z) == K(9)

        # Exponentiation
        z = x^2
        @test NonArchimedeanMachineLearning.unwrap(z) == K(9)
    end

    @testset "Division" begin
        # Create elements that divide evenly
        x = ValuedFieldPoint(K(6))
        y = ValuedFieldPoint(K(2))

        z = x / y
        @test NonArchimedeanMachineLearning.unwrap(z) == divexact(K(6), K(2))

        z = x // y
        @test NonArchimedeanMachineLearning.unwrap(z) == divexact(K(6), K(2))
    end

    @testset "Comparison and Hashing" begin
        x = ValuedFieldPoint(K(5))
        y = ValuedFieldPoint(K(5))
        z = ValuedFieldPoint(K(7))

        @test x == y
        @test !(x == z)

        # Hashing should be consistent
        @test hash(x) == hash(y)
    end

    @testset "Valuation and Absolute Value" begin
        # Test: valuation of 2^3 * 5 = 40
        x = ValuedFieldPoint(K(8))  # 8 = 2^3
        @test NonArchimedeanMachineLearning.valuation(x) == 3

        # Absolute value |2^v|_2 = 2^(-v)
        @test abs(x) == Float64(2)^(-3)

        y = ValuedFieldPoint(K(5))  # 5 is a unit in 2-adics
        @test NonArchimedeanMachineLearning.valuation(y) == 0
        @test abs(y) == 1.0
    end

    @testset "Zero and One" begin
        x = ValuedFieldPoint(K(5))

        z = zero(x)
        @test NonArchimedeanMachineLearning.unwrap(z) == K(0)
        @test z isa ValuedFieldPoint{2,20,PadicFieldElem}

        o = one(x)
        @test NonArchimedeanMachineLearning.unwrap(o) == K(1)
        @test o isa ValuedFieldPoint{2,20,PadicFieldElem}
    end

    @testset "Lift Operations" begin
        # Lift vector
        v = [K(1), K(2), K(3)]
        lifted_v = NonArchimedeanMachineLearning.lift(v)
        @test lifted_v isa Vector{ValuedFieldPoint{2,20,PadicFieldElem}}
        @test length(lifted_v) == 3
        @test NonArchimedeanMachineLearning.unwrap(lifted_v[1]) == K(1)
        @test NonArchimedeanMachineLearning.unwrap(lifted_v[2]) == K(2)
        @test NonArchimedeanMachineLearning.unwrap(lifted_v[3]) == K(3)

        # Lift tuple
        t = (K(4), K(5))
        lifted_t = NonArchimedeanMachineLearning.lift(t)
        @test lifted_t isa NTuple{2,ValuedFieldPoint{2,20,PadicFieldElem}}
        @test NonArchimedeanMachineLearning.unwrap(lifted_t[1]) == K(4)
        @test NonArchimedeanMachineLearning.unwrap(lifted_t[2]) == K(5)
    end

    @testset "Unwrap Operations" begin
        x = ValuedFieldPoint(K(1))
        y = ValuedFieldPoint(K(2))
        z = ValuedFieldPoint(K(3))

        # Unwrap tuple
        t = (x, y, z)
        unwrapped_t = NonArchimedeanMachineLearning.unwrap(t)
        @test unwrapped_t == (K(1), K(2), K(3))

        # Unwrap vector
        v = [x, y, z]
        unwrapped_v = NonArchimedeanMachineLearning.unwrap(v)
        @test unwrapped_v == [K(1), K(2), K(3)]
    end

    @testset "Type Stability" begin
        # Verify type parameters propagate correctly
        x = ValuedFieldPoint(K(5))
        y = ValuedFieldPoint(K(7))

        # Result of arithmetic should have same type parameters
        z = x + y
        @test typeof(z) == typeof(x)

        z = x * y
        @test typeof(z) == typeof(x)

        z = 3 * x
        @test typeof(z) == typeof(x)
    end

    @testset "Different Primes" begin
        # Test with different prime
        L = PadicField(3, 15)
        x = ValuedFieldPoint(L(5))

        @test NonArchimedeanMachineLearning.prime(x) == 3
        @test precision(x) == 15
        @test x isa ValuedFieldPoint{3,15,PadicFieldElem}

        # Verify 3^2 = 9 has valuation 2
        y = ValuedFieldPoint(L(9))
        @test NonArchimedeanMachineLearning.valuation(y) == 2
        @test abs(y) == Float64(3)^(-2)
    end

    @testset "Display" begin
        x = ValuedFieldPoint(K(5))
        s = sprint(show, x)
        @test occursin("ValuedFieldPoint", s)
        @test occursin("2", s)  # prime
        @test occursin("20", s)  # precision
    end
end
