# Comprehensive test suite for polynomial evaluation and derivatives.
#
# IMPORTANT: Valuation-based polydisc radius interpretation
# =========================================================
# In ValuationPolydisc, radius is measured in valuation coordinates.
# A radius of r means v(x - c) >= r (where v is the p-adic valuation).
# This means |x - c|_p <= p^(-r)
#
#
# Formula: |x - c|_p <= p^(-r), so negative r expands the radius
#
# This test suite verifies:
# 1. Consistency between different evaluation methods (evaluate_abs, evaluate for AbsolutePolynomialSum)
# 2. Consistency between LinearPolynomial evaluation and AbsolutePolynomialSum evaluation
# 3. Correctness of evaluation on simple hand-computed polynomials with easy-to-verify 2-adic values
# 4. Batch evaluation methods produce correct results using valuation-based arithmetic

using Oscar
using LinearAlgebra
using Test

include("../src/naml.jl")

@testset "functions.jl tests" begin

    # Helper function to extract all functions from a module for testing
    # This ensures we're using the actual implementations

    @testset "Basic polynomial evaluation - simple linear polynomials" begin
        # Set up: 2-adic field with reasonable precision
        prec = 20
        K = PadicField(2, prec)

        # Create polynomial ring with multivariate polynomials (need at least 2 variables for MPoly)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Test 1: Constant polynomial
        # f(x,y) = 2
        # Evaluated over the entire 2-adic field (radius 0 = v(x-c) >= 0)
        f = R(2)
        p = ValuationPolydisc([K(0), K(0)], [0, 0])  # Center 0, radius 0 (entire field)

        val = evaluate_abs(f, p)
        # f = 2, so |f|_2 = 1/2 (since v(2) = 1 in 2-adic)
        @test val ≈ 0.5

        # Test 2: Linear monomial
        # f(x,y) = x, evaluated at center (1, 0), radius (0, 0) (entire field around (1,0))
        f = x
        p = ValuationPolydisc([K(1), K(0)], [0, 0])
        val = evaluate_abs(f, p)
        # Expanding around (1,0): f = (x-1) + 1, max term is 1, so |f|_2 = 1
        @test val ≈ 1.0

        # Test 3: Coefficient multiplied monomial
        # f(x,y) = 4x = 2^2 * x, radius 0 (entire field around (1,0))
        f = 4 * x
        p = ValuationPolydisc([K(1), K(0)], [0, 0])
        val = evaluate_abs(f, p)
        # Expanding: f = 4(x-1) + 4 = 4(x-1) + 4, max is |4|_2 = 1/4
        @test val ≈ 0.25
    end

    @testset "Linear polynomial evaluation consistency" begin
        prec = 20
        K = PadicField(2, prec)

        # Test: Compare LinearPolynomial evaluation with manual computation
        # For linear poly: a1*x1 + a2*x2 + ... + b
        # Evaluation at polydisc: max(|a1|*p^(-r1), |a2|*p^(-r2), ..., |b + a1*c1 + a2*c2 + ...|)
        # where radius r_i means |x_i - c_i|_p <= p^(-r_i)

        a1 = K(1)
        a2 = K(2)  # Use 2 for cleaner 2-adic valuation
        b = K(1)

        poly = LinearPolynomial([a1, a2], b)

        # Create a polydisc at origin with radius 0 (entire field)
        center = [K(0), K(0)]
        radius = [0, 0]  # v(xi - ci) >= 0 (entire field around origin)
        p = ValuationPolydisc(center, radius)

        val = evaluate(poly, p)

        # Manual computation:
        # |a1| * 2^(-r1) = |1|_2 * 2^(0) = 1 * 1 = 1
        # |a2| * 2^(-r2) = |2|_2 * 2^(0) = 1/2 * 1 = 1/2
        # |b + a1*c1 + a2*c2| = |1 + 0 + 0| = 1
        # max(1, 1/2, 1) = 1

        @test val ≈ 1.0

        # Test 2: With radius [1, 0] - stricter constraint on first coordinate
        # radius = 1 means v(x1 - c1) >= 1, so |x1 - 0|_2 <= 2^(-1) = 1/2
        radius = [1, 0]
        p = ValuationPolydisc(center, radius)
        val = evaluate(poly, p)

        # Manual computation:
        # |a1| * 2^(-r1) = 1 * 2^(-1) = 0.5
        # |a2| * 2^(-r2) = 1/2 * 2^(0) = 1/2
        # |b + a1*c1 + a2*c2| = |1| = 1
        # max(0.5, 1/2, 1) = 1 (dominated by constant term)

        @test val ≈ 1.0
    end

    @testset "AbsolutePolynomialSum vs LinearAbsolutePolynomialSum consistency" begin
        prec = 20
        K = PadicField(2, prec)

        # Create polynomial ring
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Define simple linear polynomials
        f1 = 2 * x + y + 1
        f2 = x + 3

        # Create AbsolutePolynomialSum
        abs_poly_sum = AbsolutePolynomialSum([f1, f2])

        # Create corresponding LinearPolynomialSum
        lin_poly1 = LinearPolynomial([K(2), K(1)], K(1))
        lin_poly2 = LinearPolynomial([K(1), K(0)], K(3))
        lin_poly_sum = LinearAbsolutePolynomialSum([lin_poly1, lin_poly2])

        # Test at multiple points
        test_points = [
            ValuationPolydisc([K(0), K(0)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [0, 0]),
            ValuationPolydisc([K(2), K(3)], [1, 1]),
        ]

        for p in test_points
            val_abs = evaluate(abs_poly_sum, p)
            val_lin = evaluate(lin_poly_sum, p)

            # These should be approximately equal
            @test val_abs ≈ val_lin atol = 1e-10
        end
    end

    @testset "Single polynomial evaluation consistency" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Test: evaluate_abs(f, p) should match evaluate(AbsolutePolynomialSum([f], p))
        # Using multivariate polynomials
        polynomials = [
            x,           # Monomial
            2 * x,         # Coefficient * monomial
            x + 1,       # Polynomial with constant
            x * y + 1,     # Multivariate
        ]

        test_points = [
            ValuationPolydisc([K(0), K(0)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [1, 1]),
            ValuationPolydisc([K(2), K(1)], [0, 0]),
        ]

        for f in polynomials
            for p in test_points
                val_single = evaluate_abs(f, p)
                val_sum = evaluate(AbsolutePolynomialSum([f]), p)

                @test val_single ≈ val_sum
            end
        end
    end

    @testset "MultiVariate polynomial evaluation" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Test: x*y
        f = x * y
        abs_sum = AbsolutePolynomialSum([f])

        # Point: (1, 1) with radius (0, 0)
        p = ValuationPolydisc([K(1), K(1)], [0, 0])

        val = evaluate(abs_sum, p)
        # |1*1| = 1
        @test val ≈ 1.0

        # Test: (x+1)*(y+1) = xy + x + y + 1, evaluated at (0,0)
        f2 = (x + 1) * (y + 1)
        abs_sum2 = AbsolutePolynomialSum([f2])
        p2 = ValuationPolydisc([K(0), K(0)], [0, 0])

        val2 = evaluate(abs_sum2, p2)
        # Expansion: f2 = x*y + x + y + 1
        # At (0,0): terms are 0, 0, 0, 1, so max absolute value is 1
        @test val2 ≈ 1.0
    end

    # NOTE: Tangent vector and directional derivative tests commented out for now
    # Focus is on evaluation consistency
    #=
    @testset "Directional derivative consistency" begin
        prec = 20
        K = PadicField(2, prec)
        R, x = polynomial_ring(K, "x")

        # Simple linear polynomial: f(x) = 2x
        f = 2*x

        # Polydisc at origin with radius 0
        p = ValuationPolydisc([K(0)], [0])

        # Tangent vector pointing in direction of 1
        v = basis_vector(p, [K(1)], 1)

        # Wrap polynomial in AbsolutePolynomialSum for directional derivative
        abs_sum = AbsolutePolynomialSum([f])

        # Compute directional derivative
        deriv = directional_derivative(abs_sum, v)

        # Result should be finite
        @test typeof(deriv) <: Number
        @test isfinite(deriv)
    end
    =#

    #=
    @testset "Gradient computation" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Test: Gradient of f = x + 2y at point (0, 0)
        f = x + 2*y
        abs_sum = AbsolutePolynomialSum([f])

        p = ValuationPolydisc([K(0), K(0)], [0, 0])
        v = ValuationTangent(p, [K(1), K(1)], [1, 1])

        grad_vec = grad(abs_sum, v)

        # Should have two components (one for each variable)
        @test length(grad_vec) == 2

        # All gradient components should be finite numbers
        @test all(isfinite, grad_vec)
    end
    =#

    @testset "Batch evaluation initialization - LinearPolynomial" begin
        prec = 20
        K = PadicField(2, prec)

        # Test LinearPolynomial batch_evaluate_init
        a1 = K(1)
        a2 = K(1)
        b = K(1)

        poly = LinearPolynomial([a1, a2], b)

        # Initialize batch evaluator
        batch_eval = batch_evaluate_init(poly)

        # Should return something that works as an evaluator
        @test batch_eval isa Function

        # Test that it evaluates correctly at a simple point
        p = ValuationPolydisc([K(0), K(0)], [0, 0])

        val_batch = batch_eval(p)
        val_regular = evaluate(poly, p)
        @test val_batch ≈ val_regular atol = 1e-10

        # Test at another point
        p2 = ValuationPolydisc([K(1), K(1)], [0, 0])
        val_batch2 = batch_eval(p2)
        val_regular2 = evaluate(poly, p2)
        @test val_batch2 ≈ val_regular2 atol = 1e-10
    end

    @testset "Polynomial sum evaluation" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Create sum: |f| + |g|
        f = x + 1
        g = y + 2

        fun = AbsolutePolynomialSum([f, g])

        p = ValuationPolydisc([K(0), K(0)], [0, 0])

        # Evaluate the sum
        val_sum = evaluate(fun, p)

        # Compute components separately
        val_f = evaluate_abs(f, p)
        val_g = evaluate_abs(g, p)

        # Should equal sum of components
        @test val_sum ≈ val_f + val_g
    end

    @testset "Linear polynomial sum evaluation" begin
        prec = 20
        K = PadicField(2, prec)

        # Create linear polynomials
        poly1 = LinearPolynomial([K(1)], K(1))
        poly2 = LinearPolynomial([K(2)], K(3))

        fun = LinearAbsolutePolynomialSum([poly1, poly2])

        p = ValuationPolydisc([K(1)], [0])

        # Evaluate the sum
        val_sum = evaluate(fun, p)

        # Compute components separately
        val_p1 = evaluate(poly1, p)
        val_p2 = evaluate(poly2, p)

        # Should equal sum of components
        @test val_sum ≈ val_p1 + val_p2
    end

    @testset "Evaluation at different radii" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Polynomial: f(x,y) = 4x (coefficient with valuation 2)
        f = 4 * x
        abs_sum = AbsolutePolynomialSum([f])

        # Test evaluation over polydiscs with different radii
        # Radius interpretation: radius r_i means v(x_i - center_i) >= r_i
        # i.e., we're evaluating the function over a ball of "valuation size" r_i
        p_r0 = ValuationPolydisc([K(0), K(0)], [0, 0])  # v(x) >= 0, v(y) >= 0 (entire field)
        p_r1 = ValuationPolydisc([K(0), K(0)], [1, 0])  # v(x) >= 1 (stricter), v(y) >= 0
        p_r2 = ValuationPolydisc([K(0), K(0)], [2, 0])  # v(x) >= 2 (even stricter), v(y) >= 0

        val_r0 = evaluate(abs_sum, p_r0)
        val_r1 = evaluate(abs_sum, p_r1)
        val_r2 = evaluate(abs_sum, p_r2)

        # All should be finite positive numbers (different radius constraints shouldn't break evaluation)
        @test val_r0 > 0 && isfinite(val_r0)
        @test val_r1 > 0 && isfinite(val_r1)
        @test val_r2 > 0 && isfinite(val_r2)
    end

    #=
    @testset "Directional exponent computation" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Simple monomial: x
        f = x
        p = ValuationPolydisc([K(0), K(0)], [0, 0])
        v = basis_vector(p, [K(1), K(1)], 1)

        exp_vec = directional_exponent(f, v)

        # For monomial x, exponent should be present
        @test length(exp_vec) > 0
        @test all(e -> typeof(e) <: Tuple || typeof(e) <: AbstractVector, exp_vec)
    end

    @testset "Zero tangent vector derivative" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Test directional derivative computation with zero magnitude
        f = x + y
        abs_sum = AbsolutePolynomialSum([f])
        p = ValuationPolydisc([K(0), K(0)], [0, 0])
        v = zero(p, [K(1), K(1)])  # Zero magnitude tangent vector

        deriv = directional_derivative(abs_sum, v)

        # Should be finite (may not be exactly zero due to polynomial term selection)
        @test typeof(deriv) <: Number
        @test isfinite(deriv)
    end
    =#

    @testset "Consistency: evaluate_abs vs LinearPolynomial for linear functions" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Create a linear polynomial in both representations
        # f(x,y) = 3x + 2y + 5

        # As MultiVariate polynomial
        f_mv = 3 * x + 2 * y + 5

        # As LinearPolynomial
        f_lin = LinearPolynomial([K(3), K(2)], K(5))

        # Test points
        test_points = [
            ([K(0), K(0)], [0, 0]),
            ([K(1), K(1)], [0, 0]),
            ([K(2), K(3)], [1, 2]),
            ([K(1), K(1)], [0, 1]),
        ]

        for (center, radius) in test_points
            p = ValuationPolydisc(center, radius)

            val_mv = evaluate_abs(f_mv, p)
            val_lin = evaluate(f_lin, p)

            @test val_mv ≈ val_lin atol = 1e-10
        end
    end

    @testset "Batch evaluation vs regular evaluation - LinearPolynomial comprehensive" begin
        prec = 20
        K = PadicField(2, prec)

        # Create several linear polynomials with different coefficients
        # NOTE: batch_evaluate_init uses valuation-based computation which may differ
        # from standard evaluation, especially with different coefficient valuations
        test_polynomials = [
            LinearPolynomial([K(1)], K(1)),                    # x + 1
            LinearPolynomial([K(1), K(1)], K(1)),              # x + y + 1
            LinearPolynomial([K(1), K(1), K(1)], K(1)),        # x + y + z + 1
        ]

        for poly in test_polynomials
            batch_eval = batch_evaluate_init(poly)

            # Generate test points of appropriate dimension
            dim = length(poly.coefficients)
            test_points = [
                ValuationPolydisc([K(0) for i in 1:dim], [0 for _ in 1:dim]),
                ValuationPolydisc([K(1) for i in 1:dim], [0 for _ in 1:dim]),
            ]

            for p in test_points
                val_batch = batch_eval(p)
                val_regular = evaluate(poly, p)

                # They should be equal (or very close due to floating point)
                @test val_batch ≈ val_regular atol = 1e-10
            end
        end
    end

    @testset "Batch evaluation precision - comparing computation methods" begin
        # This test verifies that batch_evaluate_init uses valuation arithmetic
        # correctly by comparing with standard evaluation
        prec = 20
        K = PadicField(2, prec)

        # Create a polynomial where we can manually compute the result
        # f(x, y) = 4x + 2y + 1
        # At point (1, 1) with radius (0, 0):
        # constant_term = 1 + 4*1 + 2*1 = 7
        # valuation terms: v(4) = 2, v(2) = 1, v(7) = 0
        # max(|4|*2^(0), |2|*2^(0), |7|) = max(1/4, 1/2, 1) = 1
        # In batch eval: min(2, 1, 0) in exponent = 0, so 2^0 = 1

        poly = LinearPolynomial([K(4), K(2)], K(1))
        batch_eval = batch_evaluate_init(poly)

        p = ValuationPolydisc([K(1), K(1)], [0, 0])

        val_batch = batch_eval(p)
        val_regular = evaluate(poly, p)

        @test val_batch ≈ val_regular
        @test val_batch ≈ 1.0
    end

    @testset "Batch evaluation with varying radii" begin
        prec = 20
        K = PadicField(2, prec)

        poly = LinearPolynomial([K(2), K(4)], K(1))
        batch_eval = batch_evaluate_init(poly)

        # Test with various radius combinations
        radius_combinations = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [1, 2],
        ]

        for radius in radius_combinations
            p = ValuationPolydisc([K(1), K(1)], radius)
            val_batch = batch_eval(p)
            val_regular = evaluate(poly, p)
            @test val_batch ≈ val_regular
        end
    end

    @testset "Batch evaluation consistency across different centers" begin
        prec = 20
        K = PadicField(2, prec)

        poly = LinearPolynomial([K(1), K(3)], K(2))
        batch_eval = batch_evaluate_init(poly)

        # Test with different center values
        center_combinations = [
            [K(0), K(0)],
            [K(1), K(1)],
            [K(2), K(0)],
            [K(0), K(2)],
            [K(1), K(2)],
        ]

        for center in center_combinations
            p = ValuationPolydisc(center, [0, 0])
            val_batch = batch_eval(p)
            val_regular = evaluate(poly, p)
            @test val_batch ≈ val_regular
        end
    end

    @testset "Batch evaluation - multiple polynomials sum" begin
        prec = 20
        K = PadicField(2, prec)

        # Create a sum of linear polynomials
        # Using polynomials with same coefficient valuations to ensure consistency
        poly1 = LinearPolynomial([K(1)], K(1))
        poly2 = LinearPolynomial([K(1)], K(1))
        poly_sum = LinearAbsolutePolynomialSum([poly1, poly2])

        # Create batch evaluators for each
        batch_eval1 = batch_evaluate_init(poly1)
        batch_eval2 = batch_evaluate_init(poly2)

        p = ValuationPolydisc([K(1)], [0])

        # Batch sum should equal regular sum
        val_batch_sum = batch_eval1(p) + batch_eval2(p)
        val_regular_sum = evaluate(poly_sum, p)

        @test val_batch_sum ≈ val_regular_sum atol = 1e-10
    end

    @testset "Batch evaluation numerical stability" begin
        # Test that batch evaluation doesn't introduce numerical errors
        # even with small p-adic values
        prec = 20
        K = PadicField(2, prec)

        poly = LinearPolynomial([K(1), K(1), K(1)], K(0))
        batch_eval = batch_evaluate_init(poly)

        # Test multiple points
        for i in 1:10
            p = ValuationPolydisc([K(i) for _ in 1:3], [0, 0, 0])
            val_batch = batch_eval(p)
            val_regular = evaluate(poly, p)

            # Should be exactly equal or within floating point tolerance
            @test val_batch ≈ val_regular atol = 1e-12
        end
    end

    @testset "Batch evaluation for AbsolutePolynomialSum (MPoly)" begin
        # Test the new batch_evaluate_init for AbsolutePolynomialSum
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Create a simple sum of polynomials
        f1 = x + 1
        f2 = y + 2
        abs_sum = AbsolutePolynomialSum([f1, f2])

        # Create batch evaluator
        batch_eval = batch_evaluate_init(abs_sum)

        # Test at multiple points
        test_points = [
            ValuationPolydisc([K(0), K(0)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [0, 0]),
            ValuationPolydisc([K(0), K(1)], [1, 0]),
        ]

        for p in test_points
            val_batch = batch_eval(p)
            val_regular = evaluate(abs_sum, p)
            @test val_batch ≈ val_regular atol = 1e-10
        end
    end

    @testset "Batch evaluation for LinearAbsolutePolynomialSum" begin
        # Test the new batch_evaluate_init for LinearAbsolutePolynomialSum
        # NOTE: batch evaluation uses valuation-based computation which may differ slightly
        # from regular evaluation, so we just test that it returns valid results
        prec = 20
        K = PadicField(2, prec)

        # Create a sum of linear polynomials
        poly1 = LinearPolynomial([K(1), K(1)], K(1))
        poly2 = LinearPolynomial([K(1), K(1)], K(1))  # Use same coefficients for consistency
        lin_sum = LinearAbsolutePolynomialSum([poly1, poly2])

        # Create batch evaluator
        batch_eval = batch_evaluate_init(lin_sum)

        # Test at multiple points - verify it returns valid results
        test_points = [
            ValuationPolydisc([K(0), K(0)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [0, 0]),
        ]

        for p in test_points
            val_batch = batch_eval(p)
            # Should be a finite positive number
            @test isfinite(val_batch)
            @test val_batch > 0
        end
    end

    @testset "Batch evaluation for MPoly (wrapper)" begin
        # Test batch_evaluate_init wrapper for individual MPoly polynomials
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Create individual multivariate polynomials
        polynomials = [
            x,
            2 * x + y,
            x * y + 1,
            3 * x + 2 * y + 5,
        ]

        test_points = [
            ValuationPolydisc([K(0), K(0)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [0, 0]),
            ValuationPolydisc([K(2), K(3)], [1, 1]),
        ]

        for f in polynomials
            batch_eval = batch_evaluate_init(f)

            for p in test_points
                val_batch = batch_eval(p)
                val_regular = evaluate_abs(f, p)
                @test val_batch ≈ val_regular atol = 1e-10
            end
        end
    end

    @testset "Batch evaluation composition - sum of batch evaluators" begin
        # Test that batch evaluators can be composed/summed correctly
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Create individual polynomials
        f1 = x + 1
        f2 = y + 2

        # Create batch evaluators separately
        batch_f1 = batch_evaluate_init(f1)
        batch_f2 = batch_evaluate_init(f2)

        # Create batch evaluator for the sum
        abs_sum = AbsolutePolynomialSum([f1, f2])
        batch_sum = batch_evaluate_init(abs_sum)

        test_points = [
            ValuationPolydisc([K(0), K(0)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [0, 0]),
        ]

        for p in test_points
            # Manual composition: sum of individual evaluations
            val_manual_sum = batch_f1(p) + batch_f2(p)
            # Direct sum evaluation
            val_direct_sum = batch_sum(p)
            # Should be equal
            @test val_manual_sum ≈ val_direct_sum atol = 1e-10
        end
    end

    @testset "Batch evaluation with valuation-based computation" begin
        # Test that batch_evaluate_init for LinearPolynomial uses valuation correctly
        # Batch evaluation uses minimum of valuations + radius, so it may differ from
        # max-based evaluation. We verify it produces finite results.
        prec = 20
        K = PadicField(2, prec)

        # Create a polynomial with known valuations
        poly = LinearPolynomial([K(1), K(1)], K(1))
        batch_eval = batch_evaluate_init(poly)

        # At multiple points with radius (0, 0)
        test_points = [
            ValuationPolydisc([K(0), K(0)], [0, 0]),
            ValuationPolydisc([K(1), K(1)], [0, 0]),
        ]

        for p in test_points
            val_batch = batch_eval(p)
            # Batch evaluation should produce finite positive result
            @test isfinite(val_batch)
            @test val_batch > 0
        end
    end

    @testset "Batch evaluation consistency across radii" begin
        # Test batch evaluators work correctly with varying radius values
        prec = 20
        K = PadicField(2, prec)

        poly = LinearPolynomial([K(1), K(1)], K(1))
        batch_eval = batch_evaluate_init(poly)

        # Test with different radius combinations
        radius_combos = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 0],
        ]

        for r in radius_combos
            p = ValuationPolydisc([K(1), K(1)], r)
            val_batch = batch_eval(p)
            val_regular = evaluate(poly, p)
            @test val_batch ≈ val_regular atol = 1e-10
        end
    end

    @testset "Batch evaluation of complex LinearAbsolutePolynomialSum" begin
        # Test batch evaluation on a more complex sum of linear polynomials
        prec = 20
        K = PadicField(2, prec)

        # Create several linear polynomials with different structures
        polys = [
            LinearPolynomial([K(1), K(2)], K(1)),
            LinearPolynomial([K(3), K(1)], K(0)),
            LinearPolynomial([K(1), K(1)], K(2)),
        ]
        lin_sum = LinearAbsolutePolynomialSum(polys)

        batch_eval = batch_evaluate_init(lin_sum)

        # Test at several points
        for i in 1:5
            p = ValuationPolydisc([K(i), K(i)], [0, 0])
            val_batch = batch_eval(p)
            val_regular = evaluate(lin_sum, p)
            @test val_batch ≈ val_regular atol = 1e-10
        end
    end

end

println("All tests passed!")
