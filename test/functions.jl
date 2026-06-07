# Test file for polynomial evaluation and derivatives.
#
# This file tests polynomial evaluation and directional derivative computation
# on polydisc space, including absolute polynomial sums.

using Test
using Oscar
using NonArchimedeanMachineLearning

struct UnsupportedPolydiscFunction{S} <: NonArchimedeanMachineLearning.PolydiscFunction{S} end

function _raw_polydisc(center::Vector{S}, radius::Vector{T}) where {S, T}
    N = length(center)
    length(radius) == N || throw(ArgumentError("center and radius must have the same length"))
    return ValuationPolydisc{S, T, N}(tuple(center...), tuple(radius...))
end

@testset "Polynomial Functions" begin
    prec = 20
    K = PadicField(3, prec)
    a1 = [K(1), K(2)]
    r1 = Vector{Int}([1, 2])
    a2 = [K(2), K(2)]
    r2 = Vector{Int}([2, 2])
    p1 = ValuationPolydisc(a1, r1)
    p2 = ValuationPolydisc(a2, r2)

    # Create polynomial ring
    R, (x, y) = polynomial_ring(K, ["x", "y"])
    # Define polynomials
    f = 3 * x
    g = 2 * y

    f_fun = AbsolutePolynomialSum([f])
    g_fun = AbsolutePolynomialSum([g])

    f_as_linear = LinearPolynomial([K(3), K(0)], K(0))
    f_as_linear = LinearAbsolutePolynomialSum([f_as_linear])

    @testset "Polynomial Creation" begin

        # Absolute polynomial sum: |f| + |g| = |3x| + |2y|
        fun = AbsolutePolynomialSum([f, g])

        @test fun isa AbsolutePolynomialSum
        @test length(fun.polys) == 2
    end

    @testset "Polynomial Evaluation" begin
        @test NonArchimedeanMachineLearning.evaluate(f_fun, p1) == 1 / 3
        @test NonArchimedeanMachineLearning.evaluate(f_as_linear, p1) == 1 / 3
        @test NonArchimedeanMachineLearning.evaluate(g_fun, p2) == 1

    end 

    @testset "Directional Derivative" begin
        f = 3 * x
        # Create tangent vector (direction polydisc coincides with p1)
        v = NonArchimedeanMachineLearning.basis_vector(p1, p1, 1)

        # Test: Directional derivative
        dd = directional_derivative(f, v)
        @test dd isa Number || dd isa Vector  # Result should be numeric
    end
end

@testset "PolydiscFunction algebra and internal evaluators" begin
    NAML = NonArchimedeanMachineLearning
    K = PadicField(2, 20)
    make_pd(cs, rs) = ValuationPolydisc{PadicFieldElem, Int, length(cs)}(tuple(cs...), tuple(rs...))

    p = make_pd([K(1)], [1])
    direction = make_pd([K(1)], [2])
    tangent = ValuationTangent(p, direction, [1])
    linear = LinearAbsolutePolynomialSum([LinearPolynomial([K(1)], K(1))])
    shifted = linear + 1

    @testset "operator constructors evaluate exact composed values" begin
        @test NAML.evaluate(linear, p) == 1 / 2
        @test NAML.evaluate(zero(linear), p) == 0
        @test NAML.evaluate(zero(PolydiscFunction{PadicFieldElem}), p) == 0
        @test NAML.evaluate(linear + 2, p) == 5 / 2
        @test NAML.evaluate(2 + linear, p) == 5 / 2
        @test NAML.evaluate(linear - 1 / 4, p) == 1 / 4
        @test NAML.evaluate(3 - linear, p) == 5 / 2
        @test NAML.evaluate(linear * (linear + 3), p) == 7 / 4
        @test NAML.evaluate((linear + 3) / (linear + 1), p) == 7 / 3
        @test NAML.evaluate(3 * linear, p) == 3 / 2
        @test NAML.evaluate(linear / 2, p) == 1 / 4
        @test NAML.evaluate(-linear, p) == -1 / 2
    end

    @testset "power and reciprocal powers" begin
        @test linear^0 == 1
        @test NAML.evaluate(linear^2, p) == 1 / 4
        @test NAML.evaluate(inv(linear), p) == 2.0
        @test NAML.evaluate(1 / linear, p) == 2.0
        @test NAML.evaluate(linear^-1, p) == 2.0
        @test NAML.evaluate(linear^-2, p) == 4.0
        @test NAML.evaluate(linear^-3, p) == 8.0
    end

    @testset "constant, lambda, and composition" begin
        constant = NAML.Constant{PadicFieldElem}(7.0)
        lambda = NAML.Lambda{PadicFieldElem}(q -> Float64(NAML.radius(q)[1] + NAML.dim(q)))
        outer = DifferentiableFunction(x -> x + 10, _ -> 1.0)

        @test NAML.evaluate(constant, p) == 7.0
        @test NAML.evaluate(lambda, p) == 2.0
        @test outer(3.5) == 13.5
        @test NAML.evaluate(NAML.comp(outer, linear), p) == 10.5
    end

    @testset "linear rational functions" begin
        numerator = LinearPolynomial([K(1)], K(1))
        denominator = LinearPolynomial([K(0)], K(1))
        second_num = LinearPolynomial([K(2)], K(0))
        rational = NAML.LinearRationalFunction(numerator, denominator)
        rational_sum = NAML.LinearRationalFunctionSum([
            rational,
            NAML.LinearRationalFunction(second_num, denominator)
        ])

        @test NAML.evaluate(rational, p) == 1 / 2
        @test NAML.evaluate(rational_sum, p) == 1.0
    end

    @testset "typed evaluator values and construction errors" begin
        unsupported = UnsupportedPolydiscFunction{PadicFieldElem}()

        @test batch_evaluate_init(shifted, p)(p) == 3 / 2
        @test batch_evaluate_init(linear - 1, typeof(p))(p) == -1 / 2
        @test batch_evaluate_init((linear + 3) / shifted, typeof(p))(p) == 7 / 3
        @test batch_evaluate_init(3 * linear, typeof(p))(p) == 3 / 2
        @test batch_evaluate_init(NAML.comp(DifferentiableFunction(x -> 2x, _ -> 2.0), linear),
            typeof(p))(p) == 1.0

        bad_poly = LinearPolynomial([K(1), K(1)], K(0))
        @test_throws AssertionError batch_evaluate_init(
            bad_poly, ValuationPolydisc{PadicFieldElem, Int, 1})
        @test_throws ErrorException batch_evaluate_init(
            unsupported,
            ValuationPolydisc{PadicFieldElem, Int, 1}
        )
        @test_throws MethodError batch_evaluate_init(unsupported)
        @test_throws MethodError NAML.parent(unsupported)
        @test_throws MethodError NAML.evaluate(unsupported, p)
        @test_throws MethodError directional_derivative(unsupported, tangent)
    end

    @testset "directional derivative and partial gradient helpers" begin
        slope = LinearAbsolutePolynomialSum([LinearPolynomial([K(1)], K(0))])
        slope_point = make_pd([K(0)], [1])
        slope_direction = make_pd([K(0)], [2])
        slope_tangent = ValuationTangent(slope_point, slope_direction, [1])

        @test directional_derivative(NAML.Constant{PadicFieldElem}(3.0), tangent) == 0.0
        @test directional_derivative(2 * slope, slope_tangent) == -1.0
        @test directional_derivative(slope + slope, slope_tangent) == -1.0
        @test directional_derivative(slope - slope, slope_tangent) == 0.0
        @test directional_derivative(slope * (slope + 1), slope_tangent) ≈ -1.0
        @test directional_derivative((slope + 1) / (slope + 3), slope_tangent) ≈ -4 / 49
        @test directional_derivative(
            NAML.comp(DifferentiableFunction(x -> x^2, x -> 2x), slope),
            slope_tangent
        ) ≈ -1 / 2
        @test directional_derivative(
            NAML.Lambda{PadicFieldElem}(_ -> 1.0, _ -> 0.25),
            slope_tangent
        ) == 0.25
        @test_throws ErrorException directional_derivative(
            NAML.Lambda{PadicFieldElem}(_ -> 1.0),
            slope_tangent
        )
        @test grad(slope, slope_tangent) == [-1 / 2]
        @test NAML.partial_gradient(slope, slope_tangent, [1]) == [-1 / 2]
        @test NAML.partial_gradient(NAML.Constant{PadicFieldElem}(3.0), tangent, [1]) == [0.0]
    end

    @testset "ValuedFieldPoint lifting adapters preserve exact values" begin
        p_vfp = ValuationPolydisc([K(1)], [1])
        direction_vfp = ValuationPolydisc([K(1)], [2])
        tangent_vfp = ValuationTangent(p_vfp, direction_vfp, [1])
        slope = LinearAbsolutePolynomialSum([LinearPolynomial([K(1)], K(0))])
        slope_point = ValuationPolydisc([K(0)], [1])
        slope_direction = ValuationPolydisc([K(0)], [2])
        slope_tangent = ValuationTangent(slope_point, slope_direction, [1])

        @test NAML.evaluate(linear, p_vfp) == 1 / 2
        @test batch_evaluate_init(linear + 1, typeof(p_vfp))(p_vfp) == 3 / 2
        @test batch_evaluate_init((linear + 1) * (linear + 3), typeof(p_vfp))(p_vfp) == 21 / 4
        @test batch_evaluate_init((linear + 3) / (linear + 1), typeof(p_vfp))(p_vfp) == 7 / 3
        @test batch_evaluate_init(3 * linear, typeof(p_vfp))(p_vfp) == 3 / 2
        @test batch_evaluate_init(NAML.comp(DifferentiableFunction(x -> x + 10, _ -> 1.0), linear),
            typeof(p_vfp))(p_vfp) == 10.5
        @test directional_derivative(linear, tangent_vfp) == 0.0
        @test directional_derivative(slope, slope_tangent) == -1 / 2
    end
end

@testset "directional_exponent" begin
    K2 = PadicField(2, 20)
    make_pd(cs, rs) = ValuationPolydisc{PadicFieldElem, Int, length(cs)}(tuple(cs...), tuple(rs...))
    R2, (x2, y2) = polynomial_ring(K2, ["x", "y"])

    @testset "valuation weight determines winner" begin
        # f = x + 4y: v(1)+0=0 vs v(4)+0=2 (2-adic) → x wins
        f = x2 + K2(4) * y2
        p = make_pd([K2(0), K2(0)], [0, 0])
        dir = make_pd([K2(0), K2(0)], [1, 0])
        v = ValuationTangent(p, dir, [1, 0])
        @test directional_exponent(f, v) == [1, 0]
    end

    @testset "radius contributes to valuation weight" begin
        # f = x + y, radius (3,0): x gets penalty → v(1)+3=3 vs v(1)+0=0 → y wins
        f = x2 + y2
        p = make_pd([K2(0), K2(0)], [3, 0])
        dir = make_pd([K2(0), K2(0)], [4, 0])
        v = ValuationTangent(p, dir, [1, 0])
        @test directional_exponent(f, v) == [0, 1]
    end

    @testset "magnitude breaks tie" begin
        # f = x + y, radius (0,0): both val_weight=0; magnitude [1,0] → dot([0,1],[1,0])=0 < dot([1,0],[1,0])=1 → y wins
        f = x2 + y2
        p = make_pd([K2(0), K2(0)], [0, 0])
        dir = make_pd([K2(0), K2(0)], [1, 0])
        v = ValuationTangent(p, dir, [1, 0])
        @test directional_exponent(f, v) == [0, 1]
    end
end

@testset "directional_derivative for LinearPolynomial" begin
    K2 = PadicField(2, 20)
    make_pd(cs, rs) = ValuationPolydisc{PadicFieldElem, Int, length(cs)}(tuple(cs...), tuple(rs...))

    @testset "linear wins (zero constant)" begin
        # f = T₁, radius 0: c₀=0, linear val_weight=0+0=0 → d_v = -2^0 = -1.0
        poly = LinearPolynomial([K2(1)], K2(0))
        p = make_pd([K2(0)], [0])
        dir = make_pd([K2(0)], [1])
        v = ValuationTangent(p, dir, [1])
        @test directional_derivative(poly, v) ≈ -1.0
    end

    @testset "linear wins over constant by valuation" begin
        # f = T₁ + 4, radius 1: v(4)=2 > v(1)+1=1 → linear wins → d_v = -2^{-1} = -0.5
        poly = LinearPolynomial([K2(1)], K2(4))
        p = make_pd([K2(0)], [1])
        dir = make_pd([K2(0)], [2])
        v = ValuationTangent(p, dir, [1])
        @test directional_derivative(poly, v) ≈ -0.5
    end

    @testset "constant dominates due to large radius" begin
        # f = T₁ + 1, radius 3: v(1)=0 < v(1)+3=3 → constant wins → d_v = 0.0
        poly = LinearPolynomial([K2(1)], K2(1))
        p = make_pd([K2(0)], [3])
        dir = make_pd([K2(0)], [4])
        v = ValuationTangent(p, dir, [1])
        @test directional_derivative(poly, v) ≈ 0.0
    end

    @testset "2D: magnitude breaks tie between linear terms" begin
        # f = T₁ + T₂, radius (0,0): both val_weight=0; magnitude [1,0] → T₂ wins (mag 0 < 1) → d_v = -1.0
        poly = LinearPolynomial([K2(1), K2(1)], K2(0))
        p = make_pd([K2(0), K2(0)], [0, 0])
        dir = make_pd([K2(0), K2(0)], [1, 0])
        v = ValuationTangent(p, dir, [1, 0])
        @test directional_derivative(poly, v) ≈ -1.0
    end
end

@testset "directional_derivative for LinearAbsolutePolynomialSum" begin
    K2 = PadicField(2, 20)
    make_pd(cs, rs) = ValuationPolydisc{PadicFieldElem, Int, length(cs)}(tuple(cs...), tuple(rs...))

    @testset "one linear winner, one constant winner" begin
        # poly1 = T₁ (zero constant), radius 0: linear wins → -1.0
        # poly2 = 1 (zero linear), radius 0: constant wins → 0.0
        # sum = -1.0
        poly1 = LinearPolynomial([K2(1)], K2(0))
        poly2 = LinearPolynomial([K2(0)], K2(1))
        f = LinearAbsolutePolynomialSum([poly1, poly2])
        p = make_pd([K2(0)], [0])
        dir = make_pd([K2(0)], [1])
        v = ValuationTangent(p, dir, [1])
        @test directional_derivative(f, v) ≈ -1.0
    end

    @testset "both linear winners" begin
        # poly1 = T₁,     radius 1: linear val_weight=1 < constant (none) → d_v = -0.5
        # poly2 = T₁ + 4, radius 1: v(4)=2 > 0+1=1 → linear wins → d_v = -0.5
        # sum = -1.0
        poly1 = LinearPolynomial([K2(1)], K2(0))
        poly2 = LinearPolynomial([K2(1)], K2(4))
        f = LinearAbsolutePolynomialSum([poly1, poly2])
        p = make_pd([K2(0)], [1])
        dir = make_pd([K2(0)], [2])
        v = ValuationTangent(p, dir, [1])
        @test directional_derivative(f, v) ≈ -1.0
    end
end


# Evaluator coverage tests.

@testset "Polynomial function evaluation" begin
    NAML = NonArchimedeanMachineLearning

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
        p = _raw_polydisc([K(0), K(0)], [0, 0])  # Center 0, radius 0 (entire field)

        val = NAML.evaluate(f, p)
        # f = 2, so |f|_2 = 1/2 (since v(2) = 1 in 2-adic)
        @test val ≈ 0.5

        # Test 2: Linear monomial
        # f(x,y) = x, evaluated at center (1, 0), radius (0, 0) (entire field around (1,0))
        f = x
        p = _raw_polydisc([K(1), K(0)], [0, 0])
        val = NAML.evaluate(f, p)
        # Expanding around (1,0): f = (x-1) + 1, max term is 1, so |f|_2 = 1
        @test val ≈ 1.0

        # Test 3: Coefficient multiplied monomial
        # f(x,y) = 4x = 2^2 * x, radius 0 (entire field around (1,0))
        f = 4 * x
        p = _raw_polydisc([K(1), K(0)], [0, 0])
        val = NAML.evaluate(f, p)
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
        p = _raw_polydisc(center, radius)

        val = NAML.evaluate(poly, p)

        # Manual computation:
        # |a1| * 2^(-r1) = |1|_2 * 2^(0) = 1 * 1 = 1
        # |a2| * 2^(-r2) = |2|_2 * 2^(0) = 1/2 * 1 = 1/2
        # |b + a1*c1 + a2*c2| = |1 + 0 + 0| = 1
        # max(1, 1/2, 1) = 1

        @test val ≈ 1.0

        # Test 2: With radius [1, 0] - stricter constraint on first coordinate
        # radius = 1 means v(x1 - c1) >= 1, so |x1 - 0|_2 <= 2^(-1) = 1/2
        radius = [1, 0]
        p = _raw_polydisc(center, radius)
        val = NAML.evaluate(poly, p)

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
            _raw_polydisc([K(0), K(0)], [0, 0]),
            _raw_polydisc([K(1), K(1)], [0, 0]),
            _raw_polydisc([K(2), K(3)], [1, 1])
        ]

        for p in test_points
            val_abs = NAML.evaluate(abs_poly_sum, p)
            val_lin = NAML.evaluate(lin_poly_sum, p)

            # These should be approximately equal
            @test val_abs ≈ val_lin atol = 1e-10
        end
    end

    @testset "Single polynomial evaluation consistency" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Test: NAML.evaluate(f, p) should match NAML.evaluate(AbsolutePolynomialSum([f], p))
        # Using multivariate polynomials
        polynomials = [
            x,           # Monomial
            2 * x,         # Coefficient * monomial
            x + 1,       # Polynomial with constant
            x * y + 1     # Multivariate
        ]

        test_points = [
            _raw_polydisc([K(0), K(0)], [0, 0]),
            _raw_polydisc([K(1), K(1)], [0, 0]),
            _raw_polydisc([K(1), K(1)], [1, 1]),
            _raw_polydisc([K(2), K(1)], [0, 0])
        ]

        for f in polynomials
            for p in test_points
                val_single = NAML.evaluate(f, p)
                val_sum = NAML.evaluate(AbsolutePolynomialSum([f]), p)

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
        p = _raw_polydisc([K(1), K(1)], [0, 0])

        val = NAML.evaluate(abs_sum, p)
        # |1*1| = 1
        @test val ≈ 1.0

        # Test: (x+1)*(y+1) = xy + x + y + 1, evaluated at (0,0)
        f2 = (x + 1) * (y + 1)
        abs_sum2 = AbsolutePolynomialSum([f2])
        p2 = _raw_polydisc([K(0), K(0)], [0, 0])

        val2 = NAML.evaluate(abs_sum2, p2)
        # Expansion: f2 = x*y + x + y + 1
        # At (0,0): terms are 0, 0, 0, 1, so max absolute value is 1
        @test val2 ≈ 1.0
    end

    @testset "Polynomial sum evaluation" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Create sum: |f| + |g|
        f = x + 1
        g = y + 2

        fun = AbsolutePolynomialSum([f, g])

        p = _raw_polydisc([K(0), K(0)], [0, 0])

        # Evaluate the sum
        val_sum = NAML.evaluate(fun, p)

        # Compute components separately
        val_f = NAML.evaluate(f, p)
        val_g = NAML.evaluate(g, p)

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

        p = _raw_polydisc([K(1)], [0])

        # Evaluate the sum
        val_sum = NAML.evaluate(fun, p)

        # Compute components separately
        val_p1 = NAML.evaluate(poly1, p)
        val_p2 = NAML.evaluate(poly2, p)

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
        p_r0 = _raw_polydisc([K(0), K(0)], [0, 0])  # v(x) >= 0, v(y) >= 0 (entire field)
        p_r1 = _raw_polydisc([K(0), K(0)], [1, 0])  # v(x) >= 1 (stricter), v(y) >= 0
        p_r2 = _raw_polydisc([K(0), K(0)], [2, 0])  # v(x) >= 2 (even stricter), v(y) >= 0

        val_r0 = NAML.evaluate(abs_sum, p_r0)
        val_r1 = NAML.evaluate(abs_sum, p_r1)
        val_r2 = NAML.evaluate(abs_sum, p_r2)

        @test val_r0 ≈ 0.25
        @test val_r1 ≈ 0.125
        @test val_r2 ≈ 0.0625
    end

    @testset "Consistency: MPoly evaluate vs LinearPolynomial for linear functions" begin
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
            ([K(1), K(1)], [0, 1])
        ]

        for (center, radius) in test_points
            p = _raw_polydisc(center, radius)

            val_mv = NAML.evaluate(f_mv, p)
            val_lin = NAML.evaluate(f_lin, p)

            @test val_mv ≈ val_lin atol = 1e-10
        end
    end

    @testset "Batch evaluation vs regular evaluation - LinearPolynomial cases" begin
        prec = 20
        K = PadicField(2, prec)

        # Create several linear polynomials with different dimensions.
        test_polynomials = [
            LinearPolynomial([K(1)], K(1)),                    # x + 1
            LinearPolynomial([K(1), K(1)], K(1)),              # x + y + 1
            LinearPolynomial([K(1), K(1), K(1)], K(1))        # x + y + z + 1
        ]

        for poly in test_polynomials
            batch_eval = batch_evaluate_init(poly)
            @test batch_eval isa Function

            # Generate test points of appropriate dimension
            dim = length(poly.coefficients)
            test_points = [
                _raw_polydisc([K(0) for i in 1:dim], [0 for _ in 1:dim]),
                _raw_polydisc([K(1) for i in 1:dim], [0 for _ in 1:dim])
            ]

            for p in test_points
                val_batch = batch_eval(p)
                val_regular = NAML.evaluate(poly, p)

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

        p = _raw_polydisc([K(1), K(1)], [0, 0])

        val_batch = batch_eval(p)
        val_regular = NAML.evaluate(poly, p)

        @test val_batch ≈ val_regular
        @test val_batch ≈ 1.0
    end

    @testset "Batch evaluation with varying radii" begin
        prec = 20
        K = PadicField(2, prec)

        poly = LinearPolynomial([K(2), K(4)], K(1))
        batch_eval = batch_evaluate_init(poly)

        # Test several radius combinations
        radius_combinations = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [1, 2]
        ]

        for radius in radius_combinations
            p = _raw_polydisc([K(1), K(1)], radius)
            val_batch = batch_eval(p)
            val_regular = NAML.evaluate(poly, p)
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
            [K(1), K(2)]
        ]

        for center in center_combinations
            p = _raw_polydisc(center, [0, 0])
            val_batch = batch_eval(p)
            val_regular = NAML.evaluate(poly, p)
            @test val_batch ≈ val_regular
        end
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
            p = _raw_polydisc([K(i) for _ in 1:3], [0, 0, 0])
            val_batch = batch_eval(p)
            val_regular = NAML.evaluate(poly, p)

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
            _raw_polydisc([K(0), K(0)], [0, 0]),
            _raw_polydisc([K(1), K(1)], [0, 0]),
            _raw_polydisc([K(0), K(1)], [1, 0])
        ]

        for p in test_points
            val_batch = batch_eval(p)
            val_regular = NAML.evaluate(abs_sum, p)
            @test val_batch ≈ val_regular atol = 1e-10
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
            3 * x + 2 * y + 5
        ]

        test_points = [
            _raw_polydisc([K(0), K(0)], [0, 0]),
            _raw_polydisc([K(1), K(1)], [0, 0]),
            _raw_polydisc([K(2), K(3)], [1, 1])
        ]

        for f in polynomials
            batch_eval = batch_evaluate_init(f)

            for p in test_points
                val_batch = batch_eval(p)
                val_regular = NAML.evaluate(f, p)
                @test val_batch ≈ val_regular atol = 1e-10
            end
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
            LinearPolynomial([K(1), K(1)], K(2))
        ]
        lin_sum = LinearAbsolutePolynomialSum(polys)

        batch_eval = batch_evaluate_init(lin_sum)

        # Test at several points
        for i in 1:5
            p = _raw_polydisc([K(i), K(i)], [0, 0])
            val_batch = batch_eval(p)
            val_regular = NAML.evaluate(lin_sum, p)
            @test val_batch ≈ val_regular atol = 1e-10
        end
    end
end
