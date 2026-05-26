# Test file for polynomial evaluation and derivatives.
#
# This file tests polynomial evaluation and directional derivative computation
# on polydisc space, including absolute polynomial sums.

using Test
using Oscar
using NonArchimedeanMachineLearning

struct UnsupportedPolydiscFunction{S} <: NonArchimedeanMachineLearning.PolydiscFunction{S} end

@testset "Polynomial Functions" begin
    # Set up synthetic data
    prec = 20
    K = PadicField(3, prec)
    a1 = [K(1), K(2)]
    r1 = Vector{Int}([1, 2])
    a2 = [K(2), K(2)]
    r2 = Vector{Int}([2, 2])
    p1 = ValuationPolydisc(a1, r1)
    p2 = ValuationPolydisc(a2, r2)
    data = [(p1, 1), (p2, 1)]

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
