## Test directional_derivative for typed evaluators

using Test
using Oscar
using NAML

@testset "Typed Gradient Evaluators" begin
    prec = 20
    K = PadicField(2, prec)

    # Helper to create polydiscs without auto-wrapping to ValuedFieldPoint
    make_polydisc(centers, radii) = ValuationPolydisc{PadicFieldElem,Int,length(centers)}(tuple(centers...), tuple(radii...))

    @testset "ConstantEvaluator derivative" begin
        c = NAML.Constant{PadicFieldElem}(5.0)
        eval_typed = batch_evaluate_init(c, ValuationPolydisc{PadicFieldElem,Int,2})

        p = make_polydisc([K(1), K(2)], [0, 0])
        direction = [K(1), K(0)]
        v = ValuationTangent(p, direction, [1, 0])

        @test directional_derivative(eval_typed, v) == 0.0
    end

    @testset "MPolyEvaluator derivative matches symbolic" begin
        R, (x, y) = polynomial_ring(K, ["x", "y"])
        poly = x^2 + y

        # Create evaluator
        eval_typed = batch_evaluate_init(poly, ValuationPolydisc{PadicFieldElem,Int,2})

        p = make_polydisc([K(3), K(1)], [0, 0])
        direction = [K(1), K(0)]
        v = ValuationTangent(p, direction, [1, 0])

        # Compare evaluator derivative with symbolic derivative
        eval_result = directional_derivative(eval_typed, v)
        symbolic_result = directional_derivative(poly, v)

        @test eval_result ≈ symbolic_result
    end

    @testset "SumEvaluator derivative matches symbolic" begin
        R, (x, y) = polynomial_ring(K, ["x", "y"])
        f = AbsolutePolynomialSum([x^2, y^2])

        eval_typed = batch_evaluate_init(f, ValuationPolydisc{PadicFieldElem,Int,2})

        p = make_polydisc([K(3), K(5)], [0, 0])
        direction = [K(1), K(1)]
        v = ValuationTangent(p, direction, [1, 1])

        eval_result = directional_derivative(eval_typed, v)
        symbolic_result = directional_derivative(f, v)

        @test eval_result ≈ symbolic_result
    end

    @testset "AddEvaluator derivative" begin
        R, (x,) = polynomial_ring(K, ["x"])
        f1 = AbsolutePolynomialSum([x^2])
        f2 = AbsolutePolynomialSum([x])
        f = f1 + f2

        eval_typed = batch_evaluate_init(f, ValuationPolydisc{PadicFieldElem,Int,1})

        p = make_polydisc([K(3)], [0])
        direction = [K(1)]
        v = ValuationTangent(p, direction, [1])

        eval_result = directional_derivative(eval_typed, v)
        symbolic_result = directional_derivative(f, v)

        @test eval_result ≈ symbolic_result
    end

    @testset "MulEvaluator derivative (product rule)" begin
        R, (x,) = polynomial_ring(K, ["x"])
        f1 = AbsolutePolynomialSum([x])
        f2 = AbsolutePolynomialSum([x + K(1)])
        f = f1 * f2

        eval_typed = batch_evaluate_init(f, ValuationPolydisc{PadicFieldElem,Int,1})

        p = make_polydisc([K(3)], [0])
        direction = [K(1)]
        v = ValuationTangent(p, direction, [1])

        eval_result = directional_derivative(eval_typed, v)
        symbolic_result = directional_derivative(f, v)

        @test eval_result ≈ symbolic_result
    end

    @testset "SMulEvaluator derivative" begin
        R, (x,) = polynomial_ring(K, ["x"])
        f = 3 * AbsolutePolynomialSum([x^2])

        eval_typed = batch_evaluate_init(f, ValuationPolydisc{PadicFieldElem,Int,1})

        p = make_polydisc([K(5)], [0])
        direction = [K(1)]
        v = ValuationTangent(p, direction, [1])

        eval_result = directional_derivative(eval_typed, v)
        symbolic_result = directional_derivative(f, v)

        @test eval_result ≈ symbolic_result
    end

    @testset "CompEvaluator derivative (chain rule)" begin
        R, (x,) = polynomial_ring(K, ["x"])
        inner = AbsolutePolynomialSum([x])
        outer = DifferentiableFunction(x -> x^2, x -> 2x)
        f = NAML.Comp(outer, inner)

        eval_typed = batch_evaluate_init(f, ValuationPolydisc{PadicFieldElem,Int,1})

        p = make_polydisc([K(3)], [0])
        direction = [K(1)]
        v = ValuationTangent(p, direction, [1])

        eval_result = directional_derivative(eval_typed, v)
        symbolic_result = directional_derivative(f, v)

        @test eval_result ≈ symbolic_result
    end

    @testset "LambdaEvaluator derivative" begin
        # Lambda with derivative provided
        l = NAML.Lambda{PadicFieldElem}(
            p -> abs(p.center[1]),
            v -> 0.5  # dummy derivative for testing
        )

        eval_typed = batch_evaluate_init(l, ValuationPolydisc{PadicFieldElem,Int,1})

        p = make_polydisc([K(3)], [0])
        v = ValuationTangent(p, [K(1)], [1])

        @test directional_derivative(eval_typed, v) == 0.5

        # Lambda without derivative should error
        l_no_deriv = NAML.Lambda{PadicFieldElem}(p -> abs(p.center[1]))
        eval_no_deriv = batch_evaluate_init(l_no_deriv, ValuationPolydisc{PadicFieldElem,Int,1})
        @test_throws ErrorException directional_derivative(eval_no_deriv, v)
    end

    @testset "VFP LambdaEvaluator derivative (MPoly lifting)" begin
        R, (x,) = polynomial_ring(K, ["x"])
        poly = x^2 + K(1)

        # Create VFP evaluator (should use LambdaEvaluator with derivative)
        VFP = ValuedFieldPoint{2,20,PadicFieldElem}
        eval_typed = batch_evaluate_init(poly, ValuationPolydisc{VFP,Int,1})

        # Use VFP-wrapped polydisc for VFP evaluator
        p = ValuationPolydisc([K(3)], [0])
        direction = [K(1)]
        v = ValuationTangent(p, direction, [1])

        # Compare with symbolic derivative
        eval_result = directional_derivative(eval_typed, v)

        # Compute symbolic result using unwrapped types
        p_unwrapped = make_polydisc([K(3)], [0])
        v_unwrapped = ValuationTangent(p_unwrapped, [K(1)], [1])
        symbolic_result = directional_derivative(poly, v_unwrapped)

        @test eval_result ≈ symbolic_result
    end
end
