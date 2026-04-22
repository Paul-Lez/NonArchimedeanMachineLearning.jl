## Test the new typed evaluator interface

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Typed Evaluators" begin
    @testset "Basic Setup" begin
        prec = 20
        K = PadicField(2, prec)

        # Create polydiscs with ValuedFieldPoint
        center1 = [ValuedFieldPoint(K(1)), ValuedFieldPoint(K(2))]
        radius1 = [0, 0]
        p1 = ValuationPolydisc(center1, radius1)

        @test typeof(p1) ==
              ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int, 2}
        println("✓ Basic polydisc creation works")
    end

    @testset "LinearPolynomial Evaluator" begin
        prec = 20
        K = PadicField(2, prec)

        # Create a linear polynomial: 3*T1 + 2*T2 + 1
        # Use ValuedFieldPoint for coefficients
        poly = LinearPolynomial([ValuedFieldPoint(K(3)), ValuedFieldPoint(K(2))], ValuedFieldPoint(K(1)))

        # Create a polydisc at (1,2) with zero radius
        center = [ValuedFieldPoint(K(1)), ValuedFieldPoint(K(2))]
        radius = [0, 0]
        p = ValuationPolydisc(center, radius)

        # NEW INTERFACE: typed evaluator
        eval_typed = batch_evaluate_init(
            poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2})
        result = eval_typed(p)

        # The result should be |3*1 + 2*2 + 1| = |8| = 1/8 in 2-adic (since 8 = 2^3)
        @test result > 0
        @test typeof(eval_typed) ==
              LinearPolynomialEvaluator{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}

        println("✓ LinearPolynomial evaluator works! Result: $result")
    end

    @testset "Constant Evaluator" begin
        prec = 20
        K = PadicField(2, prec)

        # Create a constant function using NonArchimedeanMachineLearning.Constant
        # Note: Constant takes a Number value, not a field element
        c = NonArchimedeanMachineLearning.Constant{PadicFieldElem}(5.0)

        # Create a polydisc
        center = [ValuedFieldPoint(K(1)), ValuedFieldPoint(K(2))]
        radius = [0, 0]
        p = ValuationPolydisc(center, radius)

        # NEW INTERFACE: typed evaluator (adapter handles type conversion)
        eval_typed = batch_evaluate_init(
            c, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2})
        result = eval_typed(p)

        @test result == 5.0
        @test typeof(eval_typed) ==
              ConstantEvaluator{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}

        println("✓ Constant evaluator works!")
    end

    @testset "MPoly Evaluator" begin
        prec = 20
        K = PadicField(2, prec)
        R, (x, y) = polynomial_ring(K, ["x", "y"])

        # Create a polynomial
        poly = x^2 + y^2

        # Create a polydisc
        center = [ValuedFieldPoint(K(1)), ValuedFieldPoint(K(2))]
        radius = [0, 0]
        p = ValuationPolydisc(center, radius)

        # NEW INTERFACE: typed evaluator
        eval_typed = batch_evaluate_init(
            poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2})
        result = eval_typed(p)

        # Result should be |1^2 + 2^2| = |5|
        @test result > 0
        # MPoly with PadicFieldElem coefficients gets wrapped in LambdaEvaluator for type compatibility
        @test typeof(eval_typed) <: PolydiscFunctionEvaluator

        println("✓ MPoly evaluator works! Result: $result (type: $(typeof(eval_typed)))")
    end

    @testset "Sum Evaluator for LinearAbsolutePolynomialSum" begin
        prec = 20
        K = PadicField(2, prec)

        # Create two linear polynomials with ValuedFieldPoint
        poly1 = LinearPolynomial([ValuedFieldPoint(K(3)), ValuedFieldPoint(K(2))], ValuedFieldPoint(K(1)))
        poly2 = LinearPolynomial([ValuedFieldPoint(K(1)), ValuedFieldPoint(K(1))], ValuedFieldPoint(K(2)))
        sum_poly = LinearAbsolutePolynomialSum([poly1, poly2])

        # Create a polydisc
        center = [ValuedFieldPoint(K(1)), ValuedFieldPoint(K(2))]
        radius = [0, 0]
        p = ValuationPolydisc(center, radius)

        # NEW INTERFACE: typed evaluator
        eval_typed = batch_evaluate_init(
            sum_poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2})
        result = eval_typed(p)

        @test result > 0
        @test typeof(eval_typed) <: SumEvaluator

        println("✓ Sum evaluator works! Result: $result")
    end
end
