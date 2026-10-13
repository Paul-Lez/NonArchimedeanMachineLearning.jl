using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Typed evaluators with wrapped coordinates" begin
    K = PadicField(2, 20)
    R, (x, y) = polynomial_ring(K, ["x", "y"])
    VFP = ValuedFieldPoint{2, 20, PadicFieldElem}
    PT = ValuationPolydisc{VFP, Int, 2}
    linear = LinearPolynomial(VFP.([K(3), K(2)]), VFP(K(1)))
    second = LinearPolynomial(VFP.([K(1), K(1)]), VFP(K(2)))
    p = ValuationPolydisc([K(1), K(2)], [3, 3])

    # Cancellation makes the linear value at the center 8, with disc norm 1/8.
    # Both native wrapped coefficients and adapters for raw coefficients are exercised.
    for (f, expected) in [(linear, 1 / 8),
            (NonArchimedeanMachineLearning.Constant{PadicFieldElem}(5.0), 5.0),
            (x^2 + y^2, 1.0),
            (LinearAbsolutePolynomialSum([linear, second]), 9 / 8)]
        @test batch_evaluate_init(f, PT)(p) == expected
    end
end

@testset "Typed Evaluator Batch Calls" begin                                 
    prec = 20                                                                
    K = PadicField(2, prec)                                                  
                                                                             
    VP = ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}
                                                                             
    poly = LinearPolynomial(                                                 
        [ValuedFieldPoint(K(3)), ValuedFieldPoint(K(2))],                    
        ValuedFieldPoint(K(1))                                               
    )                                                                        
                                                                             
    evaluator = batch_evaluate_init(poly, VP)                                
                                                                             
    p1 = ValuationPolydisc(                                                  
        [ValuedFieldPoint(K(1)), ValuedFieldPoint(K(2))],                    
        [0, 0]                                                               
    )                                                                        
                                                                             
    p2 = ValuationPolydisc(                                                  
        [ValuedFieldPoint(K(2)), ValuedFieldPoint(K(3))],                    
        [0, 0]
    )
    points = [p1, p2]
    @testset "Batch agrees with scalar evaluation" begin
        scalar_results = [evaluator(point) for point in points]
        batch_results = evaluator(points)
        @test batch_results == scalar_results
        @test length(batch_results) == length(points)
    end
    @testset "Single-element batches work" begin
        @test evaluator([p1]) == [evaluator(p1)]
    end
    @testset "Empty batches work" begin
        empty_points = Vector{typeof(p1)}()
        result = evaluator(empty_points)
        @test isempty(result)
        @test result isa Vector{Float64}
    end
    @testset "Composite evaluators support batches" begin
        poly2 = LinearPolynomial(
            [ValuedFieldPoint(K(1)), ValuedFieldPoint(K(1))],
            ValuedFieldPoint(K(2))
        )
        composed = LinearAbsolutePolynomialSum([poly, poly2])
        composed_evaluator = batch_evaluate_init(composed, VP)
        scalar_results = [composed_evaluator(point) for point in points]
        batch_results = composed_evaluator(points)
        @test batch_results == scalar_results
    end
end

