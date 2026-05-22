using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Least Squares Loss Builders" begin
    K = PadicField(2, 20)

    @testset "solve_linear_system evaluates and differentiates parameter batches" begin
        loss = solve_linear_system(
            [K(1) K(2); K(3) K(4)],
            [K(0), K(0)],
            [K(3), K(7)]
        )

        fit_param = ValuationPolydisc([K(1), K(1)], [5, 5])
        bad_param = ValuationPolydisc([K(0), K(0)], [5, 5])

        @test loss.eval([fit_param, bad_param]) == [1 / 512, 2.0]
        @test loss.eval(ValuationPolydisc[]) == Float64[]

        x_direction = ValuationPolydisc([K(1), K(1)], [6, 5])
        y_direction = ValuationPolydisc([K(1), K(1)], [5, 6])
        x_tangent = ValuationTangent(fit_param, x_direction, [1, 0])
        y_tangent = ValuationTangent(fit_param, y_direction, [0, 1])

        @test loss.grad([x_tangent, y_tangent]) == [-1 / 256, -1 / 256]
        @test loss.grad(ValuationTangent[]) == Float64[]

        unwrapped_fit = ValuationPolydisc{PadicFieldElem, Int, 2}((K(1), K(1)), (5, 5))
        unwrapped_direction = ValuationPolydisc{PadicFieldElem, Int, 2}((K(1), K(1)), (6, 5))
        unwrapped_tangent = ValuationTangent(unwrapped_fit, unwrapped_direction, [1, 0])
        @test loss.eval([unwrapped_fit]) == [1 / 512]
        @test loss.grad([unwrapped_tangent]) == [-1 / 256]
    end

    @testset "make_ordinary_least_squares_loss uses row-major A entries then b" begin
        data = [
            ([K(1), K(0)], [K(2)]),
            ([K(0), K(1)], [K(4)])
        ]
        loss = make_ordinary_least_squares_loss(data)

        fit_param = ValuationPolydisc([K(2), K(4), K(0)], [5, 5, 5])
        bad_param = ValuationPolydisc([K(1), K(1), K(1)], [5, 5, 5])

        @test loss.eval([fit_param, bad_param]) == [1 / 512, 257 / 1024]

        a11_direction = ValuationPolydisc([K(2), K(4), K(0)], [6, 5, 5])
        a12_direction = ValuationPolydisc([K(2), K(4), K(0)], [5, 6, 5])
        b_direction = ValuationPolydisc([K(2), K(4), K(0)], [5, 5, 6])
        a11_tangent = ValuationTangent(fit_param, a11_direction, [1, 0, 0])
        a12_tangent = ValuationTangent(fit_param, a12_direction, [0, 1, 0])
        b_tangent = ValuationTangent(fit_param, b_direction, [0, 0, 1])

        @test loss.grad([a11_tangent, a12_tangent, b_tangent]) ==
              [-1 / 256, -1 / 256, -1 / 256]
    end
end
