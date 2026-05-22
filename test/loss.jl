using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Loss Constructors" begin
    NAML = NonArchimedeanMachineLearning
    K = PadicField(2, 20)
    make_pd(cs, rs) = ValuationPolydisc{PadicFieldElem, Int, length(cs)}(tuple(cs...), tuple(rs...))

    parameter_only_model = AbstractModel(
        LinearAbsolutePolynomialSum([LinearPolynomial([K(0), K(1)], K(0))]),
        [true, false]
    )
    param = make_pd([K(0)], [1])
    direction = make_pd([K(0)], [2])
    tangent = ValuationTangent(param, direction, [1])

    @testset "field-valued MSE uses absolute values for targets" begin
        loss = MSE_loss_init(parameter_only_model, [(K(1), K(2)), (K(3), K(1))])

        @test loss.eval([param]) == [1 / 8]
        @test loss.grad([tangent]) == [1 / 4]
    end

    @testset "experimental composed MSE path matches the direct MSE path" begin
        loss = NAML.MSE_loss_init_new(parameter_only_model, [(K(1), 0.0), (K(3), 0.0)])

        @test loss.eval([param]) == [1 / 4]
        @test loss.eval(ValuationPolydisc{PadicFieldElem, Int, 1}[]) == Float64[]
        @test loss.grad([tangent]) == [-1 / 2]
        @test loss.grad(ValuationTangent{PadicFieldElem, Int, 1}[]) == Float64[]
    end

    @testset "field-valued MPE expects numeric targets" begin
        loss = MPE_loss_init(parameter_only_model, [(K(1), 0.0), (K(3), 0.0)], 3)

        @test loss.eval([param]) == [1 / 8]
        @test loss.grad([tangent]) == [-3 / 8]
    end

    @testset "vector-valued input MPE specialises all data variables" begin
        model = AbstractModel(
            LinearAbsolutePolynomialSum([LinearPolynomial([K(0), K(0), K(1)], K(0))]),
            [true, true, false]
        )
        loss = MPE_loss_init(model, [([K(1), K(2)], 0.0), ([K(3), K(4)], 0.0)], 2)

        @test loss.eval([param]) == [1 / 4]
        @test loss.grad([tangent]) == [-1 / 2]
    end

    @testset "polydisc-valued MSE and MPE return vector gradients" begin
        data = [
            (make_pd([K(1)], [1]), 0.0),
            (make_pd([K(3)], [1]), 0.0)
        ]
        mse = MSE_loss_init(parameter_only_model, data)
        mpe = MPE_loss_init(parameter_only_model, data, 3)

        @test mse.eval([param]) == [1 / 4]
        @test mse.grad([tangent]) == [[-1 / 2]]
        @test mpe.eval([param]) == [1 / 8]
        @test mpe.grad([tangent]) == [[-3 / 8]]
    end

    @testset "ValuedFieldPoint polydisc overloads use lifted typed evaluators" begin
        data = [
            (ValuationPolydisc([K(1)], [1]), 0.0),
            (ValuationPolydisc([K(3)], [1]), 0.0)
        ]
        param_vfp = ValuationPolydisc([K(0)], [1])
        direction_vfp = ValuationPolydisc([K(0)], [2])
        tangent_vfp = ValuationTangent(param_vfp, direction_vfp, [1])
        mse = MSE_loss_init(parameter_only_model, data)
        mpe = MPE_loss_init(parameter_only_model, data, 3)

        @test mse.eval([param_vfp]) == [1 / 4]
        @test mse.grad([tangent_vfp]) == [[-1 / 2]]
        @test mpe.eval([param_vfp]) == [1 / 8]
        @test mpe.grad([tangent_vfp]) == [[-3 / 8]]
    end
end
