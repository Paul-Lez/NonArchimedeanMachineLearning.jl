using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Model Internals" begin
    NAML = NonArchimedeanMachineLearning
    K = PadicField(2, 20)
    make_pd(cs, rs) = ValuationPolydisc{PadicFieldElem, Int, length(cs)}(tuple(cs...), tuple(rs...))

    poly = LinearPolynomial([K(2), K(3), K(4), K(5)], K(6))
    fun = LinearAbsolutePolynomialSum([poly])
    model = AbstractModel(fun, [true, false, true, false])
    data_point = make_pd([K(1), K(2)], [1, 2])
    param_point = make_pd([K(3), K(4)], [3, 4])

    @testset "variable and parameter index helpers" begin
        @test NAML.var_indices(model) == [1, 3]
        @test NAML.param_indices(model) == [2, 4]
        @test NAML.getkeys(model) == [1, 1, 2, 2]
    end

    @testset "interleaving data and parameter polydiscs" begin
        full = NAML.set_abstract_model_variable(model, data_point, param_point)

        @test typeof(full) == ValuationPolydisc{PadicFieldElem, Int, 4}
        @test collect(NAML.center(full)) == [K(1), K(3), K(2), K(4)]
        @test collect(NAML.radius(full)) == [1, 3, 2, 4]

        public_data = ValuationPolydisc([K(1), K(2)], [1, 2])
        public_param = ValuationPolydisc([K(3), K(4)], [3, 4])
        full_vfp = NAML.set_abstract_model_variable(model, public_data, public_param)

        @test typeof(full_vfp) ==
              ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int, 4}
        @test collect(NAML.unwrap(collect(NAML.center(full_vfp)))) == [K(1), K(3), K(2), K(4)]
        @test collect(NAML.radius(full_vfp)) == [1, 3, 2, 4]

        mixed_full = NAML.set_abstract_model_variable(model, data_point, public_param)
        @test typeof(mixed_full) == ValuationPolydisc{PadicFieldElem, Int, 4}
        @test collect(NAML.center(mixed_full)) == [K(1), K(3), K(2), K(4)]
    end

    @testset "specialising linear models keeps row order and constants" begin
        specialised = NAML.specialise(poly, model.param_info, [K(1), K(2)])

        @test specialised.coefficients == [K(3), K(5)]
        @test specialised.constant == K(16)

        specialised_sum = NAML.specialise(fun, model.param_info, [K(1), K(2)])
        @test length(specialised_sum.polys) == 1
        @test specialised_sum.polys[1].coefficients == [K(3), K(5)]
        @test specialised_sum.polys[1].constant == K(16)
    end

    @testset "specialising polynomial sums combines multiple data points" begin
        R, (x, theta) = polynomial_ring(K, ["x", "theta"])
        abs_model = AbstractModel(AbsolutePolynomialSum([x + theta, x + 2 * theta]),
            [true, false])
        lin_model = AbstractModel(LinearAbsolutePolynomialSum([
            LinearPolynomial([K(1), K(1)], K(0)),
            LinearPolynomial([K(2), K(3)], K(0))
        ]), [true, false])

        combined_abs = NAML.specialise(abs_model, [[K(1)], [K(3)]])
        combined_lin = NAML.specialise(lin_model, [[K(1)], [K(3)]])

        @test length(combined_abs.polys) == 4
        @test length(combined_lin.polys) == 4
        @test [p.constant for p in combined_lin.polys] == [K(1), K(2), K(3), K(6)]
        @test [p.coefficients for p in combined_lin.polys] ==
              [[K(1)], [K(3)], [K(1)], [K(3)]]
    end

    @testset "specialising composite function types" begin
        base = LinearAbsolutePolynomialSum([LinearPolynomial([K(1), K(1)], K(0))])
        base_model = AbstractModel(base, [true, false])
        param = make_pd([K(2)], [3])
        square = DifferentiableFunction(x -> x^2, x -> 2x)

        specialised_base = NAML.specialise(base_model, [K(1)])
        @test NAML.evaluate(specialised_base, param) == 1.0
        @test NAML.evaluate(NAML.specialise(base + 2, [true, false], [K(1)]), param) == 3.0
        @test NAML.evaluate(NAML.specialise(base - 1 / 4, [true, false], [K(1)]), param) == 3 / 4
        @test NAML.evaluate(NAML.specialise(base * (base + 1), [true, false], [K(1)]), param) == 2.0
        @test NAML.evaluate(NAML.specialise(base / (base + 1), [true, false], [K(1)]), param) == 1 / 2
        @test NAML.evaluate(NAML.specialise(3 * base, [true, false], [K(1)]), param) == 3.0
        @test NAML.evaluate(NAML.specialise(NAML.comp(square, base), [true, false], [K(1)]), param) == 1.0
        @test NAML.evaluate(NAML.specialise(NAML.Constant{PadicFieldElem}(5.0),
            [true, false], [K(1)]), param) == 5.0
    end

    @testset "model evaluation and batch evaluators" begin
        value = NAML.evaluate(model, data_point, param_point)
        model_with_params = Model(model, param_point)
        legacy_eval = batch_evaluate_init(model)
        model_eval = batch_evaluate_init(model_with_params)
        typed_eval = batch_evaluate_init(model, ValuationPolydisc{PadicFieldElem, Int, 4})

        @test value == 1.0
        @test NAML.evaluate(model_with_params, data_point) == value
        @test legacy_eval(data_point, param_point) == value
        @test model_eval(data_point) == value
        @test typed_eval(data_point, param_point) == value

        new_param = make_pd([K(5), K(6)], [2, 2])
        NAML.update_weights!(model_with_params, new_param)
        @test model_with_params.param == new_param
        @test NAML.set_model_variable(model_with_params, data_point) ==
              NAML.set_abstract_model_variable(model, data_point, new_param)
    end

    @testset "typed ValuedFieldPoint model evaluator" begin
        public_data = ValuationPolydisc([K(1), K(2)], [1, 2])
        public_param = ValuationPolydisc([K(3), K(4)], [3, 4])
        typed_eval = batch_evaluate_init(
            model,
            ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int, 4}
        )

        @test typed_eval(public_data, public_param) == 1.0
    end
end
