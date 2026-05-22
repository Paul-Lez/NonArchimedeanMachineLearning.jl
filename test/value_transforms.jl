using Test
using NonArchimedeanMachineLearning

@testset "Value Transforms" begin
    @testset "sigmoid_transform maps exact midpoint and shifted losses" begin
        transform = sigmoid_transform(scale = 2.0, center = 1.0)
        losses = [1.0, 0.0, 2.0]
        expected = [0.5, 1 / (1 + exp(-2.0)), 1 / (1 + exp(2.0))]

        @test transform.(losses) ≈ expected
    end

    @testset "tanh_transform maps exact midpoint and symmetric losses" begin
        transform = tanh_transform(scale = 0.5, center = 2.0)
        losses = [2.0, 4.0, 0.0]
        expected = [0.5, (1 - tanh(1.0)) / 2, (1 - tanh(-1.0)) / 2]

        @test transform.(losses) ≈ expected
    end

    @testset "negation_transform returns negated batch content" begin
        transform = negation_transform()

        @test transform.([3.0, -2.5, 0.0]) == [-3.0, 2.5, 0.0]
    end

    @testset "inverse_transform applies epsilon in denominator" begin
        transform = inverse_transform(epsilon = 0.25)

        @test transform.([0.0, 0.75, 1.75]) == [4.0, 1.0, 0.5]
    end

    @testset "DEFAULT_VALUE_TRANSFORM is the default sigmoid" begin
        losses = [-1.0, 0.0, 1.0]

        @test DEFAULT_VALUE_TRANSFORM.(losses) ≈ sigmoid_transform().(losses)
        @test DEFAULT_VALUE_TRANSFORM(0.0) == 0.5
    end
end
