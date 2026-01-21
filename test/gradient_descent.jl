# Test file for gradient descent optimization.
#
# This file demonstrates and tests the gradient descent optimization algorithm
# on a simple polynomial model in p-adic space.

using Test
using Oscar
using NAML

@testset "Gradient Descent Optimization" begin
    # Set up synthetic data
    prec = 20
    K = PadicField(2, prec)
    a1 = [K(0)]
    r1 = Vector{Int}([0])
    p1 = ValuationPolydisc(a1, r1)
    p2 = ValuationPolydisc(Vector{PadicFieldElem}(), Vector{Int}())
    data = [(p2, 1)]

    # Create polynomial ring
    R, (x,) = polynomial_ring(K, ["x"])

    @testset "Model Creation" begin
        # Define a simple model: |x| where x is a parameter
        fun = AbsolutePolynomialSum([x])
        abs_model = AbstractModel(fun, [false])  # x is a parameter
        model = Model(abs_model, p1)

        @test abs_model isa AbstractModel
        @test model isa Model
        @test abs_model.param_info == [false]
    end

    @testset "Optimization Process" begin
        fun = AbsolutePolynomialSum([x])
        abs_model = AbstractModel(fun, [false])
        model = Model(abs_model, p1)

        # Define the loss function
        loss = Loss(
            (params::Vector) -> [NAML.evaluate(abs_model, p2, param) for param in params],
            (vs::Vector) -> [gradient_param(abs_model, p2, v) for v in vs]
        )

        @test loss isa Loss

        # Initialize gradient descent optimizer
        optim = greedy_descent_init(model.param, loss, 1, (false, 1))
        @test optim isa OptimSetup

        initial_loss = eval_loss(optim)

        # Run gradient descent for fewer epochs (for testing)
        N_epochs = 5
        for i in 1:N_epochs
            step!(optim)
        end

        # Verify optimization ran
        @test optim.param isa ValuationPolydisc
        final_loss = eval_loss(optim)
        # Loss should decrease or stay the same
        @test final_loss <= initial_loss
    end
end

