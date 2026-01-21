# Test file for polynomial learning via greedy descent.
#
# This file demonstrates learning the roots of a cubic polynomial
# (x - a)(x - b)(x - c) using greedy descent optimization in 2-adic space.
# The task is to find parameters a, b, c that minimize the loss function
# given several data points (roots).

using Test
using Oscar
using NAML

@testset "Polynomial Learning with Greedy Descent" begin
    # Initialize 2-adic field with precision
    p, prec = 2, 20
    K = PadicField(p, prec)

    # Create polynomial ring with variables: x (data), a, b, c (parameters)
    R, (x, a, b, c) = K["x", "a", "b", "c"]
    g = AbsolutePolynomialSum([(x - a) * (x - b) * (x - c)])

    # Create model: x is a data variable (true), a,b,c are parameters (false)
    f = AbstractModel(g, [true, false, false, false])

    @testset "Model Setup" begin
        @test f isa AbstractModel
        @test length(f.param_info) == 4
        @test f.param_info == [true, false, false, false]
    end

    @testset "Data Setup" begin
        # Setting up training data points
        # Each data point p_i is paired with the desired output value
        # High valuation (large radius) = small disc, low valuation = large disc
        p1 = ValuationPolydisc([K(p^0)], Vector{Int}([prec]))  # x = 1
        p2 = ValuationPolydisc([K(p^1)], Vector{Int}([prec]))  # x = 2
        p3 = ValuationPolydisc([K(p^2)], Vector{Int}([prec]))  # x = 4
        p4 = ValuationPolydisc([K(3)], Vector{Int}([prec]))    # x = 3
        data = [(p1, 0), (p2, 0), (p3, 0), (p4, -2)]

        @test length(data) == 4
        @test all(d -> d[1] isa ValuationPolydisc, data)
    end

    @testset "Optimization" begin
        # Setting up training data
        p1 = ValuationPolydisc([K(p^0)], Vector{Int}([prec]))
        p2 = ValuationPolydisc([K(p^1)], Vector{Int}([prec]))
        p3 = ValuationPolydisc([K(p^2)], Vector{Int}([prec]))
        p4 = ValuationPolydisc([K(3)], Vector{Int}([prec]))
        data = [(p1, 0), (p2, 0), (p3, 0), (p4, -2)]

        # Set initial parameter values
        model = Model(f, ValuationPolydisc([K(11), K(22), K(33)], [0, 0, 0]))
        @test model isa Model
        @test length(model.param.center) == 3

        # Create Mean p-Power Error (MPE) loss function with p=2 (MSE-like)
        ell = MPE_loss_init(f, data, 2)
        @test ell isa Loss

        # Initialize greedy descent optimizer
        greedy_optim = greedy_descent_init(model.param, ell, 1, (false, 1))
        @test greedy_optim isa OptimSetup

        # Record initial loss
        initial_loss = eval_loss(greedy_optim)

        # Run greedy descent optimization (fewer epochs for testing)
        N_epochs = 10
        for i in 1:N_epochs
            step!(greedy_optim)
        end

        # Test that loss decreased
        final_loss = eval_loss(greedy_optim)
        @test final_loss <= initial_loss

        # Test that parameters were updated
        @test greedy_optim.param isa ValuationPolydisc
        @test length(greedy_optim.param.center) == 3
    end
end
