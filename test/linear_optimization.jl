# Test: Minimize |f(x, y)| for a bivariate linear polynomial over a 3-adic field
# using greedy descent and MCTS, with both LinearPolynomial and MPoly representations.

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Linear Polynomial Optimization (3-adic)" begin
    p, prec = 3, 20
    K = PadicField(p, prec)

    # Target: find (x, y) minimizing |2x + y + 1|_3
    # A root is e.g. (x, y) = (1, -3) since 2*1 + (-3) + 1 = 0

    # Both x and y are parameters (no data variables)
    param_info = [false, false]
    empty_data = ValuationPolydisc(K, Vector{PadicFieldElem}(), Vector{Int}())
    data = [(empty_data, 0)]
    param0 = ValuationPolydisc([K(0), K(0)], [0, 0])
    n_steps = 20

    @testset "LinearPolynomial + Greedy Descent" begin
        poly = LinearPolynomial([K(2), K(1)], K(1))
        fun = LinearAbsolutePolynomialSum([poly])
        model = AbstractModel(fun, param_info)
        loss = MPE_loss_init(model, data, 2)

        optim = greedy_descent_init(param0, loss, 1, (false, 1))
        initial_loss = eval_loss(optim)
        for _ in 1:n_steps; step!(optim); end
        final_loss = eval_loss(optim)

        @test final_loss <= initial_loss
        println("LinearPoly + Greedy:  $initial_loss → $final_loss")
    end

    @testset "LinearPolynomial + MCTS" begin
        poly = LinearPolynomial([K(2), K(1)], K(1))
        fun = LinearAbsolutePolynomialSum([poly])
        model = AbstractModel(fun, param_info)
        loss = MPE_loss_init(model, data, 2)

        config = MCTSConfig(num_simulations=50, degree=1)
        optim = mcts_descent_init(param0, loss, config)
        initial_loss = eval_loss(optim)
        for _ in 1:n_steps; step!(optim); end
        final_loss = eval_loss(optim)

        @test final_loss <= initial_loss
        println("LinearPoly + MCTS:    $initial_loss → $final_loss")
    end

    @testset "MPoly + Greedy Descent" begin
        R, (x, y) = K["x", "y"]
        fun = AbsolutePolynomialSum([2*x + y + R(1)])
        model = AbstractModel(fun, param_info)
        loss = MPE_loss_init(model, data, 2)

        optim = greedy_descent_init(param0, loss, 1, (false, 1))
        initial_loss = eval_loss(optim)
        for _ in 1:n_steps; step!(optim); end
        final_loss = eval_loss(optim)

        @test final_loss <= initial_loss
        println("MPoly + Greedy:       $initial_loss → $final_loss")
    end

    @testset "MPoly + MCTS" begin
        R, (x, y) = K["x", "y"]
        fun = AbsolutePolynomialSum([2*x + y + R(1)])
        model = AbstractModel(fun, param_info)
        loss = MPE_loss_init(model, data, 2)

        config = MCTSConfig(num_simulations=50, degree=1)
        optim = mcts_descent_init(param0, loss, config)
        initial_loss = eval_loss(optim)
        for _ in 1:n_steps; step!(optim); end
        final_loss = eval_loss(optim)

        @test final_loss <= initial_loss
        println("MPoly + MCTS:         $initial_loss → $final_loss")
    end
end
