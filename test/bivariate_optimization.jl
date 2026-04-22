# Test: Minimize |f(x, y)| for a bivariate cubic over a 3-adic field
# using greedy descent and MCTS.

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Bivariate Cubic Optimization (3-adic)" begin
    # 3-adic field with precision 20
    p, prec = 3, 20
    K = PadicField(p, prec)

    # Bivariate polynomial ring
    R, (x, y) = K["x", "y"]

    # A degree-3 polynomial: f(x, y) = x^3 + 2*x*y^2 + y^3 - x - y + 1
    poly = x^3 + 2*x*y^2 + y^3 - x - y + R(1)
    fun = AbsolutePolynomialSum([poly])

    # Both x and y are parameters to optimize (no data variables)
    model = AbstractModel(fun, [false, false])

    # No data variables → single dummy data point with target 0
    # We want to minimize |f(x,y) - 0| = |f(x,y)|
    empty_data = ValuationPolydisc(K, Vector{PadicFieldElem}(), Vector{Int}())
    data = [(empty_data, 0)]

    # Create loss: MPE with power 2
    loss = MPE_loss_init(model, data, 2)

    # Starting point: (x, y) = (0, 0)
    param0 = ValuationPolydisc([K(0), K(0)], [0, 0])

    @testset "Greedy Descent" begin
        optim = greedy_descent_init(param0, loss, 1, (false, 1))
        initial_loss = eval_loss(optim)

        for i in 1:20
            step!(optim)
        end

        final_loss = eval_loss(optim)
        @test final_loss <= initial_loss
        println("Greedy: loss $initial_loss → $final_loss after 20 steps")
    end

    @testset "MCTS" begin
        config = MCTSConfig(num_simulations=50, degree=1)
        optim = mcts_descent_init(param0, loss, config)
        initial_loss = eval_loss(optim)

        for i in 1:20
            step!(optim)
        end

        final_loss = eval_loss(optim)
        @test final_loss <= initial_loss
        println("MCTS:   loss $initial_loss → $final_loss after 20 steps")
    end
end
