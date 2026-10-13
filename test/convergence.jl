# Test file for convergence detection API.
#
# Tests has_converged, optimize!, and convergence behavior across optimizers.

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Convergence Detection" begin
    # Use low precision so convergence happens quickly
    p, prec = 2, 3
    K = PadicField(p, prec)

    # Simple 1D quadratic: minimize |x - a|^2
    R, (x, a) = K["x", "a"]
    g = AbsolutePolynomialSum([(x - a)])
    model = AbstractModel(g, [true, false])

    # Single data point: x=1, target=0
    data = [(ValuationPolydisc([K(1)], [prec]), 0)]
    loss = MSE_loss_init(model, data)

    # Initial param with radius 0 (will refine down to precision)
    param0 = ValuationPolydisc([K(0)], [0])

    @testset "has_converged accessor" begin
        optim = greedy_descent_init(param0, loss, GreedyDescentConfig())
        @test has_converged(optim) == false

        # Manually set converged flag
        optim.converged = true
        @test has_converged(optim) == true
    end

    @testset "Greedy descent convergence at precision boundary" begin
        optim = greedy_descent_init(param0, loss, GreedyDescentConfig())
        converged = false
        steps = 0
        for i in 1:100
            converged = step!(optim)
            steps = i
            if converged
                break
            end
        end
        # With precision 3 and starting radius 0, takes prec steps to reach
        # radius=prec, then 1 more step to detect empty children
        @test converged == true
        @test has_converged(optim) == true
        @test steps <= prec + 1
    end

    @testset "optimize! returns early on convergence" begin
        optim = greedy_descent_init(param0, loss, GreedyDescentConfig())
        steps = optimize!(optim, 100)
        @test has_converged(optim) == true
        @test steps <= prec + 1
    end

    @testset "optimize! returns max_steps when not converged" begin
        # Use high precision so convergence doesn't happen in 3 steps
        K_high = PadicField(2, 50)
        R_high, (xh, ah) = K_high["x", "a"]
        g_high = AbsolutePolynomialSum([(xh - ah)])
        model_high = AbstractModel(g_high, [true, false])
        data_high = [(ValuationPolydisc([K_high(1)], [50]), 0)]
        loss_high = MSE_loss_init(model_high, data_high)
        param_high = ValuationPolydisc([K_high(0)], [0])

        optim = greedy_descent_init(param_high, loss_high, GreedyDescentConfig())
        steps = optimize!(optim, 3)
        @test steps == 3
        @test has_converged(optim) == false
    end

    @testset "random_descent works and converges" begin
        optim = random_descent_init(param0, loss, RandomDescentConfig())
        @test has_converged(optim) == false

        # Should not crash
        steps = optimize!(optim, 100)
        @test has_converged(optim) == true
        @test steps <= prec + 1
    end

    @testset "gradient_descent convergence" begin
        # Self-contained setup to avoid scope interference
        K_gd = PadicField(2, 3)
        R_gd, (x_gd, a_gd) = K_gd["x", "a"]
        g_gd = AbsolutePolynomialSum([(x_gd - a_gd)])
        model_gd = AbstractModel(g_gd, [true, false])
        data_gd = [(ValuationPolydisc([K_gd(1)], [3]), 0)]
        loss_gd = MSE_loss_init(model_gd, data_gd)
        param_gd = ValuationPolydisc([K_gd(0)], [0])

        optim = gradient_descent_init(param_gd, loss_gd, GradientDescentConfig())
        steps = optimize!(optim, 100)
        @test has_converged(optim) == true
        @test steps <= 4
    end
end

@testset "Optimizer Configs" begin
    K = PadicField(2, 5)
    param2 = ValuationPolydisc{PadicFieldElem, Int, 2}((K(0), K(0)), (0, 0))
    flat_loss = Loss(ps -> zeros(length(ps)), ts -> zeros(length(ts)))

    @testset "$Config construction" for Config in (
        GreedyDescentConfig, GradientDescentConfig, RandomDescentConfig)
        config = Config()
        @test config isa AbstractOptimConfig
        @test config.strict == false
        @test config.degree == 1
        @test config.start_branch == 1

        config = Config(strict = true, degree = 2, start_branch = 2)
        @test (config.strict, config.degree, config.start_branch) == (true, 2, 2)

        @test_throws Exception Config(degree = 0)
        @test_throws Exception Config(start_branch = 0)
    end

    @testset "Tree-search configs share the supertype" begin
        @test DOOConfig(delta = h -> 2.0^(-h)) isa AbstractOptimConfig
        @test MCTSConfig() isa AbstractOptimConfig
        @test DAGMCTSConfig() isa AbstractOptimConfig
        @test DOOConfig(delta = h -> 2.0^(-h)).start_branch == 1
        @test MCTSConfig().start_branch == 1
        @test_throws Exception DOOConfig(delta = h -> 2.0^(-h), start_branch = 0)
        @test_throws Exception MCTSConfig(start_branch = 0)
    end

    @testset "Strict mode starts at start_branch: $init" for (init, Config) in (
        (greedy_descent_init, GreedyDescentConfig),
        (gradient_descent_init, GradientDescentConfig),
        (random_descent_init, RandomDescentConfig))
        optim = init(param2, flat_loss, Config(strict = true, start_branch = 2))
        @test optim.state == 2
        step!(optim)
        @test optim.param.radius == (0, 1)
        @test optim.state == 1
        step!(optim)
        @test optim.param.radius == (1, 1)

        @test_throws Exception init(param2, flat_loss,
            Config(strict = true, start_branch = 3))
    end

    @testset "MCTS state starts at start_branch" begin
        optim = mcts_descent_init(param2, flat_loss,
            MCTSConfig(strict = true, start_branch = 2))
        @test optim.state.next_branch == 2
    end
end
