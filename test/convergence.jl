# Test file for convergence detection API.
#
# Tests has_converged, optimize!, and convergence behavior across optimizers.

using Test
using Oscar
using NAML

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
        optim = greedy_descent_init(param0, loss, 1, (false, 1))
        @test has_converged(optim) == false

        # Manually set converged flag
        optim.converged = true
        @test has_converged(optim) == true
    end

    @testset "step! returns converged flag" begin
        optim = greedy_descent_init(param0, loss, 1, (false, 1))
        # First step should not converge (precision 3, starting radius 0)
        converged = step!(optim)
        @test converged isa Bool
    end

    @testset "Greedy descent convergence at precision boundary" begin
        optim = greedy_descent_init(param0, loss, 1, (false, 1))
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
        optim = greedy_descent_init(param0, loss, 1, (false, 1))
        steps = optimize!(optim, 100)
        @test has_converged(optim) == true
        @test steps < 100
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

        optim = greedy_descent_init(param_high, loss_high, 1, (false, 1))
        steps = optimize!(optim, 3)
        @test steps == 3
        @test has_converged(optim) == false
    end

    @testset "random_descent works and converges" begin
        optim = random_descent_init(param0, loss, 1, (false, 1))
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

        optim = gradient_descent_init(param_gd, loss_gd, 0, 1)
        steps = optimize!(optim, 100)
        @test has_converged(optim) == true
        @test steps <= 4
    end
end
