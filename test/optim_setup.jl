using Test
using Oscar
using NonArchimedeanMachineLearning

function make_param(K, center::Int, radius::Int)
    return ValuationPolydisc([K(center)], [radius])
end

@testset "Optim Setup" begin
    K = PadicField(2, 20)
    param0 = make_param(K, 0, 0)
    param1 = make_param(K, 1, 1)
    param2 = make_param(K, 2, 2)

    tangent = ValuationTangent(param0, param0, [1])

    @testset "Loss addition composes eval and grad" begin
        loss_a = Loss(
            xs -> [length(xs) + 1.0],
            xs -> [10.0],
        )
        loss_b = Loss(
            xs -> [length(xs) + 2.5],
            xs -> [-3.0],
        )

        combined = loss_a + loss_b

        @test combined.eval([param0]) == [4.5]
        @test combined.grad([tangent]) == [7.0]
    end

    @testset "update_param! and update_state! mutate setup" begin
        loss = Loss(
            xs -> [42.0],
            xs -> [0.0],
        )
        optim = OptimSetup(
            loss,
            param0,
            (l, p, st, ctx) -> (p, st, false),
            0,
            :ctx,
            false,
        )

        update_param!(optim, param1)
        @test optim.param == param1

        update_state!(optim, 99)
        @test optim.state == 99
    end

    @testset "step! updates param state and convergence flag" begin
        observed = Ref{Any}(nothing)
        loss = Loss(
            xs -> [11.0],
            xs -> [5.0],
        )

        optimiser = function (loss_arg, param_arg, state_arg, context_arg)
            observed[] = (loss_arg, param_arg, state_arg, context_arg)
            return param1, state_arg + 1, true
        end

        optim = OptimSetup(loss, param0, optimiser, 0, :token, false)
        converged = step!(optim)

        @test converged
        @test observed[][1] === loss
        @test observed[][2] == param0
        @test observed[][3] == 0
        @test observed[][4] == :token
        @test optim.param == param1
        @test optim.state == 1
        @test has_converged(optim)
    end

    @testset "eval_loss reads current parameter through loss closure" begin
        loss = Loss(
            xs -> [Float64(radius(xs[1])[1])],
            xs -> [0.0],
        )
        optim = OptimSetup(loss, param2, (l, p, st, ctx) -> (p, st, false), nothing, nothing, false)
        @test eval_loss(optim) == 2.0
    end

    @testset "optimize! returns early on convergence" begin
        loss = Loss(
            xs -> [Float64(radius(xs[1])[1])],
            xs -> [0.0],
        )

        params = [param0, param1, param2]
        optimiser = function (loss_arg, param_arg, state_arg, context_arg)
            next_state = state_arg + 1
            return params[next_state + 1], next_state, next_state >= 2
        end

        optim = OptimSetup(loss, param0, optimiser, 0, nothing, false)
        steps_taken = optimize!(optim, 10)

        @test steps_taken == 2
        @test has_converged(optim)
        @test optim.param == param2
        @test optim.state == 2
    end

    @testset "optimize! returns max_steps when not converged" begin
        loss = Loss(
            xs -> [1.0],
            xs -> [0.0],
        )
        optimiser = (loss_arg, param_arg, state_arg, context_arg) -> (param_arg, state_arg + 1, false)
        optim = OptimSetup(loss, param0, optimiser, 0, nothing, false)

        steps_taken = optimize!(optim, 3)
        @test steps_taken == 3
        @test !has_converged(optim)
        @test optim.state == 3
        @test optim.param == param0
    end

    @testset "optimize! verbose output includes convergence marker" begin
        loss = Loss(
            xs -> [Float64(radius(xs[1])[1])],
            xs -> [0.0],
        )
        params = [param0, param1]
        optimiser = function (loss_arg, param_arg, state_arg, context_arg)
            next_state = state_arg + 1
            return params[next_state + 1], next_state, true
        end

        optim = OptimSetup(loss, param0, optimiser, 0, nothing, false)
        output = sprint() do io
            redirect_stdout(io) do
                optimize!(optim, 5; verbose=true)
            end
        end

        @test occursin("Step 1: loss =", output)
        @test occursin("[converged]", output)
    end
end
