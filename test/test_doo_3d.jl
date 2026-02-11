"""
DOO 3D Test: Compare DOO with MCTS on 3D polynomial optimization
"""

using Test
using Printf
include("../src/NAML.jl")
using .NAML
using Oscar

@testset "DOO 3D Experiments" begin
    println("\n" * "="^80)
    println("3D Polynomial Optimization: DOO vs MCTS")
    println("="^80)

    # Setup: 2-adic field
    prec = 30
    p = 2
    K = PadicField(p, prec)

    # Problem: minimize |x1 + x2 - 2| + |x2 + x3 - 3| + |x1 - x3|
    # This has structure that requires exploring all 3 dimensions
    println("\nProblem: minimize |x1 + x2 - 2| + |x2 + x3 - 3| + |x1 - x3|")
    println("Field: Q_$p with precision $prec")

    linear_polys = [
        LinearPolynomial([K(1), K(1), K(0)], K(-2)),   # x1 + x2 - 2
        LinearPolynomial([K(0), K(1), K(1)], K(-3)),   # x2 + x3 - 3
        LinearPolynomial([K(1), K(0), K(-1)], K(0)),   # x1 - x3
    ]

    poly = LinearAbsolutePolynomialSum(linear_polys)
    batch_eval = batch_evaluate_init(poly)

    function loss_eval(params::Vector)
        return [batch_eval(p) for p in params]
    end
    function loss_grad(vs::Vector)
        return [directional_derivative(poly, v) for v in vs]
    end
    loss = Loss(loss_eval, loss_grad)

    # Initial parameter
    initial_center = [K(0), K(0), K(0)]
    initial_radius = [2, 2, 2]
    initial_param = ValuationPolydisc(initial_center, initial_radius)
    initial_loss = loss.eval([initial_param])[1]

    println("Initial point: (0, 0, 0) with radius (2, 2, 2)")
    @printf("Initial loss: %.6e\n\n", initial_loss)

    # Check that there's variation in children
    println("Verifying problem has structure...")
    child_losses = [batch_eval(ch) for ch in children(initial_param, 1)]
    unique_losses = unique(child_losses)
    println("Children with degree=1: $(length(child_losses)), unique losses: $(length(unique_losses))")

    if length(unique_losses) > 1
        @printf("Loss range: %.6e to %.6e\n", minimum(child_losses), maximum(child_losses))
        println("✓ Problem has variation - good for optimization")
    end
    println()

    num_steps = 100

    # @testset "DOO (degree=3, standard)" begin
    #     # DOO with degree=3 matches paper: p^n = 2^3 = 8 children per node
    #     println("-"^80)
    #     println("DOO with degree=3 (matches paper)")
    #     println("-"^80)

    #     delta = h -> Float64(p)^(-h)
    #     config = DOOConfig(delta=delta, max_depth=7, degree=3)
    #     optim = doo_descent_init(initial_param, loss, 1, config)

    #     init_loss = eval_loss(optim)

    #     for i in 1:num_steps
    #         step!(optim)
    #     end

    #     final_loss = eval_loss(optim)

    #     best_node = get_best_node(optim.state)
    #     best_radius = best_node.polydisc.radius

    #     @printf("Initial loss:     %.6e\n", init_loss)
    #     @printf("Final loss:       %.6e\n", final_loss)
    #     @printf("Improvement:      %.2f%%\n", (init_loss - final_loss) / init_loss * 100)
    #     @printf("Total samples:    %d\n", optim.state.total_samples)
    #     @printf("Tree size:        %d\n", get_tree_size(optim.state))
    #     @printf("Best node depth:  %d\n", best_node.depth)
    #     @printf("Best node radius: (%d, %d, %d)\n", best_radius[1], best_radius[2], best_radius[3])
    #     @printf("Children/node:    %d (= %d^%d)\n\n", p^3, p, 3)

    #     @test final_loss < init_loss
    #     @test optim.state.total_samples > 0
    # end

    # @testset "DOO (degree=1, efficient)" begin
    #     # DOO with degree=1: C(3,1) * 2 = 6 children per node
    #     println("-"^80)
    #     println("DOO with degree=1 (efficient variant)")
    #     println("-"^80)

    #     delta = h -> Float64(p)^(-h)
    #     config = DOOConfig(delta=delta, max_depth=7, degree=1)
    #     optim = doo_descent_init(initial_param, loss, 1, config)

    #     init_loss = eval_loss(optim)

    #     for i in 1:num_steps
    #         step!(optim)
    #     end

    #     final_loss = eval_loss(optim)

    #     best_node = get_best_node(optim.state)
    #     best_radius = best_node.polydisc.radius

    #     @printf("Initial loss:     %.6e\n", init_loss)
    #     @printf("Final loss:       %.6e\n", final_loss)
    #     @printf("Improvement:      %.2f%%\n", (init_loss - final_loss) / init_loss * 100)
    #     @printf("Total samples:    %d\n", optim.state.total_samples)
    #     @printf("Tree size:        %d\n", get_tree_size(optim.state))
    #     @printf("Best node depth:  %d\n", best_node.depth)
    #     @printf("Best node radius: (%d, %d, %d)\n", best_radius[1], best_radius[2], best_radius[3])
    #     @printf("Children/node:    %d (= C(3,1) * %d)\n\n", 3*p, p)

    #     # Note: degree=1 may not improve if all children have same loss at initial radius
    #     @test final_loss <= init_loss  # Allow no improvement for degree=1
    #     @test optim.state.total_samples > 0
    # end

    @testset "MCTS (100 sims, degree=2)" begin
        # MCTS with degree=2: C(3,2) * 2^2 = 12 children per node
        println("-"^80)
        println("MCTS with 100 simulations, degree=2")
        println("-"^80)

        config = MCTSConfig(num_simulations=100, degree=2, strict=true)
        optim = mcts_descent_init(initial_param, loss, config)

        init_loss = eval_loss(optim)

        for i in 1:num_steps
            step!(optim)
        end

        final_loss = eval_loss(optim)

        tree_sz = get_tree_size(optim.state.root)  # MCTS uses get_tree_size(node)
        # Extract radius from the optimizer's current parameter
        final_radius = optim.param.radius

        @printf("Initial loss:     %.6e\n", init_loss)
        @printf("Final loss:       %.6e\n", final_loss)
        @printf("Improvement:      %.2f%%\n", (init_loss - final_loss) / init_loss * 100)
        @printf("Tree size:        %d\n", tree_sz)
        @printf("Final radius:     (%d, %d, %d)\n", final_radius[1], final_radius[2], final_radius[3])
        @printf("Children/node:    %d (= C(3,2) * %d^2)\n\n", 3*p^2, p)

        @test final_loss < init_loss
        @test tree_sz > 0
    end

    # @testset "Performance Comparison" begin
    #     println("="^80)
    #     println("PERFORMANCE COMPARISON ($num_steps steps)")
    #     println("="^80)

    #     configs = [
    #         ("DOO (deg=3, 8 children/node)",  DOOConfig(delta=h->Float64(p)^(-h), max_depth=7, degree=3)),
    #         ("DOO (deg=1, 6 children/node)",  DOOConfig(delta=h->Float64(p)^(-h), max_depth=7, degree=1)),
    #         ("MCTS (100 sims, 12 children/node)", MCTSConfig(num_simulations=100, degree=2))
    #     ]

    #     results = []

    #     for (name, config) in configs
    #         if config isa DOOConfig
    #             optim = doo_descent_init(initial_param, loss, 1, config)
    #         else
    #             optim = mcts_descent_init(initial_param, loss, config)
    #         end

    #         init = eval_loss(optim)
    #         for i in 1:num_steps
    #             step!(optim)
    #         end
    #         final = eval_loss(optim)

    #         # get_tree_size works differently for DOO vs MCTS
    #         if config isa DOOConfig
    #             tree_size = get_tree_size(optim.state)
    #             samples = optim.state.total_samples
    #         else  # MCTS
    #             tree_size = get_tree_size(optim.state.root)
    #             samples = tree_size  # MCTS doesn't track total_samples separately
    #         end
    #         improvement = (init - final) / init * 100

    #         push!(results, (name, init, final, samples, tree_size, improvement))
    #     end

    #     # Print table
    #     println(Printf.@sprintf("\n%-35s | %12s | %12s | %8s | %9s | %10s",
    #         "Algorithm", "Init Loss", "Final Loss", "Samples", "Tree Size", "Improv."))
    #     println("-"^95)

    #     for (name, init, final, samples, tree_size, improvement) in results
    #         println(Printf.@sprintf("%-35s | %12.2e | %12.2e | %8d | %9d | %9.1f%%",
    #             name, init, final, samples, tree_size, improvement))
    #     end

    #     println("="^80)
    #     println("\nKey Observations:")
    #     println("• DOO (degree=3): Standard p^n branching from paper, uniform exploration")
    #     println("• DOO (degree=1): More efficient, refines one coordinate at a time")
    #     println("• MCTS: Uses UCB for exploration-exploitation, more samples due to simulations")
    #     println()

    #     # All should improve (or at least not get worse)
    #     for (name, init, final, _, _, _) in results
    #         @test final <= init
    #     end
    # end
end

println("\n✓ All 3D DOO tests completed!\n")
