# DAG-MCTS Benchmark: Linear Polynomial Optimization
#
# This script benchmarks DAG-MCTS against other optimizers on the problem of
# minimizing a sum of three linear polynomials in two variables over p-adic space.
#
# Problem: minimize |a₁x + b₁y + c₁| + |a₂x + b₂y + c₂| + |a₃x + b₃y + c₃|
#
# Run with: julia --project=. experiments/dag_mcts_benchmark.jl

using Printf
using Random
using Oscar

# Load NAML module
include("../src/NAML.jl")
using .NAML

##################################################
# Problem Setup
##################################################

"""
    setup_linear_polynomial_problem(K, coefficients)

Create a loss function for minimizing sum of absolute values of linear polynomials.

# Arguments
- `K`: p-adic field
- `coefficients`: Vector of tuples [(a₁, b₁, c₁), (a₂, b₂, c₂), ...] for polynomials aᵢx + bᵢy + cᵢ

# Returns
- `loss`: Loss struct for optimization
- `poly`: The polynomial function (for reference)
"""
function setup_linear_polynomial_problem(K, coefficients)
    # Create polynomial ring in two variables
    R, (x, y) = polynomial_ring(K, ["x", "y"])

    # Build linear polynomials
    linear_polys = [LinearPolynomial([K(a), K(b)], K(c)) for (a, b, c) in coefficients]

    # Sum of absolute values: |p₁| + |p₂| + |p₃|
    poly = LinearAbsolutePolynomialSum(linear_polys)

    # Create batch evaluator
    batch_eval = batch_evaluate_init(poly)

    function loss_eval(params::Vector)
        return [batch_eval(p) for p in params]
    end

    function loss_grad(vs::Vector)
        return [directional_derivative(poly, v) for v in vs]
    end

    loss = Loss(loss_eval, loss_grad)

    return loss, poly
end

##################################################
# Benchmark Runner
##################################################

"""
    run_optimizer(name, optim, num_steps; verbose=false)

Run an optimizer for a fixed number of steps and collect metrics.

# Returns
NamedTuple with: losses, final_loss, elapsed_time
"""
function run_optimizer(name::String, optim, num_steps::Int; verbose::Bool=false)
    losses = Float64[]

    # Record initial loss
    push!(losses, eval_loss(optim))

    start_time = time()

    for i in 1:num_steps
        step!(optim)
        current_loss = eval_loss(optim)
        push!(losses, current_loss)

        if verbose && i % 10 == 0
            @printf("  %s step %3d: loss = %.6e\n", name, i, current_loss)
        end
    end

    elapsed = time() - start_time

    return (
        name = name,
        losses = losses,
        final_loss = losses[end],
        elapsed_time = elapsed,
        final_param = optim.param
    )
end

##################################################
# Main Benchmark
##################################################

function main()
    println("="^70)
    println("DAG-MCTS Benchmark: Linear Polynomial Optimization")
    println("="^70)

    # Setup
    Random.seed!(42)
    prec = 30
    p = 2  # Use 2-adic field
    K = PadicField(p, prec)

    # Problem: minimize |2x + 3y - 5| + |x - 2y + 1| + |3x + y - 4|
    # This has a unique minimum in the p-adic metric
    coefficients = [
        (2, 3, -5),   # 2x + 3y - 5
        (1, -2, 1),   # x - 2y + 1
        (3, 1, -4),   # 3x + y - 4
    ]

    println("\nProblem: minimize |2x + 3y - 5| + |x - 2y + 1| + |3x + y - 4|")
    println("Field: $(p)-adic numbers with precision $(prec)")

    # Create loss function
    loss, poly = setup_linear_polynomial_problem(K, coefficients)

    # Initial parameter (start at origin with large radius)
    initial_param = ValuationPolydisc([K(0), K(0)], [0, 0])
    initial_loss = loss.eval([initial_param])[1]

    println("\nInitial point: center = (0, 0), radius = (0, 0)")
    @printf("Initial loss: %.6e\n", initial_loss)

    # Benchmark parameters
    num_steps = 30
    num_simulations = 50  # For tree search methods

    println("\n" * "-"^70)
    println("Running optimizers for $num_steps steps...")
    println("-"^70)

    results = []

    # 1. Greedy Descent
    println("\n[1/5] Greedy Descent")
    optim_greedy = greedy_descent_init(
        initial_param,
        loss,
        1,  # next_branch
        (false, 1)  # (strict=false, degree=1)
    )
    push!(results, run_optimizer("Greedy", optim_greedy, num_steps, verbose=true))

    # Note: Gradient Descent skipped due to compatibility issue with tuple-based polydiscs

    # 2. Standard MCTS
    println("\n[2/5] Standard MCTS")
    mcts_config = MCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        selection_mode=VisitCount
    )
    optim_mcts = mcts_descent_init(initial_param, loss, mcts_config)
    push!(results, run_optimizer("MCTS", optim_mcts, num_steps, verbose=true))

    # 3. DAG-MCTS (no persistence)
    println("\n[3/5] DAG-MCTS (no persistence)")
    dag_config_no_persist = DAGMCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        persist_table=false
    )
    optim_dag_no_persist = dag_mcts_descent_init(initial_param, loss, dag_config_no_persist)
    push!(results, run_optimizer("DAG-MCTS", optim_dag_no_persist, num_steps, verbose=true))

    # 4. DAG-MCTS (with persistence)
    println("\n[4/5] DAG-MCTS (persistent table)")
    dag_config_persist = DAGMCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        persist_table=true
    )
    optim_dag_persist = dag_mcts_descent_init(initial_param, loss, dag_config_persist)
    result_dag_persist = run_optimizer("DAG-MCTS-P", optim_dag_persist, num_steps, verbose=true)
    push!(results, result_dag_persist)

    # Print DAG stats for persistent version
    dag_stats = get_dag_stats(optim_dag_persist.state)
    println("  DAG stats: $(dag_stats.unique_nodes) unique nodes, $(dag_stats.multi_parent_nodes) with multiple parents")

    # 5. UCT
    println("\n[5/5] UCT")
    uct_config = UCTConfig(
        max_depth=15,
        num_simulations=num_simulations,
        exploration_constant=sqrt(2.0),
        degree=1
    )
    optim_uct = uct_descent_init(initial_param, loss, uct_config)
    push!(results, run_optimizer("UCT", optim_uct, num_steps, verbose=true))

    ##################################################
    # Results Summary
    ##################################################

    println("\n" * "="^70)
    println("RESULTS SUMMARY")
    println("="^70)

    # Sort by final loss
    sorted_results = sort(results, by=r -> r.final_loss)

    println("\nFinal Rankings (by loss):")
    println("-"^70)
    @printf("%-15s %15s %12s %10s\n", "Optimizer", "Final Loss", "Time (s)", "Improvement")
    println("-"^70)

    for (i, r) in enumerate(sorted_results)
        improvement = (initial_loss - r.final_loss) / initial_loss * 100
        @printf("%-15s %15.6e %12.3f %9.1f%%\n",
                r.name, r.final_loss, r.elapsed_time, improvement)
    end

    println("-"^70)

    # Convergence comparison
    println("\nConvergence Progress (loss at each step):")
    println("-"^70)

    # Print header
    print(@sprintf("%-6s", "Step"))
    for r in results
        print(@sprintf(" %12s", r.name))
    end
    println()
    println("-"^70)

    # Print losses at key steps
    key_steps = [0, 5, 10, 15, 20, 25, 30]
    for step in key_steps
        if step <= num_steps
            print(@sprintf("%-6d", step))
            for r in results
                print(@sprintf(" %12.4e", r.losses[step + 1]))
            end
            println()
        end
    end

    ##################################################
    # DAG-specific Analysis
    ##################################################

    println("\n" * "="^70)
    println("DAG-MCTS ANALYSIS")
    println("="^70)

    # Re-run DAG-MCTS with detailed tracking
    println("\nDetailed DAG-MCTS run with transposition tracking:")

    dag_config_detailed = DAGMCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        persist_table=true
    )
    optim_dag_detailed = dag_mcts_descent_init(initial_param, loss, dag_config_detailed)

    println("-"^50)
    @printf("%-6s %12s %12s %12s\n", "Step", "Loss", "Nodes", "Multi-Parent")
    println("-"^50)

    for i in 0:num_steps
        current_loss = eval_loss(optim_dag_detailed)
        stats = get_dag_stats(optim_dag_detailed.state)

        if i % 5 == 0 || i == num_steps
            @printf("%-6d %12.4e %12d %12d\n",
                    i, current_loss, stats.unique_nodes, stats.multi_parent_nodes)
        end

        if i < num_steps
            step!(optim_dag_detailed)
        end
    end

    println("-"^50)

    # Final verification
    println("\nTransposition table integrity: ",
            verify_transposition_table(optim_dag_detailed.state) ? "PASSED" : "FAILED")

    ##################################################
    # Best Solution Found
    ##################################################

    println("\n" * "="^70)
    println("BEST SOLUTION")
    println("="^70)

    best = sorted_results[1]
    println("\nBest optimizer: $(best.name)")
    @printf("Final loss: %.6e\n", best.final_loss)
    println("Final parameter:")
    println("  Center: ($(best.final_param.center[1]), $(best.final_param.center[2]))")
    println("  Radius: $(best.final_param.radius)")

    println("\n" * "="^70)
    println("Benchmark complete!")
    println("="^70)

    return results
end

# Run the benchmark
results = main()
