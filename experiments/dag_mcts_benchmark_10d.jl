# DAG-MCTS Benchmark: 10-Dimensional Linear Polynomial Optimization over Q_3
#
# This script benchmarks DAG-MCTS against other optimizers on a higher-dimensional
# problem: minimizing a sum of linear polynomials in 10 variables over the 3-adic field.
#
# In higher dimensions, transpositions become more common as there are more
# paths to reach the same polydisc state.
#
# Run with: julia --project=. experiments/dag_mcts_benchmark_10d.jl

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
    setup_linear_polynomial_problem_nd(K, n_vars, n_polys; seed=42)

Create a loss function for minimizing sum of absolute values of random linear polynomials.

# Arguments
- `K`: p-adic field
- `n_vars`: Number of variables
- `n_polys`: Number of linear polynomials in the sum
- `seed`: Random seed for reproducibility

# Returns
- `loss`: Loss struct for optimization
- `poly`: The polynomial function
- `coefficients`: The generated coefficients (for reference)
"""
function setup_linear_polynomial_problem_nd(K, n_vars::Int, n_polys::Int; seed::Int=42)
    Random.seed!(seed)

    # Generate random coefficients in {-5, ..., 5}
    coefficients = []
    for _ in 1:n_polys
        # Coefficients for variables
        coeffs = [rand(-5:5) for _ in 1:n_vars]
        # Constant term
        constant = rand(-5:5)
        push!(coefficients, (coeffs, constant))
    end

    # Build linear polynomials
    linear_polys = [
        LinearPolynomial([K(c) for c in coeffs], K(constant))
        for (coeffs, constant) in coefficients
    ]

    # Sum of absolute values
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

    return loss, poly, coefficients
end

##################################################
# Benchmark Runner
##################################################

"""
    run_optimizer(name, optim, num_steps; verbose=false, print_interval=10)

Run an optimizer for a fixed number of steps and collect metrics.
"""
function run_optimizer(name::String, optim, num_steps::Int;
                       verbose::Bool=false, print_interval::Int=10)
    losses = Float64[]
    push!(losses, eval_loss(optim))

    start_time = time()

    for i in 1:num_steps
        step!(optim)
        current_loss = eval_loss(optim)
        push!(losses, current_loss)

        if verbose && i % print_interval == 0
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
    println("DAG-MCTS Benchmark: 10D Linear Polynomial Optimization over Q_3")
    println("="^70)

    # Setup
    prec = 30
    p = 3  # Use 3-adic field
    K = PadicField(p, prec)

    n_vars = 10
    n_polys = 5  # Sum of 5 linear polynomials

    println("\nProblem Setup:")
    println("  Field: Q_$p ($(p)-adic numbers) with precision $prec")
    println("  Variables: $n_vars")
    println("  Linear polynomials: $n_polys")

    # Create loss function
    loss, poly, coefficients = setup_linear_polynomial_problem_nd(K, n_vars, n_polys, seed=42)

    # Print the polynomials
    println("\nPolynomials (|p₁| + |p₂| + ... + |p$n_polys|):")
    var_names = ["x$i" for i in 1:n_vars]
    for (i, (coeffs, constant)) in enumerate(coefficients)
        terms = []
        for (j, c) in enumerate(coeffs)
            if c != 0
                if c == 1
                    push!(terms, var_names[j])
                elseif c == -1
                    push!(terms, "-$(var_names[j])")
                else
                    push!(terms, "$(c)$(var_names[j])")
                end
            end
        end
        if constant != 0
            push!(terms, string(constant))
        end
        println("  p$i = $(join(terms, " + "))")
    end

    # Initial parameter - start at deeper radius where loss differences emerge
    # (At radius=0, all children have same loss - MCTS can't distinguish!)
    initial_center = [K(0) for _ in 1:n_vars]
    initial_radius = [5 for _ in 1:n_vars]  # Start deeper in the tree
    initial_param = ValuationPolydisc(initial_center, initial_radius)
    initial_loss = loss.eval([initial_param])[1]

    println("\nInitial point: center = (0, ..., 0), radius = (0, ..., 0)")
    @printf("Initial loss: %.6e\n", initial_loss)

    # Benchmark parameters
    num_steps = 25
    num_simulations = 200  # Reduced - starting deeper gives better signal

    println("\n" * "-"^70)
    println("Running optimizers for $num_steps steps (simulations=$num_simulations)...")
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
    push!(results, run_optimizer("Greedy", optim_greedy, num_steps, verbose=true, print_interval=5))

    # 2. Standard MCTS (VisitCount - traditional)
    println("\n[2/6] Standard MCTS (VisitCount)")
    mcts_config_vc = MCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        selection_mode=VisitCount
    )
    optim_mcts_vc = mcts_descent_init(initial_param, loss, mcts_config_vc)
    push!(results, run_optimizer("MCTS-VC", optim_mcts_vc, num_steps, verbose=true, print_interval=5))

    # 3. Standard MCTS (BestValue - greedy selection)
    println("\n[3/6] Standard MCTS (BestValue)")
    mcts_config_bv = MCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        selection_mode=BestValue
    )
    optim_mcts_bv = mcts_descent_init(initial_param, loss, mcts_config_bv)
    push!(results, run_optimizer("MCTS-BV", optim_mcts_bv, num_steps, verbose=true, print_interval=5))

    # 4. DAG-MCTS (no persistence)
    println("\n[4/6] DAG-MCTS (no persistence)")
    dag_config_no_persist = DAGMCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        persist_table=false
    )
    optim_dag_no_persist = dag_mcts_descent_init(initial_param, loss, dag_config_no_persist)
    push!(results, run_optimizer("DAG-MCTS", optim_dag_no_persist, num_steps, verbose=true, print_interval=5))

    # 5. DAG-MCTS (with persistence)
    println("\n[5/6] DAG-MCTS (persistent table)")
    dag_config_persist = DAGMCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        persist_table=true
    )
    optim_dag_persist = dag_mcts_descent_init(initial_param, loss, dag_config_persist)
    result_dag_persist = run_optimizer("DAG-MCTS-P", optim_dag_persist, num_steps, verbose=true, print_interval=5)
    push!(results, result_dag_persist)

    # Print DAG stats for persistent version
    dag_stats = get_dag_stats(optim_dag_persist.state)
    println("  DAG stats: $(dag_stats.unique_nodes) unique nodes, $(dag_stats.multi_parent_nodes) with multiple parents")
    if dag_stats.unique_nodes > 0
        pct = round(dag_stats.multi_parent_nodes / dag_stats.unique_nodes * 100, digits=1)
        println("  Transposition rate: $pct%")
    end

    # 6. UCT
    println("\n[6/6] UCT")
    uct_config = UCTConfig(
        max_depth=15,
        num_simulations=num_simulations,
        exploration_constant=sqrt(2.0),
        degree=1
    )
    optim_uct = uct_descent_init(initial_param, loss, uct_config)
    push!(results, run_optimizer("UCT", optim_uct, num_steps, verbose=true, print_interval=5))

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
    key_steps = [0, 5, 10, 15, 20, 25]
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
    println("DAG-MCTS ANALYSIS (10D over Q_3)")
    println("="^70)

    println("\nIn higher dimensions, more transpositions are expected because:")
    println("  - There are $n_vars coordinates that can be refined")
    println("  - Each coordinate has $p children (branching factor)")
    println("  - Many paths can lead to the same refined polydisc")

    # Re-run DAG-MCTS with detailed tracking
    println("\nDetailed DAG-MCTS run with transposition tracking:")

    dag_config_detailed = DAGMCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=1.41,
        degree=1,
        persist_table=true
    )
    optim_dag_detailed = dag_mcts_descent_init(initial_param, loss, dag_config_detailed)

    println("-"^60)
    @printf("%-6s %12s %12s %12s %10s\n", "Step", "Loss", "Nodes", "Multi-Parent", "Trans%")
    println("-"^60)

    for i in 0:num_steps
        current_loss = eval_loss(optim_dag_detailed)
        stats = get_dag_stats(optim_dag_detailed.state)
        trans_pct = stats.unique_nodes > 0 ?
            round(stats.multi_parent_nodes / stats.unique_nodes * 100, digits=1) : 0.0

        if i % 5 == 0 || i == num_steps
            @printf("%-6d %12.4e %12d %12d %9.1f%%\n",
                    i, current_loss, stats.unique_nodes, stats.multi_parent_nodes, trans_pct)
        end

        if i < num_steps
            step!(optim_dag_detailed)
        end
    end

    println("-"^60)

    ##################################################
    # Memory/Efficiency Analysis
    ##################################################

    println("\n" * "="^70)
    println("MEMORY & EFFICIENCY ANALYSIS")
    println("="^70)

    # Compare node counts between standard MCTS tree and DAG
    final_dag_stats = get_dag_stats(optim_dag_detailed.state)

    # In a standard tree, each path creates new nodes
    # Estimate: num_simulations * num_steps * avg_path_length
    # This is a rough upper bound
    estimated_tree_nodes = num_simulations * num_steps * 5  # rough estimate

    println("\nNode comparison (persistent DAG-MCTS):")
    println("  Unique nodes in DAG: $(final_dag_stats.unique_nodes)")
    println("  Nodes with multiple parents: $(final_dag_stats.multi_parent_nodes)")
    println("  (In standard MCTS, these would be separate nodes)")

    if final_dag_stats.multi_parent_nodes > 0
        savings = final_dag_stats.multi_parent_nodes
        println("\n  Memory savings from transposition detection:")
        println("    At least $savings duplicate nodes avoided")
    end

    ##################################################
    # Best Solution Found
    ##################################################

    println("\n" * "="^70)
    println("BEST SOLUTION")
    println("="^70)

    best = sorted_results[1]
    println("\nBest optimizer: $(best.name)")
    @printf("Final loss: %.6e\n", best.final_loss)
    @printf("Improvement: %.1f%%\n", (initial_loss - best.final_loss) / initial_loss * 100)

    println("\nFinal parameter (first 3 coordinates shown):")
    for i in 1:min(3, n_vars)
        println("  x$i: center=$(best.final_param.center[i]), radius=$(best.final_param.radius[i])")
    end
    if n_vars > 3
        println("  ...")
    end

    println("\n" * "="^70)
    println("Benchmark complete!")
    println("="^70)

    return results
end

# Run the benchmark
results = main()
