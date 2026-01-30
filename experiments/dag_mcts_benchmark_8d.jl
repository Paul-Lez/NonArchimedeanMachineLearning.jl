# DAG-MCTS Benchmark: 8D Linear Polynomial Optimization over Q_3
#
# 8 variables, BestValue selection, 10K simulations
#
# Run with: julia --project=. experiments/dag_mcts_benchmark_8d.jl

using Printf
using Random
using Oscar

include("../src/NAML.jl")
using .NAML

function main()
    println("="^70)
    println("DAG-MCTS Benchmark: 8D Linear Polynomial Optimization over Q_3")
    println("="^70)

    prec = 30
    p = 3
    K = PadicField(p, prec)

    n_vars = 8
    n_polys = 4

    # Generate random linear polynomials
    Random.seed!(123)
    coefficients = []
    for _ in 1:n_polys
        coeffs = [rand(-3:3) for _ in 1:n_vars]
        constant = rand(-3:3)
        push!(coefficients, (coeffs, constant))
    end

    linear_polys = [
        LinearPolynomial([K(c) for c in coeffs], K(constant))
        for (coeffs, constant) in coefficients
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

    println("\nProblem Setup:")
    println("  Field: Q_$p with precision $prec")
    println("  Variables: $n_vars")
    println("  Polynomials: $n_polys")

    # Print polynomials
    println("\nPolynomials:")
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
        if isempty(terms)
            push!(terms, "0")
        end
        println("  p$i = $(join(terms, " + "))")
    end

    # Start at deeper radius for better signal
    # Need radius deep enough that loss varies among children
    initial_center = [K(0) for _ in 1:n_vars]
    initial_radius = [6 for _ in 1:n_vars]  # Deeper starting point
    initial_param = ValuationPolydisc(initial_center, initial_radius)
    initial_loss = loss.eval([initial_param])[1]

    println("\nInitial radius: (3, 3, ..., 3)")
    @printf("Initial loss: %.6e\n", initial_loss)

    # Check for variation
    println("\nChecking loss variation among children...")
    child_losses = [batch_eval(ch) for ch in children(initial_param, 1)]
    unique_losses = unique(child_losses)
    println("Children: $(length(child_losses)), Unique losses: $(length(unique_losses))")
    if length(unique_losses) > 1
        println("Loss range: $(minimum(child_losses)) to $(maximum(child_losses))")
    end

    # Benchmark parameters
    num_steps = 15
    num_simulations = 5000  # Reduced for tractability

    println("\n" * "-"^70)
    println("Running optimizers for $num_steps steps (simulations=$num_simulations)...")
    println("This may take a few minutes...")
    println("-"^70)

    results = []

    # Greedy
    println("\n[1/5] Greedy Descent")
    optim_greedy = greedy_descent_init(initial_param, loss, 1, (false, 1))
    losses_greedy = [eval_loss(optim_greedy)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_greedy)
        push!(losses_greedy, eval_loss(optim_greedy))
        if i % 5 == 0
            @printf("  Greedy step %2d: loss = %.6e (%.1fs)\n", i, losses_greedy[end], time()-t0)
        end
    end
    push!(results, (name="Greedy", losses=losses_greedy, time=time()-t0))

    # MCTS BestValue
    println("\n[2/5] MCTS (BestValue)")
    mcts_config = MCTSConfig(num_simulations=num_simulations, exploration_constant=1.41,
                             degree=1, selection_mode=BestValue)
    optim_mcts = mcts_descent_init(initial_param, loss, mcts_config)
    losses_mcts = [eval_loss(optim_mcts)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_mcts)
        push!(losses_mcts, eval_loss(optim_mcts))
        if i % 5 == 0
            @printf("  MCTS-BV step %2d: loss = %.6e (%.1fs)\n", i, losses_mcts[end], time()-t0)
        end
    end
    push!(results, (name="MCTS-BV", losses=losses_mcts, time=time()-t0))

    # DAG-MCTS BestValue
    println("\n[3/5] DAG-MCTS (BestValue, persistent)")
    dag_config = DAGMCTSConfig(num_simulations=num_simulations, exploration_constant=1.41,
                               degree=1, persist_table=true, selection_mode=BestValue)
    optim_dag = dag_mcts_descent_init(initial_param, loss, dag_config)
    losses_dag = [eval_loss(optim_dag)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_dag)
        push!(losses_dag, eval_loss(optim_dag))
        if i % 5 == 0
            stats = get_dag_stats(optim_dag.state)
            @printf("  DAG-BV step %2d: loss = %.6e, nodes=%d, trans=%.1f%% (%.1fs)\n",
                    i, losses_dag[end], stats.unique_nodes,
                    stats.multi_parent_nodes/stats.unique_nodes*100, time()-t0)
        end
    end
    dag_time = time() - t0
    push!(results, (name="DAG-BV", losses=losses_dag, time=dag_time))

    dag_stats = get_dag_stats(optim_dag.state)
    println("  Final DAG: $(dag_stats.unique_nodes) nodes, $(dag_stats.multi_parent_nodes) multi-parent ($(round(dag_stats.multi_parent_nodes/dag_stats.unique_nodes*100, digits=1))%)")

    # DAG-MCTS BestValue (no persistence for comparison)
    println("\n[4/5] DAG-MCTS (BestValue, no persistence)")
    dag_config_np = DAGMCTSConfig(num_simulations=num_simulations, exploration_constant=1.41,
                                   degree=1, persist_table=false, selection_mode=BestValue)
    optim_dag_np = dag_mcts_descent_init(initial_param, loss, dag_config_np)
    losses_dag_np = [eval_loss(optim_dag_np)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_dag_np)
        push!(losses_dag_np, eval_loss(optim_dag_np))
        if i % 5 == 0
            @printf("  DAG-BV-NP step %2d: loss = %.6e (%.1fs)\n", i, losses_dag_np[end], time()-t0)
        end
    end
    push!(results, (name="DAG-BV-NP", losses=losses_dag_np, time=time()-t0))

    # UCT
    println("\n[5/5] UCT")
    uct_config = UCTConfig(max_depth=12, num_simulations=num_simulations, degree=1)
    optim_uct = uct_descent_init(initial_param, loss, uct_config)
    losses_uct = [eval_loss(optim_uct)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_uct)
        push!(losses_uct, eval_loss(optim_uct))
        if i % 5 == 0
            @printf("  UCT step %2d: loss = %.6e (%.1fs)\n", i, losses_uct[end], time()-t0)
        end
    end
    push!(results, (name="UCT", losses=losses_uct, time=time()-t0))

    # Summary
    println("\n" * "="^70)
    println("RESULTS SUMMARY")
    println("="^70)

    println("\nFinal Rankings:")
    println("-"^60)
    @printf("%-12s %15s %12s %10s\n", "Optimizer", "Final Loss", "Time(s)", "Improv%")
    println("-"^60)

    sorted = sort(results, by=r->r.losses[end])
    for r in sorted
        improv = (initial_loss - r.losses[end]) / initial_loss * 100
        @printf("%-12s %15.6e %12.2f %9.1f%%\n", r.name, r.losses[end], r.time, improv)
    end

    println("\nConvergence (every 5 steps):")
    println("-"^60)
    print(@sprintf("%-6s", "Step"))
    for r in results
        print(@sprintf(" %12s", r.name))
    end
    println()
    println("-"^60)
    for step in [0, 5, 10, 15]
        print(@sprintf("%-6d", step))
        for r in results
            print(@sprintf(" %12.4e", r.losses[step+1]))
        end
        println()
    end

    println("\n" * "="^70)
    println("Benchmark complete!")
    println("="^70)

    return results
end

results = main()
