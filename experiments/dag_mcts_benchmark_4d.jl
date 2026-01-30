# DAG-MCTS Benchmark: 4D Linear Polynomial Optimization over Q_3
#
# A simpler problem where MCTS can make progress.
# 4 variables, 3 polynomials, starting at a point where loss varies.
#
# Run with: julia --project=. experiments/dag_mcts_benchmark_4d.jl

using Printf
using Random
using Oscar

include("../src/NAML.jl")
using .NAML

function main()
    println("="^70)
    println("DAG-MCTS Benchmark: 4D Linear Polynomial Optimization over Q_3")
    println("="^70)

    prec = 30
    p = 3
    K = PadicField(p, prec)

    n_vars = 4

    # Simpler problem: minimize |x1 + x2 - 1| + |x3 + x4 - 1| + |x1 - x3|
    # This has a clear minimum at x1=x3=0.5, x2=x4=0.5 (in real terms)
    # In p-adic terms, we're looking for x values that make each term small

    linear_polys = [
        LinearPolynomial([K(1), K(1), K(0), K(0)], K(-1)),   # x1 + x2 - 1
        LinearPolynomial([K(0), K(0), K(1), K(1)], K(-1)),   # x3 + x4 - 1
        LinearPolynomial([K(1), K(0), K(-1), K(0)], K(0)),   # x1 - x3
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

    println("\nProblem: minimize |x1 + x2 - 1| + |x3 + x4 - 1| + |x1 - x3|")
    println("Field: Q_$p with precision $prec")
    println("Variables: $n_vars")

    # Start at a point where there's signal
    initial_center = [K(0), K(0), K(0), K(0)]
    initial_radius = [2, 2, 2, 2]
    initial_param = ValuationPolydisc(initial_center, initial_radius)
    initial_loss = loss.eval([initial_param])[1]

    println("\nInitial radius: (2, 2, 2, 2)")
    @printf("Initial loss: %.6e\n", initial_loss)

    # Check children to verify there's variation
    println("\nChecking for loss variation among children...")
    child_losses = [batch_eval(ch) for ch in children(initial_param, 1)]
    unique_losses = unique(child_losses)
    println("Unique losses among $(length(child_losses)) children: $(length(unique_losses))")
    if length(unique_losses) > 1
        println("Loss range: $(minimum(child_losses)) to $(maximum(child_losses))")
        println("Good - there's variation for MCTS to exploit!")
    else
        println("Warning: all children have same loss")
    end

    # Benchmark
    num_steps = 20
    num_simulations = 100

    println("\n" * "-"^70)
    println("Running optimizers for $num_steps steps (simulations=$num_simulations)...")
    println("-"^70)

    results = []

    # Greedy
    println("\n[1/4] Greedy Descent")
    optim_greedy = greedy_descent_init(initial_param, loss, 1, (false, 1))

    losses_greedy = [eval_loss(optim_greedy)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_greedy)
        push!(losses_greedy, eval_loss(optim_greedy))
        if i % 5 == 0
            @printf("  Greedy step %2d: loss = %.6e\n", i, losses_greedy[end])
        end
    end
    push!(results, (name="Greedy", losses=losses_greedy, time=time()-t0))

    # MCTS BestValue
    println("\n[2/4] MCTS (BestValue)")
    mcts_config = MCTSConfig(num_simulations=num_simulations, exploration_constant=1.41,
                             degree=1, selection_mode=BestValue)
    optim_mcts = mcts_descent_init(initial_param, loss, mcts_config)

    losses_mcts = [eval_loss(optim_mcts)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_mcts)
        push!(losses_mcts, eval_loss(optim_mcts))
        if i % 5 == 0
            @printf("  MCTS step %2d: loss = %.6e\n", i, losses_mcts[end])
        end
    end
    push!(results, (name="MCTS-BV", losses=losses_mcts, time=time()-t0))

    # DAG-MCTS (VisitCount)
    println("\n[3/5] DAG-MCTS (VisitCount)")
    dag_config_vc = DAGMCTSConfig(num_simulations=num_simulations, exploration_constant=1.41,
                                   degree=1, persist_table=true, selection_mode=VisitCount)
    optim_dag_vc = dag_mcts_descent_init(initial_param, loss, dag_config_vc)

    losses_dag_vc = [eval_loss(optim_dag_vc)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_dag_vc)
        push!(losses_dag_vc, eval_loss(optim_dag_vc))
        if i % 5 == 0
            @printf("  DAG-VC step %2d: loss = %.6e\n", i, losses_dag_vc[end])
        end
    end
    push!(results, (name="DAG-VC", losses=losses_dag_vc, time=time()-t0))

    dag_stats_vc = get_dag_stats(optim_dag_vc.state)
    println("  DAG stats: $(dag_stats_vc.unique_nodes) nodes, $(dag_stats_vc.multi_parent_nodes) multi-parent ($(round(dag_stats_vc.multi_parent_nodes/dag_stats_vc.unique_nodes*100, digits=1))%)")

    # DAG-MCTS (BestValue)
    println("\n[4/5] DAG-MCTS (BestValue)")
    dag_config_bv = DAGMCTSConfig(num_simulations=num_simulations, exploration_constant=1.41,
                                   degree=1, persist_table=true, selection_mode=BestValue)
    optim_dag_bv = dag_mcts_descent_init(initial_param, loss, dag_config_bv)

    losses_dag_bv = [eval_loss(optim_dag_bv)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_dag_bv)
        push!(losses_dag_bv, eval_loss(optim_dag_bv))
        if i % 5 == 0
            @printf("  DAG-BV step %2d: loss = %.6e\n", i, losses_dag_bv[end])
        end
    end
    push!(results, (name="DAG-BV", losses=losses_dag_bv, time=time()-t0))

    dag_stats_bv = get_dag_stats(optim_dag_bv.state)
    println("  DAG stats: $(dag_stats_bv.unique_nodes) nodes, $(dag_stats_bv.multi_parent_nodes) multi-parent ($(round(dag_stats_bv.multi_parent_nodes/dag_stats_bv.unique_nodes*100, digits=1))%)")

    # UCT
    println("\n[5/5] UCT")
    uct_config = UCTConfig(max_depth=10, num_simulations=num_simulations, degree=1)
    optim_uct = uct_descent_init(initial_param, loss, uct_config)

    losses_uct = [eval_loss(optim_uct)]
    t0 = time()
    for i in 1:num_steps
        step!(optim_uct)
        push!(losses_uct, eval_loss(optim_uct))
        if i % 5 == 0
            @printf("  UCT step %2d: loss = %.6e\n", i, losses_uct[end])
        end
    end
    push!(results, (name="UCT", losses=losses_uct, time=time()-t0))

    # Summary
    println("\n" * "="^70)
    println("RESULTS SUMMARY")
    println("="^70)

    println("\nFinal Rankings:")
    println("-"^50)
    @printf("%-12s %15s %10s %10s\n", "Optimizer", "Final Loss", "Time(s)", "Improv%")
    println("-"^50)

    sorted = sort(results, by=r->r.losses[end])
    for r in sorted
        improv = (initial_loss - r.losses[end]) / initial_loss * 100
        @printf("%-12s %15.6e %10.2f %9.1f%%\n", r.name, r.losses[end], r.time, improv)
    end

    println("\n" * "="^70)
end

main()
