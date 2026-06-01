#!/usr/bin/env julia

using Oscar
using Plots
using Random
using WGLMakie
using NonArchimedeanMachineLearning

const NAML = NonArchimedeanMachineLearning

function run_more_simulations!(optim, loss, count::Int)
    config = DAGMCTSConfig(
        num_simulations = count,
        exploration_constant = optim.context.exploration_constant,
        degree = optim.context.degree,
        value_transform = optim.context.value_transform,
        persist_table = optim.context.persist_table,
        selection_mode = optim.context.selection_mode,
        track_parents = optim.context.track_parents,
        variant = optim.context.variant,
    )

    NAML.dag_mcts_search(
        optim.state.root,
        optim.state.transposition_table,
        loss,
        config,
        optim.state,
    )
    return optim
end

function main()
    serve_live = ("--serve" in ARGS) || ("--live" in ARGS)
    Random.seed!(2026)

    # DAG-MCTS search over a 2D parameter polydisc in Q_2.
    K = PadicField(2, 20)
    root_disc = ValuationPolydisc([K(0), K(0)], [0, 0])

    # A small target-shaped objective: |a - 1|_2 + |b - 3|_2.
    l1 = LinearPolynomial([K(1), K(0)], K(-1))
    l2 = LinearPolynomial([K(0), K(1)], K(-3))
    f = LinearAbsolutePolynomialSum([l1, l2])

    VP = ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int, 2}
    batch_eval = batch_evaluate_init(f, VP)
    loss = Loss(
        params -> map(batch_eval, params),
        tangents -> zeros(Float64, length(tangents)),
    )

    base_config = DAGMCTSConfig(
        num_simulations = 1,
        exploration_constant = 1.41,
        degree = 1,
        persist_table = true,
        selection_mode = VisitCount,
        track_parents = true,
        variant = NodeStatsDAGMCTS,
    )

    optim = dag_mcts_descent_init(root_disc, loss, base_config)
    snapshot_simulations = [4, 50, 200, 1000]
    last_snapshot = last(snapshot_simulations)
    completed = 0

    for target in snapshot_simulations
        run_more_simulations!(optim, loss, target - completed)
        completed = target

        stats = get_dag_stats(optim.state)
        println(
            "After $target simulations: ",
            "root visits = $(optim.state.root.visits), ",
            "unique nodes = $(stats.unique_nodes), ",
            "multi-parent nodes = $(stats.multi_parent_nodes)",
        )

        plt = plot_search_tree_cone(optim;
            max_depth = 7,
            max_nodes = 5000,
            title = "",
            node_size = 3,
            edge_width = 0.9,
            edge_gradient_steps = 8,
            visited_only = true,
            visit_color_max = last_snapshot,
            visit_color_scale = :log,
            visit_color_scheme = :rainbow,
            show_surface = false,
            camera = (45, 25),
        )

        filename = "cone_search_tree_viz_after_$(lpad(target, 3, '0')).png"
        savefig(plt, filename)
        println("  Saved $filename")

        if target == last_snapshot
            savefig(plt, "cone_search_tree_viz.png")
            println("  Saved cone_search_tree_viz.png")

            interactive_title = "DAG-MCTS after $target simulations"
            fig = plot_search_tree_cone_interactive(optim;
                max_depth = 7,
                max_nodes = 5000,
                title = interactive_title,
                node_size = 7,
                edge_width = 1.2,
                edge_gradient_steps = 3,
                visited_only = true,
                visit_color_max = last_snapshot,
                visit_color_scale = :log,
                visit_color_scheme = :rainbow,
                show_surface = false,
                offline = !serve_live,
            )
            if serve_live
                serve_search_tree_cone_interactive(fig;
                    title = interactive_title,
                    port = 8765,
                    block = true)
            else
                save_search_tree_cone_interactive_html("cone_search_tree_viz_interactive.html", fig;
                    title = interactive_title)
                save_search_tree_cone_interactive_html("cone_search_tree_viz_after_$(target)_interactive.html", fig;
                    title = interactive_title)
                println("  Saved cone_search_tree_viz_interactive.html")
                println("  For the live rotatable view, run this demo with --serve")
            end
        end
    end
end

main()
