# Tests for search tree visualization (src/visualization/search_tree_viz.jl)
#
# Tests cover:
# - Internal helper functions (_value_to_color, _truncate_padic, accessors)
# - _flatten_search_tree: BFS ordering, max_depth/max_nodes limits, DAG deduplication
# - visualize_search_tree: dispatches for node / state / OptimSetup

using Test
using Oscar
using D3Trees
using NonArchimedeanMachineLearning

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

function _make_loss(K)
    R, x = polynomial_ring(K, ["x"])
    poly = AbsolutePolynomialSum([x[1]^2])
    batch_eval = batch_evaluate_init(poly, ValuationPolydisc{
        ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})
    Loss(
        params -> [batch_eval(p) for p in params],
        vs -> [directional_derivative(poly, v) for v in vs]
    )
end

@testset "Search Tree Visualization" begin
    K = PadicField(2, 20)

    # -------------------------------------------------------------------
    @testset "Internal helpers" begin
        @testset "_value_to_color" begin
            # Unvisited node → grey
            @test NonArchimedeanMachineLearning._value_to_color(0.0, 0.0, 1.0, 0) == "#cccccc"

            # Visited, at minimum → blue-ish (hue ≈ 240)
            color_min = NonArchimedeanMachineLearning._value_to_color(0.0, 0.0, 1.0, 1)
            @test startswith(color_min, "hsl(")
            @test occursin("240", color_min)

            # Visited, at maximum → red-ish (hue ≈ 0)
            color_max = NonArchimedeanMachineLearning._value_to_color(1.0, 0.0, 1.0, 1)
            @test startswith(color_max, "hsl(")
            @test occursin("0,", color_max)

            # Flat range (min == max) → clamp to 0.5 → hue 120 (mid-point)
            color_flat = NonArchimedeanMachineLearning._value_to_color(0.5, 0.5, 0.5, 1)
            @test startswith(color_flat, "hsl(120,")
        end

        @testset "_truncate_padic" begin
            short = "abc"
            @test NonArchimedeanMachineLearning._truncate_padic(short) == short

            long = "a" ^ 20
            truncated = NonArchimedeanMachineLearning._truncate_padic(long; maxlen=16)
            @test length(truncated) == 17   # 16 chars + ellipsis character
            @test endswith(truncated, "…")
            @test NonArchimedeanMachineLearning._truncate_padic(long; maxlen=20) == long
        end
    end

    # -------------------------------------------------------------------
    @testset "_flatten_search_tree" begin
        root = MCTSNode(ValuationPolydisc([K(0)], [0]))
        a = MCTSNode(ValuationPolydisc([K(0)], [1]), root)
        b = MCTSNode(ValuationPolydisc([K(1)], [1]), root)
        leaf = MCTSNode(ValuationPolydisc([K(0)], [2]), a)
        append!(root.children, [a, b])
        push!(a.children, leaf)

        edges, nodes, depths = NonArchimedeanMachineLearning._flatten_search_tree(root)
        @test nodes == [root, a, b, leaf]
        @test edges == [[2, 3], [4], Int[], Int[]]
        @test depths == [0, 1, 1, 2]

        @testset "max_depth limit" begin
            edges, nodes, depths = NonArchimedeanMachineLearning._flatten_search_tree(root; max_depth = 1)
            @test nodes == [root, a, b]
            @test edges == [[2, 3], Int[], Int[]]
            @test depths == [0, 1, 1]
        end

        @testset "max_nodes limit" begin
            _, nodes, _ = NonArchimedeanMachineLearning._flatten_search_tree(root; max_nodes = 2)
            @test nodes == [root, a]
        end
    end

    # -------------------------------------------------------------------
    @testset "DAG deduplication in _flatten_search_tree" begin
        # Build a small DAG manually: root → A, root → B; A → child, B → child (same object)
        p_root = ValuationPolydisc([K(0), K(0)], [0, 0])
        p_a = children_along_branch(p_root, 1)[1]
        p_b = children_along_branch(p_root, 2)[1]
        p_child = ValuationPolydisc([K(0), K(0)], [1, 1])

        root = DAGMCTSNode(p_root)
        a, b, child = DAGMCTSNode(p_a), DAGMCTSNode(p_b), DAGMCTSNode(p_child)
        append!(root.children, [a, b])
        push!(a.children, child)
        push!(b.children, child)

        edges, nodes, depths = NonArchimedeanMachineLearning._flatten_search_tree(root)
        @test nodes == [root, a, b, child]
        @test edges == [[2, 3], [4], [4], Int[]]
        @test depths == [0, 1, 1, 2]
    end

    # -------------------------------------------------------------------
    @testset "visualize_search_tree dispatch" begin
        loss = _make_loss(K)
        param = ValuationPolydisc([K(4)], [0])

        @testset "MCTS: node / state / OptimSetup" begin
            config = MCTSConfig(num_simulations = 20, persist_tree = false)
            optim = mcts_descent_init(param, loss, config)
            step!(optim)

            root = optim.state.root
            state = optim.state

            t1 = visualize_search_tree(root)
            t2 = visualize_search_tree(state)
            t3 = visualize_search_tree(optim)

            for t in (t1, t2, t3)
                @test length(t.children) >= 1
                @test length(t.text) == length(t.children)
                @test length(t.tooltip) == length(t.children)
            end

            # All three dispatch routes should produce the same tree size
            @test length(t1.children) == length(t2.children) == length(t3.children)
        end

        @testset "DAG-MCTS: OptimSetup" begin
            config = DAGMCTSConfig(num_simulations = 20, persist_table = false)
            optim = dag_mcts_descent_init(param, loss, config)
            step!(optim)

            t = visualize_search_tree(optim)
            @test length(t.children) >= 1
        end
    end

    # -------------------------------------------------------------------
    @testset "D3Tree content" begin
        loss = _make_loss(K)
        param = ValuationPolydisc([K(4)], [0])
        config = MCTSConfig(num_simulations = 40, persist_tree = false)
        optim = mcts_descent_init(param, loss, config)
        step!(optim)

        t = visualize_search_tree(optim; max_depth = 3, init_expand = 2)

        @testset "label format" begin
            # Root node (index 1) text should either be "n=0" or "n=<visits>\nv=<value>"
            root_text = t.text[1]
            @test startswith(root_text, "n=")
        end

        @testset "tooltip contains expected fields" begin
            tip = t.tooltip[1]
            @test occursin("MCTS node", tip)
            @test occursin("visits:", tip)
            @test occursin("children:", tip)
        end

        @testset "style is non-empty for all nodes" begin
            @test all(!isempty, t.style)
        end

        @testset "max_depth respected" begin
            t_shallow = visualize_search_tree(optim; max_depth=1)
            _, _, depths = NonArchimedeanMachineLearning._flatten_search_tree(optim.state.root; max_depth=1)
            @test length(t_shallow.children) == length(depths)
        end

        @testset "keyword args forwarded to D3Tree" begin
            t_titled = visualize_search_tree(optim; title = "TestTitle")
            @test t_titled.title == "TestTitle"
        end
    end
end
