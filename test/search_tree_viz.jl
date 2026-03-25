# Tests for search tree visualization (src/visualization/search_tree_viz.jl)
#
# Tests cover:
# - Internal helper functions (_value_to_color, _truncate_padic, accessors)
# - _flatten_search_tree: BFS ordering, max_depth/max_nodes limits, DAG deduplication
# - visualize_search_tree: dispatches for node / state / OptimSetup, various optimizer types

using Test
using Oscar
using D3Trees
using NAML

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

function _make_loss(K)
    R, x = polynomial_ring(K, ["x"])
    poly = AbsolutePolynomialSum([x[1]^2])
    batch_eval = batch_evaluate_init(poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})
    Loss(
        params -> [batch_eval(p) for p in params],
        vs    -> [directional_derivative(poly, v) for v in vs],
    )
end

@testset "Search Tree Visualization" begin
    K = PadicField(2, 20)

    # -------------------------------------------------------------------
    @testset "Internal helpers" begin
        @testset "_value_to_color" begin
            # Unvisited node → grey
            @test NAML._value_to_color(0.0, 0.0, 1.0, 0) == "#cccccc"

            # Visited, at minimum → blue-ish (hue ≈ 240)
            color_min = NAML._value_to_color(0.0, 0.0, 1.0, 1)
            @test startswith(color_min, "hsl(")
            @test occursin("240", color_min)

            # Visited, at maximum → red-ish (hue ≈ 0)
            color_max = NAML._value_to_color(1.0, 0.0, 1.0, 1)
            @test startswith(color_max, "hsl(")
            @test occursin("0,", color_max)

            # Flat range (min == max) → clamp to 0.5 → hue 120 (mid-point)
            color_flat = NAML._value_to_color(0.5, 0.5, 0.5, 1)
            @test startswith(color_flat, "hsl(120,")
        end

        @testset "_truncate_padic" begin
            short = "abc"
            @test NAML._truncate_padic(short) == short

            long = "a" ^ 20
            truncated = NAML._truncate_padic(long; maxlen=16)
            @test length(truncated) == 17   # 16 chars + ellipsis character
            @test endswith(truncated, "…")
            @test NAML._truncate_padic(long; maxlen=20) == long
        end
    end

    # -------------------------------------------------------------------
    @testset "_flatten_search_tree" begin
        loss = _make_loss(K)
        param = ValuationPolydisc([K(4)], [0])

        config = MCTSConfig(num_simulations=30, persist_tree=false)
        optim  = mcts_descent_init(param, loss, config)
        for _ in 1:3; step!(optim); end

        root = optim.state.root

        @testset "basic BFS" begin
            children_vec, nodes_vec, depths = NAML._flatten_search_tree(root; max_depth=10)
            @test length(nodes_vec) >= 1
            @test nodes_vec[1] === root
            @test depths[1] == 0
            # Every child index should be a valid index into nodes_vec
            for (i, cs) in enumerate(children_vec)
                for c in cs
                    @test 1 <= c <= length(nodes_vec)
                end
            end
        end

        @testset "max_depth limit" begin
            _, nodes_shallow, depths_shallow = NAML._flatten_search_tree(root; max_depth=1)
            _, nodes_deep,    depths_deep    = NAML._flatten_search_tree(root; max_depth=5)
            @test maximum(depths_shallow; init=0) <= 1
            # deeper search includes at least as many nodes
            @test length(nodes_deep) >= length(nodes_shallow)
        end

        @testset "max_nodes limit" begin
            _, nodes_limited, _ = NAML._flatten_search_tree(root; max_nodes=3)
            @test length(nodes_limited) <= 3
        end
    end

    # -------------------------------------------------------------------
    @testset "DAG deduplication in _flatten_search_tree" begin
        # Build a small DAG manually: root → A, root → B; A → child, B → child (same object)
        p_root  = ValuationPolydisc([K(0), K(0)], [0, 0])
        p_a     = children_along_branch(p_root, 1)[1]
        p_b     = children_along_branch(p_root, 2)[1]
        p_child = ValuationPolydisc([K(0), K(0)], [1, 1])

        config = DAGMCTSConfig(num_simulations=5, persist_table=false)
        loss = begin
            R, (x, y) = polynomial_ring(K, ["x", "y"])
            poly = AbsolutePolynomialSum([x^2 + y^2])
            VP = ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}
            be = batch_evaluate_init(poly, VP)
            Loss(ps -> [be(p) for p in ps], vs -> [directional_derivative(poly, v) for v in vs])
        end
        optim = dag_mcts_descent_init(p_root, loss, config)
        step!(optim)

        # Verify flatten does not duplicate nodes already seen (objectid deduplicated)
        children_vec, nodes_vec, _ = NAML._flatten_search_tree(optim.state.root; max_depth=10)
        seen_ids = Set(objectid(n) for n in nodes_vec)
        @test length(seen_ids) == length(nodes_vec)
    end

    # -------------------------------------------------------------------
    @testset "visualize_search_tree dispatch" begin
        loss  = _make_loss(K)
        param = ValuationPolydisc([K(4)], [0])

        @testset "MCTS: node / state / OptimSetup" begin
            config = MCTSConfig(num_simulations=20, persist_tree=false)
            optim  = mcts_descent_init(param, loss, config)
            step!(optim)

            root  = optim.state.root
            state = optim.state

            t1 = visualize_search_tree(root)
            t2 = visualize_search_tree(state)
            t3 = visualize_search_tree(optim)

            for t in (t1, t2, t3)
                @test t isa D3Tree
                @test length(t.children) >= 1
                @test length(t.text)     == length(t.children)
                @test length(t.tooltip)  == length(t.children)
            end

            # All three dispatch routes should produce the same tree size
            @test length(t1.children) == length(t2.children) == length(t3.children)
        end

        @testset "DAG-MCTS: OptimSetup" begin
            config = DAGMCTSConfig(num_simulations=20, persist_table=false)
            optim  = dag_mcts_descent_init(param, loss, config)
            step!(optim)

            t = visualize_search_tree(optim)
            @test t isa D3Tree
            @test length(t.children) >= 1
        end

        @testset "HOO: OptimSetup" begin
            config = HOOConfig(max_depth=4)
            optim  = hoo_descent_init(param, loss, config)
            step!(optim)

            t = visualize_search_tree(optim)
            @test t isa D3Tree
            @test length(t.children) >= 1
        end

        @testset "UCT: OptimSetup" begin
            config = UCTConfig(num_simulations=20)
            optim  = uct_descent_init(param, loss, config)
            step!(optim)

            t = visualize_search_tree(optim)
            @test t isa D3Tree
            @test length(t.children) >= 1
        end
    end

    # -------------------------------------------------------------------
    @testset "D3Tree content" begin
        loss  = _make_loss(K)
        param = ValuationPolydisc([K(4)], [0])
        config = MCTSConfig(num_simulations=40, persist_tree=false)
        optim  = mcts_descent_init(param, loss, config)
        step!(optim)

        t = visualize_search_tree(optim; max_depth=3, init_expand=2)
        n = length(t.children)

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
            _, _, depths = NAML._flatten_search_tree(optim.state.root; max_depth=1)
            @test length(t_shallow.children) == length(depths)
        end

        @testset "keyword args forwarded to D3Tree" begin
            t_titled = visualize_search_tree(optim; title="TestTitle", svg_height=600)
            @test t_titled isa D3Tree
        end
    end
end
