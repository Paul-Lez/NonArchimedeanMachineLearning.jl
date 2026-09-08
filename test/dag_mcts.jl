# Test file for DAG-MCTS (Monte Carlo Tree Search with Transposition Tables)
#
# This file tests the DAG-MCTS implementation including:
# - Transposition table functionality
# - DAG node structure with multiple parents
# - UCT selection in DAG context
# - Backpropagation via explicit path
# - Integration with OptimSetup

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "DAG-MCTS" begin
    prec = 20
    K = PadicField(2, prec)

    @testset "DAGMCTSNode Creation" begin
        p = ValuationPolydisc([K(1)], [0])
        node = DAGMCTSNode(p)

        @test isempty(node.parents)
        @test isempty(node.children)
        @test node.visits == 0
        @test node.total_value == 0.0
        @test !node.is_expanded
    end

    @testset "DAG-MCTS Variant Selection" begin
        default_config = DAGMCTSConfig(num_simulations = 5)
        @test default_config.variant == PathStatsDAGMCTS
        @test default_config.selection_mode == BestValue
        @test !default_config.persist_table
        @test 0.0 < default_config.value_transform(2.0) < 1.0

        node_config = DAGMCTSConfig(num_simulations = 5, variant = NodeStatsDAGMCTS)
        @test node_config.selection_mode == VisitCount
        @test node_config.persist_table
        @test 0.0 < node_config.value_transform(2.0) < 1.0
    end

    @testset "Path Statistics and Evaluation Cache" begin
        p = ValuationPolydisc([K(0), K(0)], [0, 0])
        eval_calls = Ref(0)
        loss = Loss(
            params -> begin
                eval_calls[] += length(params)
                [0.0 for _ in params]
            end,
            vs -> [0.0 for _ in vs]
        )

        config = DAGMCTSConfig(num_simulations = 12, persist_table = true,
            track_parents = true)
        optim = dag_mcts_descent_init(p, loss, config)

        NonArchimedeanMachineLearning.dag_mcts_path_search(
            optim.state.root,
            optim.state.transposition_table,
            loss,
            config,
            optim.state
        )

        @test any(key -> length(key) > 1, keys(optim.state.path_stats))
        @test eval_calls[] == length(optim.state.evaluation_cache)
        @test length(optim.state.evaluation_cache) <=
              length(optim.state.transposition_table)
        @test verify_transposition_table(optim.state)
    end

    @testset "Transposition Table - Basic Operations" begin
        # Test get_or_create_node!
        # Note: Polydisc equality uses STRICT inequality: v(center_diff) > radius
        table = Dict{NonArchimedeanMachineLearning.HashedPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1}}()

        p1 = ValuationPolydisc([K(1)], [2])

        # First call should create new node
        node1 = NonArchimedeanMachineLearning.get_or_create_node!(table, p1)
        @test length(table) == 1
        @test node1.polydisc == p1

        # Second call with same polydisc should return same node
        node2 = NonArchimedeanMachineLearning.get_or_create_node!(table, p1)
        @test node1 === node2  # Same object reference
        @test length(table) == 1  # No new entry

        # Equivalent polydisc (v(diff) > radius) should return same node
        p1_equiv = ValuationPolydisc([K(1 + 8)], [2])  # v(8) = 3 > 2
        @test p1 == p1_equiv  # Verify they're equal
        node3 = NonArchimedeanMachineLearning.get_or_create_node!(table, p1_equiv)
        @test node1 === node3  # Same object due to transposition
        @test length(table) == 1

        # Different polydisc should create new node
        p2 = ValuationPolydisc([K(2)], [2])
        node4 = NonArchimedeanMachineLearning.get_or_create_node!(table, p2)
        @test node4 !== node1
        @test length(table) == 2
    end

    @testset "Transposition Table - Parent Linking (track_parents=true)" begin
        # Use 2D to create truly different parent nodes via different refinement paths
        table = Dict{NonArchimedeanMachineLearning.HashedPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}}()

        # Start with a root polydisc
        root = ValuationPolydisc([K(0), K(0)], [0, 0])
        root_node = NonArchimedeanMachineLearning.get_or_create_node!(table, root)

        # Create two different parents by refining different coordinates
        parent1 = children_along_branch(root, 1)[1]  # Refine coordinate 1
        parent2 = children_along_branch(root, 2)[1]  # Refine coordinate 2

        parent1_node = NonArchimedeanMachineLearning.get_or_create_node!(table, parent1, root_node)
        parent2_node = NonArchimedeanMachineLearning.get_or_create_node!(table, parent2, root_node)

        # Verify parents are different
        @test parent1 != parent2
        @test parent1_node !== parent2_node

        # Create a child that both parents can reach
        child_p = ValuationPolydisc([K(0), K(0)], [1, 1])

        # Create child with first parent
        child_node = NonArchimedeanMachineLearning.get_or_create_node!(table, child_p, parent1_node)
        @test length(child_node.parents) == 1
        @test parent1_node in child_node.parents

        # Link same child to second parent (transposition)
        child_node2 = NonArchimedeanMachineLearning.get_or_create_node!(table, child_p, parent2_node)
        @test child_node === child_node2  # Same node instance
        @test length(child_node.parents) == 2
        @test parent1_node in child_node.parents
        @test parent2_node in child_node.parents
        @test length(table) == 4
    end

    @testset "Transposition Table - No Parent Tracking (track_parents=false)" begin
        table = Dict{NonArchimedeanMachineLearning.HashedPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}}()

        child_p = ValuationPolydisc([K(0), K(0)], [1, 1])

        # Without parent tracking, parents vector stays empty
        child_node = NonArchimedeanMachineLearning.get_or_create_node!(table, child_p)
        @test isempty(child_node.parents)

        child_node2 = NonArchimedeanMachineLearning.get_or_create_node!(table, child_p)
        @test child_node === child_node2  # Transposition still detected
        @test isempty(child_node.parents)  # But no parents tracked
    end

    @testset "UCT Score Computation" begin
        p = ValuationPolydisc([K(1)], [0])
        node = DAGMCTSNode(p)

        # Unvisited node should have Inf score
        @test NonArchimedeanMachineLearning.uct_score(node, 10, sqrt(2.0)) == Inf

        # Visited node should have finite score
        node.visits = 5
        node.total_value = 2.5  # average = 0.5
        parent_visits = 100
        c = sqrt(2.0)

        score = NonArchimedeanMachineLearning.uct_score(node, parent_visits, c)
        expected = 0.5 + c * sqrt(log(parent_visits) / 5)
        @test abs(score - expected) < 1e-10
    end

    @testset "Backpropagation via Explicit Path" begin
        # Create a simple path of nodes
        p1 = ValuationPolydisc([K(0)], [0])
        p2 = ValuationPolydisc([K(0)], [1])
        p3 = ValuationPolydisc([K(0)], [2])

        node1 = DAGMCTSNode(p1)
        node2 = DAGMCTSNode(p2)
        node3 = DAGMCTSNode(p3)

        # Create a dummy state for backpropagation best-node tracking
        table = Dict{NonArchimedeanMachineLearning.HashedPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1}}()
        dummy_state = DAGMCTSState{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1}(node1, table, 0, nothing, -Inf, nothing, 0, nothing, Inf, nothing)

        path = [node1, node2, node3]
        value = 0.75

        # Backpropagate
        NonArchimedeanMachineLearning.backpropagate!(path, value, dummy_state)

        # All nodes should have 1 visit and the value
        for node in path
            @test node.visits == 1
            @test node.total_value == value
        end

        # Second backpropagation
        NonArchimedeanMachineLearning.backpropagate!(path, 0.25, dummy_state)
        for node in path
            @test node.visits == 2
            @test node.total_value == 1.0  # 0.75 + 0.25
        end

        # Best node should be tracked (best_value is the max average ever seen,
        # which was 0.75 after the first backprop; the second backprop lowered
        # averages to 0.5 which doesn't exceed the tracked best)
        @test !isnothing(dummy_state.best_node)
        @test dummy_state.best_value == 0.75
    end

    @testset "DAG-MCTS Integration: $variant" for variant in (PathStatsDAGMCTS, NodeStatsDAGMCTS)
        # Set up a simple optimization problem: minimize |x|^2
        R, x = polynomial_ring(K, ["x"])

        poly = AbsolutePolynomialSum([x[1]^2])
        batch_eval = batch_evaluate_init(
            poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})

        function loss_eval(params::Vector)
            return [batch_eval(p) for p in params]
        end
        function loss_grad(vs::Vector)
            return [directional_derivative(poly, v) for v in vs]
        end
        loss = Loss(loss_eval, loss_grad)

        # Initial parameter
        initial_param = ValuationPolydisc([K(8)], [0])  # Start at 8 = 2^3

        # Configure DAG-MCTS
        config = DAGMCTSConfig(
            variant = variant,
            num_simulations = 50,
            exploration_constant = 1.41,
            degree = 1,
            persist_table = false
        )

        # Initialize optimizer
        optim = dag_mcts_descent_init(initial_param, loss, config)

        initial_loss = eval_loss(optim)

        # Run a few optimization steps
        for _ in 1:5
            step!(optim)
        end

        final_loss = eval_loss(optim)

        # Loss should improve (or at least not get worse)
        @test final_loss <= initial_loss + 1e-6
    end

    @testset "Persist Table Option" begin
        R, x = polynomial_ring(K, ["x"])
        poly = AbsolutePolynomialSum([x[1]^2])
        batch_eval = batch_evaluate_init(
            poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})

        function loss_eval(params::Vector)
            return [batch_eval(p) for p in params]
        end
        function loss_grad(vs::Vector)
            return [directional_derivative(poly, v) for v in vs]
        end
        loss = Loss(loss_eval, loss_grad)

        initial_param = ValuationPolydisc([K(4)], [0])

        # Test with persist_table=true
        # With the DAG-first architecture, the table is rebuilt from the active subtree
        # after each step, so it only contains reachable nodes (not the whole history)
        config_persist = DAGMCTSConfig(num_simulations = 20, persist_table = true)
        optim_persist = dag_mcts_descent_init(initial_param, loss, config_persist)

        step!(optim_persist)
        table_size_after_step1 = length(optim_persist.state.transposition_table)

        step!(optim_persist)
        table_size_after_step2 = length(optim_persist.state.transposition_table)

        # Table should contain only reachable nodes (positive, bounded)
        @test table_size_after_step1 > 0
        @test table_size_after_step2 > 0

        # Verify table integrity after rebuild
        @test verify_transposition_table(optim_persist.state)

        # Test with persist_table=false
        config_no_persist = DAGMCTSConfig(num_simulations = 20, persist_table = false)
        optim_no_persist = dag_mcts_descent_init(initial_param, loss, config_no_persist)

        step!(optim_no_persist)
        step!(optim_no_persist)

        # Without persistence, table is cleared each step
        # Should only contain nodes from current search
        @test length(optim_no_persist.state.transposition_table) == 1
        @test isempty(optim_no_persist.state.root.children)
    end

    @testset "MCTS Persist Tree Option" begin
        R, x = polynomial_ring(K, ["x"])
        poly = AbsolutePolynomialSum([x[1]^2])
        batch_eval = batch_evaluate_init(
            poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})

        function mcts_loss_eval(params::Vector)
            return [batch_eval(p) for p in params]
        end
        function mcts_loss_grad(vs::Vector)
            return [directional_derivative(poly, v) for v in vs]
        end
        loss = Loss(mcts_loss_eval, mcts_loss_grad)

        initial_param = ValuationPolydisc([K(4)], [0])

        # Test with persist_tree=true
        config_persist = MCTSConfig(num_simulations = 20, persist_tree = true)
        optim_persist = mcts_descent_init(initial_param, loss, config_persist)

        step!(optim_persist)
        step!(optim_persist)

        # With persistence, subtree is reused: root should have pre-existing children/stats
        @test optim_persist.state.root.parent === nothing
        @test get_tree_size(optim_persist.state.root) > 1

        # Test with persist_tree=false (default)
        config_no_persist = MCTSConfig(num_simulations = 20, persist_tree = false)
        optim_no_persist = mcts_descent_init(initial_param, loss, config_no_persist)

        step!(optim_no_persist)
        step!(optim_no_persist)

        # Without persistence, root is fresh each step (no pre-existing subtree beyond current search)
        @test optim_no_persist.state.root.parent === nothing
    end

    ##################################################
    # Terminal / Solved Node Handling
    ##################################################

    @testset "Terminal Node Detection - MCTS" begin
        # Use very low precision so nodes become terminal quickly
        K_low = PadicField(2, 3)
        # A polydisc at radius = prec (3) is terminal: children() returns empty
        terminal_p = ValuationPolydisc([K_low(0)], [3])
        terminal_node = MCTSNode(terminal_p)

        config = MCTSConfig(num_simulations=10)
        NonArchimedeanMachineLearning.expand_node!(terminal_node, config)

        @test terminal_node.is_terminal
        @test terminal_node.is_solved
        @test isempty(terminal_node.children)
        @test terminal_node.unsolved_children_count == 0

        # A polydisc at radius 2 (< prec 3) should NOT be terminal
        non_terminal_p = ValuationPolydisc([K_low(0)], [2])
        non_terminal_node = MCTSNode(non_terminal_p)
        NonArchimedeanMachineLearning.expand_node!(non_terminal_node, config)

        @test !non_terminal_node.is_terminal
        @test !non_terminal_node.is_solved
        @test !isempty(non_terminal_node.children)
        @test non_terminal_node.unsolved_children_count ==
              length(non_terminal_node.children)
    end

    @testset "Terminal Node Detection - DAG-MCTS" begin
        K_low = PadicField(2, 3)
        terminal_p = ValuationPolydisc([K_low(0)], [3])
        terminal_node = DAGMCTSNode(terminal_p)

        table = Dict{NonArchimedeanMachineLearning.HashedPolydisc{ValuedFieldPoint{2, 3, PadicFieldElem}, Int64, 1}, DAGMCTSNode{ValuedFieldPoint{2, 3, PadicFieldElem}, Int64, 1}}()
        table[NonArchimedeanMachineLearning.HashedPolydisc(terminal_p)] = terminal_node
        config = DAGMCTSConfig(num_simulations=10)

        NonArchimedeanMachineLearning.expand_node!(terminal_node, table, config)

        @test terminal_node.is_terminal
        @test terminal_node.is_solved
        @test isempty(terminal_node.children)
        @test terminal_node.unsolved_children_count == 0
    end

    @testset "Solved Propagation - MCTS" begin
        K_low = PadicField(2, 3)

        # Build a small tree: root at radius 2, children at radius 3 (terminal)
        root_p = ValuationPolydisc([K_low(0)], [2])
        root_node = MCTSNode(root_p)

        config = MCTSConfig(num_simulations=10)
        NonArchimedeanMachineLearning.expand_node!(root_node, config)

        @test !root_node.is_terminal
        @test length(root_node.children) > 0

        # Expand all children - they should be terminal
        for child in root_node.children
            NonArchimedeanMachineLearning.expand_node!(child, config)
            @test child.is_terminal
            @test child.is_solved
            # Simulate setting proven_value (as mcts_search would)
            child.proven_value = -1.0 * rand()
        end

        # Now propagate solved status up from each child
        for child in root_node.children
            NonArchimedeanMachineLearning.propagate_solved_up!(child)
        end

        # Root should now be solved since all children are solved
        @test root_node.is_solved
        # proven_value should be the max of children's proven_values
        @test root_node.proven_value ≈ maximum(c.proven_value for c in root_node.children)
    end

    @testset "Solved Propagation - DAG Diamond" begin
        root = DAGMCTSNode(ValuationPolydisc([K(0), K(0)], [0, 0]))
        a = DAGMCTSNode(ValuationPolydisc([K(0), K(0)], [1, 0]))
        b = DAGMCTSNode(ValuationPolydisc([K(0), K(0)], [0, 1]))
        leaf = DAGMCTSNode(ValuationPolydisc([K(0), K(0)], [1, 1]))
        append!(root.children, [a, b])
        for parent in (a, b)
            push!(parent.parents, root)
            push!(parent.children, leaf)
            push!(leaf.parents, parent)
            parent.is_expanded = true
            parent.unsolved_children_count = 1
        end
        root.is_expanded = true
        root.unsolved_children_count = 2
        leaf.is_solved = true
        leaf.proven_value = -0.25

        # Only A is on the supplied path, but both parents must be notified.
        NonArchimedeanMachineLearning.propagate_solved_up_dag!(leaf, [root, a, leaf])
        for node in (root, a, b)
            @test node.is_solved
            @test node.proven_value == -0.25
            @test node.unsolved_children_count == 0
        end
    end

    @testset "check_solved! Function" begin
        K_low = PadicField(2, 3)
        root_p = ValuationPolydisc([K_low(0)], [2])
        root_node = MCTSNode(root_p)

        config = MCTSConfig(num_simulations=10)
        NonArchimedeanMachineLearning.expand_node!(root_node, config)

        # check_solved! should return false when children are unsolved
        @test !NonArchimedeanMachineLearning.check_solved!(root_node)

        # Mark all children as solved manually
        for child in root_node.children
            child.is_solved = true
            child.proven_value = -0.5
        end
        root_node.unsolved_children_count = 0

        # Now check_solved! should succeed
        @test NonArchimedeanMachineLearning.check_solved!(root_node)
        @test root_node.is_solved
        @test root_node.proven_value ≈ -0.5

        # Calling again should return false (already solved)
        @test !NonArchimedeanMachineLearning.check_solved!(root_node)
    end

    @testset "MCTS converged parameter is terminal" begin
        # Use very low precision so the entire search space is small and MCTS
        # converges because it has solved the tree, not because we ran out of steps.
        K_low = PadicField(2, 3)
        R_low, x_low = polynomial_ring(K_low, ["x"])
        poly_low = AbsolutePolynomialSum([x_low[1]^2])
        PT = ValuationPolydisc{ValuedFieldPoint{2, 3, PadicFieldElem}, Int64, 1}
        batch_eval_low = batch_evaluate_init(poly_low, PT)

        loss_low = Loss(
            params -> [batch_eval_low(p) for p in params],
            vs -> [directional_derivative(poly_low, v) for v in vs]
        )

        initial_p = ValuationPolydisc([K_low(0)], [0])
        config = MCTSConfig(num_simulations = 1000, persist_tree = false)
        optim = mcts_descent_init(initial_p, loss_low, config)

        # Run until convergence
        converged = false
        for i in 1:20
            converged = step!(optim)
            if converged
                break
            end
        end

        # Verify convergence happened because the tree was solved, not timeout
        @test converged

        # The final parameter should be terminal: all radii == precision,
        # meaning children() returns an empty vector (no further refinement possible)
        final_param = optim.param
        @test all(r -> r == 3, NonArchimedeanMachineLearning.radius(final_param))
        @test isempty(children(final_param, 1))
    end

    @testset "DAG-MCTS converged parameter is terminal" begin
        K_low = PadicField(2, 3)
        R_low, x_low = polynomial_ring(K_low, ["x"])
        poly_low = AbsolutePolynomialSum([x_low[1]^2])
        PT = ValuationPolydisc{ValuedFieldPoint{2, 3, PadicFieldElem}, Int64, 1}
        batch_eval_low = batch_evaluate_init(poly_low, PT)

        loss_low = Loss(
            params -> [batch_eval_low(p) for p in params],
            vs -> [directional_derivative(poly_low, v) for v in vs]
        )

        initial_p = ValuationPolydisc([K_low(0)], [0])
        config = DAGMCTSConfig(num_simulations = 1000, persist_table = false, track_parents = true)
        optim = dag_mcts_descent_init(initial_p, loss_low, config)

        # Run until convergence
        converged = false
        for i in 1:20
            converged = step!(optim)
            if converged
                break
            end
        end

        # Verify convergence happened because the tree was solved, not timeout
        @test converged

        # The final parameter should be terminal: all radii == precision
        final_param = optim.param
        @test all(r -> r == 3, NonArchimedeanMachineLearning.radius(final_param))
        @test isempty(children(final_param, 1))

        # Transposition table should still be consistent
        @test verify_transposition_table(optim.state)
    end

    @testset "DAG-MCTS Stats Include Solved/Terminal Counts" begin
        K_low = PadicField(2, 3)
        R_low, x_low = polynomial_ring(K_low, ["x"])
        poly_low = AbsolutePolynomialSum([x_low[1]^2])
        PT = ValuationPolydisc{ValuedFieldPoint{2, 3, PadicFieldElem}, Int64, 1}
        batch_eval_low = batch_evaluate_init(poly_low, PT)

        loss_low = Loss(
            params -> [batch_eval_low(p) for p in params],
            vs -> [directional_derivative(poly_low, v) for v in vs]
        )

        initial_p = ValuationPolydisc([K_low(0)], [0])
        config = DAGMCTSConfig(num_simulations = 500, persist_table = true, track_parents = true)
        optim = dag_mcts_descent_init(initial_p, loss_low, config)

        for _ in 1:5
            step!(optim)
        end

        stats = get_dag_stats(optim.state)
        @test stats.unique_nodes == length(optim.state.transposition_table)
        @test stats.terminal_nodes > 0
        @test stats.solved_nodes >= stats.terminal_nodes  # solved ⊇ terminal
    end
end
