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
using NAML

@testset "DAG-MCTS" begin
    prec = 20
    K = PadicField(2, prec)

    @testset "DAGMCTSNode Creation" begin
        p = ValuationPolydisc([K(1)], [0])
        node = DAGMCTSNode(p)

        @test node.polydisc == p
        @test isempty(node.parents)
        @test isempty(node.children)
        @test node.visits == 0
        @test node.total_value == 0.0
        @test !node.is_expanded
    end

    @testset "Transposition Table - Basic Operations" begin
        # Test get_or_create_node!
        # Note: Polydisc equality uses STRICT inequality: v(center_diff) > radius
        table = Dict{ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1}}()

        p1 = ValuationPolydisc([K(1)], [2])

        # First call should create new node
        node1 = NAML.get_or_create_node!(table, p1)
        @test length(table) == 1
        @test node1.polydisc == p1

        # Second call with same polydisc should return same node
        node2 = NAML.get_or_create_node!(table, p1)
        @test node1 === node2  # Same object reference
        @test length(table) == 1  # No new entry

        # Equivalent polydisc (v(diff) > radius) should return same node
        p1_equiv = ValuationPolydisc([K(1 + 8)], [2])  # v(8) = 3 > 2
        @test p1 == p1_equiv  # Verify they're equal
        node3 = NAML.get_or_create_node!(table, p1_equiv)
        @test node1 === node3  # Same object due to transposition
        @test length(table) == 1

        # Different polydisc should create new node
        p2 = ValuationPolydisc([K(2)], [2])
        node4 = NAML.get_or_create_node!(table, p2)
        @test node4 !== node1
        @test length(table) == 2
    end

    @testset "Transposition Table - Parent Linking" begin
        # Use 2D to create truly different parent nodes via different refinement paths
        table = Dict{ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}}()

        # Start with a root polydisc
        root = ValuationPolydisc([K(0), K(0)], [0, 0])
        root_node = NAML.get_or_create_node!(table, root)

        # Create two different parents by refining different coordinates
        parent1 = children_along_branch(root, 1)[1]  # Refine coordinate 1
        parent2 = children_along_branch(root, 2)[1]  # Refine coordinate 2

        parent1_node = NAML.get_or_create_node!(table, parent1, root_node)
        parent2_node = NAML.get_or_create_node!(table, parent2, root_node)

        # Verify parents are different
        @test parent1 != parent2
        @test parent1_node !== parent2_node

        # Create a child that both parents can reach
        child_p = ValuationPolydisc([K(0), K(0)], [1, 1])

        # Create child with first parent
        child_node = NAML.get_or_create_node!(table, child_p, parent1_node)
        @test length(child_node.parents) == 1
        @test parent1_node in child_node.parents

        # Link same child to second parent (transposition)
        child_node2 = NAML.get_or_create_node!(table, child_p, parent2_node)
        @test child_node === child_node2  # Same node instance
        @test length(child_node.parents) == 2
        @test parent1_node in child_node.parents
        @test parent2_node in child_node.parents
    end

    @testset "UCT Score Computation" begin
        p = ValuationPolydisc([K(1)], [0])
        node = DAGMCTSNode(p)

        # Unvisited node should have Inf score
        @test NAML.uct_score(node, 10, sqrt(2.0)) == Inf

        # Visited node should have finite score
        node.visits = 5
        node.total_value = 2.5  # average = 0.5
        parent_visits = 100
        c = sqrt(2.0)

        score = NAML.uct_score(node, parent_visits, c)
        expected = 0.5 + c * sqrt(log(parent_visits) / 5)
        @test abs(score - expected) < 1e-10
    end

    @testset "Node Expansion with Transposition Detection" begin
        table = Dict{ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}}()
        config = DAGMCTSConfig(num_simulations=10, degree=1)

        # Create 2D polydisc to test transpositions
        root_p = ValuationPolydisc([K(0), K(0)], [0, 0])
        root = NAML.get_or_create_node!(table, root_p)

        # Expand root
        NAML.expand_node!(root, table, config)
        @test root.is_expanded
        @test !isempty(root.children)

        initial_table_size = length(table)

        # Now expand one of the children - this may create nodes
        # that could be reached via a different path
        first_child = first(values(root.children))
        NAML.expand_node!(first_child, table, config)
        @test first_child.is_expanded
    end

    @testset "Backpropagation via Explicit Path" begin
        # Create a simple path of nodes
        p1 = ValuationPolydisc([K(0)], [0])
        p2 = ValuationPolydisc([K(0)], [1])
        p3 = ValuationPolydisc([K(0)], [2])

        node1 = DAGMCTSNode(p1)
        node2 = DAGMCTSNode(p2)
        node3 = DAGMCTSNode(p3)

        path = [node1, node2, node3]
        value = 0.75

        # Backpropagate
        NAML.backpropagate!(path, value)

        # All nodes should have 1 visit and the value
        for node in path
            @test node.visits == 1
            @test node.total_value == value
        end

        # Second backpropagation
        NAML.backpropagate!(path, 0.25)
        for node in path
            @test node.visits == 2
            @test node.total_value == 1.0  # 0.75 + 0.25
        end
    end

    @testset "DAG-MCTS Integration with OptimSetup" begin
        # Set up a simple optimization problem: minimize |x|^2
        R, x = polynomial_ring(K, ["x"])

        poly = AbsolutePolynomialSum([x[1]^2])
        batch_eval = batch_evaluate_init(poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})

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
            num_simulations=50,
            exploration_constant=1.41,
            degree=1,
            persist_table=false
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

    @testset "DAG Stats and Verification" begin
        R, x = polynomial_ring(K, ["x"])
        poly = AbsolutePolynomialSum([x[1]^2])
        batch_eval = batch_evaluate_init(poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})

        function loss_eval(params::Vector)
            return [batch_eval(p) for p in params]
        end
        function loss_grad(vs::Vector)
            return [directional_derivative(poly, v) for v in vs]
        end
        loss = Loss(loss_eval, loss_grad)

        initial_param = ValuationPolydisc([K(4)], [0])
        # Use persist_table=false for simpler verification
        config = DAGMCTSConfig(num_simulations=30, persist_table=false)

        optim = dag_mcts_descent_init(initial_param, loss, config)

        # Run optimization
        for _ in 1:3
            step!(optim)
        end

        # Get stats - with persist_table=false, table is reset each step
        # so we only check unique_nodes > 0 (the new root exists)
        stats = get_dag_stats(optim.state)
        @test stats.unique_nodes > 0

        # Verify table integrity
        @test verify_transposition_table(optim.state)
    end

    @testset "Transposition Detection in 2D (Diamond Pattern)" begin
        # This tests the classic DAG scenario: different paths to same state
        # We manually create the "same" polydisc to ensure transposition detection works
        # by using the hash-based lookup

        table = Dict{ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}, DAGMCTSNode{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 2}}()

        # Start point
        start = ValuationPolydisc([K(0), K(0)], [0, 0])
        start_node = NAML.get_or_create_node!(table, start)

        # Path A: shrink coord 1 first
        after_shrink_1 = children_along_branch(start, 1)[1]  # First child along branch 1
        node_a1 = NAML.get_or_create_node!(table, after_shrink_1, start_node)

        # Path B: shrink coord 2 first
        after_shrink_2 = children_along_branch(start, 2)[1]  # First child along branch 2
        node_b1 = NAML.get_or_create_node!(table, after_shrink_2, start_node)

        # Now create the "final" node that both paths should reach
        # This is radius (1, 1) with center (0, 0)
        final_polydisc = ValuationPolydisc([K(0), K(0)], [1, 1])

        # Add via path A
        node_final_via_a = NAML.get_or_create_node!(table, final_polydisc, node_a1)

        # Add via path B - should find the SAME node (transposition!)
        node_final_via_b = NAML.get_or_create_node!(table, final_polydisc, node_b1)

        # The key test: both paths should lead to the SAME node instance
        @test node_final_via_a === node_final_via_b  # Same node instance (transposition detected!)

        # The final node should have two parents
        @test length(node_final_via_a.parents) == 2
        @test node_a1 in node_final_via_a.parents
        @test node_b1 in node_final_via_a.parents

        # Check total unique nodes: start, 2 intermediate, 1 final = 4
        @test length(table) == 4
    end

    @testset "Persist Table Option" begin
        R, x = polynomial_ring(K, ["x"])
        poly = AbsolutePolynomialSum([x[1]^2])
        batch_eval = batch_evaluate_init(poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})

        function loss_eval(params::Vector)
            return [batch_eval(p) for p in params]
        end
        function loss_grad(vs::Vector)
            return [directional_derivative(poly, v) for v in vs]
        end
        loss = Loss(loss_eval, loss_grad)

        initial_param = ValuationPolydisc([K(4)], [0])

        # Test with persist_table=true
        config_persist = DAGMCTSConfig(num_simulations=20, persist_table=true)
        optim_persist = dag_mcts_descent_init(initial_param, loss, config_persist)

        step!(optim_persist)
        table_size_after_step1 = length(optim_persist.state.transposition_table)

        step!(optim_persist)
        table_size_after_step2 = length(optim_persist.state.transposition_table)

        # With persistence, table should grow (or stay same, not shrink)
        @test table_size_after_step2 >= table_size_after_step1

        # Test with persist_table=false
        config_no_persist = DAGMCTSConfig(num_simulations=20, persist_table=false)
        optim_no_persist = dag_mcts_descent_init(initial_param, loss, config_no_persist)

        step!(optim_no_persist)
        step!(optim_no_persist)

        # Without persistence, table is cleared each step
        # Should only contain nodes from current search
        @test length(optim_no_persist.state.transposition_table) > 0
    end

    @testset "MCTS Persist Tree Option" begin
        R, x = polynomial_ring(K, ["x"])
        poly = AbsolutePolynomialSum([x[1]^2])
        batch_eval = batch_evaluate_init(poly, ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 1})

        function mcts_loss_eval(params::Vector)
            return [batch_eval(p) for p in params]
        end
        function mcts_loss_grad(vs::Vector)
            return [directional_derivative(poly, v) for v in vs]
        end
        loss = Loss(mcts_loss_eval, mcts_loss_grad)

        initial_param = ValuationPolydisc([K(4)], [0])

        # Test with persist_tree=true
        config_persist = MCTSConfig(num_simulations=20, persist_tree=true)
        optim_persist = mcts_descent_init(initial_param, loss, config_persist)

        step!(optim_persist)
        step!(optim_persist)

        # With persistence, subtree is reused: root should have pre-existing children/stats
        @test optim_persist.state.root.parent === nothing
        @test get_tree_size(optim_persist.state.root) > 1

        # Test with persist_tree=false (default)
        config_no_persist = MCTSConfig(num_simulations=20, persist_tree=false)
        optim_no_persist = mcts_descent_init(initial_param, loss, config_no_persist)

        step!(optim_no_persist)
        step!(optim_no_persist)

        # Without persistence, root is fresh each step (no pre-existing subtree beyond current search)
        @test optim_no_persist.state.root.parent === nothing
    end
end
