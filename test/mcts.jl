using Test
using Oscar
using NonArchimedeanMachineLearning

const NAML = NonArchimedeanMachineLearning

function mcts_square_loss(K)
    _, x = polynomial_ring(K, ["x"])
    poly = AbsolutePolynomialSum([x[1]^2])
    polydisc_type = typeof(ValuationPolydisc([K(0)], [0]))
    batch_eval = batch_evaluate_init(poly, polydisc_type)
    return Loss(
        params -> [batch_eval(p) for p in params],
        vs -> [directional_derivative(poly, v) for v in vs],
    )
end

@testset "MCTS" begin
    @testset "Node statistics and UCB1 scoring" begin
        K = PadicField(2, 20)
        param = ValuationPolydisc([K(1)], [0])

        node = MCTSNode(param)
        @test node.visits == 0
        @test node.total_value == 0.0
        @test node.min_loss == Inf
        @test NAML.average_value(node) == 0.0
        @test NAML.ucb1_score(node, 10, sqrt(2.0)) == Inf

        node.visits = 5
        node.total_value = 2.5
        score = NAML.ucb1_score(node, 100, sqrt(2.0))
        expected = 0.5 + sqrt(2.0) * sqrt(log(100) / 5)
        @test isapprox(score, expected; atol=1e-10)
    end

    @testset "Child selection prefers highest score" begin
        K = PadicField(2, 20)
        root = MCTSNode(ValuationPolydisc([K(0)], [1]))
        root.visits = 50

        low = MCTSNode(ValuationPolydisc([K(0)], [2]), root)
        low.visits = 10
        low.total_value = 1.0

        high = MCTSNode(ValuationPolydisc([K(1)], [2]), root)
        high.visits = 10
        high.total_value = 7.0

        push!(root.children, low, high)
        @test NAML.select_child(root, 1.0) === high

        solved = MCTSNode(ValuationPolydisc([K(2)], [2]), root)
        solved.is_solved = true
        solved.proven_value = 100.0
        push!(root.children, solved)
        @test NAML.select_child(root, 1.0) === solved
    end

    @testset "Expansion creates children and detects terminal nodes" begin
        K = PadicField(2, 3)
        config = MCTSConfig(num_simulations=5, degree=1, persist_tree=false)

        root = MCTSNode(ValuationPolydisc([K(0)], [1]))
        NAML.expand_node!(root, config)
        @test root.is_expanded
        @test !root.is_terminal
        @test !root.is_solved
        @test length(root.children) == 2
        @test all(child -> child.parent === root, root.children)
        @test root.unsolved_children_count == 2

        terminal = MCTSNode(ValuationPolydisc([K(0)], [3]))
        NAML.expand_node!(terminal, config)
        @test terminal.is_expanded
        @test terminal.is_terminal
        @test terminal.is_solved
        @test isempty(terminal.children)
        @test terminal.unsolved_children_count == 0
    end

    @testset "Path selection stops at unexpanded, terminal, and solved nodes" begin
        K = PadicField(2, 20)
        root = MCTSNode(ValuationPolydisc([K(0)], [1]))

        # Unexpanded root should be returned directly.
        @test NAML.select_path(root, 1.0) === root

        config = MCTSConfig(num_simulations=5, degree=1, persist_tree=false)
        NAML.expand_node!(root, config)
        root.visits = 10

        first_child = root.children[1]
        second_child = root.children[2]
        first_child.visits = 4
        first_child.total_value = 3.0
        second_child.visits = 4
        second_child.total_value = 1.0

        # Unexpanded child should be selected as the leaf.
        @test NAML.select_path(root, 1.0) === first_child

        first_child.is_solved = true
        first_child.proven_value = 10.0
        @test NAML.select_path(root, 1.0) === first_child

        root.is_solved = true
        @test NAML.select_path(root, 1.0) === root
    end

    @testset "Backpropagation updates visits totals and min_loss" begin
        K = PadicField(2, 20)
        root = MCTSNode(ValuationPolydisc([K(0)], [0]))
        child = MCTSNode(ValuationPolydisc([K(0)], [1]), root)
        grandchild = MCTSNode(ValuationPolydisc([K(0)], [2]), child)

        NAML.backpropagate!(grandchild, 0.75, 2.0)

        for node in (root, child, grandchild)
            @test node.visits == 1
            @test node.total_value == 0.75
            @test node.min_loss == 2.0
        end

        NAML.backpropagate!(grandchild, 0.25, 3.0)
        for node in (root, child, grandchild)
            @test node.visits == 2
            @test node.total_value == 1.0
            @test node.min_loss == 2.0
        end
    end

    @testset "Solved-status checks and propagation" begin
        K = PadicField(2, 3)
        root = MCTSNode(ValuationPolydisc([K(0)], [2]))
        config = MCTSConfig(num_simulations=5, degree=1, persist_tree=false)
        NAML.expand_node!(root, config)

        @test !NAML.check_solved!(root)

        for (i, child) in enumerate(root.children)
            NAML.expand_node!(child, config)
            @test child.is_terminal
            child.proven_value = -Float64(i)
        end

        for child in root.children
            NAML.propagate_solved_up!(child)
        end

        @test root.is_solved
        @test root.unsolved_children_count == 0
        @test root.proven_value == maximum(child.proven_value for child in root.children)
        @test !NAML.check_solved!(root)
    end

    @testset "Best-child selection modes choose correct root branch" begin
        K = PadicField(2, 20)
        root = MCTSNode(ValuationPolydisc([K(0)], [0]))
        child_a = MCTSNode(ValuationPolydisc([K(0)], [1]), root)
        child_b = MCTSNode(ValuationPolydisc([K(1)], [1]), root)
        push!(root.children, child_a, child_b)

        child_a.visits = 10
        child_a.total_value = 3.0
        child_a.min_loss = 2.0

        child_b.visits = 5
        child_b.total_value = 4.0
        child_b.min_loss = 4.0

        root.visits = child_a.visits + child_b.visits
        root.min_loss = min(child_a.min_loss, child_b.min_loss)

        grandchild_b = MCTSNode(ValuationPolydisc([K(1)], [2]), child_b)
        grandchild_b.visits = 3
        grandchild_b.total_value = 9.0
        grandchild_b.min_loss = 1.0
        push!(child_b.children, grandchild_b)

        visit_config = MCTSConfig(selection_mode=VisitCount, persist_tree=false)
        value_config = MCTSConfig(selection_mode=BestValue, persist_tree=false)
        loss_config = MCTSConfig(selection_mode=BestLoss, persist_tree=false)

        @test NAML.select_best_child(root, visit_config) === child_a
        @test NAML.select_best_child(root, value_config) === child_b
        @test NAML.select_best_child(root, loss_config) === child_b
    end

    @testset "Terminal root search returns immediately" begin
        K = PadicField(2, 3)
        loss = mcts_square_loss(K)
        root = MCTSNode(ValuationPolydisc([K(0)], [3]))
        config = MCTSConfig(num_simulations=10, persist_tree=false)

        best_polydisc, best_node, converged = NAML.mcts_search(root, loss, config)
        @test converged
        @test best_node === root
        @test best_polydisc == root.polydisc
        @test root.is_terminal
        @test root.is_solved
        @test !isnan(root.proven_value)
    end

    @testset "Tiny search space converges at terminal parameter" begin
        K = PadicField(2, 3)
        loss = mcts_square_loss(K)
        initial_param = ValuationPolydisc([K(0)], [0])
        config = MCTSConfig(num_simulations=1000, persist_tree=false, selection_mode=BestLoss)
        optim = mcts_descent_init(initial_param, loss, config)

        converged = false
        for _ in 1:20
            converged = step!(optim)
            converged && break
        end

        @test converged
        @test all(r -> r == 3, NonArchimedeanMachineLearning.radius(optim.param))
        @test isempty(children(optim.param, 1))
    end

    @testset "mcts_descent respects persist_tree setting" begin
        K = PadicField(2, 20)
        loss = mcts_square_loss(K)
        initial_param = ValuationPolydisc([K(4)], [0])

        persist_optim = mcts_descent_init(
            initial_param,
            loss,
            MCTSConfig(num_simulations=20, persist_tree=true, selection_mode=BestLoss),
        )
        step!(persist_optim)
        @test persist_optim.state.root.polydisc == persist_optim.param
        @test persist_optim.state.root.parent === nothing

        fresh_optim = mcts_descent_init(
            initial_param,
            loss,
            MCTSConfig(num_simulations=20, persist_tree=false, selection_mode=BestLoss),
        )
        step!(fresh_optim)
        @test fresh_optim.state.root.polydisc == fresh_optim.param
        @test isempty(fresh_optim.state.root.children)
        @test fresh_optim.state.root.visits == 0
    end
end
