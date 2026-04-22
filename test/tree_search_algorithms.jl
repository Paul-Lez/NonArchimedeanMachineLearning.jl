using Test
using Oscar
using NAML

function square_loss(K)
    _, x = polynomial_ring(K, ["x"])
    poly = AbsolutePolynomialSum([x[1]^2])
    polydisc_type = typeof(ValuationPolydisc([K(0)], [0]))
    batch_eval = batch_evaluate_init(poly, polydisc_type)
    return Loss(
        params -> [batch_eval(p) for p in params],
        vs -> [directional_derivative(poly, v) for v in vs]
    )
end

@testset "UCT" begin
    K = PadicField(2, 20)
    test_polydisc = ValuationPolydisc([K(1)], [0])

    @testset "Node scoring and expansion" begin
        node = UCTNode(test_polydisc, nothing, 0)
        @test node.visits == 0
        @test node.depth == 0
        @test NAML.average_value(node) == 0.0

        parent = UCTNode(test_polydisc, nothing, 0)
        parent.visits = 100

        unvisited_child = UCTNode(test_polydisc, parent, 1)
        @test NAML.ucb_score(unvisited_child, parent.visits, sqrt(2.0)) == Inf

        visited_child = UCTNode(test_polydisc, parent, 1)
        visited_child.visits = 10
        visited_child.total_value = 5.0
        score = NAML.ucb_score(visited_child, parent.visits, sqrt(2.0))
        expected = 0.5 + sqrt(2.0) * sqrt(2.0 * log(100) / 10)
        @test isapprox(score, expected; atol=1e-10)

        root = UCTNode(ValuationPolydisc([K(1)], [1]), nothing, 0)
        config = UCTConfig(max_depth=3, num_simulations=10, degree=1)
        @test isempty(root.children)

        NAML.expand_node!(root, config)
        @test length(root.children) == 2
        @test all(child -> child.depth == 1, root.children)
        @test all(child -> child.parent === root, root.children)

        path = [root, root.children[1]]
        initial_visits = [path_node.visits for path_node in path]
        NAML.backpropagate!(path, 0.75)

        for (i, path_node) in enumerate(path)
            @test path_node.visits == initial_visits[i] + 1
            @test path_node.total_value == 0.75
        end

        NAML.backpropagate!(path, 0.25)
        @test root.visits == 2
        @test isapprox(NAML.average_value(root), 0.5; atol=1e-10)
    end

    @testset "Simple optimization" begin
        loss = square_loss(K)
        initial_param = ValuationPolydisc([K(16)], [0])
        config = UCTConfig(
            max_depth=8,
            num_simulations=100,
            exploration_constant=sqrt(2.0),
            degree=1,
            value_transform=loss_value -> 1.0 / (loss_value + 1e-10)
        )

        optim = uct_descent_init(initial_param, loss, config)
        initial_loss = eval_loss(optim)

        for _ in 1:15
            step!(optim)
        end

        final_loss = eval_loss(optim)
        @test isfinite(final_loss)
        @test final_loss <= initial_loss
    end
end

@testset "Modified UCT" begin
    K = PadicField(2, 20)

    @testset "Node scoring and expansion" begin
        param = ValuationPolydisc([K(5)], [3])
        node = ModifiedUCTNode(param, nothing, 0)
        @test node.visits == 0
        @test node.total_value == 0.0
        @test node.depth == 0
        @test NAML.average_value(node) == 0.0

        D = 5
        config = ModifiedUCTConfig(max_depth=D, num_simulations=10)

        sqrt2 = sqrt(2.0)
        expected_k_0 = ((1 + sqrt2) / sqrt2) * ((1 + sqrt2)^D - 1)
        @test isapprox(config.k_coeffs[1], expected_k_0; atol=1e-10)
        @test isapprox(config.k_coeffs[D + 1], 0.0; atol=1e-10)

        expected_k_prime_0 = (3.0^D - 1) / 2
        @test isapprox(config.k_prime_coeffs[1], expected_k_prime_0; atol=1e-10)

        @test isinf(NAML.modified_ucb_score(node, config))

        node.visits = 5
        node.total_value = 10.0
        score = NAML.modified_ucb_score(node, config)
        @test isfinite(score)
        @test score > NAML.average_value(node)

        NAML.expand_node!(node, config)
        @test length(node.children) == 2
        @test all(child -> child.depth == 1, node.children)
        @test all(child -> child.parent === node, node.children)

        path = [node, node.children[1]]
        initial_visits = [path_node.visits for path_node in path]
        initial_totals = [path_node.total_value for path_node in path]
        NAML.backpropagate!(path, 3.5)

        for (i, path_node) in enumerate(path)
            @test path_node.visits == initial_visits[i] + 1
            @test path_node.total_value == initial_totals[i] + 3.5
        end
    end

    @testset "Simple optimization" begin
        loss = square_loss(K)
        param = ValuationPolydisc([K(16)], [5])
        config = ModifiedUCTConfig(
            max_depth=10,
            num_simulations=50,
            beta=0.05,
            degree=1
        )

        optim = modified_uct_descent_init(param, loss, config)
        for _ in 1:15
            step!(optim)
        end

        @test isfinite(eval_loss(optim))
    end
end

@testset "Flat UCB" begin
    K = PadicField(2, 20)

    @testset "Node scoring and expansion" begin
        param = ValuationPolydisc([K(0)], [5])
        node = FlatUCBNode(param, nothing, 0)
        @test node.depth == 0
        @test node.visits == 0
        @test node.b_value == Inf
        @test NAML.is_leaf(node)

        config = FlatUCBConfig(max_depth=5, num_simulations=10, beta=0.05)
        @test NAML.compute_leaf_b_value(node, config) == Inf

        node.visits = 10
        node.total_value = 5.0
        avg = NAML.average_value(node)
        @test avg == 0.5

        b_val = NAML.compute_leaf_b_value(node, config)
        @test b_val > avg
        @test isfinite(b_val)

        children = NAML.expand_node!(node, config)
        @test length(children) == 2
        @test !NAML.is_leaf(node)
        @test all(child -> child.parent === node, children)
        @test all(child -> child.depth == node.depth + 1, children)

        for (i, child) in enumerate(children)
            child.b_value = Float64(i)
        end

        internal_b = NAML.compute_internal_b_value(node)
        @test internal_b == Float64(length(children))

        root = FlatUCBNode(param, nothing, 0)
        root.visits = 1
        root.total_value = 1.0
        NAML.update_b_value!(root, config)

        NAML.expand_node!(root, config)
        child = root.children[1]
        child.visits = 5
        child.total_value = 2.5
        NAML.update_b_value!(child, config)
        NAML.propagate_b_values!(child, config)

        @test root.b_value == maximum(root_child.b_value for root_child in root.children)
    end

    @testset "Simple optimization" begin
        loss = square_loss(K)
        param = ValuationPolydisc([K(16)], [0])
        config = FlatUCBConfig(
            max_depth=10,
            num_simulations=50,
            beta=0.05,
            degree=1,
            strict=false
        )

        optim = flat_ucb_descent_init(param, loss, config)
        initial_loss = eval_loss(optim)

        for _ in 1:20
            step!(optim)
        end

        final_loss = eval_loss(optim)
        @test isfinite(final_loss)
        @test final_loss <= initial_loss * 1.1
    end
end
