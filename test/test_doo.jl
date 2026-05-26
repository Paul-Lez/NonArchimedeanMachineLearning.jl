"""
Test file for DOO (Deterministic Optimistic Optimization) implementation.
"""

using Test
using NonArchimedeanMachineLearning
using Oscar

@testset "DOO Optimizer Tests" begin
    # Setup: 2-adic field and simple quadratic loss
    prec = 20
    K = PadicField(2, prec)
    R, (x, a) = polynomial_ring(K, ["x", "a"])

    # Model: f(x,a) = (x - a)^2
    # We want to learn a when given data where x = a (so loss should go to 0)
    fun = AbsolutePolynomialSum([(x - a)^2])
    model = AbstractModel(fun, [true, false])  # x is data, a is parameter

    # Data: single point at x = 3
    data = [(K(3), K(0))]  # (input, target)

    # Loss function
    loss = MSE_loss_init(model, data)

    # Initial parameter: start at a = 0
    # Use explicit tuple constructor to avoid auto-wrapping to ValuedFieldPoint,
    # which would cause a type mismatch with the PadicFieldElem evaluators
    param = ValuationPolydisc{PadicFieldElem, Int, 1}((K(0),), (0,))

    @testset "DOO Node Creation" begin
        node = DOONode(param, 0, 0, nothing)
        @test node.depth == 0
        @test node.position == 0
        @test node.parent === nothing
        @test isempty(node.children)
        @test node.value === nothing
        @test node.is_expanded == false
    end

    @testset "DOO Config Creation" begin
        # Define delta function: exponentially decreasing with depth
        delta = h -> 2.0^(-h)

        config = DOOConfig(
            delta = delta,
            degree = 1,
            strict = false
        )

        @test config.delta(0) == 1.0
        @test config.delta(1) == 0.5
        @test config.delta(2) == 0.25
        @test config.degree == 1
        @test config.strict == false
    end

    @testset "DOO Initialization and Basic Descent" begin
        # Define delta function
        delta = h -> 2.0^(-h)

        config = DOOConfig(
            delta = delta,
            degree = 1,
            strict = false
        )

        # Initialize optimizer
        optim = doo_descent_init(param, loss, 1, config)

        @test optim isa OptimSetup
        @test optim.state isa DOOState
        @test optim.context isa DOOConfig

        # Check initial state
        @test optim.state.root.value !== nothing  # Root should be evaluated
        @test optim.state.total_samples == 1  # Only root evaluated
        @test optim.state.step_count == 0
        @test length(optim.state.leaves) == 1  # Only root is a leaf

        # Take a few optimization steps
        initial_loss = eval_loss(optim)

        for i in 1:10
            step!(optim)
        end

        final_loss = eval_loss(optim)

        # Loss should improve
        @test final_loss < initial_loss

        # Tree should have grown
        @test get_tree_size(optim.state) > 1
        @test optim.state.total_samples > 1
    end

    @testset "DOO Utility Functions" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta, degree = 1)
        optim = doo_descent_init(param, loss, 1, config)

        # Run a few steps
        for i in 1:5
            step!(optim)
        end

        # Test utility functions
        tree_size = get_tree_size(optim.state)
        @test tree_size >= 1

        leaf_count = get_leaf_count(optim.state)
        @test leaf_count >= 1

        all_leaves = get_all_leaves(optim.state)
        @test length(all_leaves) >= 1

        best_node = get_best_node(optim.state)
        @test best_node !== nothing
        @test best_node.value !== nothing

        best_value = get_best_value(optim.state)
        @test best_value !== nothing
        @test best_value < 0  # Since value_transform converts loss to -loss
    end

    @testset "DOO B-value Computation" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta)

        # Create nodes at different depths
        node0 = DOONode(param, 0, 0, nothing)
        node0.value = 1.0

        node1 = DOONode(param, 1, 0, node0)
        node1.value = 1.0

        node2 = DOONode(param, 2, 0, node1)
        node2.value = 1.0

        # B-values should decrease with depth (since delta decreases)
        b0 = NonArchimedeanMachineLearning.b_value(node0, config)
        b1 = NonArchimedeanMachineLearning.b_value(node1, config)
        b2 = NonArchimedeanMachineLearning.b_value(node2, config)

        @test b0 == 1.0 + 1.0  # value + delta(0)
        @test b1 == 1.0 + 0.5  # value + delta(1)
        @test b2 == 1.0 + 0.25 # value + delta(2)

        @test b0 > b1 > b2

        # Unexplored node should have infinite b-value
        unexplored = DOONode(param, 0, 0, nothing)
        @test NonArchimedeanMachineLearning.b_value(unexplored, config) == Inf
    end

    @testset "DOO Strict Mode" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta, degree = 1, strict = true)
        optim = doo_descent_init(param, loss, 1, config)

        for _ in 1:3
            @test_nowarn step!(optim)
            @test 1 <= optim.state.next_branch <= NonArchimedeanMachineLearning.dim(param)
        end
        @test get_tree_size(optim.state) > 1
    end
end
