"""
Test file for DOO (Deterministic Optimistic Optimization) implementation.
"""

using Test
using NonArchimedeanMachineLearning
using Oscar
using DataStructures: PriorityQueue, peek

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

    @testset "DOO State Requires Evaluated Root" begin
        root = DOONode(param, 0, 0, nothing)
        @test_throws Exception DOOState{PadicFieldElem, Int, 1}(root)

        root.value = 0.0
        state = DOOState{PadicFieldElem, Int, 1}(root)
        @test state.best_node === root

        state.best_node = nothing
        @test_throws Exception get_best_value(state)
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

    @testset "DOO Invalid Degree Fails at Initialization" begin
        delta = h -> 2.0^(-h)
        param2 = ValuationPolydisc{PadicFieldElem, Int, 2}((K(0), K(0)), (0, 0))
        flat_loss = Loss(ps -> zeros(length(ps)), ts -> zeros(length(ts)))

        @test_throws Exception doo_descent_init(
            param2, flat_loss, 1, DOOConfig(delta = delta, degree = 0))
        @test_throws Exception doo_descent_init(
            param2, flat_loss, 1, DOOConfig(delta = delta, degree = 3))
        @test_throws Exception doo_descent_init(
            param2, flat_loss, 1, DOOConfig(delta = delta, degree = 3, strict = true))
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
        @test optim.state.best_node === optim.state.root
        @test optim.state.total_samples == 1  # Only root evaluated
        @test optim.state.step_count == 0
        @test optim.state.leaves isa PriorityQueue
        @test length(optim.state.leaves) == 1  # Only root is a leaf
        # Non-strict DOO does not use the fixed-subset coordinate schedule.
        @test isempty(optim.state.branch_sets)

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

    @testset "DOO Converges When Best Leaf Has No Children" begin
        delta = h -> 2.0^(-h)
        flat_loss = Loss(ps -> zeros(length(ps)), ts -> zeros(length(ts)))
        terminal_param = ValuationPolydisc{PadicFieldElem, Int, 1}((K(0),), (prec,))

        optim = doo_descent_init(terminal_param, flat_loss, 1, DOOConfig(delta = delta))
        @test step!(optim)
        @test has_converged(optim)
        @test optim.state.root.is_expanded
        @test isempty(optim.state.leaves)
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

        best_node = optim.state.best_node
        @test best_node !== nothing
        @test best_node === optim.state.best_node
        @test best_node.value !== nothing

        best_value = get_best_value(optim.state)
        @test best_value !== nothing
        @test best_value < 0  # Since value_transform converts loss to -loss
    end

    @testset "DOO Cached Best Node" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta, degree = 1)
        optim = doo_descent_init(param, loss, 1, config)

        for _ in 1:5
            step!(optim)
        end

        values = Float64[]
        function collect_values(node)
            if node.value !== nothing
                push!(values, node.value)
            end
            for child in node.children
                collect_values(child)
            end
        end

        collect_values(optim.state.root)
        @test optim.state.best_node.value == maximum(values)

        unevaluated = DOONode(param, 1, 0, optim.state.root)
        @test_throws Exception NonArchimedeanMachineLearning.update_best_node!(
            optim.state, unevaluated)
    end

    @testset "DOO Leaf Queue Prioritizes Greatest B-value" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta, degree = 1)
        optim = doo_descent_init(param, loss, 1, config)

        @test_nowarn step!(optim)

        queued_children = collect(keys(optim.state.leaves))
        @test Set(queued_children) == Set(optim.state.root.children)

        top_leaf = first(peek(optim.state.leaves))
        top_b = NonArchimedeanMachineLearning.b_value(top_leaf, config)
        child_b_values = [
            NonArchimedeanMachineLearning.b_value(child, config) for child in queued_children
        ]

        @test top_b == maximum(child_b_values)
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

    @testset "DOO Strict Mode Uses Coordinate-wise Bound" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta, strict = true)
        param2 = ValuationPolydisc{PadicFieldElem, Int, 2}((K(0), K(0)), (0, 0))

        node0 = DOONode(param2, 0, 0, nothing)
        node0.value = 1.0

        node1 = DOONode(param2, 1, 0, node0)
        node1.value = 1.0

        node2 = DOONode(param2, 2, 0, node1)
        node2.value = 1.0

        node3 = DOONode(param2, 3, 0, node2)
        node3.value = 1.0

        @test NonArchimedeanMachineLearning.b_value(node0, config) == 2.0
        @test NonArchimedeanMachineLearning.b_value(node1, config) == 2.0
        @test NonArchimedeanMachineLearning.b_value(node2, config) == 1.5
        @test NonArchimedeanMachineLearning.b_value(node3, config) == 1.5
    end

    @testset "DOO Strict Mode Uses Degree-d Bound" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta, degree = 2, strict = true)
        param3 = ValuationPolydisc{PadicFieldElem, Int, 3}((K(0), K(0), K(0)), (0, 0, 0))

        nodes = [DOONode(param3, depth, 0, nothing) for depth in 0:6]
        foreach(node -> node.value = 1.0, nodes)

        @test NonArchimedeanMachineLearning.doo_bound_depth(nodes[1], config) == 0
        @test NonArchimedeanMachineLearning.doo_bound_depth(nodes[4], config) == 2
        @test NonArchimedeanMachineLearning.doo_bound_depth(nodes[7], config) == 4
        @test NonArchimedeanMachineLearning.b_value(nodes[1], config) == 2.0
        @test NonArchimedeanMachineLearning.b_value(nodes[2], config) == 2.0
        @test NonArchimedeanMachineLearning.b_value(nodes[3], config) == 2.0
        @test NonArchimedeanMachineLearning.b_value(nodes[4], config) == 1.25
        @test NonArchimedeanMachineLearning.b_value(nodes[7], config) == 1.0625
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

    @testset "DOO Strict Mode Coordinate Schedule" begin
        delta = h -> 2.0^(-h)
        config = DOOConfig(delta = delta, degree = 1, strict = true)
        param2 = ValuationPolydisc{PadicFieldElem, Int, 2}((K(0), K(0)), (0, 0))
        flat_loss = Loss(ps -> zeros(length(ps)), ts -> zeros(length(ts)))

        optim = doo_descent_init(param2, flat_loss, 1, config)
        @test_nowarn step!(optim)
        @test optim.state.next_branch == 1
        @test all(child.polydisc.radius == (1, 0) for child in optim.state.root.children)

        @test_nowarn step!(optim)
        expanded_child = optim.state.root.children[1]
        @test expanded_child.is_expanded
        @test all(child.polydisc.radius == (1, 1) for child in expanded_child.children)
        @test optim.state.next_branch == 1

        offset_optim = doo_descent_init(param2, flat_loss, 2, config)
        @test_nowarn step!(offset_optim)
        @test all(child.polydisc.radius == (0, 1)
            for child in offset_optim.state.root.children)
    end

    @testset "DOO Degree-d Strict and Non-strict Expansion" begin
        delta = h -> 2.0^(-h)
        param3 = ValuationPolydisc{PadicFieldElem, Int, 3}((K(0), K(0), K(0)), (0, 0, 0))
        flat_loss = Loss(ps -> zeros(length(ps)), ts -> zeros(length(ts)))

        # Strict mode starting at the first pair of radii to shrink: [1, 2].
        strict_config = DOOConfig(delta = delta, degree = 2, strict = true)
        strict_optim = doo_descent_init(param3, flat_loss, 1, strict_config)
        @test_nowarn step!(strict_optim)
        @test length(strict_optim.state.root.children) == 4
        @test all(child.polydisc.radius == (1, 1, 0)
            for child in strict_optim.state.root.children)

        @test strict_optim.state.branch_sets == [[1, 2], [1, 3], [2, 3]]
        @test NonArchimedeanMachineLearning.strict_branch_indices(
            strict_optim.state.root.children[1], strict_optim.state) == [1, 3]

        # Starting at the second pair means we shrink radii 1 and 3.
        offset_optim = doo_descent_init(param3, flat_loss, 2, strict_config)
        @test_nowarn step!(offset_optim)
        @test all(child.polydisc.radius == (1, 0, 1)
            for child in offset_optim.state.root.children)

        # Non-strict mode allows shrinking along any degree-2 coordinate subset.
        nonstrict_config = DOOConfig(delta = delta, degree = 2, strict = false)
        nonstrict_optim = doo_descent_init(param3, flat_loss, 1, nonstrict_config)
        @test_nowarn step!(nonstrict_optim)
        @test length(nonstrict_optim.state.root.children) == 12
        @test Set(child.polydisc.radius for child in nonstrict_optim.state.root.children) ==
              Set([(1, 1, 0), (1, 0, 1), (0, 1, 1)])
    end
end
