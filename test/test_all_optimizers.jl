#!/usr/bin/env julia
########### Test All Optimizers ###########
#
# This script tests that all optimization algorithms work correctly
# with the new module structure by running each one for 10 steps
# on a simple polynomial function.

using Oscar
using NonArchimedeanMachineLearning
using Printf

# Import functions that may have conflicts
import NonArchimedeanMachineLearning: center, radius

println("="^70)
println("Testing All Optimizers")
println("="^70)
println()

# Setup: 2-adic field with precision
p, prec = 2, 20
K = PadicField(p, prec)
println("[1] Setup: 2-adic field with precision $prec")

# Create polynomial ring with variables: x (data), a (parameter)
# Simple model: f(x, a) = x - a
R, (x, a) = K["x", "a"]
g = AbsolutePolynomialSum([x - a])
println("[2] Function: f(x, a) = x - a")

# Create model: x is data variable (true), a is parameter (false)
f = AbstractModel(g, [true, false])

# Setting up training data: we want to find a such that f(x, a) ≈ 0 at x=5
# So the solution should be a ≈ 5
p1 = ValuationPolydisc([K(5)], [prec])  # x = 5
data = [(p1, 0)]  # target: f(5, a) = 0
println("[3] Training data: x=5, target=0 (solution should be a≈5)")

# Initial parameter: a = 11 (far from solution)
initial_param = ValuationPolydisc([K(11)], [0])
println("[4] Initial parameter: a=11")

# Create loss function
ell = MPE_loss_init(f, data, 2)
initial_loss = ell.eval([initial_param])[1]
@printf("[5] Initial loss: %.6f\n", initial_loss)
println()

##################################################
# Test Each Optimizer
##################################################

N_steps = 10

println("="^70)
println("Testing Greedy Descent")
println("="^70)
try
    optim = greedy_descent_init(ValuationPolydisc([K(11)], [0]), ell, 1, (false, 1))
    for i in 1:N_steps
        step!(optim)
    end
    final_loss = eval_loss(optim)
    @printf("✓ Greedy Descent: Final loss after %d steps: %.6f\n", N_steps, final_loss)
    println("  Final parameter: a = $(center(optim.param)[1])")
catch e
    println("✗ Greedy Descent failed: $e")
end
println()

println("="^70)
println("Testing Gradient Descent")
println("="^70)
try
    optim = gradient_descent_init(ValuationPolydisc([K(11)], [0]), ell, nothing, 1)
    for i in 1:N_steps
        step!(optim)
    end
    final_loss = eval_loss(optim)
    @printf("✓ Gradient Descent: Final loss after %d steps: %.6f\n", N_steps, final_loss)
    println("  Final parameter: a = $(center(optim.param)[1])")
catch e
    println("✗ Gradient Descent failed: $e")
end
println()

println("="^70)
println("Testing MCTS")
println("="^70)
try
    config = MCTSConfig(
        exploration_constant = 1.41,
        num_simulations = 10,
        degree = 1,
        strict = false,
        selection_mode = BestValue
    )
    optim = mcts_descent_init(ValuationPolydisc([K(11)], [0]), ell, config)
    for i in 1:N_steps
        step!(optim)
    end
    final_loss = eval_loss(optim)
    @printf("✓ MCTS: Final loss after %d steps: %.6f\n", N_steps, final_loss)
    println("  Final parameter: a = $(center(optim.param)[1])")
catch e
    println("✗ MCTS failed: $e")
end
println()

println("="^70)
println("Testing HOO")
println("="^70)
try
    config = HOOConfig(
        rho = 0.5,
        nu1 = 1.0,
        max_depth = 10,
        degree = 1,
        strict = false
    )
    optim = hoo_descent_init(ValuationPolydisc([K(11)], [0]), ell, config)
    for i in 1:N_steps
        step!(optim)
    end
    final_loss = eval_loss(optim)
    @printf("✓ HOO: Final loss after %d steps: %.6f\n", N_steps, final_loss)
    println("  Final parameter: a = $(center(optim.param)[1])")
catch e
    println("✗ HOO failed: $e")
end
println()

println("="^70)
println("Testing UCT")
println("="^70)
try
    config = UCTConfig(
        exploration_constant = 1.41,
        num_simulations = 10,
        degree = 1,
        strict = false
    )
    optim = uct_descent_init(ValuationPolydisc([K(11)], [0]), ell, config)
    for i in 1:N_steps
        step!(optim)
    end
    final_loss = eval_loss(optim)
    @printf("✓ UCT: Final loss after %d steps: %.6f\n", N_steps, final_loss)
    println("  Final parameter: a = $(center(optim.param)[1])")
catch e
    println("✗ UCT failed: $e")
end
println()

println("="^70)
println("Testing Modified UCT")
println("="^70)
try
    config = ModifiedUCTConfig(
        beta = 1.0,
        num_simulations = 10,
        degree = 1,
        strict = false
    )
    optim = modified_uct_descent_init(ValuationPolydisc([K(11)], [0]), ell, config)
    for i in 1:N_steps
        step!(optim)
    end
    final_loss = eval_loss(optim)
    @printf("✓ Modified UCT: Final loss after %d steps: %.6f\n", N_steps, final_loss)
    println("  Final parameter: a = $(center(optim.param)[1])")
catch e
    println("✗ Modified UCT failed: $e")
end
println()

println("="^70)
println("Testing Flat UCB")
println("="^70)
try
    config = FlatUCBConfig(
        beta = 1.0,
        num_simulations = 10,
        degree = 1,
        strict = false
    )
    optim = flat_ucb_descent_init(ValuationPolydisc([K(11)], [0]), ell, config)
    for i in 1:N_steps
        step!(optim)
    end
    final_loss = eval_loss(optim)
    @printf("✓ Flat UCB: Final loss after %d steps: %.6f\n", N_steps, final_loss)
    println("  Final parameter: a = $(center(optim.param)[1])")
catch e
    println("✗ Flat UCB failed: $e")
end
println()

println("="^70)
println("All optimizer tests complete!")
println("="^70)
