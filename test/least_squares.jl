# Test file for ordinary least squares in non-Archimedean setting.
#
# This file tests two least squares functions:
# 1. make_ordinary_least_squares_loss: Learn matrix A and bias b for linear regression
# 2. solve_linear_system: Solve Ax + b = y for x

using Oscar
include("../src/NonArchimedeanMachineLearning.jl")
using .NonArchimedeanMachineLearning

println("=" ^ 80)
println("Test 1: Linear Regression (Learning A and b)")
println("=" ^ 80)

# Initialize 2-adic field
p, prec = 2, 20
K = PadicField(p, prec)

# Generate synthetic data for 1D linear regression: y = 2x + 3
# We'll create data points and try to recover A = [2] and b = [3]
true_A = 2
true_b = 3

data_points = [
    ([K(1)], [K(true_A * 1 + true_b)]),  # x=1, y=5
    ([K(2)], [K(true_A * 2 + true_b)]),  # x=2, y=7
    ([K(3)], [K(true_A * 3 + true_b)]),  # x=3, y=9
    ([K(4)], [K(true_A * 4 + true_b)])  # x=4, y=11
]

println("\nSynthetic data (1D linear regression):")
for (i, (x, y)) in enumerate(data_points)
    println("  Point $i: x = $(x[1]), y = $(y[1])")
end
println("\nTrue parameters: A = $true_A, b = $true_b")

# Create loss function for linear regression
loss_regression = make_ordinary_least_squares_loss(data_points)

# Initial parameter guess: [A, b] = [0, 0]
# For 1D case: 1 input, 1 output, so we have 1*1 + 1 = 2 parameters
initial_param = ValuationPolydisc([K(0), K(0)], [0, 0])

# Set up greedy descent optimizer
optim_regression = greedy_descent_init(initial_param, loss_regression, 1, (true, 1))

println("\nRunning greedy descent for linear regression...")
println("Initial loss: ", eval_loss(optim_regression))

# Run optimization
N_epochs = 30
for i in 1:N_epochs
    if i % 5 == 0 || i == 1
        println("  Epoch $i: loss = ", eval_loss(optim_regression))
    end
    step!(optim_regression)
end

println("\nFinal results:")
println("  Final loss: ", eval_loss(optim_regression))
println("  Learned parameters:")
println("    A = ", optim_regression.param.center[1])
println("    b = ", optim_regression.param.center[2])
println("  Parameter radii: ", optim_regression.param.radius)

println("\n" ^ 2)
println("=" ^ 80)
println("Test 2: Solving Linear System (Finding x given A, b, y)")
println("=" ^ 80)

# Test solving a simple 2x2 system:
# 2x₁ + 3x₂ = 8
# 1x₁ + 4x₂ = 10
# Solution should be x₁ = 2, x₂ = 2 (verify: 2*2+3*2=10, wait that's wrong)
# Let me recalculate: 2x₁ + 3x₂ = 8, x₁ + 4x₂ = 10
# From eq2: x₁ = 10 - 4x₂
# Substitute: 2(10-4x₂) + 3x₂ = 8 → 20 - 8x₂ + 3x₂ = 8 → -5x₂ = -12 → x₂ = 12/5 = 2.4
# x₁ = 10 - 4(2.4) = 10 - 9.6 = 0.4

# Let me use a simpler system with integer solution:
# 2x₁ + 1x₂ = 5
# 1x₁ + 1x₂ = 3
# Solution: subtract eq2 from eq1: x₁ = 2, then x₂ = 1

A = [K(2) K(1); K(1) K(1)]
b = [K(0), K(0)]
y = [K(5), K(3)]

println("\nSolving the system:")
println("  [2  1] [x₁]   [5]")
println("  [1  1] [x₂] = [3]")
println("\nExpected solution: x₁ = 2, x₂ = 1")

# Create loss function for solving the system
loss_system = solve_linear_system(A, b, y)

# Initial guess for x: [0, 0]
initial_x = ValuationPolydisc([K(0), K(0)], [0, 0])

# Set up greedy descent optimizer
optim_system = greedy_descent_init(initial_x, loss_system, 1, (true, 1))

println("\nRunning greedy descent for system solving...")
println("Initial loss: ", eval_loss(optim_system))

# Run optimization
N_epochs_system = 30
for i in 1:N_epochs_system
    if i % 5 == 0 || i == 1
        println("  Epoch $i: loss = ", eval_loss(optim_system))
    end
    step!(optim_system)
end

println("\nFinal results:")
println("  Final loss: ", eval_loss(optim_system))
println("  Learned solution:")
println("    x₁ = ", optim_system.param.center[1])
println("    x₂ = ", optim_system.param.center[2])
println("  Parameter radii: ", optim_system.param.radius)

# Verify the solution
x1 = optim_system.param.center[1]
x2 = optim_system.param.center[2]
result1 = K(2) * x1 + K(1) * x2
result2 = K(1) * x1 + K(1) * x2
println("\nVerification:")
println("  2x₁ + 1x₂ = ", result1, " (should be 5)")
println("  1x₁ + 1x₂ = ", result2, " (should be 3)")

println("\n" ^ 2)
println("=" ^ 80)
println("All tests completed!")
println("=" ^ 80)
