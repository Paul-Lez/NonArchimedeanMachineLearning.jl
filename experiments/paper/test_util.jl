"""
Test script for util.jl functions

This script tests all the utility functions to ensure they work correctly.
"""

include("../../src/NAML.jl")
include("util.jl")

using Oscar
using .NAML

println("="^60)
println("Testing NAML Utility Functions")
println("="^60)

# Setup
prec = 20
p = 2
K = PadicField(p, prec)

println("\n" * "="^60)
println("Test 1: generate_random_padic")
println("="^60)

# Test with positive exponents
println("\nTest 1a: Positive exponents (min_exp=0, num_terms=5)")
x1 = generate_random_padic(2, 20, 0, 5)
println("Generated p-adic: $x1")
println("Type: $(typeof(x1))")

# Test with negative exponents (this is the critical case!)
println("\nTest 1b: Negative exponents (min_exp=-2, num_terms=5)")
x2 = generate_random_padic(2, 20, -2, 5)
println("Generated p-adic: $x2")
println("Valuation: $(Oscar.valuation(x2))")

# Test with large number of terms
println("\nTest 1c: Many terms (min_exp=-3, num_terms=10)")
x3 = generate_random_padic(2, 20, -3, 10)
println("Generated p-adic: $x3")

println("\n" * "="^60)
println("Test 2: mk_cutoff")
println("="^60)

cutoff = mk_cutoff(0.5)
println("Cutoff at 0.5:")
println("  cutoff(0.3) = $(cutoff(0.3)) (expected: 0)")
println("  cutoff(0.5) = $(cutoff(0.5)) (expected: 1)")
println("  cutoff(0.7) = $(cutoff(0.7)) (expected: 1)")

println("\n" * "="^60)
println("Test 3: generate_gauss_point")
println("="^60)

println("\nTest 3a: Standard Gauss point (n=5)")
gauss1 = generate_gauss_point(5, K)
println("Gauss point: $gauss1")
println("Center: $(NAML.center(gauss1))")
println("Radius: $(NAML.radius(gauss1))")

println("\nTest 3b: Gauss point with Float64 radius (n=3)")
gauss2 = generate_gauss_point(3, K, Float64)
println("Gauss point: $gauss2")
println("Radius type: $(typeof(NAML.radius(gauss2)))")

println("\n" * "="^60)
println("Test 4: polynomial_fitting_data")
println("="^60)

x = K(3)
y = 1.0
powers, output = polynomial_fitting_data(x, y, 3)
println("Input: x = $x, y = $y, degree = 3")
println("Powers: $powers")
println("Output: $output")

println("\n" * "="^60)
println("Test 5: generate_random_binary_function")
println("="^60)

binary_vals = generate_random_binary_function(10)
println("Generated 10 binary values: $binary_vals")
println("All in {0,1}? $(all(v -> v in [0.0, 1.0], binary_vals))")

println("\n" * "="^60)
println("Test 6: generate_polynomial_learning_data")
println("="^60)

data = generate_polynomial_learning_data(2, 20, 5, -2, 10)
println("Generated $(length(data)) data points:")
for (i, (x, y)) in enumerate(data)
    println("  Point $i: x = $x, y = $y")
end

# Check that all x values are distinct
x_vals = [x for (x, y) in data]
all_distinct = length(x_vals) == length(unique(x_vals))
println("\nAll x values distinct? $all_distinct")

# Test with larger dataset
println("\nGenerating larger dataset (20 points) to stress-test uniqueness...")
large_data = generate_polynomial_learning_data(2, 20, 20, -2, 15)
large_x_vals = [x for (x, y) in large_data]
large_distinct = length(large_x_vals) == length(unique(large_x_vals))
println("Generated $(length(large_data)) points, all distinct? $large_distinct")

println("\n" * "="^60)
println("Test 7: polynomial_to_linear_loss (p-adic outputs)")
println("="^60)

# Create sample data with p-adic outputs
data_padic = [(K(1), K(2)), (K(2), K(5)), (K(3), K(10))]
degree = 2

println("Data (x, y) where y is p-adic:")
for (x, y) in data_padic
    println("  ($x, $y)")
end

loss_padic = polynomial_to_linear_loss(data_padic, degree, nothing)
println("\nCreated loss function for p-adic outputs")

# Test evaluation at Gauss point
gauss_test = generate_gauss_point(degree + 1, K)
loss_val = loss_padic.eval([gauss_test])
println("Loss at Gauss point ($(degree+1) dims): $loss_val")

println("\n" * "="^60)
println("Test 8: polynomial_to_linear_loss (real outputs)")
println("="^60)

# Create sample data with real outputs
data_real = [(K(1), 0.0), (K(2), 1.0), (K(3), 1.0)]
cutoff_val = 0.25

println("Data (x, y) where y is real:")
for (x, y) in data_real
    println("  ($x, $y)")
end
println("Cutoff value: $cutoff_val")

loss_real = polynomial_to_linear_loss(data_real, degree, cutoff_val)
println("\nCreated loss function for real outputs with cutoff")

# Test evaluation at Gauss point
loss_val_real = loss_real.eval([gauss_test])
println("Loss at Gauss point: $loss_val_real")

println("\n" * "="^60)
println("Test 9: Full optimization example")
println("="^60)

# Generate synthetic data
println("\nGenerating synthetic polynomial learning data...")
n_points = 3
poly_degree = 4
test_data = generate_polynomial_learning_data(2, 20, n_points, -2, 10)

println("Generated $n_points data points for degree-$poly_degree polynomial")

# Create loss function
loss = polynomial_to_linear_loss(test_data, poly_degree, 0.25)
println("Created loss function")

# Initialize at Gauss point
initial_param = generate_gauss_point(poly_degree + 1, K)
println("Initial parameter: Gauss point in $(poly_degree + 1) dimensions")

# Evaluate initial loss
initial_loss = loss.eval([initial_param])
println("Initial loss: $initial_loss")

# Run a few steps of greedy descent
println("\nRunning 5 steps of greedy descent...")
optim = NAML.greedy_descent_init(initial_param, loss, 1, (true, 0))

for i in 1:5
    step_loss = NAML.eval_loss(optim)
    println("  Step $i: loss = $step_loss")
    NAML.step!(optim)
end

final_loss = NAML.eval_loss(optim)
println("Final loss after 5 steps: $final_loss")
println("Final parameter radius: $(NAML.radius(optim.param))")

println("\n" * "="^60)
println("All tests completed successfully!")
println("="^60)
