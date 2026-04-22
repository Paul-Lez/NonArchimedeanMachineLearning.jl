# Debug test for least squares implementation
# Let's test the basic functionality step by step

using Oscar
include("../src/NonArchimedeanMachineLearning.jl")
using .NonArchimedeanMachineLearning

println("=" ^ 80)
println("Debug Test: Building and evaluating loss function manually")
println("=" ^ 80)

# Initialize 2-adic field
p, prec = 2, 15
K = PadicField(p, prec)

# Very simple test: 1D linear regression with 1 data point
# Data: x = 1, y = 3 (so we want A*1 + b = 3, many solutions work)
data_points = [
    ([K(1)], [K(3)]),  # x=1, y=3
]

println("\n1. Creating loss function...")
loss = make_ordinary_least_squares_loss(data_points)
println("   Loss function created successfully")

# Test evaluation at different parameter values
println("\n2. Testing loss evaluation at different parameter points:")

# Test 1: A=0, b=0 → prediction = 0, residual = -3, loss = |-3|_2^2 = 1^2 = 1
param1 = ValuationPolydisc([K(0), K(0)], [100, 100])
loss_val1 = loss.eval([param1])[1]
println("   At A=0, b=0: loss = ", loss_val1)

# Test 2: A=0, b=3 → prediction = 3, residual = 0, loss = 0
param2 = ValuationPolydisc([K(0), K(3)], [100, 100])
loss_val2 = loss.eval([param2])[1]
println("   At A=0, b=3: loss = ", loss_val2)

# Test 3: A=3, b=0 → prediction = 3, residual = 0, loss = 0
param3 = ValuationPolydisc([K(3), K(0)], [100, 100])
loss_val3 = loss.eval([param3])[1]
println("   At A=3, b=0: loss = ", loss_val3)

# Test 4: A=1, b=2 → prediction = 1+2=3, residual = 0, loss = 0
param4 = ValuationPolydisc([K(1), K(2)], [100, 100])
loss_val4 = loss.eval([param4])[1]
println("   At A=1, b=2: loss = ", loss_val4)

# Test 5: A=1, b=1 → prediction = 1+1=2, residual = -1, loss = |-1|_2^2 = 1^2 = 1
param5 = ValuationPolydisc([K(1), K(1)], [100, 100])
loss_val5 = loss.eval([param5])[1]
println("   At A=1, b=1: loss = ", loss_val5)

println("\n3. Testing gradient computation:")

# # Create a tangent vector at param1 (A=0, b=0)
# # Direction: move b towards 1
# tangent1 = ValuationTangent(param1, [K(0), K(1)], [0, 0])
# grad_val1 = loss.grad([tangent1])[1]
# println("   Gradient in b-direction at A=0,b=0: ", grad_val1)

# # Direction: move A towards 1
# tangent2 = ValuationTangent(param1, [K(1), K(0)], [0, 0])
# grad_val2 = loss.grad([tangent2])[1]
# println("   Gradient in A-direction at A=0,b=0: ", grad_val2)

println("\n4. Testing solve_linear_system:")

# Simple 1D system: 2x = 4, solution x = 2
A_simple = reshape([K(2)], 1, 1)
b_simple = [K(0)]
y_simple = [K(4)]

println("   Solving: 2x = 4")
loss_system = solve_linear_system(A_simple, b_simple, y_simple)

# Test at x=0
x_param1 = ValuationPolydisc([K(0)], [100])
loss_sys1 = loss_system.eval([x_param1])[1]
println("   At x=0: loss = ", loss_sys1, " (residual = -4, loss = |-4|_2^2 = (1/4)^2 = 0.0625)")

# Test at x=2
x_param2 = ValuationPolydisc([K(2)], [100])
loss_sys2 = loss_system.eval([x_param2])[1]
println("   At x=2: loss = ", loss_sys2, " (should be 0)")

# Test at x=1
x_param3 = ValuationPolydisc([K(1)], [100])
loss_sys3 = loss_system.eval([x_param3])[1]
println("   At x=1: loss = ", loss_sys3, " (residual = -2, loss = |-2|_2^2 = (1/2)^2 = 0.25)")

println("\n" * "=" ^ 80)
println("Debug test completed!")
println("=" ^ 80)
