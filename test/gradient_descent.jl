# Test file for gradient descent optimization.
#
# This file demonstrates and tests the gradient descent optimization algorithm
# on a simple polynomial model in p-adic space.

include("../src/gradient_descent.jl")

# Set up synthetic data
prec = 20
K = PadicField(2, prec)
a1 = [K(0)]
r1 = Vector{Int}([0])
p1 = ValuationPolydisc(a1, r1)
p2 = ValuationPolydisc(Vector{PadicFieldElem}(), Vector{Int}())
data = [(p2, 1)]

# Create polynomial ring
R, (x,) = polynomial_ring(K, ["x"])

# Define a simple model: |x| where x is a parameter
fun = AbsolutePolynomialSum([x])
abs_model = AbstractModel(fun, [false])  # x is a parameter
model = Model(abs_model, p1)

# Define the loss function
loss = Loss(
    (params::Vector) -> [evaluate(abs_model, p2, param) for param in params],
    (vs::Vector) -> [gradient_param(abs_model, p2, v) for v in vs]
)

# Initialize gradient descent optimizer
optim = greedy_descent_init(model.param, loss, 1, (false, 1))

@show eval_loss(optim)

# Run gradient descent for 20 epochs
N_epochs = 20
for i in 1:N_epochs
    step!(optim)
    @show optim.param
    @show eval_loss(optim)
end

