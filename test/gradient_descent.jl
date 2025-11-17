include("../src/gradient_descent.jl")


# set up some random synthetic data
prec = 20
K = PadicField(2, prec)
a1 = [K(0)]
r1 = Vector{Int}([0])
#a2 = [K(2)]
#r2 = Vector{Int}([2, 2])
p1 = ValuationPolydisc(a1, r1)
p2 = ValuationPolydisc(Vector{PadicFieldElem}(), Vector{Int}())
data = [(p2, 1)]
R, (x,) = polynomial_ring(K, ["x"])

# Take the model to be the absolute polynomial sum f(x, y, θ, ϕ) = |3x+3θ+ϕ^15| + |2y+81θ+2ϕ|
fun = AbsolutePolynomialSum([x])
# Specify which variables are the model's parameters and which are the data variables
abs_model = AbstractModel(fun, [false])
# Set the initial value of the model's parameters to be the point p1 in the Polydisc space
model = Model(abs_model, p1)
# Pick the loss function (batch interface)
loss = Loss(
    (params::Vector) -> [eval_abs(abs_model, p2, param) for param in params],
    (vs::Vector) -> [gradient_param(abs_model, p2, v) for v in vs]
)
# Package the data of the model + loss function into an "optimiser object".
# This has a built in optimisation algorithm: greedy descent
optim = greedy_descent_init(model.param, loss, 1, (false, 1))

@show eval_loss(optim)
N_epochs = 20
# Now optimise for 20 epochs
for i in 1:N_epochs
    # Make a step in the direction given by the optimiser and update the parameters
    step!(optim)
    @show optim.param
    @show eval_loss(optim)
end

