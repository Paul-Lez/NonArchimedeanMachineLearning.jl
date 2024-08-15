include("../src/gradient_descent.jl")


# set up some random synthetic data
prec = 20
K = PadicField(3,prec)
a1 = [K(1), K(2)]
r1 = Vector{Int}([1, 2])
a2 = [K(2), K(2)]
r2 = Vector{Int}([2, 2])
p1 = ValuationPolydisc(a1, r1) 
p2 = ValuationPolydisc(a2, r2) 
data = [p1, p2]
R, (x, y, θ, ϕ) =  polynomial_ring(K, ["x", "y", "θ", "ϕ"])


# Take the model to be the absolute polynomial sum f(x, y, θ, ϕ) = |3x+3θ+ϕ^15| + !2y+81θ+2ϕ|
fun = PolydiscFunction([3*x+3*θ+ϕ^15, 2*y+81*θ+2*ϕ])
# Specify which variables are the model's parameters and which are the data variables
abs_model = AbstractModel(fun, [true, true, false, false])
# Set the initial value of the model's parameters to be the point p1 in the Polydisc space
model = Model(abs_model, p1)
# Pick the loss function to be ℓ(θ, ϕ) = f(p1, θ, ϕ) + f(p2, θ, ϕ) 
loss = (d, m, (param::ValuationPolydisc{PadicFieldElem, Int64})) -> eval_abs(m.fun, p1, param) + eval_abs(m.fun, p2, param)
# Package the data of the model + loss function into an "optimiser object". 
# This has a built in optimisation algorithm: greedy descent
optim = greedy_descent_init(data, model, loss)

N_epochs = 20
# Now optimise for 20 epochs
for i in 1:N_epochs
    # Make a step in the direction given by the optimiser and update the parameters
    step!(optim)
end 