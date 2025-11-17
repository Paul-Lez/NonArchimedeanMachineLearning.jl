include("../src/naml.jl")

p, prec = 2, 20
K = PadicField(p, prec)

R, (x, a, b, c) = K["x", "a", "b", "c"]
g = AbsolutePolynomialSum([(x - a) * (x - b) * (x - c)])
f = AbstractModel(g, [true, false, false, false])

# Setting up data points through which we want to fit f
# (high valuation = small disk, low valuation = large disk)
p1 = ValuationPolydisc([K(p^0)], Vector{Int}([prec]))
p2 = ValuationPolydisc([K(p^1)], Vector{Int}([prec]))
p3 = ValuationPolydisc([K(p^2)], Vector{Int}([prec]))
p4 = ValuationPolydisc([K(3)], Vector{Int}([prec]))
data = [(p1, 0), (p2, 0), (p3, 0), (p4, -2)]
# data = [(p1, 0), (p2, 0), (p3, 0)]

# Setting up initial parameters
model = Model(f, ValuationPolydisc([K(rand(Int8)), K(rand(Int8)), K(rand(Int8))], [0, 0, 0]))
model = Model(f, ValuationPolydisc([K(11), K(22), K(33)], [0, 0, 0]))

# Create loss function
ell = MPE_loss_init(f, data, 2)

greedy_optim = greedy_descent_init(model.param, ell, 1, (false, 1))

# Greedy descent
N_epochs = 39
t1 = time()
for i in 1:N_epochs
    println("Loss at epoch ", i, " is ", eval_loss(greedy_optim))
    step!(greedy_optim)
end
t2 = time()

# Compute the new value of the loss
println("Greedy descent finished in ", t2 - t1, " seconds.")
println("The Final parameters are: ")
display(greedy_optim.param.center)
display(greedy_optim.param.radius)
