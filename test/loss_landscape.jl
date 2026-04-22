include("../src/NonArchimedeanMachineLearning.jl")
using .NonArchimedeanMachineLearning
using Oscar

println("=== Loss Landscape Sampling Test ===\n")

# Set up the p-adic field
prec = 20
K = PadicField(2, prec)
p = 2

println("Testing with 2-adic field, precision = $prec\n")

# Create a polynomial ring with one variable x and two parameters a, b
R, (x, a, b) = K["x", "a", "b"]

# Define a sum of two polynomials: (x - a)^2 + (x - b)^2
# This creates a loss landscape over the (a, b) parameter space
poly1 = (x - a)^2
poly2 = (x - b)^2
g = AbsolutePolynomialSum([poly1, poly2])

println("Polynomial function: f(x, a, b) = (x - a)^2 + (x - b)^2")
println("This is a sum of two polynomials depending on parameters a and b\n")

# Create an abstract model where x is data and a, b are parameters
model = AbstractModel(g, [true, false, false])  # x is data (true), a and b are params (false)

println("Model setup: x is the data variable, a and b are parameters\n")

# Define some polydiscs in the 2D parameter space (a, b)
# These represent different regions in parameter space
d1 = ValuationPolydisc([K(0), K(0)], [4, 4])    # Small disc centered at (0, 0)
d2 = ValuationPolydisc([K(0), K(4)], [4, 4])    # Small disc centered at (0, 4)
d3 = ValuationPolydisc([K(4), K(0)], [4, 4])    # Small disc centered at (4, 0)

println("Created 3 polydiscs in parameter space:")
println("  d1: center = (0, 0), radius = (4, 4)")
println("  d2: center = (0, 4), radius = (4, 4)")
println("  d3: center = (4, 0), radius = (4, 4)\n")

# Build the convex hull tree
println("Building convex hull tree...")
tree = NonArchimedeanMachineLearning.convex_hull([d1, d2, d3])

println("Convex hull has $(length(tree.nodes)) nodes:")
println("  - $(length(tree.leaf_indices)) leaves (original discs)")
println("  - $(length(tree.nodes) - length(tree.leaf_indices)) join nodes")

# Count edges
num_edges = sum(length(children) for children in values(tree.children))
println("  - $num_edges parent-child edges\n")

# Define a loss function
# For a given parameter disc, evaluate at a fixed data point x = 2
data_point = ValuationPolydisc([K(2)], [0.0])  # Point disc at x = 2, using Float64 for radius type consistency

function loss_function(param_disc::ValuationPolydisc)
    # Create a model with these parameter values
    m = Model(model, param_disc)

    # Evaluate at the data point
    val = NonArchimedeanMachineLearning.evaluate(m, data_point)

    return Float64(val)
end

println("Loss function: evaluate the model at data point x = 2")
println("  For parameters (a, b), this computes |(2 - a)^2 + (2 - b)^2|\n")

# Sample the loss landscape
num_samples = 10
println("Sampling loss landscape with $num_samples points per geodesic...")
landscape = sample_loss_landscape(tree, loss_function, num_samples)

println("Sampled $(length(landscape)) geodesics\n")

# Display results for each edge
println("=== Loss Landscape Results ===\n")

for ((parent_idx, child_idx), samples) in sort(collect(landscape), by=first)
    parent_disc = tree.nodes[parent_idx]
    child_disc = tree.nodes[child_idx]

    println("Edge: Node $child_idx → Node $parent_idx")
    println("  Child center:  ($(child_disc.center[1]), $(child_disc.center[2]))")
    println("  Child radius:  ($(child_disc.radius[1]), $(child_disc.radius[2]))")
    println("  Parent center: ($(parent_disc.center[1]), $(parent_disc.center[2]))")
    println("  Parent radius: ($(parent_disc.radius[1]), $(parent_disc.radius[2]))")
    println("  Samples along geodesic:")

    for (i, (x, y)) in enumerate(samples)
        println("    x = $(round(x, digits=2)), loss = $(round(y, digits=3))")
    end
    println()
end

println("=== Test Complete ===")
