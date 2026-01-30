# Quick debug script to understand why MCTS isn't improving
using Oscar
include("../src/NAML.jl")
using .NAML

# Setup same problem
prec = 30
K = PadicField(3, prec)

coefficients = [
    ([1, -1, 0, 2, 2, -4, 1, 2, 0, -2], 2),
    ([2, -2, -3, 0, -5, -3, -3, -1, 0, -5], 0),
    ([-3, 1, -1, -2, -1, 4, -2, 1, -2, -4], 2),
    ([-1, -5, -5, 2, 0, 2, -4, 0, 0, 4], -4),
    ([-2, 5, -2, 3, 3, 1, -2, -2, 1, 1], -2),
]

linear_polys = [
    LinearPolynomial([K(c) for c in coeffs], K(constant))
    for (coeffs, constant) in coefficients
]
poly = LinearAbsolutePolynomialSum(linear_polys)
batch_eval = batch_evaluate_init(poly)

# Initial point
initial_center = [K(0) for _ in 1:10]
initial_radius = [0 for _ in 1:10]
initial_param = ValuationPolydisc(initial_center, initial_radius)

println("Initial loss: ", batch_eval(initial_param))

# Check ALL children of root
println("\nEvaluating all $(10 * 3) = 30 children of root...")
child_losses = []
for ch in children(initial_param, 1)
    push!(child_losses, (batch_eval(ch), ch.radius))
end

# Sort by loss
sort!(child_losses, by=x->x[1])

println("\nTop 10 children by loss:")
for (i, (loss, rad)) in enumerate(child_losses[1:min(10, length(child_losses))])
    println("  $i: loss = $loss, radius = $rad")
end

println("\nBottom 5 children by loss:")
for (i, (loss, rad)) in enumerate(child_losses[end-4:end])
    println("  $(length(child_losses)-5+i): loss = $loss, radius = $rad")
end

# Check: are all children the same loss?
unique_losses = unique([l for (l, _) in child_losses])
println("\nUnique loss values among children: ", length(unique_losses))
println("All losses: ", unique_losses)

# If all same, go one level deeper
if length(unique_losses) == 1
    println("\n--- All children have same loss! Going deeper... ---")
    best_child = child_losses[1][2]  # pick first one
    grandchildren_losses = []
    for ch in children(ValuationPolydisc(initial_center, collect(best_child)), 1)
        push!(grandchildren_losses, batch_eval(ch))
    end
    unique_gc = unique(grandchildren_losses)
    println("Unique losses at depth 2: ", length(unique_gc))
    println("Loss values: ", sort(unique_gc)[1:min(5, length(unique_gc))])
end
