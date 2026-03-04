# Sanity Check: Learn the One Function
# This script attempts to learn f(x) such that all outputs are 1
# This should be easy since a trivial solution is any constant function with large p-adic norm

include("../../src/NAML.jl")
include("util.jl")
using .NAML
using Oscar
using Random

println("="^80)
println("SANITY CHECK: Learning the One Function")
println("="^80)
println()

# Configuration
PREC = 20
PRIME = 2
N_POINTS = 5  # Number of random points
POLY_DEGREE = 6  # Polynomial degree
EPOCHS = 100

# Optimal hyperparameters (from previous tuning)
INITIAL_RADIUS = 0
THRESHOLD = 0.3
SCALE = .5
OPTIMIZER = "DAG_MCTS"  # Using DAG-MCTS with transposition tables

USE_REGULARIZATION = true    

# Optimizer-specific configuration
MCTS_NUM_SIMULATIONS = 1000    # MCTS simulations per step (OPTIMAL: 50-100)
MCTS_EXPLORATION = 1.41       # MCTS exploration constant
GREEDY_DEGREE = 2             # Greedy descent search degree (OPTIMAL: 2)

println("Configuration:")
println("  Prime: $PRIME")
println("  Precision: $PREC")
println("  Number of points: $N_POINTS")
println("  Polynomial degree: $POLY_DEGREE")
println("  Optimizer: $OPTIMIZER")
println("  Threshold: $THRESHOLD")
println("  Scale: $SCALE")
println("  Initial radius: $INITIAL_RADIUS")
println("  Epochs: $EPOCHS")
println()

# Setup field and polynomial ring
K = PadicField(PRIME, PREC)
var_names = ["x"]
param_names = ["a$i" for i in 0:POLY_DEGREE]
all_vars = vcat(var_names, param_names)
R, vars = polynomial_ring(K, all_vars)

x = vars[1]
params = vars[2:end]

# Create polynomial: a0 + a1*x + a2*x^2 + ... + an*x^n
poly_expr = sum(params[i] * x^(i-1) for i in 1:length(params))
println("Polynomial structure: f(x) = a0 + a1*x + a2*x^2 + ... + a$(POLY_DEGREE)*x^$(POLY_DEGREE)")
println()

# Generate random data points
println("Generating random data...")
Random.seed!(123)  # Different seed from zero function test

data_points = []
for i in 1:N_POINTS
    # Random p-adic number
    x_val = generate_random_padic(PRIME, PREC, 10, PREC)
    # x_val = K(rand(1:PRIME^PREC))

    # Target is always 1 (one function)
    y_val = 1.0

    push!(data_points, (x=x_val, y=y_val))
end

println("Data points (all targets = 1):")
for (i, point) in enumerate(data_points)
    x_norm = Float64(abs(point.x))
    println("  $i. x = $(point.x), |x|_p = $x_norm, y = $(point.y)")
end
println()

# Setup model
param_info = vcat([true], [false for _ in 1:POLY_DEGREE+1])  # x is data, params are parameters
fun = NAML.AbsolutePolynomialSum([poly_expr])
log_fn = NAML.DifferentiableFunction(
    x -> log(1/PRIME, x),
    x -> -1.0 / (x * log(PRIME))
)
fun = NAML.Comp(log_fn, fun)
model = NAML.AbstractModel(fun, param_info)

# loss = 

# Prepare data
data = [(point.x, point.y) for point in data_points]

# # Cross-entropy loss with sigmoid
# function cross_entropy_loss_init(model, data, threshold, scale)
#     # Specialize the model at each data point
#     specialized_models = [NAML.specialise(model, [val]) for (val, _) in data]

#     # Initialize batch evaluation for each specialized model
#     batch_evals = [batch_evaluate_init(specialized_models[i]) for i in eachindex(specialized_models)]

#     # Extract target values
#     y_values = [y for (_, y) in data]

#     function sigmoid(z)
#         return 1.0 / (1.0 + exp(-z))
#     end

#     function eval_fn(params::Vector{ValuationPolydisc{S,T,N}}) where {S, T, N}
#         return [begin
#             loss = 0.0
#             for i in eachindex(data)
#                 val_float = Float64(batch_evals[i](param))
#                 y = y_values[i]

#                 # Cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
#                 # where p = sigmoid((val - threshold)/scale)
#                 z = (val_float - threshold) / scale
#                 prob = sigmoid(z)

#                 # Clip probabilities to avoid log(0)
#                 prob = max(min(prob, 0.9999), 0.0001)

#                 if y > 0.5  # y = 1
#                     loss += -log(prob)
#                 else  # y = 0
#                     loss += -log(1 - prob)
#                 end
#             end
#             loss
#         end for param in params]
#     end

#     # Dummy gradient (we're using greedy descent which doesn't need it)
#     function grad_fn(vs::Vector{ValuationTangent{S,T,N}}) where {S, T, N}
#         return [0.0 for _ in vs]
#     end

#     return Loss(eval_fn, grad_fn)
# end

function compute_predictions(model, param, data)
    # Use the specialized models from the loss function
    specialized_models = [NAML.specialise(model, [val]) for (val, _) in data]
    batch_evals = [NAML.batch_evaluate_init(specialized_models[i]) for i in eachindex(specialized_models)]

    function sigmoid(z)
        return 1.0 / (1.0 + exp(-z))
    end

    correct = 0
    for (i, (_, y)) in enumerate(data)
        pred_float = Float64(batch_evals[i](param))
        # println("Evaluation yielded $(specialized_models[i])")
        z = (pred_float - THRESHOLD) / SCALE
        prob = sigmoid(z)
        pred_label = prob >= 0.5 ? 1.0 : 0.0

        is_correct = (pred_label == y)
        correct += is_correct

        check = is_correct ? "✓" : "✗"
        println("  $i. |f(x)| = $(round(pred_float, digits=4)), prob = $(round(prob, digits=4)), pred = $pred_label, actual = $y $check")
    end

end


# Create loss
# loss = cross_entropy_loss_init(model, data, THRESHOLD, SCALE)

loss = polynomial_to_valuation_crossentropy_loss(data, POLY_DEGREE, PRIME, THRESHOLD, SCALE)

# Initialize parameters at origin
param_center = [K(0) for _ in 1:POLY_DEGREE+1]
param = NAML.ValuationPolydisc(param_center, [INITIAL_RADIUS for _ in 1:POLY_DEGREE+1])

# # Setup optimizer
# if OPTIMIZER == "Greedy_deg1"
#     greedy_degree = 1
# elseif OPTIMIZER == "Greedy_deg2"
#     greedy_degree = 2
# elseif OPTIMIZER == "Greedy_deg3"
#     greedy_degree = 3
# else
#     error("Unknown optimizer: $OPTIMIZER")
# end

settings = (false, greedy_degree)  # (strict, degree)
# optim = NAML.greedy_descent_init(param, loss, 1, settings)

# Use DAG-MCTS optimizer instead of standard MCTS
dag_mcts_config = NAML.DAGMCTSConfig(
        num_simulations=MCTS_NUM_SIMULATIONS,
        exploration_constant=MCTS_EXPLORATION,
        degree=1,
        persist_table=true,  # Keep transposition table across optimization steps
        selection_mode=NAML.BestValue
    )
optim = NAML.dag_mcts_descent_init(param, loss, dag_mcts_config)

# Training
println("Training...")
initial_loss = NAML.eval_loss(optim)
initial_accuracy = compute_classification_accuracy(model, data, optim.param, THRESHOLD, SCALE)
println("Initial loss: $initial_loss")
println("Initial accuracy: $(round(initial_accuracy, digits=2))%")
println()

compute_predictions(model, optim.param, data)

# print out the initial params

for (i, coef) in enumerate(NAML.center(optim.param))
    degree = i - 1
    coef_norm = Float64(abs(coef))
    if degree == 0
        println("  a0 (constant): $coef, |a0|_p = $coef_norm, radius = $(optim.param.radius[1])")
    else
        println("  a$degree (x^$degree): $coef, |a$degree|_p = $coef_norm, radius = $(optim.param.radius[i])")
    end
end

for epoch in 1:EPOCHS
    NAML.step!(optim)

    if epoch % 100 == 0 || epoch == EPOCHS
        current_loss = NAML.eval_loss(optim)
        current_accuracy = compute_classification_accuracy(model, data, optim.param, THRESHOLD, SCALE)
        improvement = 100 * (1 - current_loss / initial_loss)

        # Get DAG statistics
        dag_stats = NAML.get_dag_stats(optim.state)

        println("Epoch $epoch: Loss = $current_loss (improvement: $(round(improvement, digits=2))%), Accuracy = $(round(current_accuracy, digits=2))%")
        println("  DAG Stats: $(dag_stats.unique_nodes) unique nodes, $(dag_stats.multi_parent_nodes) with multiple parents ($(round(100 * dag_stats.multi_parent_nodes / max(1, dag_stats.unique_nodes), digits=1))% transposition rate)")
    end
end

final_loss = NAML.eval_loss(optim)
improvement = 100 * (1 - final_loss / initial_loss)

println()
println("="^80)
println("RESULTS")
println("="^80)
println("Initial loss: $initial_loss")
println("Final loss: $final_loss")
println("Improvement: $(round(improvement, digits=2))%")
println()

# Evaluate predictions
println("Predictions:")
final_param = optim.param

# Use the specialized models from the loss function
specialized_models = [NAML.specialise(model, [val]) for (val, _) in data]
batch_evals = [NAML.batch_evaluate_init(specialized_models[i]) for i in eachindex(specialized_models)]

function sigmoid(z)
    return 1.0 / (1.0 + exp(-z))
end

global correct = 0
for (i, (_, y)) in enumerate(data)
    global correct
    pred_float = Float64(batch_evals[i](final_param))
    z = (pred_float - THRESHOLD) / SCALE
    prob = sigmoid(z)
    pred_label = prob >= 0.5 ? 1.0 : 0.0

    is_correct = (pred_label == y)
    correct += is_correct

    check = is_correct ? "✓" : "✗"
    println("  $i. |f(x)| = $(round(pred_float, digits=4)), prob = $(round(prob, digits=4)), pred = $pred_label, actual = $y $check")
end

accuracy = 100 * correct / length(data)
println()
println("Accuracy: $correct/$(length(data)) = $(round(accuracy, digits=2))%")

if accuracy == 100.0
    println()
    println("SUCCESS! Achieved 100% accuracy on one function learning.")
else
    println()
    println("WARNING: Did not achieve 100% accuracy. This suggests a problem with the learning setup.")
end

println()
# Show learned polynomial coefficients
println("Learned polynomial coefficients:")
param_center = NAML.center(final_param)
for (i, coef) in enumerate(param_center)
    degree = i - 1
    coef_norm = Float64(abs(coef))
    if degree == 0
        println("  a0 (constant): $coef, |a0|_p = $coef_norm, radius = $(optim.param.radius[1])")
    else
        println("  a$degree (x^$degree): $coef, |a$degree|_p = $coef_norm, radius = $(optim.param.radius[i])")
    end
end

println()

# Print final DAG-MCTS statistics
println("="^80)
println("DAG-MCTS STATISTICS")
println("="^80)
NAML.print_dag_stats(optim.state)
println()
println("Transposition table integrity: ", NAML.verify_transposition_table(optim.state) ? "PASSED ✓" : "FAILED ✗")
println()
println("="^80)
