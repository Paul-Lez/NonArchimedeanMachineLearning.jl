"""
Function Learning Experiment Runner

Learn polynomial approximations to target functions (zero, one, or custom) over p-adic fields.
Compare multiple optimizers across varying hyperparameters and polynomial degrees.

Usage:
    julia --project=. experiments/paper/function_learning/run_experiments.jl [flags]

Flags:
    --quick     Reduced epochs (5) and simulations (10) for quick testing
    --save      Save results to JSON file (default filename with timestamp)
    --config    Load experiment configurations from config.jl
    --epochs N  Set number of epochs (default: 20)
    --output F  Specify output JSON filename

Examples:
    julia --project=. experiments/paper/function_learning/run_experiments.jl --quick --save
    julia --project=. experiments/paper/function_learning/run_experiments.jl --config --save
    julia --project=. experiments/paper/function_learning/run_experiments.jl --epochs 50 --save --output results.json
"""

include("../../../src/NAML.jl")
include("../util.jl")

using Oscar
using .NAML
using Printf
using Dates

# ============================================================================
# Parse command line arguments
# ============================================================================

quick_mode = "--quick" in ARGS
save_results = "--save" in ARGS
use_config_file = "--config" in ARGS

# Parse --epochs N
global n_epochs = quick_mode ? 5 : 20
for (i, arg) in enumerate(ARGS)
    if arg == "--epochs" && i < length(ARGS)
        global n_epochs = parse(Int, ARGS[i+1])
    end
end

# Parse --output filename
global output_filename = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--output" && i < length(ARGS)
        global output_filename = ARGS[i+1]
    end
end

# ============================================================================
# Experiment Configuration
# ============================================================================

if use_config_file
    include("config.jl")
    configs = experiment_configs
    println("Loaded experiment configurations from config.jl")
else
    # Default configurations
    configs = [
        Dict("name" => "zero_fn_deg3_2adic", "prime" => 2, "prec" => 20,
             "degree" => 3, "n_points" => 4, "target_fn" => "zero",
             "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
        Dict("name" => "one_fn_deg3_2adic", "prime" => 2, "prec" => 20,
             "degree" => 3, "n_points" => 4, "target_fn" => "one",
             "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
    ]
end

# ============================================================================
# Optimizer definitions
# ============================================================================

"""
Return a dict of optimizer name => initializer function.

Each initializer takes (param, loss) and returns an OptimSetup.
"""
function get_optimizer_configs(; quick::Bool=false)
    return Dict(
        "Greedy" => Dict(
            "type" => "Greedy",
            "params" => Dict("strict" => false, "degree" => 1),
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "Greedy-deg2" => Dict(
            "type" => "Greedy",
            "params" => Dict("strict" => false, "degree" => 2),
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 2))
            end
        ),
        "MCTS-50" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => quick ? 10 : 50,
                             "exploration_constant" => 1.41, "degree" => 1),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick ? 10 : 50,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=1
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-100" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => quick ? 20 : 100,
                             "exploration_constant" => 1.41, "degree" => 1),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick ? 20 : 100,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=1
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-100" => Dict(
            "type" => "DAG-MCTS",
            "params" => Dict("num_simulations" => quick ? 20 : 100,
                             "exploration_constant" => 1.41, "degree" => 1,
                             "persist_table" => true),
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=quick ? 20 : 100,
                    exploration_constant=1.41,
                    degree=1,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "UCT" => Dict(
            "type" => "UCT",
            "params" => Dict("max_depth" => 10, "num_simulations" => quick ? 20 : 100,
                             "exploration_constant" => 1.41, "degree" => 1),
            "init" => (param, loss) -> begin
                config = NAML.UCTConfig(
                    max_depth=10,
                    num_simulations=quick ? 20 : 100,
                    exploration_constant=1.41,
                    degree=1
                )
                NAML.uct_descent_init(param, loss, config)
            end
        ),
        "DOO" => Dict(
            "type" => "DOO",
            "params" => Dict("max_depth" => quick ? 10 : 15, "degree" => 1),
            "init" => (param, loss) -> begin
                # Get prime from parameter polydisc
                p = Float64(NAML.prime(param))
                # Delta function: p^(-h) for depth h
                delta = h -> p^(-h)
                config = NAML.DOOConfig(
                    delta=delta,
                    max_depth=quick ? 10 : 15,
                    degree=1,
                    strict=false
                )
                NAML.doo_descent_init(param, loss, 1, config)
            end
        ),
    )
end

# Canonical ordering for display
const OPTIMIZER_ORDER = ["Greedy", "Greedy-deg2", "MCTS-50", "MCTS-100", "DAG-MCTS-100", "UCT", "DOO"]

# ============================================================================
# Create target function loss
# ============================================================================

"""
Create a loss function for learning a target function using polynomial approximation.

Uses cross-entropy loss with sigmoid activation for smooth optimization.
"""
function create_function_learning_loss(K, degree, n_points, target_fn, threshold, scale)
    # Generate random test points
    p = Int(Oscar.prime(K))
    prec = Oscar.precision(K)
    x_values = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]

    # Target values based on target function
    if target_fn == "zero"
        y_values = [0.0 for _ in 1:n_points]
    elseif target_fn == "one"
        y_values = [1.0 for _ in 1:n_points]
    else
        error("Unknown target function: $target_fn")
    end

    data = collect(zip(x_values, y_values))

    # Simple polynomial evaluation function
    # Evaluates a0 + a1*x + a2*x^2 + ... + an*x^n
    function eval_polynomial(coeffs, x)
        result = coeffs[1]  # a0
        x_power = x
        for i in 2:length(coeffs)
            result += coeffs[i] * x_power
            x_power *= x
        end
        return result
    end

    function eval_fn(param_vector::Vector{<:NAML.ValuationPolydisc})
        return [begin
            loss = 0.0
            # Extract coefficient values from parameter polydisc
            coeffs = [NAML.unwrap(c) for c in NAML.center(param)]

            for (x, y) in data
                # Evaluate polynomial at x
                poly_val = eval_polynomial(coeffs, x)
                val_float = Float64(abs(poly_val))

                # Cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
                # where p = sigmoid((val - threshold)/scale)
                z = (val_float - threshold) / scale
                prob = 1.0 / (1.0 + exp(-z))

                # Clip probabilities to avoid log(0)
                prob = max(min(prob, 0.9999), 0.0001)

                if y > 0.5  # y = 1
                    loss += -log(prob)
                else  # y = 0
                    loss += -log(1 - prob)
                end
            end
            loss
        end for param in param_vector]
    end

    # Dummy gradient (not used by greedy descent)
    function grad_fn(vs::Vector{<:NAML.ValuationTangent})
        return [0.0 for _ in vs]
    end

    return NAML.Loss(eval_fn, grad_fn), data
end

# ============================================================================
# Run a single sample (one random problem instance)
# ============================================================================

function run_single_sample(config::Dict, sample_num::Int)
    p = config["prime"]
    prec = config["prec"]
    degree = config["degree"]
    n_points = config["n_points"]
    target_fn = config["target_fn"]
    threshold = config["threshold"]
    scale = config["scale"]

    K = PadicField(p, prec)

    # Create loss function
    loss, data = create_function_learning_loss(K, degree, n_points, target_fn, threshold, scale)

    # Initialize parameters at origin with radius 0
    param_center = [K(0) for _ in 1:degree+1]
    initial_param = NAML.ValuationPolydisc(param_center, [0 for _ in 1:degree+1])
    initial_loss = loss.eval([initial_param])[1]

    # Get optimizer configs
    opt_configs = get_optimizer_configs(quick=quick_mode)

    # Results for this sample
    sample_results = Dict{String, Any}()
    sample_results["sample_num"] = sample_num
    sample_results["initial_loss"] = initial_loss

    # Store data info (as floats for JSON serializability)
    sample_results["data"] = Dict(
        "x_abs_values" => [Float64(abs(x)) for (x, _) in data],
        "y_values" => [y for (_, y) in data],
    )

    sample_results["optimizers"] = Dict{String, Any}()

    # Run each optimizer
    for opt_name in OPTIMIZER_ORDER
        if !haskey(opt_configs, opt_name)
            continue
        end

        opt_setup = opt_configs[opt_name]
        try
            optim = opt_setup["init"](initial_param, loss)

            losses = Float64[]
            t_start = time()

            for epoch in 1:n_epochs
                current_loss = NAML.eval_loss(optim)
                push!(losses, current_loss)
                NAML.step!(optim)
            end

            t_end = time()
            elapsed = t_end - t_start

            final_loss = NAML.eval_loss(optim)
            push!(losses, final_loss)

            sample_results["optimizers"][opt_name] = Dict(
                "time" => elapsed,
                "final_loss" => final_loss,
                "losses" => losses,
                "improvement" => initial_loss - final_loss,
                "improvement_ratio" => (initial_loss > 0) ?
                    (initial_loss - final_loss) / initial_loss : 0.0,
                "hyperparameters" => opt_setup["params"],
            )

        catch e
            sample_results["optimizers"][opt_name] = Dict("error" => string(e))
        end
    end

    return sample_results
end

# ============================================================================
# Run a single experiment (multiple samples for one config)
# ============================================================================

function run_single_experiment(config::Dict)
    println("\n" * "="^70)
    println("Experiment: $(config["name"])")
    println("="^70)
    println("  Prime: $(config["prime"]), Degree: $(config["degree"]), " *
            "Points: $(config["n_points"]), Target: $(config["target_fn"])")
    println("  Samples: $(config["num_samples"]), Threshold: $(config["threshold"]), Scale: $(config["scale"])")
    println("-"^70)

    results = Dict{String, Any}()
    results["config"] = config
    results["samples"] = []

    for sample in 1:config["num_samples"]
        println("\n  [Sample $sample/$(config["num_samples"])]")

        try
            sample_result = run_single_sample(config, sample)
            push!(results["samples"], sample_result)

            # Brief summary
            println(@sprintf("    Initial loss: %.6e", sample_result["initial_loss"]))
            for opt_name in OPTIMIZER_ORDER
                if haskey(sample_result["optimizers"], opt_name)
                    opt_result = sample_result["optimizers"][opt_name]
                    if !haskey(opt_result, "error")
                        println(@sprintf("    %-15s Final: %.6e  (%.1f%% improvement, %.2fs)",
                            opt_name, opt_result["final_loss"],
                            opt_result["improvement_ratio"] * 100,
                            opt_result["time"]))
                    else
                        println("    $opt_name: ERROR - $(opt_result["error"])")
                    end
                end
            end
        catch e
            println("    Sample $sample failed: $e")
            push!(results["samples"], Dict("sample_num" => sample, "error" => string(e)))
        end
    end

    # Compute aggregate statistics
    compute_aggregate_stats!(results)

    return results
end

# ============================================================================
# Aggregate statistics across samples
# ============================================================================

_mean(x) = sum(x) / length(x)
function _std(x)
    m = _mean(x)
    sqrt(sum((xi - m)^2 for xi in x) / length(x))
end

function compute_aggregate_stats!(results::Dict)
    samples = results["samples"]
    valid_samples = filter(s -> !haskey(s, "error"), samples)

    if isempty(valid_samples)
        results["aggregate"] = Dict("error" => "No valid samples")
        return
    end

    results["aggregate"] = Dict{String, Any}()

    for opt_name in OPTIMIZER_ORDER
        opt_data = []
        for sample in valid_samples
            if haskey(sample["optimizers"], opt_name)
                opt_result = sample["optimizers"][opt_name]
                if !haskey(opt_result, "error")
                    push!(opt_data, opt_result)
                end
            end
        end

        if !isempty(opt_data)
            final_losses = [d["final_loss"] for d in opt_data]
            results["aggregate"][opt_name] = Dict(
                "mean_final_loss" => _mean(final_losses),
                "std_final_loss" => length(opt_data) > 1 ? _std(final_losses) : 0.0,
                "min_final_loss" => minimum(final_losses),
                "max_final_loss" => maximum(final_losses),
                "mean_improvement" => _mean([d["improvement"] for d in opt_data]),
                "mean_improvement_ratio" => _mean([d["improvement_ratio"] for d in opt_data]),
                "mean_time" => _mean([d["time"] for d in opt_data]),
                "n_valid" => length(opt_data),
                "hyperparameters" => opt_data[1]["hyperparameters"],
            )
        end
    end
end

# ============================================================================
# Main execution
# ============================================================================

println("\n" * "="^70)
println("FUNCTION LEARNING EXPERIMENT RUNNER")
println("="^70)
println("Start time: $(Dates.now())")
println("Number of experiments: $(length(configs))")
println("Epochs per optimizer: $n_epochs")
println("Quick mode: $quick_mode")
println("="^70)

all_results = []

for (i, config) in enumerate(configs)
    println("\n\n" * "#"^70)
    println("# EXPERIMENT $i/$(length(configs))")
    println("#"^70)

    try
        result = run_single_experiment(config)
        push!(all_results, result)
    catch e
        println("\nExperiment $(config["name"]) failed: $e")
        push!(all_results, Dict("config" => config, "error" => string(e)))
    end
end

# ============================================================================
# Summary Table
# ============================================================================

println("\n\n" * "="^70)
println("SUMMARY TABLE")
println("="^70)

for (i, result) in enumerate(all_results)
    if haskey(result, "error")
        println("\nExperiment $(i): $(result["config"]["name"]) - FAILED")
        continue
    end

    config = result["config"]
    println("\nExperiment $(i): $(config["name"])")
    println("  p=$(config["prime"]), degree=$(config["degree"]), " *
            "points=$(config["n_points"]), target=$(config["target_fn"])")

    if haskey(result, "aggregate") && !haskey(result["aggregate"], "error")
        println()
        println(@sprintf("  %-15s %15s %12s %12s %12s",
            "Optimizer", "Mean Final", "Std", "Improv %", "Time (s)"))
        println("  " * "-"^65)

        for opt_name in OPTIMIZER_ORDER
            if haskey(result["aggregate"], opt_name)
                agg = result["aggregate"][opt_name]
                println(@sprintf("  %-15s %15.6e %12.2e %11.1f%% %12.2f",
                    opt_name, agg["mean_final_loss"], agg["std_final_loss"],
                    agg["mean_improvement_ratio"] * 100, agg["mean_time"]))
            end
        end
    end
end

println("\n" * "="^70)
println("End time: $(Dates.now())")
println("="^70)

# ============================================================================
# Save results to JSON
# ============================================================================

if save_results
    try
        using JSON

        # Build output
        json_output = Dict{String, Any}()
        json_output["metadata"] = Dict(
            "timestamp" => string(Dates.now()),
            "n_epochs" => n_epochs,
            "quick_mode" => quick_mode,
            "optimizer_order" => OPTIMIZER_ORDER,
        )

        json_experiments = []
        for result in all_results
            json_result = Dict{String, Any}()
            json_result["config"] = result["config"]

            if haskey(result, "error")
                json_result["error"] = result["error"]
            else
                json_result["samples"] = result["samples"]
                if haskey(result, "aggregate")
                    json_result["aggregate"] = result["aggregate"]
                end
            end

            push!(json_experiments, json_result)
        end
        json_output["experiments"] = json_experiments

        # Determine filename
        if isnothing(output_filename)
            timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            global output_filename = "function_learning_results_$(timestamp).json"
        end
        filepath = joinpath(@__DIR__, output_filename)

        open(filepath, "w") do f
            JSON.print(f, json_output, 2)
        end

        println("\nResults saved to: $filepath")
    catch e
        if e isa ArgumentError && occursin("Package JSON not found", string(e))
            println("\nWarning: JSON package not installed. Cannot save results.")
            println("  Install with: julia -e 'using Pkg; Pkg.add(\"JSON\")'")
        else
            println("\nError saving results: $e")
        end
    end
end

println("\nAll experiments complete!")
