"""
Function Learning Experiment Runner

Learn polynomial approximations to target functions (zero, one, or custom) over p-adic fields.
Compare multiple optimizers across varying hyperparameters and polynomial degrees.

Usage:
    julia --project=. experiments/paper/function_learning/run_experiments.jl [flags]

Flags:
    --quick              Reduced epochs (5) and simulations (10) for quick testing
    --save               Save results to JSON file (default filename with timestamp)
    --config             Load experiment configurations from config.jl
    --paper              Use paper-ready configurations from paper_config.jl
    --epochs N           Set number of epochs (default: 20)
    --samples N          Override number of samples per config
    --output F           Specify output JSON filename
    --selection-mode M   MCTS/DAG-MCTS selection mode: BestValue, VisitCount, or BestLoss (default: BestValue)
    --degree D           Override tree branching degree for MCTS/DAG-MCTS/DOO optimizers (default: auto from num_params: 1 if 1 param, 2 if ≥2 params)

Examples:
    julia --project=. experiments/paper/function_learning/run_experiments.jl --quick --save
    julia --project=. experiments/paper/function_learning/run_experiments.jl --config --save
    julia --project=. experiments/paper/function_learning/run_experiments.jl --paper --save
    julia --project=. experiments/paper/function_learning/run_experiments.jl --epochs 50 --save --output results.json
    julia --project=. experiments/paper/function_learning/run_experiments.jl --paper --save --selection-mode VisitCount
"""

include("../../../src/NAML.jl")
include("../util.jl")

using Oscar
using .NAML
using Printf
using Dates
using Random

# ============================================================================
# Parse command line arguments
# ============================================================================

quick_mode = "--quick" in ARGS
save_results = "--save" in ARGS
use_config_file = "--config" in ARGS
use_paper_config = "--paper" in ARGS

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

# Parse --samples N
global n_samples_override = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--samples" && i < length(ARGS)
        global n_samples_override = parse(Int, ARGS[i+1])
    end
end

# Parse --selection-mode M (BestValue, VisitCount, BestLoss)
global selection_mode = NAML.BestValue
for (i, arg) in enumerate(ARGS)
    if arg == "--selection-mode" && i < length(ARGS)
        mode_str = ARGS[i+1]
        if mode_str == "BestValue"
            global selection_mode = NAML.BestValue
        elseif mode_str == "VisitCount"
            global selection_mode = NAML.VisitCount
        elseif mode_str == "BestLoss"
            global selection_mode = NAML.BestLoss
        else
            error("Invalid selection mode: $mode_str. Must be BestValue, VisitCount, or BestLoss")
        end
    end
end

# Parse --degree D (or --degree=D)
global mcts_degree_override = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--degree" && i < length(ARGS)
        global mcts_degree_override = parse(Int, ARGS[i+1])
    elseif startswith(arg, "--degree=")
        global mcts_degree_override = parse(Int, arg[10:end])
    end
end

# ============================================================================
# Experiment Configuration
# ============================================================================

if use_paper_config
    include("paper_config.jl")
    configs = experiment_configs
    println("Loaded PAPER-READY experiment configurations from paper_config.jl")
elseif use_config_file
    include("config.jl")
    configs = experiment_configs
    println("Loaded experiment configurations from config.jl")
else
    # Default configurations
    configs = [
        Dict("name" => "p2_zero_deg3", "prime" => 2, "prec" => 20,
             "degree" => 3, "n_points" => 4, "target_fn" => "zero",
             "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
        Dict("name" => "p2_one_deg3", "prime" => 2, "prec" => 20,
             "degree" => 3, "n_points" => 4, "target_fn" => "one",
             "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
    ]
end

# Apply samples override if specified
if !isnothing(n_samples_override)
    for config in configs
        config["num_samples"] = n_samples_override
    end
    println("Overriding num_samples to $n_samples_override for all configs")
end

# ============================================================================
# Optimizer definitions
# ============================================================================

"""
Return a dict of optimizer name => initializer function.

Each initializer takes (param, loss) and returns an OptimSetup.
"""
function get_optimizer_configs(; quick::Bool=false, selection_mode=NAML.BestValue, degree::Int=1, prime::Int=2, dim::Int=1)
    # k = number of children of a polydisc = binomial(dim, degree) * prime^degree
    k = binomial(dim, degree) * prime^degree
    sims_k   = quick ? 50 : k
    sims_5k  = quick ? 100 : 5 * k
    sims_10k = quick ? 200 : 10 * k
    return Dict(
        "Random" => Dict(
            "type" => "Random",
            "params" => Dict("degree" => 1),
            "init" => (param, loss) -> begin
                NAML.random_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "Best-First" => Dict(
            "type" => "Best-First",
            "params" => Dict("strict" => false, "degree" => 1),
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "Best-First-branch2" => Dict(
            "type" => "Best-First",
            "params" => Dict("strict" => false, "degree" => 2),
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 2))
            end
        ),
        "MCTS-k" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => sims_k,
                             "exploration_constant" => 1.41, "degree" => degree),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=sims_k,
                    exploration_constant=1.41,
                    selection_mode=selection_mode,
                    degree=degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-5k" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => sims_5k,
                             "exploration_constant" => 1.41, "degree" => degree),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=sims_5k,
                    exploration_constant=1.41,
                    selection_mode=selection_mode,
                    degree=degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-10k" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => sims_10k,
                             "exploration_constant" => 1.41, "degree" => degree),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=sims_10k,
                    exploration_constant=1.41,
                    selection_mode=selection_mode,
                    degree=degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-k" => Dict(
            "type" => "DAG-MCTS",
            "params" => Dict("num_simulations" => sims_k,
                             "exploration_constant" => 1.41, "degree" => degree,
                             "persist_table" => true),
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=sims_k,
                    exploration_constant=1.41,
                    degree=degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-5k" => Dict(
            "type" => "DAG-MCTS",
            "params" => Dict("num_simulations" => sims_5k,
                             "exploration_constant" => 1.41, "degree" => degree,
                             "persist_table" => true),
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=sims_5k,
                    exploration_constant=1.41,
                    degree=degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-10k" => Dict(
            "type" => "DAG-MCTS",
            "params" => Dict("num_simulations" => sims_10k,
                             "exploration_constant" => 1.41, "degree" => degree,
                             "persist_table" => true),
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=sims_10k,
                    exploration_constant=1.41,
                    degree=degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DOO" => Dict(
            "type" => "DOO",
            "params" => Dict("max_depth" => quick ? 10 : 15, "degree" => degree),
            "init" => (param, loss) -> begin
                # Get prime from parameter polydisc
                p = Float64(NAML.prime(param))
                # Delta function: p^(-h) for depth h
                delta = h -> p^(-h)
                config = NAML.DOOConfig(
                    delta=delta,
                    max_depth=quick ? 10 : 15,
                    degree=degree,
                    strict=false
                )
                NAML.doo_descent_init(param, loss, 1, config)
            end
        ),
        "Best-First-Gradient" => Dict(
            "type" => "Best-First-Gradient",
            "params" => Dict("degree" => 1),
            "init" => (param, loss) -> begin
                NAML.gradient_descent_init(param, loss, 1, (false, 1))
            end
        ),
    )
end

# Canonical ordering for display (shared across all experiments)
const OPTIMIZER_ORDER = ["Random", "Best-First", "Best-First-branch2", "Best-First-Gradient", "MCTS-k", "MCTS-5k", "MCTS-10k", "DAG-MCTS-k", "DAG-MCTS-5k", "DAG-MCTS-10k", "DOO"]
const NAME_WIDTH = maximum(length(n) for n in OPTIMIZER_ORDER)

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
    elseif target_fn == "random"
        # Random binary labels (0 or 1)
        y_values = [Float64(rand(0:1)) for _ in 1:n_points]
    else
        error("Unknown target function: $target_fn")
    end

    data = collect(zip(x_values, y_values))

    loss = polynomial_to_crossentropy_loss(data, degree, threshold, scale)

    return loss, data
end

# ============================================================================
# Compute classification accuracy
# ============================================================================

"""
Compute classification accuracy for a polynomial classifier.

Given polynomial coefficients, evaluates accuracy on the data using:
- Polynomial evaluation: f(x) = a0 + a1*x + ... + an*x^n
- Sigmoid activation: p = 1/(1 + exp(-(|f(x)| - threshold)/scale))
- Binary prediction: predict 1 if p > 0.5, else 0
"""
function compute_accuracy(coeffs, data, threshold, scale)
    # Simple polynomial evaluation function
    function eval_polynomial(coeffs, x)
        result = coeffs[1]  # a0
        x_power = x
        for i in 2:length(coeffs)
            result += coeffs[i] * x_power
            x_power *= x
        end
        return result
    end

    correct = 0
    total = length(data)

    for (x, y) in data
        # Evaluate polynomial at x
        poly_val = eval_polynomial(coeffs, x)
        val_float = Float64(abs(poly_val))

        # Sigmoid activation
        z = (val_float - threshold) / scale
        prob = 1.0 / (1.0 + exp(-z))

        # Predict: 1 if prob > 0.5, else 0
        prediction = prob > 0.5 ? 1.0 : 0.0

        if prediction == y
            correct += 1
        end
    end

    return correct / total
end

# ============================================================================
# Per-sample optimizer ranking
# ============================================================================

"""
Rank optimizers within a sample by final_loss (lower = rank 1).
Adds a "rank" field to each valid optimizer result. Ties share the average rank.
"""
function compute_sample_rankings!(sample_results::Dict)
    optimizers = sample_results["optimizers"]
    valid_opts = [(name, res["final_loss"]) for (name, res) in optimizers
                  if !haskey(res, "error")]
    isempty(valid_opts) && return
    sort!(valid_opts, by=x -> x[2])
    n = length(valid_opts)
    i = 1
    while i <= n
        j = i
        while j <= n && valid_opts[j][2] == valid_opts[i][2]
            j += 1
        end
        avg_rank = (i + j - 1) / 2.0
        for k in i:j-1
            optimizers[valid_opts[k][1]]["rank"] = avg_rank
        end
        i = j
    end
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

    # Compute initial accuracy
    initial_coeffs = [NAML.unwrap(c) for c in NAML.center(initial_param)]
    initial_accuracy = compute_accuracy(initial_coeffs, data, threshold, scale)

    # Get optimizer configs
    num_params = degree + 1  # polynomial has degree+1 coefficients
    auto_degree = num_params >= 2 ? 2 : 1
    effective_mcts_degree = isnothing(mcts_degree_override) ? auto_degree : mcts_degree_override
    opt_configs = get_optimizer_configs(quick=quick_mode, selection_mode=selection_mode, degree=effective_mcts_degree, prime=p, dim=num_params)

    # Results for this sample
    sample_results = Dict{String, Any}()
    sample_results["sample_num"] = sample_num
    sample_results["initial_loss"] = initial_loss
    sample_results["initial_accuracy"] = initial_accuracy

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
            # Wrap loss with evaluation counting
            counted_loss, eval_counter = wrap_loss_with_counting(loss)

            optim = opt_setup["init"](initial_param, counted_loss)

            losses = Float64[]
            t_start = time()

            for epoch in 1:n_epochs
                current_loss = NAML.eval_loss(optim)
                push!(losses, current_loss)
                NAML.step!(optim)
                NAML.has_converged(optim) && break
            end

            t_end = time()
            elapsed = t_end - t_start

            final_loss = NAML.eval_loss(optim)
            push!(losses, final_loss)

            # Compute final accuracy
            final_coeffs = [NAML.unwrap(c) for c in NAML.center(optim.param)]
            final_accuracy = compute_accuracy(final_coeffs, data, threshold, scale)

            # Subtract monitoring eval_loss calls: length(losses) in-loop + 1 final
            monitoring_evals = length(losses)
            total_optimizer_evals = eval_counter.eval_count - monitoring_evals + eval_counter.grad_count

            sample_results["optimizers"][opt_name] = Dict(
                "time" => elapsed,
                "final_loss" => final_loss,
                "final_accuracy" => final_accuracy,
                "losses" => losses,
                "improvement" => initial_loss - final_loss,
                "improvement_ratio" => (initial_loss > 0) ?
                    (initial_loss - final_loss) / initial_loss : 0.0,
                "accuracy_improvement" => final_accuracy - initial_accuracy,
                "hyperparameters" => opt_setup["params"],
                "total_evals" => total_optimizer_evals,
            )

        catch e
            sample_results["optimizers"][opt_name] = Dict("error" => string(e))
        end
    end

    compute_sample_rankings!(sample_results)
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
            println(@sprintf("    Initial loss: %.6e, accuracy: %.2f%%",
                sample_result["initial_loss"], sample_result["initial_accuracy"] * 100))
            for opt_name in OPTIMIZER_ORDER
                if haskey(sample_result["optimizers"], opt_name)
                    opt_result = sample_result["optimizers"][opt_name]
                    if !haskey(opt_result, "error")
                        acc_imp = opt_result["accuracy_improvement"] * 100
                        acc_imp_str = acc_imp >= 0 ? "+$(Printf.@sprintf("%.2f", acc_imp))" : Printf.@sprintf("%.2f", acc_imp)
                        println(Printf.format(Printf.Format("    %-$(NAME_WIDTH)s Final: %.6e, accuracy: %.2f%%  (loss: %.1f%%, acc: %s%%, %.2fs)"),
                            opt_name, opt_result["final_loss"],
                            opt_result["final_accuracy"] * 100,
                            opt_result["improvement_ratio"] * 100,
                            acc_imp_str,
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
            final_accuracies = [d["final_accuracy"] for d in opt_data]
            accuracy_improvements = [d["accuracy_improvement"] for d in opt_data]

            agg = Dict(
                "mean_final_loss" => _mean(final_losses),
                "std_final_loss" => length(opt_data) > 1 ? _std(final_losses) : 0.0,
                "min_final_loss" => minimum(final_losses),
                "max_final_loss" => maximum(final_losses),
                "mean_final_accuracy" => _mean(final_accuracies),
                "std_final_accuracy" => length(opt_data) > 1 ? _std(final_accuracies) : 0.0,
                "min_final_accuracy" => minimum(final_accuracies),
                "max_final_accuracy" => maximum(final_accuracies),
                "mean_accuracy_improvement" => _mean(accuracy_improvements),
                "mean_improvement" => _mean([d["improvement"] for d in opt_data]),
                "mean_improvement_ratio" => _mean([d["improvement_ratio"] for d in opt_data]),
                "mean_time" => _mean([d["time"] for d in opt_data]),
                "std_time" => length(opt_data) > 1 ? _std([d["time"] for d in opt_data]) : 0.0,
                "n_valid" => length(opt_data),
                "hyperparameters" => opt_data[1]["hyperparameters"],
            )
            if haskey(opt_data[1], "total_evals")
                agg["mean_total_evals"] = _mean([d["total_evals"] for d in opt_data])
            end
            ranks = [d["rank"] for d in opt_data if haskey(d, "rank")]
            if !isempty(ranks)
                agg["mean_rank"] = _mean(ranks)
                agg["std_rank"] = length(ranks) > 1 ? _std(ranks) : 0.0
            end
            results["aggregate"][opt_name] = agg
        end
    end
end

# ============================================================================
# Main execution
# ============================================================================

# Set random seed for reproducibility
Random.seed!(42)

println("\n" * "="^70)
println("FUNCTION LEARNING EXPERIMENT RUNNER")
println("="^70)
println("Start time: $(Dates.now())")
println("Number of experiments: $(length(configs))")
println("Epochs per optimizer: $n_epochs")
println("Quick mode: $quick_mode")
if !isnothing(mcts_degree_override)
    println("MCTS/DAG-MCTS/DOO degree override: $mcts_degree_override")
end
println("Random seed: 42 (for reproducibility)")
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
        println(Printf.format(Printf.Format("  %-$(NAME_WIDTH)s %10s %15s %12s %12s %12s %12s"),
            "Optimizer", "Mean Rank", "Mean Final", "Std", "Accuracy", "Acc Δ", "Time (s)"))
        println("  " * "-"^(NAME_WIDTH + 77))

        for opt_name in OPTIMIZER_ORDER
            if haskey(result["aggregate"], opt_name)
                agg = result["aggregate"][opt_name]
                rank_str = haskey(agg, "mean_rank") ? @sprintf("%.2f", agg["mean_rank"]) : "N/A"
                acc_delta = agg["mean_accuracy_improvement"] * 100
                acc_delta_str = acc_delta >= 0 ? "+$(Printf.@sprintf("%.1f", acc_delta))" : Printf.@sprintf("%.1f", acc_delta)
                println(Printf.format(Printf.Format("  %-$(NAME_WIDTH)s %10s %15.6e %12.2e %11.1f%% %12s%% %12.2f"),
                    opt_name, rank_str, agg["mean_final_loss"], agg["std_final_loss"],
                    agg["mean_final_accuracy"] * 100,
                    acc_delta_str, agg["mean_time"]))
            end
        end
    end
end

# Compute global ranking across all configs
global_ranks = Dict{String, Vector{Float64}}()
for result in all_results
    if !haskey(result, "error") && haskey(result, "aggregate") && !haskey(result["aggregate"], "error")
        for opt_name in OPTIMIZER_ORDER
            if haskey(result["aggregate"], opt_name) && haskey(result["aggregate"][opt_name], "mean_rank")
                if !haskey(global_ranks, opt_name)
                    global_ranks[opt_name] = Float64[]
                end
                push!(global_ranks[opt_name], result["aggregate"][opt_name]["mean_rank"])
            end
        end
    end
end

println("\n" * "-"^70)
println("OPTIMIZER RANKING (average rank across all configs)")
println("-"^70)
if !isempty(global_ranks)
    ranked_opts = sort([(opt, _mean(ranks)) for (opt, ranks) in global_ranks], by=x -> x[2])
    println()
    println(Printf.format(Printf.Format("  %-$(NAME_WIDTH)s %12s %10s"),
        "Optimizer", "Avg Rank", "# Configs"))
    println("  " * "-"^(NAME_WIDTH + 26))
    for (opt_name, avg_rank) in ranked_opts
        n_configs = length(global_ranks[opt_name])
        println(Printf.format(Printf.Format("  %-$(NAME_WIDTH)s %12.2f %10d"),
            opt_name, avg_rank, n_configs))
    end
else
    println("  No ranking data available")
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

        # Add global ranking to JSON
        json_output["global_ranking"] = Dict{String, Any}(
            opt => Dict("avg_rank" => _mean(ranks), "n_configs" => length(ranks))
            for (opt, ranks) in global_ranks if !isempty(ranks)
        )

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
        save_to_logs(filepath)
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
