"""
Polynomial Interpolation Benchmark Runner

Run multiple polynomial interpolation experiments with varying:
- Polynomial degrees
- Base fields (different primes)
- Number of data points

Usage:
    julia run_benchmarks.jl [--quick] [--save] [--config]

Flags:
    --quick: Use reduced epochs (5) and simulations (50) for quick testing
    --save: Save results to JSON file
    --config: Use experiment configurations from benchmark_config.jl

Examples:
    julia run_benchmarks.jl --quick                    # Quick test with default configs
    julia run_benchmarks.jl --config                   # Use benchmark_config.jl
    julia run_benchmarks.jl --config --quick --save    # Custom config, quick mode, save results
"""

include("../../src/NAML.jl")
include("util.jl")

using Oscar
using .NAML
using Printf
using Dates

# Parse command line arguments
quick_mode = "--quick" in ARGS
save_results = "--save" in ARGS
use_config_file = "--config" in ARGS

# ============================================================================
# Experiment Configuration
# ============================================================================

# Load configurations
if use_config_file
    include("benchmark_config.jl")
    configs = experiment_configs
    println("Loaded experiment configurations from benchmark_config.jl")
else
    # Default configurations (small, fast experiments)
    configs = [
        Dict("prime" => 2, "prec" => 20, "degree" => 2, "n_points" => 3, "name" => "2adic_deg2"),
        Dict("prime" => 2, "prec" => 20, "degree" => 3, "n_points" => 4, "name" => "2adic_deg3"),
        Dict("prime" => 3, "prec" => 15, "degree" => 3, "n_points" => 4, "name" => "3adic_deg3"),
    ]
end

# Set epochs based on mode
if quick_mode
    n_epochs = 5
    println("="^70)
    println("QUICK MODE: Running with only $n_epochs epochs per optimizer")
    println("="^70)
else
    n_epochs = 15
end

# ============================================================================
# Optimizer configurations
# ============================================================================

function get_optimizer_configs(K)
    return Dict(
        "Greedy" => Dict(
            "init" => (param, loss) -> begin
                state = 1
                config = (true, 0)
                NAML.greedy_descent_init(param, loss, state, config)
            end
        ),
        "MCTS" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick_mode ? 50 : 100,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=3
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "UCT" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.UCTConfig(
                    max_depth=10,
                    num_simulations=quick_mode ? 50 : 100,
                    exploration_constant=1.41,
                    degree=3
                )
                NAML.uct_descent_init(param, loss, config)
            end
        ),
        "HOO" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.HOOConfig(
                    rho=0.5,
                    nu1=0.1,
                    max_depth=quick_mode ? 10 : 15
                )
                NAML.hoo_descent_init(param, loss, config)
            end
        )
    )
end

# ============================================================================
# Run single experiment
# ============================================================================

function run_single_experiment(config::Dict)
    println("\n" * "="^70)
    println("Experiment: $(config["name"])")
    println("="^70)
    println("Prime: $(config["prime"]), Degree: $(config["degree"]), Points: $(config["n_points"])")
    println("-"^70)

    # Setup field
    p = config["prime"]
    prec = config["prec"]
    K = PadicField(p, prec)

    degree = config["degree"]
    n_points = config["n_points"]

    # Generate data
    println("\n[1] Generating data...")
    x_values = Vector{PadicFieldElem}()
    max_attempts = n_points * 100
    attempts = 0

    while length(x_values) < n_points && attempts < max_attempts
        x = generate_random_padic(p, prec, 0, 8)
        if !any(existing_x -> existing_x == x, x_values)
            push!(x_values, x)
        end
        attempts += 1
    end

    if length(x_values) < n_points
        error("Could not generate $n_points distinct points")
    end

    y_values = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]
    data = [(x, y) for (x, y) in zip(x_values, y_values)]

    # Create loss
    println("[2] Creating loss function...")
    loss = polynomial_to_linear_loss(data, degree, nothing)

    # Initialize starting point
    initial_param = generate_gauss_point(degree + 1, K)
    initial_loss = loss.eval([initial_param])[1]
    println("[3] Initial loss: $initial_loss")

    # Get optimizer configs
    opt_configs = get_optimizer_configs(K)

    # Results storage
    results = Dict{String, Any}()
    results["config"] = config
    results["initial_loss"] = initial_loss
    results["optimizers"] = Dict{String, Any}()

    # Run each optimizer
    for (opt_name, opt_setup) in opt_configs
        println("\n[Running: $opt_name]")

        try
            optim = opt_setup["init"](initial_param, loss)

            losses = Float64[]
            t_start = time()

            for epoch in 1:n_epochs
                current_loss = NAML.eval_loss(optim)
                push!(losses, current_loss)
                if epoch <= 3 || epoch == n_epochs
                    println(@sprintf("  Epoch %2d: loss = %.6e", epoch, current_loss))
                elseif epoch == 4
                    println("  ...")
                end
                NAML.step!(optim)
            end

            t_end = time()
            elapsed = t_end - t_start

            final_loss = NAML.eval_loss(optim)
            push!(losses, final_loss)

            results["optimizers"][opt_name] = Dict(
                "time" => elapsed,
                "final_loss" => final_loss,
                "losses" => losses,
                "improvement" => initial_loss - final_loss
            )

            println(@sprintf("  ✓ Completed in %.2f s, final loss = %.6e", elapsed, final_loss))

        catch e
            println("  ✗ Error: $e")
            results["optimizers"][opt_name] = Dict("error" => string(e))
        end
    end

    return results
end

# ============================================================================
# Run all experiments
# ============================================================================

println("\n" * "="^70)
println("POLYNOMIAL INTERPOLATION BENCHMARK RUNNER")
println("="^70)
println("Start time: $(Dates.now())")
println("Number of experiments: $(length(configs))")
println("Epochs per optimizer: $n_epochs")
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
        println("\n✗ Experiment $(config["name"]) failed with error:")
        println("  $e")
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
    println("  Prime: $(config["prime"]), Degree: $(config["degree"]), Points: $(config["n_points"])")
    println("  Initial loss: $(result["initial_loss"])")
    println()
    println(@sprintf("  %-15s %12s %15s %15s", "Optimizer", "Time (s)", "Final Loss", "Improvement"))
    println("  " * "-"^60)

    for opt_name in ["Greedy", "MCTS", "UCT", "HOO"]
        if haskey(result["optimizers"], opt_name)
            opt_result = result["optimizers"][opt_name]
            if haskey(opt_result, "error")
                println(@sprintf("  %-15s %s", opt_name, "ERROR"))
            else
                println(@sprintf("  %-15s %12.2f %15.6e %15.6e",
                    opt_name, opt_result["time"], opt_result["final_loss"], opt_result["improvement"]))
            end
        end
    end
end

println("\n" * "="^70)
println("End time: $(Dates.now())")
println("="^70)

# ============================================================================
# Save results if requested
# ============================================================================

if save_results
    try
        using JSON

        timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        filename = "benchmark_results_$(timestamp).json"

        # Convert results to JSON-serializable format
        json_results = []
        for result in all_results
            json_result = Dict{String, Any}()

            # Copy config
            json_result["config"] = result["config"]

            if haskey(result, "error")
                json_result["error"] = result["error"]
            else
                json_result["initial_loss"] = result["initial_loss"]
                json_result["optimizers"] = Dict{String, Any}()

                for (opt_name, opt_data) in result["optimizers"]
                    json_result["optimizers"][opt_name] = opt_data
                end
            end

            push!(json_results, json_result)
        end

        open(filename, "w") do f
            JSON.print(f, json_results, 2)
        end

        println("\n✓ Results saved to: $filename")
    catch e
        if e isa ArgumentError && occursin("Package JSON not found", string(e))
            println("\n⚠ Warning: JSON package not installed. Cannot save results.")
            println("  To enable JSON saving, run: julia -e 'using Pkg; Pkg.add(\"JSON\")'")
        else
            println("\n✗ Error saving results: $e")
        end
    end
end

println("\n✓ All experiments complete!")
