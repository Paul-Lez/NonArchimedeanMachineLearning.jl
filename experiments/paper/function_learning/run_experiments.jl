"""
Function Learning Experiment Runner

Learn binary classification functions using polynomial approximation with
cross-entropy loss. Tests learning various target functions (zero, one, random)
over p-adic inputs.

Usage:
    julia --project=. experiments/paper/function_learning/run_experiments.jl [FLAGS]

Flags:
    --quick       Reduced epochs and simulations for smoke testing
    --save        Save results to JSON file
    --config      Use configurations from config.jl
    --paper       Use paper-ready configurations from paper_config.jl
    --epochs N    Override number of epochs
    --output FILE Override output filename
    --samples N   Override number of samples per config
    --selection-mode M   MCTS selection mode: BestValue, VisitCount, or BestLoss
    --degree D    Override tree branching degree
    --description TEXT   Experiment description
    --git-commit HASH   Git commit hash
"""

# ============================================================================
# Setup
# ============================================================================

using Distributed

# Ensure we have workers
if nworkers() == 1
    addprocs()
end

# Synchronize dependencies on all workers
@everywhere using Oscar
@everywhere include(joinpath(@__DIR__, "../../../src/NAML.jl"))
@everywhere using .NAML
@everywhere begin
    using Random, Dates, Printf
    include(joinpath(@__DIR__, "../util.jl"))
    include(joinpath(@__DIR__, "../experiment_utils.jl"))
end

args = parse_experiment_args(ARGS)

# ============================================================================
# Default configurations
# ============================================================================

default_configs = [
    Dict("name" => "p2_deg2_zero", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 10, "target_fn" => "zero",
         "threshold" => 0.5, "scale" => 0.1, "num_samples" => 2),
    Dict("name" => "p2_deg2_one", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 10, "target_fn" => "one",
         "threshold" => 0.5, "scale" => 0.1, "num_samples" => 2),
]

load_config_file(@__DIR__, args)
configs = load_configs(args, default_configs)

# ============================================================================
# Classification helpers
# ============================================================================

@everywhere begin
    """
    Create a loss function for learning a target function using polynomial
    approximation with cross-entropy loss and sigmoid activation.
    """
    function create_function_learning_loss(K, degree, n_points, target_fn, threshold, scale)
        p = Int(Oscar.prime(K))
        prec = Oscar.precision(K)
        x_values = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]

        if target_fn == "zero"
            y_values = [0.0 for _ in 1:n_points]
        elseif target_fn == "one"
            y_values = [1.0 for _ in 1:n_points]
        elseif target_fn == "random"
            y_values = [Float64(rand(0:1)) for _ in 1:n_points]
        else
            error("Unknown target function: $target_fn")
        end

        data = collect(zip(x_values, y_values))
        loss = polynomial_to_crossentropy_loss(data, degree, threshold, scale)
        return loss, data
    end

    """
    Compute classification accuracy for a polynomial classifier.
    """
    function compute_accuracy(coeffs, data, threshold, scale)
        function eval_polynomial(coeffs, x)
            result = coeffs[1]
            x_power = x
            for i in 2:length(coeffs)
                result += coeffs[i] * x_power
                x_power *= x
            end
            return result
        end

        correct = 0
        for (x, y) in data
            poly_val = eval_polynomial(coeffs, x)
            val_float = Float64(abs(poly_val))
            z = (val_float - threshold) / scale
            prob = 1.0 / (1.0 + exp(-z))
            prediction = prob > 0.5 ? 1.0 : 0.0
            if prediction == y
                correct += 1
            end
        end
        return correct / length(data)
    end
end

# ============================================================================
# Run single sample
# ============================================================================

@everywhere function run_single_sample(config::Dict, sample_num::Int, args)
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
    num_params = degree + 1
    param_center = [K(0) for _ in 1:num_params]
    initial_param = NAML.ValuationPolydisc(param_center, [0 for _ in 1:num_params])
    initial_loss = loss.eval([initial_param])[1]

    # Compute initial accuracy
    initial_coeffs = [NAML.unwrap(c) for c in NAML.center(initial_param)]
    initial_accuracy = compute_accuracy(initial_coeffs, data, threshold, scale)

    # Get suite configs (SuiteName => {OptName => Setup})
    suite_configs = get_optimizer_configs(config, args)

    # Holder for all results: SuiteName => {OptName => Result}
    suite_results = Dict{String, Any}()

    # Run each suite independently
    for (suite_name, opt_configs) in suite_configs
        # Run all optimizers in this suite serially
        suite_results[suite_name] = run_all_optimizers_serial(
            opt_configs, initial_param, loss, args.n_epochs;
            post_run_fn=post_run_fn
        )
    end

    return Dict{String, Any}(
        "sample_num" => sample_num,
        "initial_loss" => initial_loss,
        "initial_accuracy" => initial_accuracy,
        "data" => Dict(
            "x_abs_values" => [Float64(abs(x)) for (x, _) in data],
            "y_values" => [y for (_, y) in data],
        ),
        "suites" => suite_results,
    )
end

@everywhere function run_experiment_suite(configs::Vector{Dict{String, Any}}, sample_num::Int, args)
    results = []
    for (config_idx, config) in enumerate(configs)
        sample_result = try
            run_single_sample(config, sample_num, args)
        catch e 
            Dict{String, Any}("sample_num" => sample_num, "config_idx" => config_idx, "error" => string(e))
        end
        push!(results, sample_result)
    end 
    return results
end

@everywhere function run_experiment_suites(configs, sample_num::AbstractVector{Int}, args)
    results = Dict{Int, Any}()
    for s in sample_num
        results[s] = run_experiment_suite(configs, s, args)
    end
    return results
end

@everywhere dispatch_job(job) = run_experiment_suites(job[1], job[2], job[3])

# ============================================================================
# Print helpers
# ============================================================================

function print_sample_result(config::Dict, sample_num::Int, sample_result::Dict)
    io = IOBuffer()
    println(io, "\n  [$(config["name"]) sample $sample_num/$(config["num_samples"])]")
    println(io, @sprintf("    Initial loss: %.6e, accuracy: %.2f%%",
        sample_result["initial_loss"], sample_result["initial_accuracy"] * 100))
    
    suites = get(sample_result, "suites", Dict())
    for suite_name in sort(collect(keys(suites)))
        println(io, "    » Suite: $suite_name")
        opt_results = suites[suite_name]
        for opt_name in sort(collect(keys(opt_results)))
            r = opt_results[opt_name]
            if !haskey(r, "error")
                acc_imp = r["accuracy_improvement"] * 100
                acc_imp_str = acc_imp >= 0 ? "+$(@sprintf("%.2f", acc_imp))" : @sprintf("%.2f", acc_imp)
                println(io, Printf.format(
                    Printf.Format("      %-$(NAME_WIDTH)s Final: %.6e, acc: %.2f%%  (loss: %.1f%%, acc: %s%%, %.2fs)"),
                    opt_name, r["final_loss"],
                    r["final_accuracy"] * 100,
                    r["improvement_ratio"] * 100,
                    acc_imp_str,
                    r["time"]))
            else
                println(io, "      %-$(NAME_WIDTH)s ERROR: $(r["error"])")
            end
        end
    end
    write(stdout, take!(io))
    flush(stdout)
end

# ============================================================================
# Main execution
# ============================================================================

Random.seed!(42)

println("="^70)
println("Function Learning Experiments")
println("Start time: $(Dates.now())")
println("Epochs: $(args.n_epochs), Quick: $(args.quick_mode)")
println("="^70)

# Announce each experiment up front
for (i, config) in enumerate(configs)
    println("\n" * "#"^70)
    println("# EXPERIMENT $i/$(length(configs)): $(config["name"])")
    println("#"^70)
    println("  Prime: $(config["prime"]), Degree: $(config["degree"]), " *
            "Points: $(config["n_points"]), Target: $(config["target_fn"])")
    println("  Samples: $(config["num_samples"]), Threshold: $(config["threshold"]), Scale: $(config["scale"])")
end

# Per-config result holders (preserves config order in output)
results_by_config = [Dict{String, Any}("config" => config, "samples" => Any[])
                     for config in configs]

# Hacky, but currently all experiments have the same number of samples.
num_samples = first([config["num_samples"] for config in configs])

# Divide samples into chunks for each worker to minimize redundant compilation
chunk_size = ceil(Int, num_samples / nworkers())
worker_samples = Base.Iterators.partition(1:num_samples, chunk_size)

println("We use $(nworkers()) workers, and each worker will run $chunk_size samples.")

# Flatten (configs, sample_num) into a single task list for pmap.
experiments = [(configs, samples, args) for samples in worker_samples]

println("Running on $(nworkers()) workers...")

results = pmap(dispatch_job, experiments)

println("\nAll experiments finished. Collecting results...")

for (worker_idx, (_, samples, _)) in enumerate(experiments)
    for sample_num in samples
        sample_suite_rs = results[worker_idx][sample_num]
        
        for (ci, sample_result) in enumerate(sample_suite_rs)
            push!(results_by_config[ci]["samples"], sample_result)
        
            if !haskey(sample_result, "error")
                print_sample_result(configs[ci], sample_num, sample_result)
            else
                println("    ✗ $(configs[ci]["name"]) sample $sample_num failed: $(sample_result["error"])")
            end
        end
    end
end

all_results = results_by_config

# ============================================================================
# Save results
# ============================================================================

if args.save_results
    metadata = build_metadata(
        experiment_type="function_learning",
        n_epochs=args.n_epochs,
        quick_mode=args.quick_mode,
        suites=collect(keys(get_optimizer_configs(configs[1], args))),
        description=args.description,
        git_commit=args.git_commit,
    )

    output_fn = if !isnothing(args.output_filename)
        args.output_filename
    else
        timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        "function_learning_results_$(timestamp)_raw.json"
    end
    filepath = joinpath(@__DIR__, output_fn)

    save_raw_results(all_results, metadata, filepath)
    save_to_logs(filepath)
end

println("\n✓ All experiments complete!")
