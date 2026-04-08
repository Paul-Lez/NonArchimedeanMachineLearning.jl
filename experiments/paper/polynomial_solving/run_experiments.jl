"""
Polynomial Solving Experiment Runner

Minimize |f(z)| where f is a random polynomial with a guaranteed root in Z_p^n
(the Gauss point / unit ball). Tests varying numbers of variables and degrees.

Usage:
    julia --project=. experiments/paper/polynomial_solving/run_experiments.jl [FLAGS]

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
    include(joinpath(@__DIR__, "util.jl"))
end

args = parse_experiment_args(ARGS)

# ============================================================================
# Default configurations
# ============================================================================

default_configs = [
    Dict("name" => "p2_1var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 2, "num_samples" => 2),
    Dict("name" => "p2_2var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 2, "num_samples" => 2),
]

load_config_file(@__DIR__, args)
configs = load_configs(args, default_configs)

# ============================================================================
# Run single sample
# ============================================================================

@everywhere function run_single_sample(config::Dict, sample_num::Int, args)
    p = config["prime"]
    prec = config["prec"]
    num_vars = config["num_vars"]
    degree = config["degree"]

    K = PadicField(p, prec)

    # Generate random polynomial with guaranteed root
    loss, root = generate_polynomial_solving_problem(p, prec, num_vars, degree)

    # Initialize starting point (Gauss point)
    initial_param = generate_initial_point(num_vars, K)
    initial_loss = loss.eval([initial_param])[1]

    # Get suite configs (SuiteName => {OptName => Setup})
    suite_configs = get_optimizer_configs(config, args)

    # Holder for all results: SuiteName => {OptName => Result}
    suite_results = Dict{String, Any}()

    # Run each suite independently
    for (suite_name, opt_configs) in suite_configs
        # Run all optimizers in this suite serially
        suite_results[suite_name] = run_all_optimizers_serial(
            opt_configs, initial_param, loss, args.n_epochs
        )
    end

    return Dict{String, Any}(
        "sample_num" => sample_num,
        "initial_loss" => initial_loss,
        "num_vars" => num_vars,
        "degree" => degree,
        "suites" => suite_results,
    )
end

@everywhere function run_experiment_suite(configs::Vector{Dict{String, Any}}, sample_num::Int, args)
    results = []
    for (config_idx, config) in enumerate(configs)
        sample_result = try
            run_single_sample(config, sample_num, args)
        catch e 
            # Capture error and backtrace for proper debugging on master
            # bt = catch_backtrace()
            # @error "Sample failed" exception=(e, bt)
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


# ============================================================================
# Print helpers
# ============================================================================

function print_sample_result(config::Dict, sample_num::Int, sample_result::Dict)
    io = IOBuffer()
    println(io, "\n  [$(config["name"]) sample $sample_num/$(config["num_samples"])]")
    println(io, @sprintf("    Initial: %.6e", sample_result["initial_loss"]))
    
    suites = get(sample_result, "suites", Dict())
    # Sort suite names for consistent output
    for suite_name in sort(collect(keys(suites)))
        println(io, "    » Suite: $suite_name")
        opt_results = suites[suite_name]
        # Sort optimizer names for consistent output
        for opt_name in sort(collect(keys(opt_results)))
            r = opt_results[opt_name]
            if !haskey(r, "error")
                println(io, Printf.format(
                    Printf.Format("      %-$(NAME_WIDTH)s Final: %.6e (Δ: %.6e, %.1f%%)"),
                    opt_name, r["final_loss"], r["improvement"],
                    r["improvement_ratio"] * 100))
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
println("Polynomial Solving Experiments")
println("Start time: $(Dates.now())")
println("Epochs: $(args.n_epochs), Quick: $(args.quick_mode)")
println("="^70)

# Announce each experiment up front
for (i, config) in enumerate(configs)
    println("\n" * "#"^70)
    println("# EXPERIMENT $i/$(length(configs)): $(config["name"])")
    println("#"^70)
    println("Prime: $(config["prime"]), Variables: $(config["num_vars"]), " *
            "Degree: $(config["degree"])")
    println("Samples: $(config["num_samples"]), Epochs: $(args.n_epochs)")
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
# We pass (configs, sample_num, args) to each worker.
experiments = [(configs, samples, args) for samples in worker_samples]

println("Running on $(nworkers()) workers...")

# Helper to destructure pmap arguments
@everywhere dispatch_job(job) = run_experiment_suites(job[1], job[2], job[3])

results = pmap(dispatch_job, experiments)

println("\nAll experiments finished. Collecting results...")

for (worker_idx, (configs, samples, args)) in enumerate(experiments)
    for sample_num in samples
        # collect the results of sample `sample_num` from worker `worker_idx`
        # This is whole experiment run over all possible configurations for a single sample per configuration
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

# for (ci, sample) in tasks
#     config = configs[ci]
#     sample_result = try
#         run_single_sample(config, sample)
#     catch e
#         println("    ✗ $(config["name"]) sample $sample failed: $e")
#         flush(stdout)
#         Dict{String, Any}("sample_num" => sample, "error" => string(e))
#     end

#     push!(results_by_config[ci]["samples"], sample_result)

#     if !haskey(sample_result, "error")
#         print_sample_result(config, sample, sample_result)
#     end
# end

all_results = results_by_config

# ============================================================================
# Save results
# ============================================================================

if args.save_results
    metadata = build_metadata(
        experiment_type="polynomial_solving",
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
        "polynomial_solving_results_$(timestamp)_raw.json"
    end
    filepath = joinpath(@__DIR__, output_fn)

    save_raw_results(all_results, metadata, filepath)
    save_to_logs(filepath)
end

println("\n✓ All experiments complete!")
