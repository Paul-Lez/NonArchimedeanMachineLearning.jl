"""
Statistical aggregation for NAML paper experiments.

Reads a raw JSON file produced by run_experiments.jl (which contains only
per-sample results and no aggregate statistics) and produces a stats JSON file
with computed aggregates: mean/std/min/max across samples, per-sample rankings,
and cross-experiment global rankings.

Usage:
    julia --project=. experiments/paper/make_stats.jl <raw_results.json> [--output stats.json]

The output file defaults to replacing "_raw.json" with "_stats.json" in the input filename,
or appending "_stats" before ".json" if no "_raw" suffix is found.
"""

using JSON
using Printf
using Dates

# Load shared stats utilities
include(joinpath(@__DIR__, "stats_utils.jl"))

# ============================================================================
# Parse arguments
# ============================================================================

if length(ARGS) < 1
    println("Usage: julia make_stats.jl <raw_results.json> [--output stats.json]")
    exit(1)
end

input_file = ARGS[1]

# Determine output filename
output_file = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--output" && i < length(ARGS)
        output_file = ARGS[i+1]
    end
end

if isnothing(output_file)
    if endswith(input_file, "_raw.json")
        output_file = replace(input_file, "_raw.json" => "_stats.json")
    else
        output_file = replace(input_file, ".json" => "_stats.json")
    end
end

# ============================================================================
# Load raw results
# ============================================================================

if !isfile(input_file)
    println("Error: File not found: $input_file")
    exit(1)
end

println("Loading raw results from: $input_file")
data = JSON.parsefile(input_file)

metadata = get(data, "metadata", Dict())
experiments = get(data, "experiments", data)

if !isa(experiments, AbstractVector)
    experiments = [experiments[k] for k in sort(collect(keys(experiments)))]
end

# Get experiment type for type-specific processing
experiment_type = get(metadata, "experiment_type", "unknown")

# Get optimizer order
optimizer_order = get(metadata, "optimizer_order", String[])
if isempty(optimizer_order)
    # Try to infer from data
    for exp in experiments
        if haskey(exp, "samples")
            for sample in exp["samples"]
                if haskey(sample, "optimizers")
                    optimizer_order = collect(keys(sample["optimizers"]))
                    break
                end
            end
        end
        !isempty(optimizer_order) && break
    end
end

println("Experiment type: $experiment_type")
println("Found $(length(experiments)) experiment configurations")
println("Optimizers: $(join(optimizer_order, ", "))")
println()

# ============================================================================
# Determine extra fields based on experiment type
# ============================================================================

extra_fields = String[]
if experiment_type == "function_learning"
    extra_fields = ["final_accuracy", "accuracy_improvement"]
end

# ============================================================================
# Compute statistics for each experiment
# ============================================================================

println("Computing statistics...")

for (i, exp) in enumerate(experiments)
    if haskey(exp, "error")
        println("  [$(i)/$(length(experiments))] Skipping errored experiment: $(get(exp, "config", Dict()))")
        continue
    end

    config_name = get(get(exp, "config", Dict()), "name", "unknown")
    samples = get(exp, "samples", [])
    valid_samples = filter(s -> !haskey(s, "error"), samples)

    println("  [$(i)/$(length(experiments))] $(config_name): $(length(valid_samples))/$(length(samples)) valid samples")

    # suites_aggregate will store: SuiteName => {OptName => AggStats}
    exp["suites_aggregate"] = Dict{String, Any}()

    # 1. First, compute per-sample rankings for EACH suite independently
    for sample in valid_samples
        if haskey(sample, "suites")
            for (suite_name, opt_results) in sample["suites"]
                compute_sample_rankings!(opt_results)
            end
        end
    end

    # 2. Identify all suites present in this experiment
    all_suites = Set{String}()
    for sample in valid_samples
        if haskey(sample, "suites")
            for suite_name in keys(sample["suites"])
                push!(all_suites, suite_name)
            end
        end
    end

    # 3. Compute aggregate statistics for EACH suite
    for suite_name in sort(collect(all_suites))
        # Extract the results for this suite across all samples
        samples_in_suite = [s["suites"][suite_name] for s in valid_samples if haskey(s["suites"], suite_name)]
        
        if !isempty(samples_in_suite)
            exp["suites_aggregate"][suite_name] = compute_aggregate_stats(
                samples_in_suite, suite_name; extra_fields=extra_fields
            )

            # Print summary for this suite
            println("    » Suite: $suite_name")
            agg = exp["suites_aggregate"][suite_name]
            # Sort opt names for consistent output
            for opt_name in sort(collect(keys(agg)))
                if !haskey(agg[opt_name], "error")
                    stats = agg[opt_name]
                    loss_str = @sprintf("%.4e ± %.4e", stats["mean_final_loss"],
                                        get(stats, "std_final_loss", 0.0))
                    rank_str = haskey(stats, "mean_rank") ? @sprintf("%.2f", stats["mean_rank"]) : "N/A"
                    println("      $(rpad(opt_name, 25)) loss: $loss_str  rank: $rank_str")
                end
            end
        end
    end
end

# ============================================================================
# Compute global ranking across experiments (per suite)
# ============================================================================

println("\nComputing global ranking per suite...")
global_ranking = compute_global_ranking(experiments)

if !isempty(global_ranking)
    for (suite_name, opt_ranks) in sort(collect(global_ranking); by=x -> x[1])
        println("\nGlobal ranking for suite: $suite_name")
        sorted = sort(collect(opt_ranks), by=x -> x[2]["avg_rank"])
        for (opt_name, stats) in sorted
            println("  $(rpad(opt_name, 25)) avg_rank: $(@sprintf("%.2f", stats["avg_rank"]))  (n_configs: $(stats["n_configs"]))")
        end
    end
end

# ============================================================================
# Write stats JSON
# ============================================================================

# Update metadata to indicate this is a stats file
metadata["stats_computed"] = true
metadata["stats_timestamp"] = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

output = Dict{String, Any}(
    "metadata" => metadata,
    "experiments" => experiments,
    "global_ranking" => global_ranking,
)

open(output_file, "w") do f
    JSON.print(f, output, 2)
end

println("\n✓ Stats written to: $output_file")
println("Done!")
