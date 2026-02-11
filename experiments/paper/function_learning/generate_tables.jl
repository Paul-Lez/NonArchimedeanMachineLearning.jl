"""
LaTeX Table Generator for Function Learning Results

Reads JSON results from run_experiments.jl and produces LaTeX tables
in a single unified document suitable for inclusion in a paper.

Usage:
    julia --project=. experiments/paper/function_learning/generate_tables.jl <results.json>
    julia --project=. experiments/paper/function_learning/generate_tables.jl <results.json> --output tables.tex
    julia --project=. experiments/paper/function_learning/generate_tables.jl <results.json> --stdout

Flags:
    --output FILE Write to specified .tex file (default: function_learning_tables.tex)
    --stdout      Print tables to stdout instead of writing file
"""

using JSON
using Printf

# ============================================================================
# Parse arguments
# ============================================================================

if length(ARGS) < 1
    println("Usage: julia generate_tables.jl <results.json> [--output FILE] [--stdout]")
    exit(1)
end

json_file = ARGS[1]
print_stdout = "--stdout" in ARGS

output_file = "function_learning_tables.tex"
for (i, arg) in enumerate(ARGS)
    if arg == "--output" && i < length(ARGS)
        global output_file = ARGS[i+1]
    end
end

# ============================================================================
# Load results
# ============================================================================

if !isfile(json_file)
    println("Error: File not found: $json_file")
    exit(1)
end

data = JSON.parsefile(json_file)
# Handle both cases: direct array or dict with "experiments" key
if isa(data, AbstractDict)
    experiments = haskey(data, "experiments") ? data["experiments"] : data
    metadata = haskey(data, "metadata") ? data["metadata"] : Dict()
else
    experiments = data
    metadata = Dict()
end

# Ensure experiments is a proper array/vector
if !isa(experiments, AbstractVector)
    # If it's a dict, it might be indexed experiments - convert to array
    experiments = [experiments[k] for k in sort(collect(keys(experiments)))]
end

println("Loaded $(length(experiments)) experiments from $json_file")
println()

# ============================================================================
# Helper: format a number in scientific notation for LaTeX
# ============================================================================

function latex_sci_compact(x::Float64; digits::Int=1)
    if x == 0.0
        return "0"
    end
    exp = floor(Int, log10(abs(x)))
    mantissa = x / 10.0^exp
    if exp == 0
        return Printf.format(Printf.Format("%.$(digits)f"), x)
    else
        return Printf.format(Printf.Format("%.$(digits)f{\\scriptstyle\\times 10^{%d}}"), mantissa, exp)
    end
end

# ============================================================================
# Helper: Get optimizer names from first valid experiment
# ============================================================================

function get_optimizer_names(experiments)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return []
    end

    agg = valid[1]["aggregate"]
    if haskey(agg, "error")
        return []
    end

    # Return optimizer names in a consistent order
    opt_names = collect(keys(agg))
    preferred_order = ["Greedy", "Greedy-deg2", "MCTS-50", "MCTS-100", "DAG-MCTS-100", "UCT", "HOO", "DOO"]
    ordered = []
    for name in preferred_order
        if name in opt_names
            push!(ordered, name)
        end
    end
    # Add any remaining optimizers not in preferred order
    for name in opt_names
        if !(name in ordered)
            push!(ordered, name)
        end
    end
    return ordered
end

optimizer_order = get_optimizer_names(experiments)
println("Optimizers: $(join(optimizer_order, ", "))")
println()

# ============================================================================
# Table 1: Summary table (mean final loss across all experiments)
# ============================================================================

function generate_summary_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments to tabulate\n"
    end

    n_opts = length(optimizer_order)

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Function learning: mean final loss across optimizers. " *
                 "Lower is better. Values are averaged over multiple random problem instances.}")
    push!(lines, "\\label{tab:funclearn-summary}")

    # Build column spec
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    # Header row
    header = "Configuration"
    for opt_name in optimizer_order
        header *= " & $opt_name"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    # Data rows
    for exp in valid
        config = exp["config"]
        agg = exp["aggregate"]

        # Find best (minimum) mean final loss for bolding
        best_loss = Inf
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                loss = agg[opt_name]["mean_final_loss"]
                if loss < best_loss
                    best_loss = loss
                end
            end
        end

        name = replace(config["name"], "_" => "\\_")
        row = name
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                loss = agg[opt_name]["mean_final_loss"]
                formatted = "\$$(latex_sci_compact(loss))\$"
                if loss == best_loss
                    row *= " & \\textbf{$formatted}"
                else
                    row *= " & $formatted"
                end
            else
                row *= " & ---"
            end
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table 2: Timing comparison
# ============================================================================

function generate_timing_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for timing table\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Mean wall-clock time (seconds) per optimizer for function learning.}")
    push!(lines, "\\label{tab:funclearn-timing}")

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "Configuration"
    for opt_name in optimizer_order
        header *= " & $opt_name"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    for exp in valid
        config = exp["config"]
        agg = exp["aggregate"]
        name = replace(config["name"], "_" => "\\_")

        row = name
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                t = agg[opt_name]["mean_time"]
                row *= " & " * @sprintf("%.2f", t)
            else
                row *= " & ---"
            end
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table 3: Detailed results (per configuration)
# ============================================================================

function generate_detailed_tables(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for detailed tables\n"
    end

    all_lines = String[]

    for (idx, exp) in enumerate(valid)
        config = exp["config"]
        agg = exp["aggregate"]
        name = replace(config["name"], "_" => "\\_")

        lines = String[]
        push!(lines, "\\begin{table}[ht]")
        push!(lines, "\\centering")
        push!(lines, "\\caption{Detailed results for configuration: $(name). " *
                     "Shows mean final loss, improvement ratio (\\%), and wall-clock time.}")
        push!(lines, "\\label{tab:funclearn-detail-$(idx)}")
        push!(lines, "\\begin{tabular}{lrrr}")
        push!(lines, "\\toprule")
        push!(lines, "Optimizer & Final Loss & Improv.~(\\%) & Time (s) \\\\")
        push!(lines, "\\midrule")

        # Find best loss for bolding
        best_loss = Inf
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                loss = agg[opt_name]["mean_final_loss"]
                if loss < best_loss
                    best_loss = loss
                end
            end
        end

        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                stats = agg[opt_name]
                loss = stats["mean_final_loss"]
                loss_str = @sprintf("\$%.2e\$", loss)
                if loss == best_loss
                    loss_str = "\\textbf{$loss_str}"
                end
                improv_str = @sprintf("%.1f", stats["mean_improvement_ratio"] * 100)
                time_str = @sprintf("%.2f", stats["mean_time"])

                push!(lines, "$opt_name & $loss_str & $improv_str & $time_str \\\\")
            end
        end

        push!(lines, "\\bottomrule")
        push!(lines, "\\end{tabular}")
        push!(lines, "\\end{table}")

        push!(all_lines, join(lines, "\n"))
        push!(all_lines, "")  # Blank line between tables
    end

    return join(all_lines, "\n") * "\n"
end

# ============================================================================
# Table 4: Overall optimizer comparison (aggregate across all configs)
# ============================================================================

function generate_optimizer_aggregate_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for optimizer aggregate\n"
    end

    # Compute aggregate statistics per optimizer across all experiments
    optimizer_stats = Dict{String, Dict{String, Vector{Float64}}}()

    for opt_name in optimizer_order
        optimizer_stats[opt_name] = Dict(
            "final_loss" => Float64[],
            "improvement_ratio" => Float64[],
            "time" => Float64[]
        )
    end

    for exp in valid
        agg = exp["aggregate"]
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                stats = agg[opt_name]
                push!(optimizer_stats[opt_name]["final_loss"], stats["mean_final_loss"])
                push!(optimizer_stats[opt_name]["improvement_ratio"], stats["mean_improvement_ratio"])
                push!(optimizer_stats[opt_name]["time"], stats["mean_time"])
            end
        end
    end

    # Compute means
    _mean(x) = isempty(x) ? 0.0 : sum(x) / length(x)

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Overall optimizer comparison aggregated across all function learning configurations. " *
                 "Shows mean performance metrics.}")
    push!(lines, "\\label{tab:funclearn-optimizer-aggregate}")
    push!(lines, "\\begin{tabular}{lrrrr}")
    push!(lines, "\\toprule")
    push!(lines, "Optimizer & Mean Loss & Improv.~(\\%) & Mean Time (s) & Configs \\\\")
    push!(lines, "\\midrule")

    # Find best mean loss for bolding
    best_mean_loss = Inf
    for opt_name in optimizer_order
        if !isempty(optimizer_stats[opt_name]["final_loss"])
            mean_loss = _mean(optimizer_stats[opt_name]["final_loss"])
            if mean_loss < best_mean_loss
                best_mean_loss = mean_loss
            end
        end
    end

    for opt_name in optimizer_order
        if !isempty(optimizer_stats[opt_name]["final_loss"])
            mean_loss = _mean(optimizer_stats[opt_name]["final_loss"])
            mean_improv = _mean(optimizer_stats[opt_name]["improvement_ratio"])
            mean_time = _mean(optimizer_stats[opt_name]["time"])
            n_configs = length(optimizer_stats[opt_name]["final_loss"])

            loss_str = @sprintf("\$%.2e\$", mean_loss)
            if mean_loss == best_mean_loss
                loss_str = "\\textbf{$loss_str}"
            end
            improv_str = @sprintf("%.1f", mean_improv * 100)
            time_str = @sprintf("%.2f", mean_time)

            push!(lines, "$opt_name & $loss_str & $improv_str & $time_str & $n_configs \\\\")
        end
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table 5: Function type comparison (zero vs one)
# ============================================================================

function generate_function_type_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for function type table\n"
    end

    # Group by target function type
    by_type = Dict{String, Vector}()
    for exp in valid
        if haskey(exp["config"], "target_fn")
            fn_type = exp["config"]["target_fn"]
            if !haskey(by_type, fn_type)
                by_type[fn_type] = []
            end
            push!(by_type[fn_type], exp)
        end
    end

    if length(by_type) < 2
        return "% Not enough function types for comparison table\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Performance comparison by target function type. " *
                 "Shows mean final loss aggregated over configurations for each function type.}")
    push!(lines, "\\label{tab:funclearn-function-type}")

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "Function Type"
    for opt_name in optimizer_order
        header *= " & $opt_name"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    _mean(x) = isempty(x) ? 0.0 : sum(x) / length(x)

    for fn_type in sort(collect(keys(by_type)))
        exps = by_type[fn_type]
        row = replace(fn_type, "_" => "\\_")

        for opt_name in optimizer_order
            losses = Float64[]
            for exp in exps
                if haskey(exp["aggregate"], opt_name) && !haskey(exp["aggregate"][opt_name], "error")
                    push!(losses, exp["aggregate"][opt_name]["mean_final_loss"])
                end
            end

            if !isempty(losses)
                mean_loss = _mean(losses)
                row *= " & \$$(latex_sci_compact(mean_loss))\$"
            else
                row *= " & ---"
            end
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Generate unified document
# ============================================================================

function generate_unified_document(experiments, optimizer_order)
    lines = String[]

    # Document header
    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Function Learning Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "%")
    push!(lines, "% This file contains all tables for the function learning experiments.")
    push!(lines, "% Include in your paper with: \\input{function_learning_tables.tex}")
    push!(lines, "%")
    push!(lines, "% Required packages: booktabs")
    push!(lines, "% ============================================================================")
    push!(lines, "")

    # Generate all tables
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Summary (mean final loss)")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_summary_table(experiments, optimizer_order))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Timing comparison")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_timing_table(experiments, optimizer_order))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Overall optimizer comparison")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_optimizer_aggregate_table(experiments, optimizer_order))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Function type comparison")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_function_type_table(experiments, optimizer_order))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Tables: Detailed results per configuration")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_detailed_tables(experiments, optimizer_order))

    return join(lines, "\n")
end

# ============================================================================
# Main execution
# ============================================================================

using Dates

document = generate_unified_document(experiments, optimizer_order)

if print_stdout
    println(document)
else
    filepath = joinpath(dirname(json_file), output_file)
    open(filepath, "w") do f
        write(f, document)
    end
    println("✓ Wrote unified tables to: $filepath")
end

println("\nDone!")
