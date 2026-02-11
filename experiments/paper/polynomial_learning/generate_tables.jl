"""
LaTeX Table Generator for Polynomial Learning Results

Reads JSON results from run_experiments.jl and produces LaTeX tables
in a single unified document suitable for inclusion in a paper.

Usage:
    julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <results.json>
    julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <results.json> --output tables.tex
    julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <results.json> --stdout

Flags:
    --output FILE Write to specified .tex file (default: polynomial_learning_tables.tex)
    --stdout      Print tables to stdout instead of writing file
"""

using JSON
using Printf
using Dates

# ============================================================================
# Parse arguments
# ============================================================================

if length(ARGS) < 1
    println("Usage: julia generate_tables.jl <results.json> [--output FILE] [--stdout]")
    exit(1)
end

json_file = ARGS[1]
print_stdout = "--stdout" in ARGS

output_file = "polynomial_learning_tables.tex"
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
metadata = data["metadata"]
experiments = data["experiments"]
optimizer_order = metadata["optimizer_order"]

println("Loaded $(length(experiments)) experiments from $json_file")
println("Optimizers: $(join(optimizer_order, ", "))")
println("Epochs: $(metadata["n_epochs"]), Quick mode: $(metadata["quick_mode"])")
println()

# ============================================================================
# Helper: format a number in scientific notation for LaTeX
# ============================================================================

function latex_sci(x::Float64; digits::Int=2)
    if x == 0.0
        return "0"
    end
    exp = floor(Int, log10(abs(x)))
    mantissa = x / 10.0^exp
    if exp == 0
        return Printf.format(Printf.Format("%.$(digits)f"), x)
    else
        return Printf.format(Printf.Format("%.$(digits)f \\times 10^{%d}"), mantissa, exp)
    end
end

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
# Table 1: Summary table (all experiments, all optimizers)
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
    push!(lines, "\\caption{Polynomial learning: mean final loss across optimizers. " *
                 "Lower is better. Values are averaged over multiple random problem instances.}")
    push!(lines, "\\label{tab:poly-learning-summary}")

    # Build column spec
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    # Header row
    header = "Experiment"
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
# Table 2: Detailed table with time and improvement ratio
# ============================================================================

function generate_detailed_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments to tabulate\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Detailed polynomial learning results: " *
                 "mean final loss, improvement ratio (\\%), and wall-clock time (s).}")
    push!(lines, "\\label{tab:poly-learning-detailed}")
    push!(lines, "\\begin{tabular}{llrrr}")
    push!(lines, "\\toprule")
    push!(lines, "Experiment & Optimizer & Final Loss & Improv.~(\\%) & Time (s) \\\\")
    push!(lines, "\\midrule")

    for (ei, exp) in enumerate(valid)
        config = exp["config"]
        agg = exp["aggregate"]
        name = replace(config["name"], "_" => "\\_")

        for (oi, opt_name) in enumerate(optimizer_order)
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                stats = agg[opt_name]
                loss_str = @sprintf("\$%.2e\$", stats["mean_final_loss"])
                improv_str = @sprintf("%.1f", stats["mean_improvement_ratio"] * 100)
                time_str = @sprintf("%.2f", stats["mean_time"])

                exp_label = oi == 1 ? name : ""
                push!(lines, "$exp_label & $opt_name & $loss_str & $improv_str & $time_str \\\\")
            end
        end

        if ei < length(valid)
            push!(lines, "\\midrule")
        end
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table 3: Degree sweep table (if applicable)
# ============================================================================

function generate_degree_sweep_table(experiments, optimizer_order)
    # Filter experiments that vary degree with the same prime
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if length(valid) < 2
        return "% Not enough experiments for degree sweep table\n"
    end

    # Group by prime
    by_prime = Dict{Int, Vector}()
    for exp in valid
        p = exp["config"]["prime"]
        if !haskey(by_prime, p)
            by_prime[p] = []
        end
        push!(by_prime[p], exp)
    end

    # Find primes with multiple degrees
    sweep_primes = filter(p -> length(by_prime[p]) >= 2, collect(keys(by_prime)))
    if isempty(sweep_primes)
        return "% No degree sweep detected (need multiple degrees for same prime)\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Effect of polynomial degree on learning performance " *
                 "(mean final loss). Each row is a different degree.}")
    push!(lines, "\\label{tab:poly-learning-degree}")

    n_opts = length(optimizer_order)
    col_spec = "cc" * "c"^n_opts
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "\$p\$ & Degree"
    for opt_name in optimizer_order
        header *= " & $opt_name"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    for p in sort(sweep_primes)
        exps = sort(by_prime[p], by=e -> e["config"]["degree"])
        for (i, exp) in enumerate(exps)
            config = exp["config"]
            agg = exp["aggregate"]

            prime_label = i == 1 ? string(p) : ""
            row = "$prime_label & $(config["degree"])"

            for opt_name in optimizer_order
                if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                    loss = agg[opt_name]["mean_final_loss"]
                    row *= " & \$$(latex_sci_compact(loss))\$"
                else
                    row *= " & ---"
                end
            end
            row *= " \\\\"
            push!(lines, row)
        end

        if p != last(sort(sweep_primes))
            push!(lines, "\\midrule")
        end
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table 4: Prime sweep table (if applicable)
# ============================================================================

function generate_prime_sweep_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if length(valid) < 2
        return "% Not enough experiments for prime sweep table\n"
    end

    # Group by degree
    by_degree = Dict{Int, Vector}()
    for exp in valid
        d = exp["config"]["degree"]
        if !haskey(by_degree, d)
            by_degree[d] = []
        end
        push!(by_degree[d], exp)
    end

    # Find degrees with multiple primes
    sweep_degrees = filter(d -> begin
        primes = unique([e["config"]["prime"] for e in by_degree[d]])
        length(primes) >= 2
    end, collect(keys(by_degree)))

    if isempty(sweep_degrees)
        return "% No prime sweep detected (need multiple primes for same degree)\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Effect of base prime on learning performance " *
                 "(mean final loss). Each row is a different prime.}")
    push!(lines, "\\label{tab:poly-learning-prime}")

    n_opts = length(optimizer_order)
    col_spec = "cc" * "c"^n_opts
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "Degree & \$p\$"
    for opt_name in optimizer_order
        header *= " & $opt_name"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    for d in sort(sweep_degrees)
        exps = sort(by_degree[d], by=e -> e["config"]["prime"])
        # Only include experiments that have distinct primes
        seen_primes = Set{Int}()
        for (i, exp) in enumerate(exps)
            config = exp["config"]
            if config["prime"] in seen_primes
                continue
            end
            push!(seen_primes, config["prime"])
            agg = exp["aggregate"]

            deg_label = i == 1 ? string(d) : ""
            row = "$deg_label & $(config["prime"])"

            for opt_name in optimizer_order
                if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                    loss = agg[opt_name]["mean_final_loss"]
                    row *= " & \$$(latex_sci_compact(loss))\$"
                else
                    row *= " & ---"
                end
            end
            row *= " \\\\"
            push!(lines, row)
        end

        if d != last(sort(sweep_degrees))
            push!(lines, "\\midrule")
        end
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table 5: Timing comparison
# ============================================================================

function generate_timing_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for timing table\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[ht]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Mean wall-clock time (seconds) per optimizer across experiments.}")
    push!(lines, "\\label{tab:poly-learning-timing}")

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "Experiment"
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
# Generate unified document
# ============================================================================

function generate_unified_document(experiments, optimizer_order)
    lines = String[]

    # Document header
    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Polynomial Learning Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "%")
    push!(lines, "% This file contains all tables for the polynomial learning experiments.")
    push!(lines, "% Include in your paper with: \\input{polynomial_learning_tables.tex}")
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
    push!(lines, "% Table: Detailed results")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_detailed_table(experiments, optimizer_order))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Degree sweep")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_degree_sweep_table(experiments, optimizer_order))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Prime sweep")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_prime_sweep_table(experiments, optimizer_order))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Timing comparison")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_timing_table(experiments, optimizer_order))

    return join(lines, "\n")
end

# ============================================================================
# Main execution
# ============================================================================

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
