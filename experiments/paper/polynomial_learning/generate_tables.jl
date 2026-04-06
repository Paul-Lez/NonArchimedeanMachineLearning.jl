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
    --verbose     Include per-configuration detailed tables (default: aggregate only)
"""

using JSON
using Printf
using Dates

# ============================================================================
# Display names, ordering, and landscape helper
# ============================================================================

const DISPLAY_NAMES = Dict(
    "Random"           => "Random",
    "Best-First"       => "Best First Value",
    "Best-First-branch2"  => "Best First Branch 2",
    "Best-First-Gradient" => "Best First Gradient",
    "DOO"              => "DOO",
    "MCTS-k"           => "MCTS-\$k\$",
    "MCTS-5k"          => "MCTS-\$5k\$",
    "MCTS-10k"         => "MCTS-\$10k\$",
    "DAG-MCTS-k"       => "DAG-MCTS-\$k\$",
    "DAG-MCTS-5k"      => "DAG-MCTS-\$5k\$",
    "DAG-MCTS-10k"     => "DAG-MCTS-\$10k\$",
)
display_name(n) = get(DISPLAY_NAMES, n, n)

const DISPLAY_ORDER = ["Random", "Best-First", "Best-First-branch2", "Best-First-Gradient",
                       "MCTS-k", "MCTS-5k", "MCTS-10k",
                       "DAG-MCTS-k", "DAG-MCTS-5k", "DAG-MCTS-10k"]

# Wrap wide LaTeX tables in landscape environment with footnotesize font.
# Requires \usepackage{lscape} (or pdflscape) in document preamble.
function as_landscape(tex)
    result = String[]
    for line in split(tex, "\n")
        if line == "\\begin{table}[H]"
            push!(result, "\\begin{landscape}")
            push!(result, line)
        elseif line == "\\end{table}"
            push!(result, line)
            push!(result, "\\end{landscape}")
        elseif line == "\\centering"
            push!(result, line)
            push!(result, "\\footnotesize")
        else
            push!(result, line)
        end
    end
    return join(result, "\n")
end

# ============================================================================
# Parse arguments
# ============================================================================

if length(ARGS) < 1
    println("Usage: julia generate_tables.jl <results.json> [--output FILE] [--stdout]")
    exit(1)
end

json_file = ARGS[1]
print_stdout = "--stdout" in ARGS
verbose = "--verbose" in ARGS

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
# Reorder to preferred display order (extras appended at end)
optimizer_order = vcat(
    [n for n in DISPLAY_ORDER if n in optimizer_order],
    [n for n in optimizer_order if !(n in DISPLAY_ORDER)]
)

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
        return Printf.format(Printf.Format("%.$(digits)f\\text{e}%d"), mantissa, exp)
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
        return Printf.format(Printf.Format("%.$(digits)f\\text{e}%d"), mantissa, exp)
    end
end

# ============================================================================
# Helper: escape special characters for LaTeX
# ============================================================================

function escape_latex(s::String)
    # Escape LaTeX special characters in order
    # Note: backslash must be first, and we need to be careful with replacement order
    s = replace(s, "\\" => "\\textbackslash{}")
    s = replace(s, "~" => "\\~{}")
    s = replace(s, "^" => "\\^{}")
    s = replace(s, "_" => "\\_")
    s = replace(s, "#" => "\\#")
    s = replace(s, "&" => "\\&")
    s = replace(s, "%" => "\\%")
    s = replace(s, "\$" => "\\\$")
    s = replace(s, "{" => "\\{")
    s = replace(s, "}" => "\\}")
    return s
end

# ============================================================================
# Table 0: Configuration summary table
# ============================================================================

function generate_config_table(experiments)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "config"), experiments)
    if isempty(valid)
        return "% No valid experiments to tabulate\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Polynomial learning experiment configurations. " *
                 "Each row describes one experimental setup.}")
    push!(lines, "\\label{tab:poly-learning-config}")
    push!(lines, "\\adjustbox{max width=\\textwidth}{%")
    push!(lines, "\\begin{tabular}{lcccccc}")
    push!(lines, "\\toprule")
    push!(lines, "Experiment & Prime (\$p\$) & Precision & Degree & \\#Points & \\#Samples \\\\")
    push!(lines, "\\midrule")

    for exp in valid
        config = exp["config"]
        name = "\\texttt{" * escape_latex(config["name"]) * "}"
        prime = config["prime"]
        prec = config["prec"]
        degree = config["degree"]
        n_points = config["n_points"]
        num_samples = config["num_samples"]

        row = "$name & $prime & $prec & $degree & $n_points & $num_samples \\\\"
        push!(lines, row)
        push!(lines, "\\hline")
    end

    if !isempty(lines) && lines[end] == "\\hline"
        pop!(lines)
    end
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "}% end adjustbox")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
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
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Polynomial learning: mean final loss across optimizers. " *
                 "Lower is better. Values are averaged over multiple random problem instances.}")
    push!(lines, "\\label{tab:poly-learning-summary}")

    # Build column spec
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\adjustbox{max width=\\textwidth}{%")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    # Header row
    header = "Experiment"
    for opt_name in optimizer_order
        header *= " & $(display_name(opt_name))"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    # Data rows
    for exp in valid
        config = exp["config"]
        agg = exp["aggregate"]

        # Find best (minimum) mean final loss for bolding (excluding Random)
        best_loss = Inf
        for opt_name in optimizer_order
            opt_name == "Random" && continue
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                loss = agg[opt_name]["mean_final_loss"]
                if loss < best_loss
                    best_loss = loss
                end
            end
        end

        name = "\\texttt{" * escape_latex(config["name"]) * "}"
        row = name
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                loss = agg[opt_name]["mean_final_loss"]
                std_loss = get(agg[opt_name], "std_final_loss", 0.0)
                formatted = if std_loss > 0
                    "\$$(latex_sci_compact(loss)) {\\scriptstyle \\pm $(latex_sci_compact(std_loss))}\$"
                else
                    "\$$(latex_sci_compact(loss))\$"
                end
                if latex_sci_compact(loss) == latex_sci_compact(best_loss)
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
        push!(lines, "\\hline")
    end

    if !isempty(lines) && lines[end] == "\\hline"
        pop!(lines)
    end
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "}% end adjustbox")
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

    all_lines = String[]

    for (idx, exp) in enumerate(valid)
        config = exp["config"]
        agg = exp["aggregate"]
        name = "\\texttt{" * escape_latex(config["name"]) * "}"

        lines = String[]
        push!(lines, "\\begin{table}[H]")
        push!(lines, "\\centering")
        push!(lines, "\\caption{Detailed results for configuration: $(name). " *
                     "Shows mean final loss, improvement ratio (\\%), and wall-clock time.}")
        push!(lines, "\\label{tab:poly-learning-detail-$(idx)}")
        push!(lines, "\\begin{tabular}{lrrrrr}")
        push!(lines, "\\toprule")
        push!(lines, "Optimizer & Final Loss & Std Loss & Improv.~(\\%) & Time (s) & Rank \$\\pm\$ Std \\\\")
        push!(lines, "\\midrule")

        # Find best loss for bolding (excluding Random)
        best_loss = Inf
        for opt_name in optimizer_order
            opt_name == "Random" && continue
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
                if @sprintf("%.2e", loss) == @sprintf("%.2e", best_loss)
                    loss_str = "\\textbf{$loss_str}"
                end
                std_loss_str = haskey(stats, "std_final_loss") ? @sprintf("\$%.2e\$", stats["std_final_loss"]) : "---"
                improv_str = @sprintf("%.1f", stats["mean_improvement_ratio"] * 100)
                std_t = get(stats, "std_time", 0.0)
                time_str = std_t > 0 ?
                    @sprintf("\$%.4f {\\scriptstyle \\pm %.4f}\$", stats["mean_time"], std_t) :
                    @sprintf("\$%.4f\$", stats["mean_time"])
                rank_str = if haskey(stats, "mean_rank")
                    std_r = get(stats, "std_rank", 0.0)
                    @sprintf("%.2f \$\\pm\$ %.2f", stats["mean_rank"], std_r)
                else
                    "---"
                end

                push!(lines, "$(display_name(opt_name)) & $loss_str & $std_loss_str & $improv_str & $time_str & $rank_str \\\\")
                push!(lines, "\\hline")
            end
        end

        if !isempty(lines) && lines[end] == "\\hline"
            pop!(lines)
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
# Table 3: Timing comparison
# ============================================================================

function generate_timing_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for timing table\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Mean wall-clock time (seconds) per optimizer across experiments.}")
    push!(lines, "\\label{tab:poly-learning-timing}")

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\adjustbox{max width=\\textwidth}{%")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "Experiment"
    for opt_name in optimizer_order
        header *= " & $(display_name(opt_name))"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    for exp in valid
        config = exp["config"]
        agg = exp["aggregate"]
        name = "\\texttt{" * escape_latex(config["name"]) * "}"

        # Find best (minimum) time for bolding (excluding Random)
        best_time = Inf
        for opt_name in optimizer_order
            opt_name == "Random" && continue
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                t = agg[opt_name]["mean_time"]
                if t < best_time
                    best_time = t
                end
            end
        end

        row = name
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                t = agg[opt_name]["mean_time"]
                t_str = @sprintf("%.4f", t)
                if @sprintf("%.4f", t) == @sprintf("%.4f", best_time)
                    row *= " & \\textbf{$t_str}"
                else
                    row *= " & $t_str"
                end
            else
                row *= " & ---"
            end
        end
        row *= " \\\\"
        push!(lines, row)
        push!(lines, "\\hline")
    end

    if !isempty(lines) && lines[end] == "\\hline"
        pop!(lines)
    end
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "}% end adjustbox")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table 6: Evaluation count comparison
# ============================================================================

function generate_eval_count_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for eval count table\n"
    end

    # Check if eval count data is available
    has_evals = false
    for exp in valid
        agg = exp["aggregate"]
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error") && haskey(agg[opt_name], "mean_total_evals")
                has_evals = true
                break
            end
        end
        has_evals && break
    end
    if !has_evals
        return "% No evaluation count data available\n"
    end

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Mean number of function evaluations per optimizer (excluding monitoring calls). " *
                 "Lower is more efficient.}")
    push!(lines, "\\label{tab:poly-learning-evals}")

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\adjustbox{max width=\\textwidth}{%")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "Experiment"
    for opt_name in optimizer_order
        header *= " & $(display_name(opt_name))"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    for exp in valid
        config = exp["config"]
        agg = exp["aggregate"]
        name = "\\texttt{" * escape_latex(config["name"]) * "}"

        row = name
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error") && haskey(agg[opt_name], "mean_total_evals")
                evals = agg[opt_name]["mean_total_evals"]
                row *= " & $(Int(round(evals)))"
            else
                row *= " & ---"
            end
        end
        row *= " \\\\"
        push!(lines, row)
        push!(lines, "\\hline")
    end

    if !isempty(lines) && lines[end] == "\\hline"
        pop!(lines)
    end
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "}% end adjustbox")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Table: Optimizer ranking summary
# ============================================================================

function generate_ranking_table(experiments, optimizer_order)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    if isempty(valid)
        return "% No valid experiments for ranking table\n"
    end

    # Check if ranking data is available
    has_ranks = any(
        haskey(exp["aggregate"], opt) &&
        !haskey(exp["aggregate"][opt], "error") &&
        haskey(exp["aggregate"][opt], "mean_rank")
        for exp in valid for opt in optimizer_order
        if haskey(exp["aggregate"], opt)
    )
    if !has_ranks
        return "% No ranking data available (re-run experiments to generate ranks)\n"
    end

    n_opts = length(optimizer_order)
    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Polynomial learning: optimizer ranking by mean final loss per configuration " *
                 "(rank 1 = best, lower is better). Averaged over random samples. " *
                 "Bold marks the best-ranked optimizer per row (excluding Random). " *
                 "The \\textit{Average} row shows the mean rank across all configurations.}")
    push!(lines, "\\label{tab:poly-learning-ranking}")
    col_spec = "l" * "c"^n_opts
    push!(lines, "\\adjustbox{max width=\\textwidth}{%")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header = "Configuration"
    for opt_name in optimizer_order
        header *= " & $(display_name(opt_name))"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    rank_sums   = Dict{String, Float64}(opt => 0.0 for opt in optimizer_order)
    rank_counts = Dict{String, Int}(opt => 0 for opt in optimizer_order)

    for exp in valid
        config = exp["config"]
        agg = exp["aggregate"]

        best_rank = Inf
        for opt_name in optimizer_order
            opt_name == "Random" && continue
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error") && haskey(agg[opt_name], "mean_rank")
                r = agg[opt_name]["mean_rank"]
                if r < best_rank; best_rank = r; end
            end
        end

        name = "\\texttt{" * escape_latex(config["name"]) * "}"
        row = name
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error") && haskey(agg[opt_name], "mean_rank")
                r = agg[opt_name]["mean_rank"]
                std_r = get(agg[opt_name], "std_rank", 0.0)
                cell = std_r > 0 ?
                    @sprintf("\$%.2f {\\scriptstyle \\pm %.2f}\$", r, std_r) :
                    @sprintf("\$%.2f\$", r)
                if @sprintf("%.2f", r) == @sprintf("%.2f", best_rank)
                    row *= " & \\textbf{$cell}"
                else
                    row *= " & $cell"
                end
                rank_sums[opt_name] += r
                rank_counts[opt_name] += 1
            else
                row *= " & ---"
            end
        end
        row *= " \\\\"
        push!(lines, row)
        push!(lines, "\\hline")
    end

    # Average row
    best_avg = Inf
    for opt_name in optimizer_order
        opt_name == "Random" && continue
        if rank_counts[opt_name] > 0
            avg = rank_sums[opt_name] / rank_counts[opt_name]
            if avg < best_avg; best_avg = avg; end
        end
    end

    push!(lines, "\\midrule")
    avg_row = "\\textit{Average}"
    for opt_name in optimizer_order
        if rank_counts[opt_name] > 0
            avg = rank_sums[opt_name] / rank_counts[opt_name]
            avg_str = @sprintf("%.2f", avg)
            if @sprintf("%.2f", avg) == @sprintf("%.2f", best_avg)
                avg_row *= " & \\textbf{$avg_str}"
            else
                avg_row *= " & $avg_str"
            end
        else
            avg_row *= " & ---"
        end
    end
    avg_row *= " \\\\"
    push!(lines, avg_row)

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "}% end adjustbox")
    push!(lines, "\\end{table}")

    return join(lines, "\n") * "\n"
end

# ============================================================================
# Generate unified document
# ============================================================================

function generate_unified_document(experiments, optimizer_order; verbose=true)
    lines = String[]

    # Document header
    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Polynomial Learning Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "%")
    push!(lines, "% This file contains all tables for the polynomial learning experiments.")
    push!(lines, "% ============================================================================")
    push!(lines, "")

    # Generate all tables
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Configuration Summary")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, generate_config_table(experiments))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Summary (mean final loss)")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, as_landscape(generate_summary_table(experiments, optimizer_order)))

    if verbose
        push!(lines, "% ----------------------------------------------------------------------------")
        push!(lines, "% Table: Detailed results")
        push!(lines, "% ----------------------------------------------------------------------------")
        push!(lines, "")
        push!(lines, generate_detailed_table(experiments, optimizer_order))
    end

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Timing comparison")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, as_landscape(generate_timing_table(experiments, optimizer_order)))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Evaluation count comparison")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, as_landscape(generate_eval_count_table(experiments, optimizer_order)))

    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "% Table: Optimizer ranking summary")
    push!(lines, "% ----------------------------------------------------------------------------")
    push!(lines, "")
    push!(lines, as_landscape(generate_ranking_table(experiments, optimizer_order)))

    return join(lines, "\n")
end

# ============================================================================
# Main execution
# ============================================================================

document = generate_unified_document(experiments, optimizer_order; verbose=verbose)

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
