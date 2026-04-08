"""
Shared LaTeX table utilities for the NAML paper experiment infrastructure.

Provides:
1. Optimizer display names and ordering
2. LaTeX formatting helpers (escaping, scientific notation)
3. Generic table generators (summary, timing, ranking, eval count, etc.)
4. CLI parsing for table generators
5. Unified document assembly

Used by all generate_tables.jl scripts.
"""

using JSON
using Printf
using Dates

# ============================================================================
# Display names and ordering
# ============================================================================

const DISPLAY_NAMES = Dict(
    "Random"              => "Random",
    "Best-First"          => "Best First Value",
    "Best-First-branch2"  => "Best First Branch 2",
    "Best-First-Gradient" => "Best First Gradient",
    "DOO"                 => "DOO",
    "MCTS-k"              => "MCTS-\$k\$",
    "MCTS-5k"             => "MCTS-\$5k\$",
    "MCTS-10k"            => "MCTS-\$10k\$",
    "DAG-MCTS-k"          => "DAG-MCTS-\$k\$",
    "DAG-MCTS-5k"         => "DAG-MCTS-\$5k\$",
    "DAG-MCTS-10k"        => "DAG-MCTS-\$10k\$",
    
    # New suite-based names
    "MCTS-10k-deg1"       => "MCTS-\$10k\$ (deg 1)",
    "MCTS-10k-deg2"       => "MCTS-\$10k\$ (deg 2)",
    "DAG-MCTS-10k-deg1"   => "DAG-MCTS-\$10k\$ (deg 1)",
    "DAG-MCTS-10k-deg2"   => "DAG-MCTS-\$10k\$ (deg 2)",
    "Greedy-deg1"         => "Greedy (deg 1)",
    "Greedy-deg2"         => "Greedy (deg 2)",
    "Gradient-deg1"       => "Gradient (deg 1)",
    "Gradient-deg2"       => "Gradient (deg 2)",
)
display_name(n) = get(DISPLAY_NAMES, n, n)

const DISPLAY_ORDER = [
    "Random", "Best-First", "Best-First-branch2", "Best-First-Gradient",
    "MCTS-k", "MCTS-5k", "MCTS-10k",
    "DAG-MCTS-k", "DAG-MCTS-5k", "DAG-MCTS-10k", 
    "MCTS-10k-deg1", "MCTS-10k-deg2", "DAG-MCTS-10k-deg1", "DAG-MCTS-10k-deg2",
    "Greedy-deg1", "Greedy-deg2", "Gradient-deg1", "Gradient-deg2",
    "DOO"
]

# ============================================================================
# CLI parsing for table generators
# ============================================================================

"""
    parse_table_args(ARGS, default_output) -> NamedTuple

Parse CLI arguments for generate_tables.jl scripts.
"""
function parse_table_args(args, default_output::String)
    if length(args) < 1
        println("Usage: julia generate_tables.jl <stats.json> [--output FILE] [--stdout] [--verbose]")
        exit(1)
    end

    json_file = args[1]
    print_stdout = "--stdout" in args
    verbose = "--verbose" in args

    output_file = default_output
    for (i, arg) in enumerate(args)
        if arg == "--output" && i < length(args)
            output_file = args[i+1]
        end
    end

    return (
        json_file = json_file,
        print_stdout = print_stdout,
        verbose = verbose,
        output_file = output_file,
    )
end


# ============================================================================
# Load and validate stats JSON
# ============================================================================

"""
    load_stats_json(filepath) -> (experiments, metadata, optimizer_order)

Load a stats JSON file and extract experiments, metadata, and optimizer ordering.
"""
function load_stats_json(filepath::String)
    if !isfile(filepath)
        println("Error: File not found: $filepath")
        exit(1)
    end

    data = JSON.parsefile(filepath)

    # Handle both formats: dict with "experiments" key or direct array
    if isa(data, AbstractDict)
        experiments = haskey(data, "experiments") ? data["experiments"] : data
        metadata = haskey(data, "metadata") ? data["metadata"] : Dict()
    else
        experiments = data
        metadata = Dict()
    end

    # Ensure experiments is a proper array
    if !isa(experiments, AbstractVector)
        experiments = [experiments[k] for k in sort(collect(keys(experiments)))]
    end

    # Determine optimizer order (legacy; for suite-based runs use get_optimizer_names per suite)
    optimizer_order = if haskey(metadata, "optimizer_order") && !isempty(metadata["optimizer_order"])
        metadata["optimizer_order"]
    else
        names = get_optimizer_names(experiments)
        if isempty(names)
            # Suite-based: take union across all suites
            suite_names = list_suites(experiments)
            union_names = String[]
            for s in suite_names
                for n in get_optimizer_names(experiments; suite_name=s)
                    n in union_names || push!(union_names, n)
                end
            end
            union_names
        else
            names
        end
    end

    # Reorder to preferred display order
    optimizer_order = vcat(
        [n for n in DISPLAY_ORDER if n in optimizer_order],
        [n for n in optimizer_order if !(n in DISPLAY_ORDER)]
    )

    suites = list_suites(experiments)

    println("Loaded $(length(experiments)) experiments from $filepath")
    if !isempty(suites)
        println("Suites: $(join(suites, ", "))")
    end
    println("Optimizers: $(join(optimizer_order, ", "))")
    println()

    return experiments, metadata, optimizer_order
end


"""
    list_suites(experiments) -> Vector{String}

Return the sorted list of suite names present in `experiments` (looking in
`suites_aggregate`). Returns an empty vector if no suites are found (legacy
flat-aggregate format).
"""
function list_suites(experiments)
    suites = Set{String}()
    for exp in experiments
        haskey(exp, "error") && continue
        if haskey(exp, "suites_aggregate")
            for name in keys(exp["suites_aggregate"])
                push!(suites, name)
            end
        end
    end
    # Prefer a canonical order: optimizer-comparison first, then alphabetical
    ordered = String[]
    if "optimizer-comparison" in suites
        push!(ordered, "optimizer-comparison")
        delete!(suites, "optimizer-comparison")
    end
    append!(ordered, sort(collect(suites)))
    return ordered
end

"""
    suite_display_name(suite_name) -> String

Return a human-readable display name for a suite (for LaTeX captions/sections).
"""
function suite_display_name(suite_name::String)
    mapping = Dict(
        "optimizer-comparison"              => "Optimizer Comparison",
        "mcts-branching"                    => "MCTS Branching Sweep",
        "dag-mcts-branching"                => "DAG-MCTS Branching Sweep",
        "greedy-descent-branching"          => "Greedy Descent Branching Sweep",
        "gradient-descent-branching"        => "Gradient Descent Branching Sweep",
        "mcts-number-of-simulations"        => "MCTS Simulation Count Sweep",
        "dag-mcts-number-of-simulations"    => "DAG-MCTS Simulation Count Sweep",
        "mcts-exploration-constant"         => "MCTS Exploration Constant Sweep",
        "dag-mcts-exploration-constant"     => "DAG-MCTS Exploration Constant Sweep",
    )
    return get(mapping, suite_name, suite_name)
end

"""
    get_optimizer_names(experiments; suite_name=nothing) -> Vector{String}

Extract optimizer names from the first valid experiment's aggregate data.
If suite_name is provided, look in suites_aggregate[suite_name].
"""
function get_optimizer_names(experiments; suite_name=nothing)
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    
    if isempty(valid)
        return String[]
    end

    agg = if isnothing(suite_name)
        valid[1]["aggregate"]
    else
        valid[1]["suites_aggregate"][suite_name]
    end
    if haskey(agg, "error")
        return String[]
    end

    opt_names = collect(keys(agg))
    ordered = String[]
    for name in DISPLAY_ORDER
        if name in opt_names
            push!(ordered, name)
        end
    end
    for name in opt_names
        if !(name in ordered)
            push!(ordered, name)
        end
    end
    return ordered
end


# ============================================================================
# LaTeX formatting helpers
# ============================================================================

"""Format a number in compact scientific notation for LaTeX."""
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

"""Format a number in scientific notation for LaTeX (more digits)."""
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

"""Escape special characters for LaTeX."""
function escape_latex(s::String)
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

"""Wrap wide LaTeX tables in landscape environment with footnotesize font."""
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
# Generic table generators
# ============================================================================

"""
    generate_config_table(experiments, label, caption, columns_fn) -> String

Generate a configuration summary table.
`columns_fn(config)` should return (header_string, row_string) for experiment-specific columns.
"""
function generate_config_table(experiments, label::String, caption::String,
                                headers::String, row_fn::Function)
    valid = filter(e -> !haskey(e, "error") && haskey(e, "config"), experiments)
    if isempty(valid)
        return "% No valid experiments to tabulate\n"
    end

    ncols = count(c -> c == '&', headers) + 1
    col_spec = "l" * "c"^(ncols - 1)

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
    push!(lines, "\\adjustbox{max width=\\textwidth}{%")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")
    push!(lines, headers * " \\\\")
    push!(lines, "\\midrule")

    for exp in valid
        config = exp["config"]
        name = "\\texttt{" * escape_latex(config["name"]) * "}"
        row = name * " & " * row_fn(config) * " \\\\"
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


"""
    generate_summary_table(experiments, optimizer_order, label, caption; suite_name=nothing) -> String

Generate a grid table of mean final loss per optimizer per configuration.
Best result per row (excluding Random) is bolded.
"""
function generate_summary_table(experiments, optimizer_order, label::String, caption::String; suite_name=nothing)
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    if isempty(valid)
        return "% No valid experiments to tabulate\n"
    end

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
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

    for exp in valid
        config = exp["config"]
        agg = isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name]

        # Find best (excluding Random)
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


"""
    generate_timing_table(experiments, optimizer_order, label, caption; suite_name=nothing) -> String

Generate a grid table of mean wall-clock time per optimizer per configuration.
"""
function generate_timing_table(experiments, optimizer_order, label::String, caption::String; suite_name=nothing)
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    if isempty(valid)
        return "% No valid experiments for timing table\n"
    end

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
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

    for exp in valid
        config = exp["config"]
        agg = isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name]
        name = "\\texttt{" * escape_latex(config["name"]) * "}"

        best_time = Inf
        for opt_name in optimizer_order
            opt_name == "Random" && continue
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                t = agg[opt_name]["mean_time"]
                if t < best_time; best_time = t; end
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


"""
    generate_eval_count_table(experiments, optimizer_order, label, caption; suite_name=nothing) -> String

Generate a grid table of mean function evaluations per optimizer per configuration.
"""
function generate_eval_count_table(experiments, optimizer_order, label::String, caption::String; suite_name=nothing)
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    if isempty(valid)
        return "% No valid experiments for eval count table\n"
    end

    has_evals = false
    for exp in valid
        agg = isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name]
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

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
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

    for exp in valid
        config = exp["config"]
        agg = isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name]
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


"""
    generate_ranking_table(experiments, optimizer_order, label, caption; suite_name=nothing) -> String

Generate a grid table of optimizer rankings per configuration with an average row.
"""
function generate_ranking_table(experiments, optimizer_order, label::String, caption::String; suite_name=nothing)
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    if isempty(valid)
        return "% No valid experiments for ranking table\n"
    end

    has_ranks = any(
        haskey(isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name], opt) &&
        !haskey((isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name])[opt], "error") &&
        haskey((isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name])[opt], "mean_rank")
        for exp in valid for opt in optimizer_order
        if haskey(isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name], opt)
    )
    if !has_ranks
        return "% No ranking data available (re-run experiments to generate ranks)\n"
    end

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
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
        agg = isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name]

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


"""
    generate_optimizer_aggregate_table(experiments, optimizer_order, label, caption) -> String

Generate an overall optimizer comparison table aggregated across all configurations.
"""
function generate_optimizer_aggregate_table(experiments, optimizer_order, label::String, caption::String; suite_name=nothing)
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    if isempty(valid)
        return "% No valid experiments for optimizer aggregate\n"
    end

    # Collect per-optimizer stats across experiments
    _m(x) = isempty(x) ? 0.0 : sum(x) / length(x)
    optimizer_stats = Dict{String, Dict{String, Vector{Float64}}}()
    for opt_name in optimizer_order
        optimizer_stats[opt_name] = Dict(
            "final_loss" => Float64[],
            "improvement_ratio" => Float64[],
            "time" => Float64[]
        )
    end

    for exp in valid
        agg = isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name]
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                stats = agg[opt_name]
                push!(optimizer_stats[opt_name]["final_loss"], stats["mean_final_loss"])
                push!(optimizer_stats[opt_name]["improvement_ratio"], stats["mean_improvement_ratio"])
                push!(optimizer_stats[opt_name]["time"], stats["mean_time"])
            end
        end
    end

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
    push!(lines, "\\adjustbox{max width=\\textwidth}{%")
    push!(lines, "\\begin{tabular}{lrrrr}")
    push!(lines, "\\toprule")
    push!(lines, "Optimizer & Mean Loss & Improv.~(\\%) & Mean Time (s) & Configs \\\\")
    push!(lines, "\\midrule")

    best_mean_loss = Inf
    for opt_name in optimizer_order
        opt_name == "Random" && continue
        if !isempty(optimizer_stats[opt_name]["final_loss"])
            mean_loss = _m(optimizer_stats[opt_name]["final_loss"])
            if mean_loss < best_mean_loss; best_mean_loss = mean_loss; end
        end
    end

    for opt_name in optimizer_order
        if !isempty(optimizer_stats[opt_name]["final_loss"])
            mean_loss = _m(optimizer_stats[opt_name]["final_loss"])
            mean_improv = _m(optimizer_stats[opt_name]["improvement_ratio"])
            mean_time = _m(optimizer_stats[opt_name]["time"])
            n_configs = length(optimizer_stats[opt_name]["final_loss"])

            loss_str = @sprintf("\$%.2e\$", mean_loss)
            if @sprintf("%.2e", mean_loss) == @sprintf("%.2e", best_mean_loss)
                loss_str = "\\textbf{$loss_str}"
            end
            improv_str = @sprintf("%.1f", mean_improv * 100)
            time_str = @sprintf("%.2f", mean_time)

            push!(lines, "$(display_name(opt_name)) & $loss_str & $improv_str & $time_str & $n_configs \\\\")
            push!(lines, "\\hline")
        end
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


"""
    generate_detailed_tables(experiments, optimizer_order, label_prefix, caption_prefix; extra_cols=[]) -> String

Generate per-configuration detailed result tables.
"""
function generate_detailed_tables(experiments, optimizer_order, label_prefix::String,
                                   caption_prefix::String;
                                   extra_cols::Vector{Tuple{String,String,Function}}=Tuple{String,String,Function}[],
                                   suite_name=nothing)
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    if isempty(valid)
        return "% No valid experiments for detailed tables\n"
    end

    all_lines = String[]

    for (idx, exp) in enumerate(valid)
        config = exp["config"]
        agg = isnothing(suite_name) ? exp["aggregate"] : exp["suites_aggregate"][suite_name]
        name = "\\texttt{" * escape_latex(config["name"]) * "}"

        # Base columns
        ncols = 4 + length(extra_cols)  # optimizer, loss, improv%, time, rank + extras
        col_spec = "l" * "r"^ncols

        lines = String[]
        push!(lines, "\\begin{table}[H]")
        push!(lines, "\\centering")
        push!(lines, "\\caption{$caption_prefix: $(name).}")
        push!(lines, "\\label{$label_prefix-$(idx)}")
        push!(lines, "\\begin{tabular}{$col_spec}")
        push!(lines, "\\toprule")

        header = "Optimizer & Final Loss & Improv.~(\\%) & Time (s) & Mean Rank"
        for (col_header, _, _) in extra_cols
            header *= " & $col_header"
        end
        header *= " \\\\"
        push!(lines, header)
        push!(lines, "\\midrule")

        # Find best loss for bolding
        best_loss = Inf
        for opt_name in optimizer_order
            opt_name == "Random" && continue
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                loss = agg[opt_name]["mean_final_loss"]
                if loss < best_loss; best_loss = loss; end
            end
        end

        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error")
                stats = agg[opt_name]
                loss = stats["mean_final_loss"]
                std_loss = get(stats, "std_final_loss", 0.0)
                loss_str = std_loss > 0 ?
                    "\$$(latex_sci_compact(loss)) {\\scriptstyle \\pm $(latex_sci_compact(std_loss))}\$" :
                    @sprintf("\$%.2e\$", loss)
                if @sprintf("%.2e", loss) == @sprintf("%.2e", best_loss)
                    loss_str = "\\textbf{$loss_str}"
                end
                improv_str = @sprintf("%.1f", stats["mean_improvement_ratio"] * 100)
                std_t = get(stats, "std_time", 0.0)
                time_str = std_t > 0 ?
                    @sprintf("\$%.4f {\\scriptstyle \\pm %.4f}\$", stats["mean_time"], std_t) :
                    @sprintf("\$%.4f\$", stats["mean_time"])
                rank_str = haskey(stats, "mean_rank") ? @sprintf("%.2f", stats["mean_rank"]) : "---"

                row = "$(display_name(opt_name)) & $loss_str & $improv_str & $time_str & $rank_str"
                for (_, _, format_fn) in extra_cols
                    row *= " & " * format_fn(stats)
                end
                row *= " \\\\"
                push!(lines, row)
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
        push!(all_lines, "")
    end

    return join(all_lines, "\n") * "\n"
end


# ============================================================================
# Per-suite section helper
# ============================================================================

"""
    generate_suite_section(experiments, suite_name, label_prefix, caption_prefix;
                           extra_table_fn=nothing, include_aggregate=false,
                           verbose=false, detailed_extra_cols=[]) -> String

Generate a full LaTeX section (summary + timing + eval counts + ranking, plus
optionally the aggregate and verbose detailed tables) for a single suite.

`extra_table_fn(experiments, optimizer_order, suite_name) -> String` may be
passed to insert an experiment-specific table (e.g. accuracy) right after the
timing table.
"""
function generate_suite_section(experiments, suite_name::String,
                                 label_prefix::String, caption_prefix::String;
                                 extra_table_fn=nothing,
                                 include_aggregate::Bool=false,
                                 verbose::Bool=false,
                                 detailed_extra_cols::Vector{Tuple{String,String,Function}}=Tuple{String,String,Function}[])
    opt_order = get_optimizer_names(experiments; suite_name=suite_name)
    if isempty(opt_order)
        return "% Suite $suite_name: no data\n"
    end

    # Ensure preferred order
    opt_order = vcat(
        [n for n in DISPLAY_ORDER if n in opt_order],
        [n for n in opt_order if !(n in DISPLAY_ORDER)]
    )

    suite_label = replace(suite_name, "-" => "")
    suite_title = suite_display_name(suite_name)

    lines = String[]
    push!(lines, "")
    push!(lines, "\\subsection*{$caption_prefix: $suite_title}")
    push!(lines, "")

    push!(lines, as_landscape(generate_summary_table(experiments, opt_order,
        "$label_prefix-summary-$suite_label",
        "$caption_prefix ($suite_title): mean final loss per optimizer. Lower is better.";
        suite_name=suite_name)))

    push!(lines, as_landscape(generate_timing_table(experiments, opt_order,
        "$label_prefix-timing-$suite_label",
        "$caption_prefix ($suite_title): mean wall-clock time (seconds) per optimizer.";
        suite_name=suite_name)))

    if extra_table_fn !== nothing
        extra = extra_table_fn(experiments, opt_order, suite_name)
        if !isempty(extra)
            push!(lines, as_landscape(extra))
        end
    end

    if include_aggregate
        push!(lines, generate_optimizer_aggregate_table(experiments, opt_order,
            "$label_prefix-optimizer-aggregate-$suite_label",
            "$caption_prefix ($suite_title): overall comparison aggregated across all configurations.";
            suite_name=suite_name))
    end

    push!(lines, as_landscape(generate_eval_count_table(experiments, opt_order,
        "$label_prefix-evals-$suite_label",
        "$caption_prefix ($suite_title): mean number of function evaluations per optimizer.";
        suite_name=suite_name)))

    if verbose
        push!(lines, generate_detailed_tables(experiments, opt_order,
            "$label_prefix-detail-$suite_label",
            "$caption_prefix ($suite_title) detailed results";
            extra_cols=detailed_extra_cols,
            suite_name=suite_name))
    end

    push!(lines, as_landscape(generate_ranking_table(experiments, opt_order,
        "$label_prefix-ranking-$suite_label",
        "$caption_prefix ($suite_title): optimizer ranking per configuration (rank 1 = best). " *
        "Bold marks the best-ranked optimizer per row (excluding Random).";
        suite_name=suite_name)))

    return join(lines, "\n")
end


# ============================================================================
# Document assembly and output
# ============================================================================

"""
    write_or_print(document, json_file, output_file, print_stdout)

Write the document to a file or print to stdout.
"""
function write_or_print(document::String, json_file::String, output_file::String, print_stdout::Bool)
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
end
