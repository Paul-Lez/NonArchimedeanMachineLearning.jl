"""
LaTeX Table Generator for Absolute Sum Minimization Results

Reads stats JSON (produced by make_stats.jl) and generates LaTeX tables.

Usage:
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json>
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json> --output tables.tex
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json> --stdout
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json> --verbose
"""

include("../table_utils.jl")

# Parse arguments
targs = parse_table_args(ARGS, "absolute_sum_tables.tex")
experiments, metadata, optimizer_order = load_stats_json(targs.json_file)

# ============================================================================
# Experiment-specific: Configuration table
# ============================================================================

function abssum_config_table(experiments)
    generate_config_table(
        experiments,
        "tab:abssum-config",
        "Absolute sum minimization experiment configurations. Each row describes one experimental setup.",
        "Experiment & Prime (\$p\$) & Precision & \\#Polys & \\#Vars & Poly Deg. & \\#Samples",
        config -> "$(config["prime"]) & $(config["prec"]) & $(config["num_polys"]) & $(config["num_vars"]) & $(config["degree"]) & $(config["num_samples"])"
    )
end

# ============================================================================
# Generate unified document
# ============================================================================

function generate_document(experiments, optimizer_order; verbose=false)
    lines = String[]

    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Absolute Sum Minimization Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "% ============================================================================")
    push!(lines, "")

    # Config table
    push!(lines, abssum_config_table(experiments))

    suites = list_suites(experiments)
    if isempty(suites)
        push!(lines, as_landscape(generate_summary_table(experiments, optimizer_order,
            "tab:abssum-summary",
            "Absolute sum minimization: mean final loss across optimizers.")))
        push!(lines, as_landscape(generate_timing_table(experiments, optimizer_order,
            "tab:abssum-timing",
            "Mean wall-clock time (seconds) per optimizer.")))
        push!(lines, generate_optimizer_aggregate_table(experiments, optimizer_order,
            "tab:abssum-optimizer-aggregate",
            "Overall optimizer comparison aggregated across configurations."))
        push!(lines, as_landscape(generate_eval_count_table(experiments, optimizer_order,
            "tab:abssum-evals",
            "Mean number of function evaluations per optimizer.")))
        if verbose
            push!(lines, generate_detailed_tables(experiments, optimizer_order,
                "tab:abssum-detail", "Detailed results for configuration"))
        end
        push!(lines, as_landscape(generate_ranking_table(experiments, optimizer_order,
            "tab:abssum-ranking",
            "Absolute sum minimization: optimizer ranking by mean final loss.")))
    else
        for suite in suites
            push!(lines, generate_suite_section(experiments, suite,
                "tab:abssum", "Absolute Sum Minimization";
                include_aggregate=true, verbose=verbose))
        end
    end

    return join(lines, "\n")
end

# ============================================================================
# Main
# ============================================================================

document = generate_document(experiments, optimizer_order; verbose=targs.verbose)
write_or_print(document, targs.json_file, targs.output_file, targs.print_stdout)
