"""
LaTeX Table Generator for Polynomial Learning Results

Reads stats JSON (produced by make_stats.jl) and generates LaTeX tables.

Usage:
    julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <stats.json>
    julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <stats.json> --output tables.tex
    julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <stats.json> --stdout
    julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <stats.json> --verbose
"""

include("../table_utils.jl")

targs = parse_table_args(ARGS, "polynomial_learning_tables.tex")
experiments, metadata, optimizer_order = load_stats_json(targs.json_file)

# ============================================================================
# Experiment-specific: Configuration table
# ============================================================================

function polylearn_config_table(experiments)
    generate_config_table(
        experiments,
        "tab:poly-learning-config",
        "Polynomial learning experiment configurations. Each row describes one experimental setup.",
        "Experiment & Prime (\$p\$) & Precision & Degree & \\#Points & \\#Samples",
        config -> "$(config["prime"]) & $(config["prec"]) & $(config["degree"]) & $(config["n_points"]) & $(config["num_samples"])"
    )
end

# ============================================================================
# Generate unified document
# ============================================================================

function generate_document(experiments, optimizer_order; verbose=false)
    lines = String[]

    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Polynomial Learning Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "% ============================================================================")
    push!(lines, "")

    push!(lines, polylearn_config_table(experiments))

    suites = list_suites(experiments)
    if isempty(suites)
        # Legacy flat-aggregate fallback
        push!(lines, as_landscape(generate_summary_table(experiments, optimizer_order,
            "tab:poly-learning-summary",
            "Polynomial learning: mean final loss across optimizers.")))
        push!(lines, as_landscape(generate_timing_table(experiments, optimizer_order,
            "tab:poly-learning-timing",
            "Mean wall-clock time (seconds) per optimizer across experiments.")))
        push!(lines, as_landscape(generate_eval_count_table(experiments, optimizer_order,
            "tab:poly-learning-evals",
            "Mean number of function evaluations per optimizer.")))
        push!(lines, as_landscape(generate_ranking_table(experiments, optimizer_order,
            "tab:poly-learning-ranking",
            "Polynomial learning: optimizer ranking by mean final loss per configuration.")))
        if verbose
            push!(lines, generate_detailed_tables(experiments, optimizer_order,
                "tab:poly-learning-detail",
                "Detailed results for configuration"))
        end
    else
        for suite in suites
            push!(lines, generate_suite_section(experiments, suite,
                "tab:poly-learning", "Polynomial Learning"; verbose=verbose))
        end
    end

    return join(lines, "\n")
end

document = generate_document(experiments, optimizer_order; verbose=targs.verbose)
write_or_print(document, targs.json_file, targs.output_file, targs.print_stdout)
