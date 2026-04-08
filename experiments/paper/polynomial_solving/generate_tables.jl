"""
LaTeX Table Generator for Polynomial Solving Results

Reads stats JSON (produced by make_stats.jl) and generates LaTeX tables.

Usage:
    julia --project=. experiments/paper/polynomial_solving/generate_tables.jl <stats.json>
    julia --project=. experiments/paper/polynomial_solving/generate_tables.jl <stats.json> --output tables.tex
    julia --project=. experiments/paper/polynomial_solving/generate_tables.jl <stats.json> --stdout
    julia --project=. experiments/paper/polynomial_solving/generate_tables.jl <stats.json> --verbose
"""

include("../table_utils.jl")

targs = parse_table_args(ARGS, "polynomial_solving_tables.tex")
experiments, metadata, optimizer_order = load_stats_json(targs.json_file)

# ============================================================================
# Experiment-specific: Configuration table
# ============================================================================

function polysolve_config_table(experiments)
    generate_config_table(
        experiments,
        "tab:poly-solving-config",
        "Polynomial solving experiment configurations.",
        "Experiment & Prime (\$p\$) & Precision & Variables & Degree & \\#Samples",
        config -> "$(config["prime"]) & $(config["prec"]) & $(config["num_vars"]) & $(config["degree"]) & $(config["num_samples"])"
    )
end

# ============================================================================
# Generate unified document
# ============================================================================

function generate_document(experiments, optimizer_order; verbose=false)
    lines = String[]

    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Polynomial Solving Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "% Minimize |f(z)| where f has a known root in Z_p^n.")
    push!(lines, "% ============================================================================")
    push!(lines, "")

    push!(lines, polysolve_config_table(experiments))

    suites = list_suites(experiments)
    if isempty(suites)
        push!(lines, as_landscape(generate_summary_table(experiments, optimizer_order,
            "tab:poly-solving-summary",
            "Polynomial solving: mean final loss across optimizers.")))
        push!(lines, as_landscape(generate_timing_table(experiments, optimizer_order,
            "tab:poly-solving-timing",
            "Mean wall-clock time (seconds) per optimizer.")))
        push!(lines, as_landscape(generate_eval_count_table(experiments, optimizer_order,
            "tab:poly-solving-evals",
            "Mean number of function evaluations per optimizer.")))
        push!(lines, as_landscape(generate_ranking_table(experiments, optimizer_order,
            "tab:poly-solving-ranking",
            "Polynomial solving: optimizer ranking by mean final loss.")))
        if verbose
            push!(lines, generate_detailed_tables(experiments, optimizer_order,
                "tab:poly-solving-detail", "Detailed results for configuration"))
        end
    else
        for suite in suites
            push!(lines, generate_suite_section(experiments, suite,
                "tab:poly-solving", "Polynomial Solving"; verbose=verbose))
        end
    end

    return join(lines, "\n")
end

document = generate_document(experiments, optimizer_order; verbose=targs.verbose)
write_or_print(document, targs.json_file, targs.output_file, targs.print_stdout)
