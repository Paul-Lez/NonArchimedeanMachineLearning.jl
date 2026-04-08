"""
LaTeX Table Generator for Function Learning Results

Reads stats JSON (produced by make_stats.jl) and generates LaTeX tables.
Includes experiment-specific accuracy tables.

Usage:
    julia --project=. experiments/paper/function_learning/generate_tables.jl <stats.json>
    julia --project=. experiments/paper/function_learning/generate_tables.jl <stats.json> --output tables.tex
    julia --project=. experiments/paper/function_learning/generate_tables.jl <stats.json> --stdout
    julia --project=. experiments/paper/function_learning/generate_tables.jl <stats.json> --verbose
"""

include("../table_utils.jl")

targs = parse_table_args(ARGS, "function_learning_tables.tex")
experiments, metadata, optimizer_order = load_stats_json(targs.json_file)

# ============================================================================
# Experiment-specific tables
# ============================================================================

function funclearn_config_table(experiments)
    generate_config_table(
        experiments,
        "tab:funclearn-config",
        "Function learning experiment configurations. Each row describes one experimental setup.",
        "Experiment & Prime (\$p\$) & Precision & Degree & \\#Points & Target & Threshold & Scale & \\#Samples",
        config -> "$(config["prime"]) & $(config["prec"]) & $(config["degree"]) & $(config["n_points"]) & $(config["target_fn"]) & $(config["threshold"]) & $(config["scale"]) & $(config["num_samples"])"
    )
end

"""Generate accuracy comparison table (experiment-specific)."""
function generate_accuracy_table(experiments, optimizer_order; suite_name=nothing, label="tab:funclearn-accuracy")
    valid = if isnothing(suite_name)
        filter(e -> !haskey(e, "error") && haskey(e, "aggregate"), experiments)
    else
        filter(e -> !haskey(e, "error") && haskey(e, "suites_aggregate") && haskey(e["suites_aggregate"], suite_name), experiments)
    end
    if isempty(valid)
        return "% No valid experiments for accuracy table\n"
    end

    n_opts = length(optimizer_order)
    col_spec = "l" * "c"^n_opts

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Mean classification accuracy (\\%) per optimizer for function learning. " *
                 "Higher is better. Accuracy improvement (delta) shows change from initial accuracy.}")
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

        # Find best accuracy (excluding Random)
        best_accuracy = -Inf
        for opt_name in optimizer_order
            opt_name == "Random" && continue
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error") && haskey(agg[opt_name], "mean_final_accuracy")
                acc = agg[opt_name]["mean_final_accuracy"]
                if acc > best_accuracy; best_accuracy = acc; end
            end
        end

        name = "\\texttt{" * escape_latex(config["name"]) * "}"
        row = name
        for opt_name in optimizer_order
            if haskey(agg, opt_name) && !haskey(agg[opt_name], "error") && haskey(agg[opt_name], "mean_final_accuracy")
                acc = agg[opt_name]["mean_final_accuracy"]
                acc_pct = acc * 100
                std_acc = get(agg[opt_name], "std_final_accuracy", 0.0)
                acc_delta = get(agg[opt_name], "mean_accuracy_improvement", 0.0) * 100

                formatted = std_acc > 0 ?
                    @sprintf("\$%.1f {\\scriptstyle \\pm %.1f}\$ \\%% (%+.1f)", acc_pct, std_acc * 100, acc_delta) :
                    @sprintf("%.1f\\%% (%+.1f)", acc_pct, acc_delta)
                if @sprintf("%.1f", acc * 100) == @sprintf("%.1f", best_accuracy * 100)
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
# Generate unified document
# ============================================================================

function generate_document(experiments, optimizer_order; verbose=false)
    lines = String[]

    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Function Learning Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "% ============================================================================")
    push!(lines, "")

    push!(lines, funclearn_config_table(experiments))

    # Accuracy column formatter for verbose detailed tables
    acc_col = ("Accuracy (\\%)", "mean_final_accuracy",
        stats -> begin
            if haskey(stats, "mean_final_accuracy")
                acc = stats["mean_final_accuracy"]
                std_acc = get(stats, "std_final_accuracy", 0.0)
                std_acc > 0 ?
                    @sprintf("\$%.1f {\\scriptstyle \\pm %.1f}\$", acc * 100, std_acc * 100) :
                    @sprintf("%.1f", acc * 100)
            else
                "---"
            end
        end
    )

    # Insert an experiment-specific accuracy table in each suite section
    accuracy_table_fn = (exps, opts, suite) -> generate_accuracy_table(
        exps, opts; suite_name=suite,
        label="tab:funclearn-accuracy-$(replace(suite, "-" => ""))")

    suites = list_suites(experiments)
    if isempty(suites)
        push!(lines, as_landscape(generate_summary_table(experiments, optimizer_order,
            "tab:funclearn-summary",
            "Function learning: mean final loss across optimizers.")))
        push!(lines, as_landscape(generate_timing_table(experiments, optimizer_order,
            "tab:funclearn-timing",
            "Mean wall-clock time (seconds) per optimizer.")))
        push!(lines, as_landscape(generate_accuracy_table(experiments, optimizer_order)))
        push!(lines, generate_optimizer_aggregate_table(experiments, optimizer_order,
            "tab:funclearn-optimizer-aggregate",
            "Overall optimizer comparison aggregated across configurations."))
        push!(lines, as_landscape(generate_eval_count_table(experiments, optimizer_order,
            "tab:funclearn-evals",
            "Mean number of function evaluations per optimizer.")))
        if verbose
            push!(lines, generate_detailed_tables(experiments, optimizer_order,
                "tab:funclearn-detail",
                "Detailed results for configuration";
                extra_cols=[acc_col]))
        end
        push!(lines, as_landscape(generate_ranking_table(experiments, optimizer_order,
            "tab:funclearn-ranking",
            "Function learning: optimizer ranking by mean final loss.")))
    else
        for suite in suites
            push!(lines, generate_suite_section(experiments, suite,
                "tab:funclearn", "Function Learning";
                extra_table_fn=accuracy_table_fn,
                include_aggregate=true,
                verbose=verbose,
                detailed_extra_cols=[acc_col]))
        end
    end

    return join(lines, "\n")
end

document = generate_document(experiments, optimizer_order; verbose=targs.verbose)
write_or_print(document, targs.json_file, targs.output_file, targs.print_stdout)
