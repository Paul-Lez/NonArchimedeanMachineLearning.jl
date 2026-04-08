"""
Shared statistics utilities for the NAML paper experiment infrastructure.

Provides:
1. Basic statistics (mean, std)
2. Per-sample optimizer ranking
3. Per-experiment aggregate statistics
4. Cross-experiment global ranking

Used by make_stats.jl to compute all stats from raw experiment JSON.
"""

# ============================================================================
# Basic statistics
# ============================================================================

_mean(x) = sum(x) / length(x)

function _std(x)
    m = _mean(x)
    sqrt(sum((xi - m)^2 for xi in x) / length(x))
end

# ============================================================================
# Per-sample rankings
# ============================================================================

"""
    compute_sample_rankings!(optimizer_results::Dict)

Rank optimizers within a dictionary by final_loss (lower = rank 1).
Adds a "rank" field to each valid optimizer result.
Ties share the minimum rank (competition ranking: 1,1,3).
"""
function compute_sample_rankings!(optimizer_results::AbstractDict)
    valid_opts = [(name, res["final_loss"]) for (name, res) in optimizer_results
                  if !haskey(res, "error")]
    isempty(valid_opts) && return
    sort!(valid_opts, by=x -> x[2])
    n = length(valid_opts)
    i = 1
    while i <= n
        j = i
        while j <= n && valid_opts[j][2] == valid_opts[i][2]
            j += 1
        end
        for k in i:j-1
            optimizer_results[valid_opts[k][1]]["rank"] = Float64(i)
        end
        i = j
    end
end


# ============================================================================
# Per-experiment aggregate statistics
# ============================================================================

"""
    compute_aggregate_stats(samples_in_suite::Vector{Dict}, suite_name::String;
                            extra_fields::Vector{String}=String[]) -> Dict

Compute aggregate statistics across samples for a specific suite.
`samples_in_suite` is a Vector of optimizer results for this suite (one per sample).
"""
function compute_aggregate_stats(samples_in_suite::AbstractVector, suite_name::String;
                                  extra_fields::Vector{String}=String[])
    if isempty(samples_in_suite)
        return Dict("error" => "No samples in suite")
    end

    # Identify all optimizers in this suite
    all_opt_names = Set{String}()
    for sample in samples_in_suite
        for opt_name in keys(sample)
            push!(all_opt_names, opt_name)
        end
    end

    aggregate = Dict{String, Any}()

    for opt_name in sort(collect(all_opt_names))
        opt_data = []
        for sample in samples_in_suite
            if haskey(sample, opt_name)
                opt_result = sample[opt_name]
                if !haskey(opt_result, "error")
                    push!(opt_data, opt_result)
                end
            end
        end

        if !isempty(opt_data)
            final_losses = [d["final_loss"] for d in opt_data]

            agg = Dict{String, Any}(
                "mean_final_loss" => _mean(final_losses),
                "std_final_loss" => length(opt_data) > 1 ? _std(final_losses) : 0.0,
                "min_final_loss" => minimum(final_losses),
                "max_final_loss" => maximum(final_losses),
                "mean_improvement" => _mean([d["improvement"] for d in opt_data]),
                "mean_improvement_ratio" => _mean([d["improvement_ratio"] for d in opt_data]),
                "mean_time" => _mean([d["time"] for d in opt_data]),
                "std_time" => length(opt_data) > 1 ? _std([d["time"] for d in opt_data]) : 0.0,
                "n_valid" => length(opt_data),
            )

            # Eval counts
            if haskey(opt_data[1], "total_evals")
                agg["mean_total_evals"] = _mean([d["total_evals"] for d in opt_data])
            end

            # Rankings
            ranks = [d["rank"] for d in opt_data if haskey(d, "rank")]
            if !isempty(ranks)
                agg["mean_rank"] = _mean(ranks)
                agg["std_rank"] = length(ranks) > 1 ? _std(ranks) : 0.0
            end

            # Extra fields
            for field in extra_fields
                mean_key = "mean_$field"
                std_key = "std_$field"
                vals = [d[field] for d in opt_data if haskey(d, field)]
                if !isempty(vals)
                    agg[mean_key] = _mean(vals)
                    agg[std_key] = length(vals) > 1 ? _std(vals) : 0.0
                    agg["min_$field"] = minimum(vals)
                    agg["max_$field"] = maximum(vals)
                end
            end

            aggregate[opt_name] = agg
        end
    end

    return aggregate
end


# ============================================================================
# Global ranking across experiments
# ============================================================================

"""
    compute_global_ranking(experiments) -> Dict{String, Dict}

Compute average rank across all experiment configurations, grouped by suite.
Returns Dict{SuiteName => Dict{OptName => {avg_rank, n_configs}}}.
"""
function compute_global_ranking(experiments::Vector)
    # SuiteName => OptName => Vector{Ranks}
    suite_global_ranks = Dict{String, Dict{String, Vector{Float64}}}()

    for result in experiments
        if haskey(result, "error") || !haskey(result, "suites_aggregate")
            continue
        end
        
        suites_agg = result["suites_aggregate"]
        for (suite_name, agg) in suites_agg
            if haskey(agg, "error")
                continue
            end
            
            if !haskey(suite_global_ranks, suite_name)
                suite_global_ranks[suite_name] = Dict{String, Vector{Float64}}()
            end
            
            opt_ranks = suite_global_ranks[suite_name]
            for (opt_name, stats) in agg
                if haskey(stats, "mean_rank")
                    if !haskey(opt_ranks, opt_name)
                        opt_ranks[opt_name] = Float64[]
                    end
                    push!(opt_ranks[opt_name], stats["mean_rank"])
                end
            end
        end
    end

    # Convert to results
    output = Dict{String, Any}()
    for (suite_name, opt_ranks) in suite_global_ranks
        suite_res = Dict{String, Any}()
        for (opt_name, ranks) in opt_ranks
            suite_res[opt_name] = Dict(
                "avg_rank" => _mean(ranks),
                "n_configs" => length(ranks)
            )
        end
        output[suite_name] = suite_res
    end

    return output
end
