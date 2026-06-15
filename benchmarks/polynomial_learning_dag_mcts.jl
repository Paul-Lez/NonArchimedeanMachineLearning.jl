#!/usr/bin/env julia

# Benchmark DAG-MCTS variants on the polynomial-learning experiment:
# learn parameters a_1, ..., a_d for prod_i (x - a_i) from root data.
#
# Default run requested in the Codex thread:
#   julia --project=. benchmarks/polynomial_learning_dag_mcts.jl
#
# Optional flags:
#   --samples=20
#   --steps=3
#   --simulations=50
#   --precision=12
#   --max-dim=5
#   --primes=2,3,5

using BenchmarkTools
using NonArchimedeanMachineLearning
using Oscar
using Printf
using Random
using Statistics

const VARIANTS = [
    ("pathstats", PathStatsDAGMCTS),
    ("node_stats", NodeStatsDAGMCTS),
    ("uct1", UCT1DAGMCTS),
    ("uct2", UCT2DAGMCTS),
    ("uctmax", UCTMaxDAGMCTS)
]

function arg_value(name::String)
    prefix = "--$name="
    for arg in ARGS
        startswith(arg, prefix) && return arg[(lastindex(prefix) + 1):end]
    end
    return nothing
end

function option_int(name::String, default::Int)
    value = arg_value(name)
    return parse(Int, isnothing(value) ? string(default) : value)
end

function option_primes(default::Vector{Int})
    value = arg_value("primes")
    isnothing(value) && return default
    return [parse(Int, part) for part in split(value, ",") if !isempty(part)]
end

function polynomial_learning_problem(p::Int, dimension::Int, precision::Int)
    K = PadicField(p, precision)
    names = vcat(["x"], ["a$i" for i in 1:dimension])
    _, vars = polynomial_ring(K, names)
    x = vars[1]
    params = vars[2:end]

    polynomial = reduce(*, (x - param for param in params))
    model = AbstractModel(AbsolutePolynomialSum([polynomial]),
        vcat([true], fill(false, dimension)))

    data = [
        (ValuationPolydisc([K(p^i)], [precision]), 0)
        for i in 0:(dimension - 1)
    ]
    loss = MPE_loss_init(model, data, 2)

    initial_param = ValuationPolydisc(
        [K((p + 7) * (i + 1)) for i in 1:dimension],
        fill(0, dimension)
    )

    return initial_param, loss
end

function run_variant(
        variant::DAGMCTSVariant,
        p::Int,
        dimension::Int,
        precision::Int,
        steps::Int,
        num_simulations::Int,
        seed::Int
)
    Random.seed!(seed)
    param, loss = polynomial_learning_problem(p, dimension, precision)
    config = DAGMCTSConfig(
        num_simulations = num_simulations,
        degree = 1,
        value_transform = negation_transform(),
        persist_table = true,
        track_parents = true,
        selection_mode = VisitCount,
        variant = variant
    )
    optim = dag_mcts_descent_init(param, loss, config)
    initial_loss = eval_loss(optim)
    best_loss = initial_loss

    for _ in 1:steps
        step!(optim) && break
        best_loss = min(best_loss, eval_loss(optim))
    end

    return (
        optim = optim,
        initial_loss = initial_loss,
        best_loss = best_loss,
        final_loss = eval_loss(optim)
    )
end

function measure_variant(args...; samples::Int)
    times = Float64[]
    bytes = Int[]

    run_variant(args...)

    for _ in 1:samples
        GC.gc()
        result = @timed run_variant(args...)
        push!(times, result.time)
        push!(bytes, result.bytes)
    end

    return (
        min_time = minimum(times),
        median_time = median(times),
        min_bytes = minimum(bytes)
    )
end

function summarize_variant(args...)
    result = run_variant(args...)
    stats = get_dag_stats(result.optim.state)
    return (
        initial_loss = result.initial_loss,
        best_loss = result.best_loss,
        final_loss = result.final_loss,
        unique_nodes = hasproperty(stats, :unique_nodes) ? getproperty(stats, :unique_nodes) : 0,
        multi_parent_nodes = hasproperty(stats, :multi_parent_nodes) ?
                             getproperty(stats, :multi_parent_nodes) : 0
    )
end

function tied_average_ranks(values::Vector{Float64}; atol::Float64 = 1e-12)
    order = sortperm(values)
    ranks = zeros(Float64, length(values))
    i = 1
    while i <= length(order)
        j = i
        while j < length(order) &&
              isapprox(values[order[j]], values[order[j + 1]], atol = atol, rtol = 0.0)
            j += 1
        end
        rank = (i + j) / 2
        for k in i:j
            ranks[order[k]] = rank
        end
        i = j + 1
    end
    return ranks
end

function print_rank_summary(records, primes::Vector{Int}, max_dim::Int)
    labels = first.(VARIANTS)
    metrics = (
        (:best_loss, "best loss rank"),
        (:final_loss, "final loss rank"),
        (:min_time, "min time rank"),
        (:median_time, "median time rank")
    )
    rank_sums = Dict(metric => Dict(label => 0.0 for label in labels)
        for (metric, _) in metrics)
    best_counts = Dict(metric => Dict(label => 0 for label in labels)
        for (metric, _) in metrics)
    unique_best_counts = Dict(metric => Dict(label => 0 for label in labels)
        for (metric, _) in metrics)
    case_count = 0

    for p in primes
        for dimension in 1:max_dim
            case_records = [record for record in records
                            if record.p == p && record.dimension == dimension]
            isempty(case_records) && continue
            case_count += 1

            for (metric, _) in metrics
                values = [Float64(getproperty(record, metric)) for record in case_records]
                ranks = tied_average_ranks(values)
                best_value = minimum(values)
                best_labels = String[]
                for (record, rank) in zip(case_records, ranks)
                    rank_sums[metric][record.label] += rank
                    if isapprox(Float64(getproperty(record, metric)), best_value,
                        atol = 1e-12, rtol = 0.0)
                        push!(best_labels, record.label)
                    end
                end
                for label in best_labels
                    best_counts[metric][label] += 1
                end
                if length(best_labels) == 1
                    unique_best_counts[metric][only(best_labels)] += 1
                end
            end
        end
    end

    println()
    println("Average rank summary")
    println("  lower ranks are better; ranks are averaged across $(case_count) (prime, dimension) cases")
    println("  tied values receive the average of their tied rank positions")
    println()
    @printf("%-12s %15s %15s %15s %17s\n",
        "variant", "best loss", "final loss", "min time", "median time")
    println("-"^80)
    for label in labels
        @printf("%-12s %15.3f %15.3f %15.3f %17.3f\n",
            label,
            rank_sums[:best_loss][label] / case_count,
            rank_sums[:final_loss][label] / case_count,
            rank_sums[:min_time][label] / case_count,
            rank_sums[:median_time][label] / case_count)
    end

    println()
    println("Best-loss wins")
    @printf("%-12s %12s %12s\n", "variant", "tied best", "unique best")
    println("-"^38)
    for label in labels
        @printf("%-12s %12d %12d\n",
            label,
            best_counts[:best_loss][label],
            unique_best_counts[:best_loss][label])
    end
end

function main()
    samples = option_int("samples", 20)
    steps = option_int("steps", 3)
    num_simulations = option_int("simulations", 50)
    precision = option_int("precision", 12)
    max_dim = option_int("max-dim", 5)
    seed = option_int("seed", 2026)
    primes = option_primes([2, 3, 5])
    prime_list = join(primes, ",")

    println("Polynomial-learning DAG-MCTS benchmark")
    println("  primes=$prime_list dimensions=1:$max_dim samples=$samples steps=$steps simulations=$num_simulations precision=$precision seed=$seed")
    println()
    @printf("%3s %3s %-12s %11s %11s %11s %12s %12s %12s %8s %8s\n",
        "p", "dim", "variant", "min time", "median", "min alloc",
        "initial", "best", "final", "nodes", "multi")
    println("-"^112)

    records = []
    for p in primes
        for dimension in 1:max_dim
            for (label, variant) in VARIANTS
                args = (variant, p, dimension, precision, steps, num_simulations, seed)
                measurement = measure_variant(args...; samples = samples)
                summary = summarize_variant(args...)
                @printf("%3d %3d %-12s %11s %11s %11s %12.4e %12.4e %12.4e %8d %8d\n",
                    p,
                    dimension,
                    label,
                    BenchmarkTools.prettytime(measurement.min_time * 1e9),
                    BenchmarkTools.prettytime(measurement.median_time * 1e9),
                    BenchmarkTools.prettymemory(measurement.min_bytes),
                    summary.initial_loss,
                    summary.best_loss,
                    summary.final_loss,
                    summary.unique_nodes,
                    summary.multi_parent_nodes)
                push!(records, (
                    p = p,
                    dimension = dimension,
                    label = label,
                    min_time = measurement.min_time,
                    median_time = measurement.median_time,
                    best_loss = summary.best_loss,
                    final_loss = summary.final_loss
                ))
            end
        end
    end

    print_rank_summary(records, primes, max_dim)
end

main()
