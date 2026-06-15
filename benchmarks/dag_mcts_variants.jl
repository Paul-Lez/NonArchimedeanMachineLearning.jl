#!/usr/bin/env julia

# Benchmark DAG-MCTS variants on a small synthetic p-adic polydisc problem.
#
# Run from the repository root:
#   julia --project=. benchmarks/dag_mcts_variants.jl
#
# Optional environment knobs:
#   NAML_BENCH_SAMPLES       number of measured repetitions (default: 5)
#   NAML_BENCH_STEPS         optimizer steps per repetition (default: 4)
#   NAML_BENCH_SIMULATIONS   MCTS simulations per step (default: 100)
#   NAML_BENCH_DIMENSION     polydisc dimension (default: 3)
#   NAML_BENCH_PRECISION     p-adic precision (default: 12)
#   NAML_BENCH_PERSIST       whether to persist tables across steps (default: true)
#   NAML_BENCH_SEED          RNG seed for each repetition (default: 2026)
#
# The same values can be supplied as CLI flags, for example:
#   julia --project=. benchmarks/dag_mcts_variants.jl --samples=3 --simulations=50

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

function option_value(arg_name::String, env_name::String, default)
    value = arg_value(arg_name)
    return isnothing(value) ? get(ENV, env_name, string(default)) : value
end

function option_int(arg_name::String, env_name::String, default::Int)
    return parse(Int, option_value(arg_name, env_name, default))
end

function option_bool(arg_name::String, env_name::String, default::Bool)
    value = lowercase(option_value(arg_name, env_name, default))
    return value in ("1", "true", "yes", "y", "on")
end

function make_benchmark_problem(dimension::Int, precision::Int)
    K = PadicField(2, precision)
    param = ValuationPolydisc(fill(K(0), dimension), fill(0, dimension))

    function loss_eval(params)
        return [begin
                    centers = NonArchimedeanMachineLearning.canonical_center(p)
                    radii = NonArchimedeanMachineLearning.radius(p)
                    center_penalty = sum(abs(Float64(c)) /
                                         (Float64(NonArchimedeanMachineLearning.prime(p))^max(Int(r), 1))
                                         for (c, r) in zip(centers, radii))
                    precision_reward = 0.25 * sum(Float64, radii)
                    center_penalty - precision_reward
                end for p in params]
    end

    return param, Loss(loss_eval, tangents -> zeros(length(tangents)))
end

function run_variant(
        variant::DAGMCTSVariant,
        steps::Int,
        num_simulations::Int,
        dimension::Int,
        precision::Int,
        persist_table::Bool,
        seed::Int
)
    Random.seed!(seed)
    param, loss = make_benchmark_problem(dimension, precision)
    config = DAGMCTSConfig(
        num_simulations = num_simulations,
        degree = 1,
        value_transform = negation_transform(),
        persist_table = persist_table,
        track_parents = true,
        selection_mode = VisitCount,
        variant = variant
    )
    optim = dag_mcts_descent_init(param, loss, config)

    for _ in 1:steps
        step!(optim) && break
    end

    return optim
end

function run_single_search(
        variant::DAGMCTSVariant,
        num_simulations::Int,
        dimension::Int,
        precision::Int,
        persist_table::Bool,
        seed::Int
)
    Random.seed!(seed)
    param, loss = make_benchmark_problem(dimension, precision)
    config = DAGMCTSConfig(
        num_simulations = num_simulations,
        degree = 1,
        value_transform = negation_transform(),
        persist_table = persist_table,
        track_parents = true,
        selection_mode = VisitCount,
        variant = variant
    )
    optim = dag_mcts_descent_init(param, loss, config)

    if optim.state isa DAGMCTSPathState
        NonArchimedeanMachineLearning.dag_mcts_path_search(
            optim.state.root,
            optim.state.transposition_table,
            loss,
            config,
            optim.state
        )
    else
        NonArchimedeanMachineLearning.dag_mcts_search(
            optim.state.root,
            optim.state.transposition_table,
            loss,
            config,
            optim.state
        )
    end

    return optim
end

function measure_variant(args...; samples::Int)
    times = Float64[]
    bytes = Int[]

    # Warm up outside the measured loop.
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
        min_bytes = minimum(bytes),
        median_bytes = median(bytes)
    )
end

function stat_value(stats, field::Symbol)
    return hasproperty(stats, field) ? string(getproperty(stats, field)) : "-"
end

function summarize_variant(
        variant::DAGMCTSVariant,
        steps::Int,
        num_simulations::Int,
        dimension::Int,
        precision::Int,
        persist_table::Bool,
        seed::Int
)
    optim = run_variant(
        variant, steps, num_simulations, dimension, precision, persist_table, seed)
    search_optim = run_single_search(
        variant, num_simulations, dimension, precision, persist_table, seed)
    stats = get_dag_stats(search_optim.state)
    return (
        final_loss = eval_loss(optim),
        unique_nodes = stat_value(stats, :unique_nodes),
        multi_parent_nodes = stat_value(stats, :multi_parent_nodes),
        total_visits = stat_value(stats, :total_visits),
        path_count = stat_value(stats, :path_count)
    )
end

function main()
    samples = option_int("samples", "NAML_BENCH_SAMPLES", 5)
    steps = option_int("steps", "NAML_BENCH_STEPS", 4)
    num_simulations = option_int("simulations", "NAML_BENCH_SIMULATIONS", 100)
    dimension = option_int("dimension", "NAML_BENCH_DIMENSION", 3)
    precision = option_int("precision", "NAML_BENCH_PRECISION", 12)
    persist_table = option_bool("persist", "NAML_BENCH_PERSIST", true)
    seed = option_int("seed", "NAML_BENCH_SEED", 2026)

    println("DAG-MCTS variant benchmark")
    println("  samples=$samples steps=$steps simulations=$num_simulations dimension=$dimension precision=$precision persist_table=$persist_table seed=$seed")
    println("  timings measure full optimizer runs; DAG counters are from one root search")
    println()
    @printf("%-12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "variant", "min time", "median", "min alloc", "loss",
        "nodes", "multi-par", "visits", "paths")
    println("-"^120)

    for (label, variant) in VARIANTS
        args = (variant, steps, num_simulations, dimension, precision, persist_table, seed)
        measurement = measure_variant(args...; samples = samples)
        summary = summarize_variant(args...)
        @printf("%-12s %12s %12s %12s %12.4f %12s %12s %12s %12s\n",
            label,
            BenchmarkTools.prettytime(measurement.min_time * 1e9),
            BenchmarkTools.prettytime(measurement.median_time * 1e9),
            BenchmarkTools.prettymemory(measurement.min_bytes),
            summary.final_loss,
            summary.unique_nodes,
            summary.multi_parent_nodes,
            summary.total_visits,
            summary.path_count)
    end
end

main()
