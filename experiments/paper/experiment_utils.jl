"""
Shared experiment utilities for the NAML paper experiment infrastructure.

Provides:
1. Unified CLI argument parsing
2. Canonical optimizer factory (all optimizers in one place)
3. Threaded experiment runner (parallelizes across optimizers within a sample)
4. Raw JSON serialization (no statistics — that's make_stats.jl's job)

Usage from any run_experiments.jl:
    include("../experiment_utils.jl")
    args = parse_experiment_args(ARGS)
    ...
"""

using JSON
using Printf
using Dates
using Random

# ============================================================================
# CLI Argument Parsing
# ============================================================================

"""
    parse_experiment_args(ARGS) -> NamedTuple

Unified CLI argument parser for all run_experiments.jl scripts.

Returns a NamedTuple with fields:
- quick_mode::Bool
- save_results::Bool
- use_config_file::Bool
- use_paper_config::Bool
- n_epochs::Int
- output_filename::Union{String, Nothing}
- n_samples_override::Union{Int, Nothing}
- selection_mode  (NAML.BestValue, NAML.VisitCount, or NAML.BestLoss)
- mcts_degree_override::Union{Int, Nothing}
- description::String
- git_commit::String
"""
function parse_experiment_args(args)
    quick_mode = "--quick" in args
    save_results = "--save" in args
    use_config_file = "--config" in args
    
    # Paper-ready flags
    use_optimizer_comparison = "--paper-optimizer-comparison" in args || "--paper" in args
    use_mcts_branching = "--paper-mcts-branching" in args
    use_dag_mcts_branching = "--paper-dag-mcts-branching" in args
    use_greedy_branching = "--paper-greedy-descent-branching" in args
    use_gradient_branching = "--paper-gradient-descent-branching" in args
    use_mcts_sims = "--paper-mcts-number-of-simulations" in args
    use_dag_mcts_sims = "--paper-dag-mcts-number-of-simulations" in args
    use_mcts_exp = "--paper-mcts-exploration-constant" in args
    use_dag_mcts_exp = "--paper-dag-mcts-exploration-constant" in args

    use_paper_config = use_optimizer_comparison || use_mcts_branching || 
                      use_dag_mcts_branching || use_greedy_branching || 
                      use_gradient_branching || use_mcts_sims || 
                      use_dag_mcts_sims || use_mcts_exp || use_dag_mcts_exp

    # Default epochs
    n_epochs = quick_mode ? 5 : 20

    output_filename = nothing
    n_samples_override = nothing
    selection_mode = NAML.BestValue
    mcts_degree_override = nothing
    description = ""
    git_commit = ""

    for (i, arg) in enumerate(args)
        if arg == "--epochs" && i < length(args)
            n_epochs = parse(Int, args[i+1])
        elseif arg == "--output" && i < length(args)
            output_filename = args[i+1]
        elseif arg == "--samples" && i < length(args)
            n_samples_override = parse(Int, args[i+1])
        elseif arg == "--selection-mode" && i < length(args)
            mode_str = args[i+1]
            if mode_str == "BestValue"
                selection_mode = NAML.BestValue
            elseif mode_str == "VisitCount"
                selection_mode = NAML.VisitCount
            elseif mode_str == "BestLoss"
                selection_mode = NAML.BestLoss
            else
                error("Invalid selection mode: $mode_str. Must be BestValue, VisitCount, or BestLoss")
            end
        elseif arg == "--degree" && i < length(args)
            mcts_degree_override = parse(Int, args[i+1])
        elseif startswith(arg, "--degree=")
            mcts_degree_override = parse(Int, arg[10:end])
        elseif arg == "--description" && i < length(args)
            description = args[i+1]
        elseif arg == "--git-commit" && i < length(args)
            git_commit = args[i+1]
        end
    end

    return (
        quick_mode = quick_mode,
        save_results = save_results,
        use_config_file = use_config_file,
        use_paper_config = use_paper_config,
        use_optimizer_comparison = use_optimizer_comparison,
        use_mcts_branching = use_mcts_branching,
        use_dag_mcts_branching = use_dag_mcts_branching,
        use_greedy_branching = use_greedy_branching,
        use_gradient_branching = use_gradient_branching,
        use_mcts_sims = use_mcts_sims,
        use_dag_mcts_sims = use_dag_mcts_sims,
        use_mcts_exp = use_mcts_exp,
        use_dag_mcts_exp = use_dag_mcts_exp,
        n_epochs = n_epochs,
        output_filename = output_filename,
        n_samples_override = n_samples_override,
        selection_mode = selection_mode,
        mcts_degree_override = mcts_degree_override,
        description = description,
        git_commit = git_commit,
    )
end


# ============================================================================
# Load configurations
# ============================================================================

function load_config_file(experiment_dir::String, args)
    if args.use_paper_config
        include(joinpath(experiment_dir, "paper_config.jl"))
    elseif args.use_config_file
        include(joinpath(experiment_dir, "config.jl"))
    end
end

"""
    load_configs(experiment_dir, args, default_configs) -> Vector{Dict}

Load experiment configurations based on CLI flags.
"""
function load_configs(args, default_configs::Vector)
    configs = if args.use_paper_config
        println("Loaded PAPER-READY experiment configurations")
        paper_experiments
    elseif args.use_config_file
        println("Loaded experiment configurations from config.jl")
        experiment_configs
    else
        println("Using default configurations")
        default_configs
    end

    # Apply samples override
    if !isnothing(args.n_samples_override)
        for config in configs
            config["num_samples"] = args.n_samples_override
        end
        println("Overriding num_samples to $(args.n_samples_override) for all configs")
    end

    return configs
end


# ============================================================================
# Optimizer Factory
# ============================================================================

"""Canonical display ordering for all experiments."""
const OPTIMIZER_ORDER = [
    "Random", "Best-First", "Best-First-branch2", "Best-First-Gradient",
    "MCTS-k", "MCTS-5k", "MCTS-10k",
    "DAG-MCTS-k", "DAG-MCTS-5k", "DAG-MCTS-10k",
    "DOO"
]

const NAME_WIDTH = maximum(length(n) for n in OPTIMIZER_ORDER)

"""
    get_optimizer_configs(config::Dict, args::NamedTuple) -> Dict{String, Dict{String, Any}}

Return a nested Dict of SuiteName => { OptimizerName => OptimizerSetup }.
Each Setup is Dict("init" => (param, loss) -> OptimSetup).

Results are organized by suite to allow rigorous comparison. Optimizers may 
appear in multiple suites and will be run independently for each.
"""
function get_optimizer_configs(config::Dict, args::NamedTuple)
    suites = Dict{String, Dict{String, Any}}()

    prime = config["prime"]
    # num_vars is the parameter-polydisc dimension. Some experiments (polynomial
    # learning, function learning) do not set it explicitly — they learn
    # degree+1 polynomial coefficients.
    dim = if haskey(config, "num_vars")
        config["num_vars"]
    elseif haskey(config, "degree")
        config["degree"] + 1
    else
        error("Cannot determine num_vars for config $(get(config, "name", "?"))")
    end
    quick = args.quick_mode
    selection_mode = args.selection_mode
    
    p_float = Float64(prime)

    # Helper to create standard MCTS config
    function mk_mcts(sims, deg, exp=1.41)
        return Dict(
            "init" => (param, loss) -> begin
                c = NAML.MCTSConfig(
                    num_simulations=sims,
                    exploration_constant=exp,
                    selection_mode=selection_mode,
                    degree=deg
                )
                NAML.mcts_descent_init(param, loss, c)
            end
        )
    end

    # Helper to create standard DAG-MCTS config
    function mk_dag_mcts(sims, deg, exp=1.41)
        return Dict(
            "init" => (param, loss) -> begin
                c = NAML.DAGMCTSConfig(
                    num_simulations=sims,
                    exploration_constant=exp,
                    degree=deg,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, c)
            end
        )
    end

    # suite 0: Standard Optimizer Comparison (legacy --paper)
    if args.use_optimizer_comparison
        s = Dict{String, Any}()
        deg = effective_degree(dim, args.mcts_degree_override)
        k = binomial(dim, deg) * prime^deg
        sims_10k = quick ? 200 : 10 * k

        s["Random"] = Dict("init" => (param, loss) -> NAML.random_descent_init(param, loss, 1, (false, 1)))
        s["Best-First"] = Dict("init" => (param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, deg)))
        s["Best-First-Gradient"] = Dict("init" => (param, loss) -> NAML.gradient_descent_init(param, loss, 1, (false, deg)))
        s["MCTS-10k"] = mk_mcts(sims_10k, deg)
        s["DAG-MCTS-10k"] = mk_dag_mcts(sims_10k, deg)
        s["DOO"] = Dict(
            "init" => (param, loss) -> begin
                delta = h -> p_float^(-h)
                c = NAML.DOOConfig(delta=delta, degree=deg, strict=false)
                NAML.doo_descent_init(param, loss, 1, c)
            end
        )
        suites["optimizer-comparison"] = s
    end

    # suite 1: MCTS Branching (2+ vars, deg 1 & 2, 10k sims)
    if args.use_mcts_branching && dim >= 2
        s = Dict{String, Any}()
        for deg in [1, 2]
            k = binomial(dim, deg) * prime^deg
            sims = quick ? 200 : 10 * k
            s["MCTS-10k-deg$deg"] = mk_mcts(sims, deg)
        end
        suites["mcts-branching"] = s
    end

    # suite 2: DAG-MCTS Branching (2+ vars, deg 1 & 2, 10k sims)
    if args.use_dag_mcts_branching && dim >= 2
        s = Dict{String, Any}()
        for deg in [1, 2]
            k = binomial(dim, deg) * prime^deg
            sims = quick ? 200 : 10 * k
            s["DAG-MCTS-10k-deg$deg"] = mk_dag_mcts(sims, deg)
        end
        suites["dag-mcts-branching"] = s
    end

    # suite 3: Greedy Branching (2+ vars, deg 1 & 2)
    if args.use_greedy_branching && dim >= 2
        s = Dict{String, Any}()
        s["Greedy-deg1"] = Dict("init" => (param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, 1)))
        s["Greedy-deg2"] = Dict("init" => (param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, 2)))
        suites["greedy-descent-branching"] = s
    end

    # suite 4: Gradient Branching (2+ vars, deg 1 & 2)
    if args.use_gradient_branching && dim >= 2
        s = Dict{String, Any}()
        s["Gradient-deg1"] = Dict("init" => (param, loss) -> NAML.gradient_descent_init(param, loss, 1, (false, 1)))
        s["Gradient-deg2"] = Dict("init" => (param, loss) -> NAML.gradient_descent_init(param, loss, 1, (false, 2)))
        suites["gradient-descent-branching"] = s
    end

    # suite 5: MCTS Number of Simulations (2+ vars, deg 2, k/5k/10k sims)
    if args.use_mcts_sims && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        s["MCTS-k"] = mk_mcts(quick ? 50 : k, deg)
        s["MCTS-5k"] = mk_mcts(quick ? 100 : 5 * k, deg)
        s["MCTS-10k"] = mk_mcts(quick ? 200 : 10 * k, deg)
        suites["mcts-number-of-simulations"] = s
    end

    # suite 6: DAG-MCTS Number of Simulations (2+ vars, deg 2, k/5k/10k sims)
    if args.use_dag_mcts_sims && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        s["DAG-MCTS-k"] = mk_dag_mcts(quick ? 50 : k, deg)
        s["DAG-MCTS-5k"] = mk_dag_mcts(quick ? 100 : 5 * k, deg)
        s["DAG-MCTS-10k"] = mk_dag_mcts(quick ? 200 : 10 * k, deg)
        suites["dag-mcts-number-of-simulations"] = s
    end

    # suite 7: MCTS Exploration Constant (2+ vars, deg 2, 10k sims, sweep exp)
    if args.use_mcts_exp && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        sims = quick ? 200 : 10 * k
        for exp in 1.4:0.1:2.4
            name = @sprintf("MCTS-10k-exp%.1f", exp)
            s[name] = mk_mcts(sims, deg, exp)
        end
        suites["mcts-exploration-constant"] = s
    end

    # suite 8: DAG-MCTS Exploration Constant (2+ vars, deg 2, 10k sims, sweep exp)
    if args.use_dag_mcts_exp && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        sims = quick ? 200 : 10 * k
        for exp in 1.4:0.1:2.4
            name = @sprintf("DAG-MCTS-10k-exp%.1f", exp)
            s[name] = mk_dag_mcts(sims, deg, exp)
        end
        suites["dag-mcts-exploration-constant"] = s
    end

    return suites
end

"""
    effective_degree(num_dims, mcts_degree_override) -> Int

Compute the effective MCTS/tree branching degree.
Default: 1 for 1-dimensional, 2 for ≥2 dimensions.
"""
function effective_degree(num_dims::Int, mcts_degree_override)
    auto_degree = num_dims >= 2 ? 2 : 1
    return isnothing(mcts_degree_override) ? auto_degree : mcts_degree_override
end


# ============================================================================
# Single optimizer run
# ============================================================================

"""
    run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                         post_run_fn=nothing) -> Dict

Run a single optimizer on a single problem instance.
Returns a Dict with raw results (no ranking or aggregate stats).

Deep-copies `initial_param` to avoid mutation issues when running in parallel.

If `post_run_fn` is provided, it is called as `post_run_fn(optim)` and
the returned Dict is merged into the result. Use this for experiment-specific
fields like classification accuracy.
"""
function run_single_optimizer(opt_name::String, opt_setup::Dict,
                               initial_param, loss, n_epochs::Int;
                               post_run_fn::Union{Function,Nothing}=nothing)
    # Deep copy starting parameter to avoid cross-thread mutation
    param_copy = deepcopy(initial_param)
    initial_loss_val = loss.eval([param_copy])[1]

    try
        # Wrap loss with evaluation counting
        counted_loss, eval_counter = wrap_loss_with_counting(loss)

        optim = opt_setup["init"](param_copy, counted_loss)

        losses = Float64[]
        t_start = time()

        for epoch in 1:n_epochs
            current_loss = NAML.eval_loss(optim)
            push!(losses, current_loss)
            NAML.step!(optim)
            NAML.has_converged(optim) && break
        end

        t_end = time()
        elapsed = t_end - t_start

        final_loss = NAML.eval_loss(optim)
        push!(losses, final_loss)

        # Subtract monitoring eval_loss calls
        monitoring_evals = length(losses)
        total_optimizer_evals = eval_counter.eval_count - monitoring_evals + eval_counter.grad_count

        result = Dict{String, Any}(
            "time" => elapsed,
            "final_loss" => final_loss,
            "losses" => losses,
            "improvement" => initial_loss_val - final_loss,
            "improvement_ratio" => (initial_loss_val > 0) ?
                (initial_loss_val - final_loss) / initial_loss_val : 0.0,
            "total_evals" => total_optimizer_evals,
        )

        # Run experiment-specific post-processing (e.g., accuracy computation)
        if !isnothing(post_run_fn)
            extra = post_run_fn(optim)
            merge!(result, extra)
        end

        return result
    catch e
        return Dict{String, Any}("error" => string(e))
    end
end


# ============================================================================
# Threaded run across all optimizers for one sample
# ============================================================================

"""
    run_all_optimizers_serial(opt_configs, initial_param, loss, n_epochs;
                              post_run_fn=nothing) -> Dict{String,Any}

Run all optimizers on a single problem instance, serially (no threading).

Use this when the caller is already parallelizing at a coarser level (e.g. over
`(config, sample)` pairs) and each task owns its own `loss`. Running optimizers
serially here avoids nested threading and contention on shared evaluator state.
"""
function run_all_optimizers_serial(opt_configs::Dict, initial_param, loss, n_epochs::Int;
                                    post_run_fn::Union{Function,Nothing}=nothing)
    results = Dict{String, Any}()
    for opt_name in keys(opt_configs)
        opt_setup = opt_configs[opt_name]
        results[opt_name] = run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                                                  post_run_fn=post_run_fn)
    end
    return results
end


"""
    run_all_optimizers_threaded(opt_configs, initial_param, loss, n_epochs;
                                post_run_fn=nothing) -> Dict{String,Any}

Run all optimizers on a single problem instance, using threads.
Returns a Dict mapping optimizer name => result Dict.
"""
function run_all_optimizers_threaded(opt_configs::Dict, initial_param, loss, n_epochs::Int;
                                     post_run_fn::Union{Function,Nothing}=nothing)
    opt_names = collect(keys(opt_configs))

    results = Dict{String, Any}()
    result_lock = ReentrantLock()

    Threads.@threads for i in 1:length(opt_names)
        opt_name = opt_names[i]
        opt_setup = opt_configs[opt_name]
        result = run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                                       post_run_fn=post_run_fn)
        lock(result_lock) do
            results[opt_name] = result
        end
    end

    return results
end


# ============================================================================
# JSON serialization
# ============================================================================

"""
    build_metadata(; experiment_type, n_epochs, quick_mode, optimizer_order,
                     description, git_commit, extra...) -> Dict

Build metadata dict for JSON output.
"""
function build_metadata(; experiment_type::String,
                          n_epochs::Int,
                          quick_mode::Bool,
                          optimizer_order::Vector{String}=String[],
                          suites::Vector{String}=String[],
                          description::String="",
                          git_commit::String="",
                          extra::Dict{String,Any}=Dict{String,Any}())
    metadata = Dict{String, Any}(
        "experiment_type" => experiment_type,
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        "n_epochs" => n_epochs,
        "quick_mode" => quick_mode,
        "optimizer_order" => optimizer_order,
        "suites" => suites,
        "description" => description,
        "git_commit" => git_commit,
    )
    merge!(metadata, extra)
    return metadata
end

"""
    save_raw_results(all_results, metadata, filepath)

Save raw experiment results to JSON. No aggregate stats — just raw per-sample data.
"""
function save_raw_results(all_results::Vector, metadata::Dict, filepath::String)
    json_experiments = []
    for result in all_results
        json_result = Dict{String, Any}()
        json_result["config"] = result["config"]

        if haskey(result, "error")
            json_result["error"] = result["error"]
        else
            json_result["samples"] = result["samples"]
        end

        push!(json_experiments, json_result)
    end

    json_output = Dict{String, Any}(
        "metadata" => metadata,
        "experiments" => json_experiments,
    )

    open(filepath, "w") do f
        JSON.print(f, json_output, 2)
    end

    println("\n✓ Raw results saved to: $filepath")
end


# ============================================================================
# Progress printing
# ============================================================================

"""Print a brief per-sample summary to stdout."""
function print_sample_summary(sample_result::Dict, initial_loss::Float64)
    println(@sprintf("    Initial: %.6e", initial_loss))
    for opt_name in OPTIMIZER_ORDER
        if haskey(sample_result, opt_name)
            opt_result = sample_result[opt_name]
            if !haskey(opt_result, "error")
                println(Printf.format(Printf.Format("    %-$(NAME_WIDTH)s Final: %.6e (Δ: %.6e, %.1f%%)"),
                    opt_name, opt_result["final_loss"], opt_result["improvement"],
                    opt_result["improvement_ratio"] * 100))
            end
        end
    end
end
