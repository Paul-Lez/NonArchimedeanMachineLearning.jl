"""
Worked Example: Minimizing |x² - 1|₂ over Q₂

The function to minimize is:
    f(x) = |x² - 1|₂

over x ∈ Q₂. The global minimizers are x = ±1 (where f(±1) = 0).

This is the same function whose loss landscape is visualized in
experiments/paper/pictures/x2_minus_1_landscape.jl.

Three optimizers are compared:
  - Greedy Descent (Best-First)
  - MCTS (Monte Carlo Tree Search)
  - DOO (Deterministic Optimistic Optimization)

Usage:
    julia --project=. experiments/paper/worked_examples/x2_minus_1_minimization.jl
"""

include("../../../src/NAML.jl")
using .NAML
using Oscar
using Printf
using Random
using Plots
Random.seed!(42)

println("=== Worked Example: Minimizing |x² - 1|₂ over Q₂ ===\n")

# ── Display helper ─────────────────────────────────────────────────────────────
function padic_display(x::PadicFieldElem)
    NAML.valuation(x) >= 0 || return string(x)
    K    = Oscar.parent(x)
    pprec = Int(Oscar.prime(K))^Int(Oscar.precision(K))
    n    = Int(Oscar.lift(ZZ, x))
    n    = n > pprec ÷ 2 ? n - pprec : n
    return string(n)
end

# ── Setup ──────────────────────────────────────────────────────────────────────

prec = 20
p    = 2
K    = PadicField(p, prec)

R, (x,) = polynomial_ring(K, ["x"])
f = AbsolutePolynomialSum([x^2 - 1])

# Build a typed loss using the typed-evaluator interface
VP    = ValuationPolydisc{PadicFieldElem, Int, 1}
batch = NAML.batch_evaluate_init(f, VP)
loss  = Loss(params -> map(batch, params), _ -> 0)

# Starting polydisc: center at 1, radius 0 (the 2-adic unit ball centered at 1)
# This disc contains both minimizers x = 1 and x = -1.
initial_param = VP((K(1),), (0,))

println("Initial parameter: center = $(NAML.center(initial_param)[1]), radius = $(NAML.radius(initial_param)[1])")
println("Initial loss:      $(@sprintf("%.6e", loss.eval([initial_param])[1]))\n")
println("Known minimizers:  x = ±1  (f(±1) = 0)\n")

# ── Helper: run an optimizer, return (best_loss, best_param, elapsed_seconds) ──

function run_optimizer(optim, n_steps)
    best_loss  = eval_loss(optim)
    best_param = optim.param
    loss_history = Float64[best_loss]
    t_start    = time()
    for _ in 1:n_steps
        step!(optim)
        l = eval_loss(optim)
        if l < best_loss
            best_loss  = l
            best_param = optim.param
        end
        push!(loss_history, best_loss)
        has_converged(optim) && break
    end
    return best_loss, best_param, time() - t_start, loss_history
end

n_steps = 60

# ── Greedy Descent ─────────────────────────────────────────────────────────────

println("Running Greedy Descent ($n_steps steps)...")
greedy_optim = greedy_descent_init(initial_param, loss, 1, (false, 1))
greedy_loss, greedy_param, greedy_time, greedy_history = run_optimizer(greedy_optim, n_steps)

# ── MCTS ───────────────────────────────────────────────────────────────────────

println("Running MCTS ($n_steps steps, 50 simulations/step)...")
mcts_config = MCTSConfig(
    num_simulations      = 50,
    exploration_constant = 1.41,
    degree               = 1,
    selection_mode       = BestValue
)
mcts_optim = mcts_descent_init(initial_param, loss, mcts_config)
mcts_loss, mcts_param, mcts_time, mcts_history = run_optimizer(mcts_optim, n_steps)

# ── DOO ────────────────────────────────────────────────────────────────────────

println("Running DOO ($n_steps steps)...")
p_val = Float64(p)
doo_config = DOOConfig(
    delta     = h -> p_val^(-h),
    max_depth = 15,
    degree    = 1
)
doo_optim = doo_descent_init(initial_param, loss, 1, doo_config)
doo_loss, doo_param, doo_time, doo_history = run_optimizer(doo_optim, 8 * n_steps)

# ── Console summary ────────────────────────────────────────────────────────────

println()
println("="^70)
println("RESULTS")
println("="^70)
@printf("  %-16s  %-6s  %-10s  %-14s  %s\n",
        "Optimizer", "Radius", "Loss", "Time (s)", "Center")
println("-"^70)
for (name, lss, prm, t) in [
        ("Greedy Descent", greedy_loss, greedy_param, greedy_time),
        ("MCTS",           mcts_loss,   mcts_param,   mcts_time),
        ("DOO",            doo_loss,    doo_param,    doo_time)]
    c = NAML.center(prm)[1]
    r = NAML.radius(prm)[1]
    @printf("  %-16s  %-6d  %-10.3e  %-14.3f  %s\n", name, r, lss, t, string(c))
end
println("="^70)
println("\nTrue minimizers: x = 1 or x = -1,  f(±1) = 0")

# ── LaTeX table ────────────────────────────────────────────────────────────────

println()
println("LaTeX table:")
println()
println(raw"\begin{tabular}{lcrrc}")
println(raw"  \toprule")
println(raw"  Optimizer & Center $c$ & Radius $r$ & Loss $f\!\left(B(c,\,2^{-r})\right)$ & Time\,(s) \\\\")
println(raw"  \midrule")
for (name, lss, prm, t) in [
        ("Greedy Descent", greedy_loss, greedy_param, greedy_time),
        ("MCTS",           mcts_loss,   mcts_param,   mcts_time),
        ("DOO",            doo_loss,    doo_param,    doo_time)]
    c = NAML.center(prm)[1]
    r = NAML.radius(prm)[1]
    loss_str = let e = floor(Int, log10(max(lss, 1e-300))),
                   m = lss / 10.0^e
        @sprintf("\$%.2f\\text{e}%d\$", m, e)
    end
    println(@sprintf("  %-16s & \$%s\$ & \$%d\$ & %s & \$%.3f\$ \\\\",
                     name, padic_display(c), r, loss_str, t))
end
println(raw"  \bottomrule")
println(raw"\end{tabular}")
println()

# ── Loss curve plot ───────────────────────────────────────────────────────────

println("Generating loss curve plot...")
plt = plot(
    0:length(greedy_history)-1, greedy_history,
    label="Greedy Descent", linewidth=2, yscale=:log10,
    xlabel="Epoch", ylabel="Best Loss",
    title="|x² - 1|₂ Minimization — Loss Curves",
    legend=:topright
)
plot!(plt, 0:length(mcts_history)-1, mcts_history, label="MCTS", linewidth=2)
plot!(plt, 0:length(doo_history)-1, doo_history, label="DOO", linewidth=2)

outpath = joinpath(@__DIR__, "x2_minus_1_loss_curves.png")
savefig(plt, outpath)
println("Loss curve saved to $outpath")

println("Done!")
