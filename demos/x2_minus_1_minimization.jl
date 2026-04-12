#!/usr/bin/env julia
"""
Quick demo: minimizing |x² - 1|₂ over Q₂.

The true minimizers are x = ±1 (where f(±1) = 0). Three optimizers are
compared: Greedy Descent, MCTS, and DOO.

Usage:
    julia --project=. demos/x2_minus_1_minimization.jl
"""

include("../src/NAML.jl")
using .NAML
using Oscar

# ── Setup ─────────────────────────────────────────────────────────────────────

K = PadicField(2, 20)
R, (x,) = polynomial_ring(K, ["x"])

# Objective: f(x) = |x² - 1|₂
f = AbsolutePolynomialSum([x^2 - 1])

# Build a typed loss
VP    = ValuationPolydisc{PadicFieldElem, Int, 1}
batch = NAML.batch_evaluate_init(f, VP)
loss  = Loss(params -> map(batch, params), _ -> 0)

# Starting polydisc: the 2-adic unit ball centered at 1
initial_param = VP((K(1),), (0,))

println("Objective:    f(x) = |x² - 1|₂")
println("Initial loss: ", loss.eval([initial_param])[1])
println()

# ── Helper ────────────────────────────────────────────────────────────────────

function run_optimizer(optim, n_steps)
    best_loss  = eval_loss(optim)
    best_param = optim.param
    for _ in 1:n_steps
        step!(optim)
        l = eval_loss(optim)
        if l < best_loss
            best_loss  = l
            best_param = optim.param
        end
        has_converged(optim) && break
    end
    return best_loss, best_param
end

n_steps = 60

# ── Greedy Descent ────────────────────────────────────────────────────────────

greedy_optim = greedy_descent_init(initial_param, loss, 1, (false, 1))
greedy_loss, greedy_param = run_optimizer(greedy_optim, n_steps)

# ── MCTS ──────────────────────────────────────────────────────────────────────

mcts_optim = mcts_descent_init(initial_param, loss,
    MCTSConfig(num_simulations=50, exploration_constant=1.41,
               degree=1, selection_mode=BestValue))
mcts_loss, mcts_param = run_optimizer(mcts_optim, n_steps)

# ── DOO ───────────────────────────────────────────────────────────────────────

doo_optim = doo_descent_init(initial_param, loss, 1,
    DOOConfig(delta=h -> 2.0^(-h), degree=1))
doo_loss, doo_param = run_optimizer(doo_optim, 8 * n_steps)

# ── Results ───────────────────────────────────────────────────────────────────

println("Results (true minimizers: x = ±1, f(±1) = 0):\n")
for (name, l, prm) in [
        ("Greedy Descent", greedy_loss, greedy_param),
        ("MCTS",           mcts_loss,   mcts_param),
        ("DOO",            doo_loss,    doo_param)]
    c = NAML.center(prm)[1]
    r = NAML.radius(prm)[1]
    println("  $name:  loss = $l,  radius = $r,  center = $c")
end
