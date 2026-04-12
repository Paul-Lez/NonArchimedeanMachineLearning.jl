# NAML — Non-Archimedean Optimisation (and a bit of ML)

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Paul-Lez.github.io/naml/dev/)
[![Build Status](https://github.com/Paul-Lez/naml/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Paul-Lez/naml/actions/workflows/CI.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/Paul-Lez/naml/branch/main/graph/badge.svg)](https://codecov.io/gh/Paul-Lez/naml)

NAML is a Julia package for optimization over non-Archimedean fields. It provides tools to define objective functions on p-adic polydiscs and minimize them using a range of optimizers, from greedy descent to tree-search methods (MCTS, DOO, plus other experimental implementations).

## Installation

NAML depends on [Oscar.jl](https://github.com/oscar-system/Oscar.jl). From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/Paul-Lez/NAML.jl")
```

Or clone the repository and activate the project locally:

```bash
git clone https://github.com/Paul-Lez/NAML.jl.git
cd NAML.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

Minimize |x² - 1|₂ over the 2-adic integers — the true minimizers are x = ±1.

```julia
using NAML
using Oscar

# Set up the 2-adic field with precision 20
K = PadicField(2, 20)
R, (x,) = polynomial_ring(K, ["x"])

# Define the objective: f(x) = |x² - 1|₂
f = AbsolutePolynomialSum([x^2 - 1])

# Build a typed loss function
VP    = ValuationPolydisc{PadicFieldElem, Int, 1}
batch = NAML.batch_evaluate_init(f, VP)
loss  = Loss(params -> map(batch, params), _ -> 0)

# Starting polydisc: the 2-adic unit ball centered at 1
initial_param = VP((K(1),), (0,))

# Initialize an optimizer (e.g. Greedy Descent) and run
optim = greedy_descent_init(initial_param, loss, 1, (false, 1))
for _ in 1:60
    step!(optim)
    has_converged(optim) && break
end

println("Best loss: ", eval_loss(optim))       # ≈ 0
println("Center:    ", NAML.center(optim.param)) # ≈ ±1
```

Other optimizers can be swapped in just as easily:

```julia
# MCTS
mcts_optim = mcts_descent_init(initial_param, loss,
    MCTSConfig(num_simulations=50, exploration_constant=1.41,
               degree=1, selection_mode=BestValue))

# Deterministic Optimistic Optimization (DOO)
doo_optim = doo_descent_init(initial_param, loss, 1,
    DOOConfig(delta=h -> 2.0^(-h), degree=1))
```

See [experiments/paper/worked_examples/](experiments/paper/worked_examples/) for complete runnable scripts.

## Documentation

Full documentation is available at **[paul-lez.github.io/naml/dev/](https://Paul-Lez.github.io/naml/dev/)**.

## Repository structure

```
src/
├── basic/              Core algebraic structures
│   ├── valuation.jl        Valuations and absolute values
│   ├── polydisc.jl         Polydiscs and their operations
│   ├── tangent_vector.jl   Tangent vectors on polydisc spaces
│   ├── functions.jl        Compositional function algebra
│   └── valued_point.jl     Valued field point wrapper
├── optimization/       Optimization framework
│   ├── model.jl            Abstract model interface
│   ├── optim_setup.jl      Core optimizer API (step!, has_converged, ...)
│   ├── loss.jl             Standard losses (MSE, MPE)
│   └── optimizers/
│       ├── greedy_descent.jl
│       ├── gradient_descent.jl
│       ├── random_descent.jl
│       └── tree_search/    MCTS, DOO, DAG-MCTS
├── statistics/         Frechet means, least squares
└── visualization/      Loss landscape plots, search tree rendering

test/                   Test suite
docs/                   Documenter.jl source
```

## Citing
If this package was useful for your research, please consider adding the following citation (preprint to be uploaded to the arXiv very soon)
```
@article{lezeau2026nonarchimedean,
  title={Non-Archimedean Polydisc Spaces and Applications to Optimisation},
  author={Lezeau, Paul and Fam, Yiannis and Monod, Anthea and Ren, Yue},
  year={2026}
}
```
