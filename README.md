# NonArchimedeanMachineLearning — Non-Archimedean Optimisation (and a bit of ML)

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Paul-Lez.github.io/NonArchimedeanMachineLearning.jl/dev/)
[![Build Status](https://github.com/Paul-Lez/NonArchimedeanMachineLearning.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Paul-Lez/NonArchimedeanMachineLearning.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/Paul-Lez/NonArchimedeanMachineLearning.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Paul-Lez/NonArchimedeanMachineLearning.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2606.07782-b31b1b.svg)](https://arxiv.org/abs/2606.07782)

NonArchimedeanMachineLearning is a Julia package for optimization over non-Archimedean fields. It provides tools to define objective functions on p-adic polydiscs and minimize them using a range of optimizers, from greedy descent to tree-search methods (MCTS, DOO, plus other experimental implementations).

## Installation

NonArchimedeanMachineLearning depends on [Oscar.jl](https://github.com/oscar-system/Oscar.jl). From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/Paul-Lez/NonArchimedeanMachineLearning.jl")
```

Or clone the repository and activate the project locally:

```bash
git clone https://github.com/Paul-Lez/NonArchimedeanMachineLearning.jl.git
cd NonArchimedeanMachineLearning.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

Minimize |x² - 1|₂ over the 2-adic integers — the true minimizers are x = ±1.

```julia
using NonArchimedeanMachineLearning
using Oscar

# Set up the 2-adic field with precision 20
K = PadicField(2, 20)
R, (x,) = polynomial_ring(K, ["x"])

# Define the objective: f(x) = |x² - 1|₂
f = AbsolutePolynomialSum([x^2 - 1])

# Build a typed loss function
VP    = ValuationPolydisc{PadicFieldElem, Int, 1}
batch = NonArchimedeanMachineLearning.batch_evaluate_init(f, VP)
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
println("Center:    ", NonArchimedeanMachineLearning.center(optim.param)) # ≈ ±1
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

See [demos/](demos/) for complete runnable scripts.

## Documentation

Full documentation is available at **[paul-lez.github.io/NonArchimedeanMachineLearning.jl/dev/](https://Paul-Lez.github.io/NonArchimedeanMachineLearning.jl/dev/)**.

## Citing
If this package was useful for your research, please consider adding the following citation (preprint to be uploaded to the arXiv very soon)
```
@article{lezeau2026nonarchimedean,
  title={Non-Archimedean Polydisc Spaces and Applications to Optimisation},
  author={Lezeau, Paul and Fam, Yiannis and Monod, Anthea and Ren, Yue},
  year={2026}
  url = {https://arxiv.org/abs/2606.07782},
}
```
