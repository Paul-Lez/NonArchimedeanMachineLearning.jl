# Paper Experiments - Benchmarks and Utilities

This directory contains benchmark infrastructure, experiment utilities, and organized subdirectories for paper-ready experiments using the NAML library.

## Directory Structure

```
experiments/paper/
├── util.jl                          # Shared utility functions
├── test_util.jl                     # Tests for utilities
├── polynomial_learning/             # Polynomial interpolation experiments
├── absolute_sum_minimization/       # Absolute sum optimization experiments
└── function_learning/               # Binary classification experiments
```

## Core Utilities

### `util.jl` - Shared Experiment API

Provides reusable functions for all experiments:

- **`generate_random_padic(p, prec, min_exp, num_terms)`**: Generate random p-adic numbers with configurable exponent range
- **`polynomial_to_linear_loss(data, degree, cutoff_val)`**: Transform polynomial learning into linear optimization
  - Handles both p-adic outputs (`cutoff_val=nothing`) and real outputs (with cutoff)
- **`polynomial_to_crossentropy_loss(data, degree, threshold, scale)`**: Smooth binary classification loss
- **`polynomial_to_valuation_crossentropy_loss(data, degree, prime, threshold, scale)`**: Classification with p-adic valuation
- **`polynomial_to_mse_loss(data, degree)`**: Direct MSE regression
- **`generate_gauss_point(n, K)`**: Create Gauss points (standard starting point)
- **`generate_polynomial_learning_data(p, prec, n_points)`**: Generate training data with **guaranteed distinct x values**
- **`compute_classification_accuracy(model, data, param, threshold, scale)`**: Compute classification accuracy

### `test_util.jl` - Test Suite

Comprehensive tests for all utility functions:
- Random p-adic generation (including negative exponents)
- Uniqueness of generated data points
- Loss function creation (p-adic and real outputs)
- End-to-end optimization examples

## Experiment Subdirectories

### `polynomial_learning/` - **Production-Ready Benchmark** (NEW)

Polynomial interpolation experiments with multi-sample averaging and LaTeX table generation.

**Files:**
- `run_experiments.jl` - Main launcher with JSON output
- `config.jl` - Experiment configurations (degree sweep, prime sweep, comprehensive)
- `generate_tables.jl` - LaTeX table generator (5 table types)
- `polynomial_learning.ipynb` - Interactive Jupyter notebook
- `README.md` - Documentation

**Features:**
- Multi-sample averaging (mean, std, min, max)
- 6 optimizers: Greedy, MCTS-50/100, DAG-MCTS-100, UCT, HOO
- Hyperparameters stored in JSON
- Aggregate statistics across samples
- Paper-ready LaTeX tables

**Quick Start:**
```bash
julia --project=. experiments/paper/polynomial_learning/run_experiments.jl --quick --save
julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <results.json> --stdout
```

See [`polynomial_learning/README.md`](polynomial_learning/README.md) for details.

### `absolute_sum_minimization/`

Minimize sums of absolute polynomials: |f₁(x)| + |f₂(x)| + ... + |fₙ(x)|

**Files:**
- `run_experiments.jl` - Main launcher
- `config.jl`, `paper_config.jl` - Experiment configurations
- `util.jl` - Problem-specific utilities
- `test_setup.jl` - Setup testing

**Features:**
- Multi-sample averaging
- Random problem generation
- Optimizers: Random, Greedy, MCTS (50/100/200)

### `function_learning/`

Binary classification tasks using polynomial functions.

**Files:**
- `function_learning.ipynb` - Large-scale notebook experiments
- `learn_zero_function.jl` - Learn constant 0 function
- `learn_one_function.jl` - Learn constant 1 function (uses DAG-MCTS)

**Features:**
- Cross-entropy loss with sigmoid
- Classification accuracy computation
