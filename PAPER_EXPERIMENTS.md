# Paper Experiments - Benchmarking Infrastructure

This document describes the experiment infrastructure in `experiments/paper/` for research and benchmarking.

**Note:** This infrastructure is under active development and will change frequently.

---

## Directory Structure

```
experiments/paper/
├── run_benchmarks.jl        # Main benchmarking script
├── benchmark_config.jl       # Configuration templates
├── util.jl                   # General utilities
└── test_util.jl             # Testing utilities
```

---

## Running Benchmarks

**Basic usage:**
```bash
julia --project=. experiments/paper/run_benchmarks.jl
```

This script orchestrates large-scale benchmarking experiments comparing different optimizers.

---

## Utility Functions (`util.jl`)

### Random P-adic Generation

```julia
generate_random_padic(p::Int, prec::Int, min_exp::Int, num_terms::Int) -> PadicFieldElem
```

Generate random p-adic number with specified precision and exponent range.

**Parameters:**
- `p::Int` - Prime for p-adic field
- `prec::Int` - P-adic precision
- `min_exp::Int` - Minimal exponent (can be negative, e.g., -2)
- `num_terms::Int` - Number of terms in expansion

**Returns:** p-adic number `Σ aᵢ * p^(min_exp + i)` where `aᵢ ∈ {0, ..., p-1}`

**Example:**
```julia
using Oscar
include("experiments/paper/util.jl")

# Generate 2-adic number: a₀*2^(-2) + a₁*2^(-1) + a₂*2^0 + a₃*2^1 + a₄*2^2
x = generate_random_padic(2, 20, -2, 5)
```

**Use cases:**
- Generate random training data
- Initialize parameters randomly
- Test robustness across input distributions

---

### Cutoff Functions

```julia
mk_cutoff(cutoff_val::Float64) -> Function
```

Create step function: returns 0 below `cutoff_val`, 1 above.

**Example:**
```julia
cutoff = mk_cutoff(0.5)
cutoff(0.3)  # Returns 0
cutoff(0.7)  # Returns 1
```

**Use in loss functions:**
```julia
# Loss with cutoff: stops penalizing below threshold
loss_with_cutoff = base_loss * cutoff(error_value)
```

---

### Additional Utilities

The `util.jl` file contains more functions for:
- Polynomial-to-linear transformations
- Gauss point generation
- Problem setup helpers

**Refer to the file for current implementation.**

---

## Benchmark Configuration (`benchmark_config.jl`)

This file contains configuration templates for experiments.

**Typical structure:**
```julia
# Optimizer configurations
mcts_config = MCTSConfig(
    num_simulations=100,
    exploration_constant=1.41
)

dag_mcts_config = DAGMCTSConfig(
    num_simulations=100,
    exploration_constant=1.41,
    persist_table=true
)

# Problem configurations
problems = [
    (name="Cubic", dim=3, degree=1),
    (name="Quartic", dim=4, degree=1),
    # ...
]

# Experiment parameters
num_trials = 50
max_steps = 100
```

**Customize for your experiments.**

---

## Testing Utilities (`test_util.jl`)

Helper functions for testing and debugging experiments.

**Example functions:**
- Setup test problems
- Validate configurations
- Generate synthetic data
- Run quick sanity checks

**Usage:**
```julia
include("experiments/paper/test_util.jl")

# Run quick test
test_problem = setup_test_polynomial()
test_optimizer_on_problem(test_problem, mcts_config)
```

---

## Example Workflow

### 1. Define Problem

```julia
using Oscar
include("src/NAML.jl")
using .NAML
include("experiments/paper/util.jl")

# Setup
prec = 20
p = 2
K = PadicField(p, prec)

# Generate random polynomial coefficients
a = generate_random_padic(p, prec, -2, 5)
b = generate_random_padic(p, prec, -2, 5)
c = generate_random_padic(p, prec, -2, 5)

# Create model
R, (x, α, β, γ) = polynomial_ring(K, ["x", "a", "b", "c"])
f = AbsolutePolynomialSum([(x-α)*(x-β)*(x-γ)])
model = AbstractModel(f, [true, false, false, false])
```

### 2. Generate Training Data

```julia
# True parameters (to be learned)
true_params = [a, b, c]

# Generate input points
train_x = [generate_random_padic(p, prec, 0, 3) for _ in 1:20]

# Evaluate model at true parameters
train_data = [(
    [x],
    eval_abs(model, [x, true_params...])
) for x in train_x]
```

### 3. Setup Loss

```julia
loss = MSE_loss_init(model, train_data)
```

### 4. Compare Optimizers

```julia
# Initial parameter guess
init_param = ValuationPolydisc([K(0), K(0), K(0)], [5, 5, 5])

# Configure optimizers
configs = Dict(
    "Greedy" => (greedy_descent_init, (false, 1)),
    "MCTS" => (mcts_descent_init, MCTSConfig(num_simulations=100)),
    "DAG-MCTS" => (dag_mcts_descent_init, DAGMCTSConfig(num_simulations=100))
)

# Run experiments
results = Dict()
for (name, (init_fn, config)) in configs
    println("Running $name...")
    optim = init_fn(init_param, loss, 1, config)

    losses = Float64[]
    for i in 1:100
        step!(optim)
        push!(losses, eval_loss(optim))
    end

    results[name] = losses
end
```

### 5. Analyze Results

```julia
using Plots

# Plot convergence
plot(xlabel="Step", ylabel="Loss", title="Optimizer Comparison")
for (name, losses) in results
    plot!(losses, label=name, linewidth=2)
end
savefig("convergence_comparison.png")

# Print final losses
for (name, losses) in results
    println("$name final loss: $(losses[end])")
end
```

---

## Benchmarking Best Practices

### 1. Use Multiple Seeds
```julia
seeds = 1:50
results = []
for seed in seeds
    Random.seed!(seed)
    # Run experiment
    push!(results, ...)
end
```

### 2. Track Multiple Metrics
```julia
metrics = Dict(
    :loss => [],
    :wall_time => [],
    :tree_size => [],  # For tree search
    :memory => []
)
```

### 3. Save Results
```julia
using JSON

results_dict = Dict(
    "config" => config,
    "losses" => losses,
    "timestamp" => now()
)

open("results/experiment_$(timestamp).json", "w") do f
    JSON.print(f, results_dict, 4)
end
```

### 4. Use BenchmarkTools for Timing
```julia
using BenchmarkTools

@btime step!(optim)  # Micro-benchmark
```

---

## Common Experiment Types

### 1. Convergence Comparison
Compare optimizer convergence rates on same problem.

### 2. Scaling Analysis
Test how algorithms scale with problem dimension.

### 3. Robustness Testing
Vary initialization, noise levels, data size.

### 4. Hyperparameter Sensitivity
Grid search over configuration parameters.

### 5. Memory Profiling
Track memory usage during optimization (especially for tree search).

---

## Generating Paper Figures

**Example script:**
```julia
# Load results
results = load_experiment_results("results/")

# Create publication-quality plots
using Plots
using LaTeXStrings

plot(
    results[:steps],
    results[:losses],
    xlabel=L"\text{Optimization Step}",
    ylabel=L"\text{Loss}",
    label="DAG-MCTS",
    linewidth=2,
    legend=:topright,
    size=(600, 400),
    dpi=300
)

savefig("paper_figure_1.pdf")
```

---

## Notes

- **Reproducibility**: Always set random seed
- **Validation**: Use separate test set for final evaluation
- **Logging**: Print progress and save intermediate results
- **Configuration**: Document all hyperparameters used
- **Data**: Save problem setups for reproducibility

---

## Future Additions

This infrastructure will likely include:
- Automated hyperparameter tuning
- Distributed experiment running
- Result aggregation and statistical analysis
- Visualization templates
- Experiment management system

Check the files directly for the latest implementation.

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Main library documentation
- [OPTIMIZERS.md](OPTIMIZERS.md) - Optimizer details
- `logs_local/` - Additional experiment scripts (research-grade, not documented)
