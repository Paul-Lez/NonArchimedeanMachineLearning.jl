# Polynomial Learning Experiments

Learn polynomial coefficients over p-adic fields, comparing multiple optimization methods.

## Quick Start

```bash
# Quick test (few epochs, saves JSON)
julia --project=. experiments/paper/polynomial_learning/run_experiments.jl --quick --save

# Full run with config file
julia --project=. experiments/paper/polynomial_learning/run_experiments.jl --config --save

# Generate LaTeX tables from results
julia --project=. experiments/paper/polynomial_learning/generate_tables.jl <results.json> --stdout
```

## Files

| File | Description |
|------|-------------|
| `run_experiments.jl` | Main launcher script. Outputs JSON with per-sample and aggregate results. |
| `config.jl` | Experiment configurations (degree sweep, prime sweep, comprehensive). |
| `generate_tables.jl` | Reads JSON results and produces LaTeX tables. |
| `polynomial_learning.ipynb` | Interactive Jupyter notebook with the same experiment. |

## Optimizers Compared

- **Greedy** — Greedy tree descent (baseline)
- **MCTS-50** — Monte Carlo Tree Search (50 simulations/step)
- **MCTS-100** — Monte Carlo Tree Search (100 simulations/step)
- **DAG-MCTS-100** — MCTS with transposition tables (100 simulations/step)
- **UCT** — Upper Confidence Trees (depth 10, 100 simulations/step)
- **HOO** — Hierarchical Optimistic Optimization (rho=0.5, nu1=0.1)

## JSON Output Structure

```json
{
  "metadata": { "timestamp", "n_epochs", "quick_mode", "optimizer_order" },
  "experiments": [
    {
      "config": { "name", "prime", "prec", "degree", "n_points", "num_samples" },
      "samples": [
        {
          "sample_num": 1,
          "initial_loss": 1.0,
          "optimizers": {
            "Greedy": { "final_loss", "losses", "time", "improvement", "hyperparameters" }
          }
        }
      ],
      "aggregate": {
        "Greedy": { "mean_final_loss", "std_final_loss", "mean_time", ... }
      }
    }
  ]
}
```

## LaTeX Tables Generated

1. **Summary** — Mean final loss per optimizer, best bolded
2. **Detailed** — Final loss, improvement %, and time for each experiment
3. **Degree sweep** — Performance vs polynomial degree (same prime)
4. **Prime sweep** — Performance vs base prime (same degree)
5. **Timing** — Wall-clock time comparison
