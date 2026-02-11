"""
Function Learning Experiment Configuration

Define experiment configurations for learning target functions over p-adic fields.
For each configuration, we learn polynomial coefficients (a₀, ..., aₙ) such that
    f(x) = a₀ + a₁x + ... + aₙxⁿ ≈ target_function(x)
for random p-adic inputs x.

Two main tasks:
1. Zero Function: Learn f(x) = 0 for all x (trivial solution exists)
2. One Function: Learn f(x) = 1 for all x (requires constant polynomial)

Each config should have:
- name: Descriptive name for the experiment
- prime: The prime p for the p-adic field
- prec: The p-adic precision
- degree: Polynomial degree to fit
- n_points: Number of random test points
- target_fn: Target function ("zero" or "one" or custom)
- num_samples: Number of random problem instances to average over
- threshold: Threshold for cross-entropy loss (optional)
- scale: Scale parameter for cross-entropy loss (optional)
"""

# ============================================================================
# SMALL EXPERIMENTS (fast, for testing)
# ============================================================================
small_experiments = [
    Dict("name" => "zero_fn_deg3_3adic", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg3_3adic", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# ZERO FUNCTION EXPERIMENTS (varying degree)
# ============================================================================
zero_function_sweep = [
    Dict("name" => "zero_fn_deg2_2adic", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg4_2adic", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg5_2adic", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 7, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# ONE FUNCTION EXPERIMENTS (varying degree)
# ============================================================================
one_function_sweep = [
    Dict("name" => "one_fn_deg2_2adic", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg4_2adic", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg5_2adic", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 7, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# PRIME SWEEP (same function, different primes)
# ============================================================================
prime_sweep = [
    Dict("name" => "zero_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_3adic", "prime" => 3, "prec" => 15,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_5adic", "prime" => 5, "prec" => 12,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_7adic", "prime" => 7, "prec" => 10,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# COMPREHENSIVE (both functions, multiple settings)
# ============================================================================
comprehensive = [
    # Zero function - 2-adic
    Dict("name" => "zero_fn_deg2_2adic", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg4_2adic", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 2-adic
    Dict("name" => "one_fn_deg2_2adic", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg4_2adic", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Zero function - 3-adic
    Dict("name" => "zero_fn_deg3_3adic", "prime" => 3, "prec" => 15,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 3-adic
    Dict("name" => "one_fn_deg3_3adic", "prime" => 3, "prec" => 15,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# SELECT WHICH SET TO USE
# ============================================================================

# Default: use small experiments for quick testing
experiment_configs = small_experiments

# Uncomment one of these to use a different experiment set:
# experiment_configs = zero_function_sweep
# experiment_configs = one_function_sweep
# experiment_configs = prime_sweep
# experiment_configs = comprehensive
