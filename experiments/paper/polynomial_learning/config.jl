"""
Polynomial Learning Experiment Configuration

Define experiment configurations for polynomial interpolation in p-adic fields.
For each configuration, we learn polynomial coefficients (a₀, ..., aₙ) such that
    a₀ + a₁x + ... + aₙxⁿ ≈ y
for given (x, y) training data.

Each config should have:
- name: Descriptive name for the experiment
- prime: The prime p for the p-adic field
- prec: The p-adic precision
- degree: Polynomial degree to fit
- n_points: Number of training data points
- num_samples: Number of random problem instances to average over
"""

# ============================================================================
# SMALL EXPERIMENTS (fast, for testing)
# ============================================================================
small_experiments = [
    Dict("name" => "2adic_deg2_3pts", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 3, "num_samples" => 3),
    Dict("name" => "2adic_deg3_4pts", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 3),
]

# ============================================================================
# DEGREE SWEEP (vary polynomial degree, 2-adic)
# ============================================================================
degree_sweep = [
    Dict("name" => "2adic_deg2_3pts", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),
    Dict("name" => "2adic_deg3_4pts", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "2adic_deg4_5pts", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 5, "num_samples" => 5),
    Dict("name" => "2adic_deg5_6pts", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 6, "num_samples" => 5),
]

# ============================================================================
# PRIME SWEEP (vary base field, fixed degree 3)
# ============================================================================
prime_sweep = [
    Dict("name" => "2adic_deg3_4pts", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "3adic_deg3_4pts", "prime" => 3, "prec" => 15,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "5adic_deg3_4pts", "prime" => 5, "prec" => 12,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "7adic_deg3_4pts", "prime" => 7, "prec" => 10,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
]

# ============================================================================
# COMPREHENSIVE (degree x prime)
# ============================================================================
comprehensive = [
    # 2-adic
    Dict("name" => "2adic_deg2_3pts", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),
    Dict("name" => "2adic_deg3_4pts", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "2adic_deg4_5pts", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 5, "num_samples" => 5),
    Dict("name" => "2adic_deg5_6pts", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 6, "num_samples" => 5),

    # 3-adic
    Dict("name" => "3adic_deg2_3pts", "prime" => 3, "prec" => 15,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),
    Dict("name" => "3adic_deg3_4pts", "prime" => 3, "prec" => 15,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "3adic_deg4_5pts", "prime" => 3, "prec" => 15,
         "degree" => 4, "n_points" => 5, "num_samples" => 5),

    # 5-adic
    Dict("name" => "5adic_deg2_3pts", "prime" => 5, "prec" => 12,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),
    Dict("name" => "5adic_deg3_4pts", "prime" => 5, "prec" => 12,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
]

# ============================================================================
# SELECT WHICH SET TO USE
# ============================================================================

# Default: use small experiments for quick testing
experiment_configs = small_experiments

# Uncomment one of these to use a different experiment set:
# experiment_configs = degree_sweep
# experiment_configs = prime_sweep
# experiment_configs = comprehensive
