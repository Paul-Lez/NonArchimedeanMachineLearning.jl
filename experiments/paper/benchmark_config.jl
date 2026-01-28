"""
Benchmark Configuration File

Define experiment configurations here to easily run custom benchmarks.
Modify the `experiment_configs` array to add/remove experiments.

Each config should have:
- prime: The prime p for the p-adic field
- prec: The p-adic precision
- degree: Polynomial degree to fit
- n_points: Number of data points
- name: Descriptive name for the experiment
"""

# Example experiment sets you can uncomment and use:

# ============================================================================
# SMALL EXPERIMENTS (fast, for testing)
# ============================================================================
small_experiments = [
    Dict("prime" => 2, "prec" => 20, "degree" => 2, "n_points" => 3, "name" => "2adic_deg2"),
    Dict("prime" => 2, "prec" => 20, "degree" => 3, "n_points" => 4, "name" => "2adic_deg3"),
    Dict("prime" => 3, "prec" => 15, "degree" => 2, "n_points" => 3, "name" => "3adic_deg2"),
]

# ============================================================================
# DEGREE SWEEP (vary polynomial degree)
# ============================================================================
degree_sweep_2adic = [
    Dict("prime" => 2, "prec" => 20, "degree" => 2, "n_points" => 3, "name" => "2adic_deg2"),
    Dict("prime" => 2, "prec" => 20, "degree" => 3, "n_points" => 4, "name" => "2adic_deg3"),
    Dict("prime" => 2, "prec" => 20, "degree" => 4, "n_points" => 5, "name" => "2adic_deg4"),
    Dict("prime" => 2, "prec" => 20, "degree" => 5, "n_points" => 6, "name" => "2adic_deg5"),
    Dict("prime" => 2, "prec" => 20, "degree" => 6, "n_points" => 7, "name" => "2adic_deg6"),
]

# ============================================================================
# PRIME SWEEP (vary base field)
# ============================================================================
prime_sweep_deg3 = [
    Dict("prime" => 2, "prec" => 20, "degree" => 3, "n_points" => 4, "name" => "2adic_deg3"),
    Dict("prime" => 3, "prec" => 15, "degree" => 3, "n_points" => 4, "name" => "3adic_deg3"),
    Dict("prime" => 5, "prec" => 12, "degree" => 3, "n_points" => 4, "name" => "5adic_deg3"),
    Dict("prime" => 7, "prec" => 10, "degree" => 3, "n_points" => 4, "name" => "7adic_deg3"),
]

# ============================================================================
# LARGE EXPERIMENTS (more data points)
# ============================================================================
large_experiments = [
    Dict("prime" => 2, "prec" => 20, "degree" => 5, "n_points" => 10, "name" => "2adic_deg5_10pts"),
    Dict("prime" => 2, "prec" => 20, "degree" => 7, "n_points" => 15, "name" => "2adic_deg7_15pts"),
    Dict("prime" => 3, "prec" => 15, "degree" => 5, "n_points" => 10, "name" => "3adic_deg5_10pts"),
]

# ============================================================================
# COMPREHENSIVE COMPARISON (degree and prime variations)
# ============================================================================
comprehensive = [
    # 2-adic experiments
    Dict("prime" => 2, "prec" => 20, "degree" => 2, "n_points" => 3, "name" => "2adic_deg2"),
    Dict("prime" => 2, "prec" => 20, "degree" => 3, "n_points" => 4, "name" => "2adic_deg3"),
    Dict("prime" => 2, "prec" => 20, "degree" => 4, "n_points" => 5, "name" => "2adic_deg4"),
    Dict("prime" => 2, "prec" => 20, "degree" => 5, "n_points" => 6, "name" => "2adic_deg5"),

    # 3-adic experiments
    Dict("prime" => 3, "prec" => 15, "degree" => 2, "n_points" => 3, "name" => "3adic_deg2"),
    Dict("prime" => 3, "prec" => 15, "degree" => 3, "n_points" => 4, "name" => "3adic_deg3"),
    Dict("prime" => 3, "prec" => 15, "degree" => 4, "n_points" => 5, "name" => "3adic_deg4"),

    # 5-adic experiments
    Dict("prime" => 5, "prec" => 12, "degree" => 2, "n_points" => 3, "name" => "5adic_deg2"),
    Dict("prime" => 5, "prec" => 12, "degree" => 3, "n_points" => 4, "name" => "5adic_deg3"),
    Dict("prime" => 5, "prec" => 12, "degree" => 4, "n_points" => 5, "name" => "5adic_deg4"),
]

# ============================================================================
# SELECT WHICH SET TO USE
# ============================================================================

# Default: use small experiments for quick testing
experiment_configs = small_experiments

# Uncomment one of these to use a different experiment set:
# experiment_configs = degree_sweep_2adic
# experiment_configs = prime_sweep_deg3
# experiment_configs = large_experiments
# experiment_configs = comprehensive

# Or define your own custom experiments:
# experiment_configs = [
#     Dict("prime" => 2, "prec" => 20, "degree" => 10, "n_points" => 20, "name" => "custom_large"),
# ]
