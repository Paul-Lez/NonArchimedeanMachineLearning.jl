# Paper Experiments - Utility API and Clean Implementations

This directory contains clean implementations and utilities for polynomial function learning experiments using the NAML library.

## Files

### Core Utilities

- **`util.jl`**: Clean utility API for experiments
  - `generate_random_padic()`: Generate random p-adic numbers with configurable exponent range
  - `polynomial_to_linear_loss()`: Transform polynomial learning problems into linear optimization
  - `generate_gauss_point()`: Create Gauss points with generic dimensions/types
  - `generate_polynomial_learning_data()`: Generate training data with **guaranteed distinct x values**
  - Helper functions for data transformation and loss creation

### Experiments

- **`function_learning_clean.jl`**: Standalone script for polynomial function learning
  - Clean reimplementation of `function_learning.ipynb`
  - Well-documented with progress reporting
  - Demonstrates learning polynomials with binary outputs

- **`function_learning_clean.ipynb`**: Interactive Jupyter notebook version
  - Same functionality as the script
  - Includes markdown documentation
  - Easy to experiment with different parameters

### Testing

- **`test_util.jl`**: Comprehensive test suite for all utilities
  - Tests random p-adic generation (including negative exponents)
  - Verifies uniqueness of generated data points
  - Tests both p-adic and real output loss functions
  - Includes end-to-end optimization example

## Quick Start

### Run Tests
```bash
cd experiments/paper
julia test_util.jl
```

### Run Function Learning Experiment
```bash
julia function_learning_clean.jl
```

### Interactive Exploration
```bash
jupyter notebook function_learning_clean.ipynb
```

## Key Features

### 1. Guaranteed Distinct Data Points

The `generate_polynomial_learning_data()` function now ensures all x values are distinct:

```julia
data = generate_polynomial_learning_data(2, 20, 10)
# Generates 10 data points with guaranteed distinct x values
```

**How it works:**
- Generates random p-adic numbers
- Checks for duplicates before adding to dataset
- Continues until requested number of distinct points is reached
- Raises an error if too many collisions occur (suggests adjusting parameters)

### 2. Full Precision P-adic Numbers

Fixed precision issue in `generate_random_padic()`:
- **Before**: Used `O(K, p)` which limited precision to O(p^1)
- **After**: Uses `K(p)` which preserves full precision O(p^prec)

This means generated numbers now have full precision (e.g., O(2^20) instead of O(2^1)), vastly increasing the space of possible distinct values.

### 3. Polynomial-to-Linear Transformation

The `polynomial_to_linear_loss()` function automatically handles two cases:

**Case 1: P-adic outputs**
```julia
data = [(K(1), K(2)), (K(2), K(5))]  # y is p-adic
loss = polynomial_to_linear_loss(data, degree, nothing)
```
Uses `LinearAbsolutePolynomialSum` directly for loss: |a₀ + a₁x + ... + aₙx^n - y|²

**Case 2: Real outputs with cutoff**
```julia
data = [(K(1), 0.0), (K(2), 1.0)]  # y is real
loss = polynomial_to_linear_loss(data, degree, 0.25)
```
Uses cutoff composition for loss: |cutoff(|a₀ + a₁x + ... + aₙx^n|) - y|²

## Example Usage

```julia
include("../../src/NAML.jl")
include("util.jl")

using Oscar
using .NAML

# Setup
prec = 20
p = 2
K = PadicField(p, prec)

# Generate distinct training data
data = generate_polynomial_learning_data(p, prec, 10)

# Create loss function
loss = polynomial_to_linear_loss(data, degree=6, cutoff_val=0.25)

# Initialize at Gauss point
param = generate_gauss_point(7, K)

# Optimize
optim = NAML.greedy_descent_init(param, loss, 1, (true, 0))
for i in 1:20
    NAML.step!(optim)
    println("Epoch $i: loss = $(NAML.eval_loss(optim))")
end
```

## Parameter Recommendations

### For `generate_random_padic(p, prec, min_exp, num_terms)`

- **More distinct values**: Increase `num_terms` (space size ≈ p^num_terms)
- **Smaller numbers**: Use `min_exp=0` (default)
- **Fractional p-adic numbers**: Use `min_exp<0` (e.g., -2)

Typical values:
- `num_terms=10`: Up to ~1000 distinct values for p=2
- `num_terms=15`: Up to ~32000 distinct values for p=2

### For `generate_polynomial_learning_data(p, prec, n_points, min_exp, num_terms)`

Default parameters work well for most experiments:
- `min_exp=0`: Start from integers (not fractions)
- `num_terms=10`: Enough for dozens of distinct points

For larger datasets:
```julia
# Generate 100 distinct points
data = generate_polynomial_learning_data(2, 20, 100, 0, 15)
```

## Improvements Over Original Notebook

1. **Modular design**: Utilities separated from experiments
2. **Guaranteed uniqueness**: No duplicate x values in training data
3. **Full precision**: P-adic numbers use complete precision
4. **Better documentation**: Clear docstrings and examples
5. **Automatic type handling**: Handles p-adic and real outputs seamlessly
6. **Progress reporting**: Clear output during optimization
7. **Comprehensive tests**: Full test coverage for all utilities

## Troubleshooting

### "Could not generate N distinct p-adic numbers"

**Solution**: Increase `num_terms` parameter
```julia
# Instead of:
data = generate_polynomial_learning_data(2, 20, 100, 0, 10)  # May fail

# Use:
data = generate_polynomial_learning_data(2, 20, 100, 0, 15)  # More distinct values possible
```

### Low precision in generated numbers

**Issue**: Getting O(p^1) instead of O(p^20)

**Cause**: Using `O(K, p)` instead of `K(p)` - this has been fixed in `util.jl`

### Loss not improving

This is expected behavior! The optimization is exploring the parameter space, not necessarily improving the loss at every step. Greedy descent progressively refines each coordinate to find regions with better loss values.
