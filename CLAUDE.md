# NAML - Non-Archimedean Machine Learning

A Julia library for machine learning over p-adic numbers and polydisc spaces.

## Quick Start

### Running Scripts
```bash
# Run any Julia script with the project
julia --project=. path/to/script.jl

# Run tests
julia --project=. test/runtests.jl

# Run paper experiments
julia --project=. experiments/paper/run_benchmarks.jl
```

### Basic Usage
```julia
# Load the module
include("src/NAML.jl")
using .NAML

# Set up p-adic field
prec = 20
p = 2
K = PadicField(p, prec)

# Create a polydisc
param = ValuationPolydisc([K(1), K(2)], [0, 0])

# Define a polynomial function
R, (x, a, b) = polynomial_ring(K, ["x", "a", "b"])
f = AbsolutePolynomialSum([(x-a)*(x-b)])

# Create a model (x is data, a,b are parameters)
model = AbstractModel(f, [true, false, false])

# Create loss and optimize
loss = MSE_loss_init(model, data)
optim = greedy_descent_init(param, loss, 1, (false, 1))

# Run optimization
for i in 1:100
    step!(optim)
    println("Step $i: Loss = ", eval_loss(optim))
end
```

## Repository Structure

```
naml-experiments/
├── src/
│   ├── NAML.jl                    # Main module with exports
│   ├── basic/                     # Core mathematical structures
│   │   ├── valuation.jl           # Valuation and absolute value
│   │   ├── polydisc.jl            # Polydisc definitions
│   │   ├── tangent_vector.jl      # Tangent vectors
│   │   └── functions.jl           # Function algebra
│   ├── optimization/
│   │   ├── optim_setup.jl         # Core optimization API
│   │   ├── loss.jl                # Loss functions
│   │   ├── model.jl               # Model structures
│   │   └── optimizers/            # Optimization algorithms
│   │       ├── greedy_descent.jl
│   │       ├── gradient_descent.jl
│   │       └── tree_search/       # MCTS, UCT, HOO, DAG-MCTS, etc.
│   ├── statistics/
│   │   ├── frechet.jl             # Frechet mean
│   │   └── least_squares.jl       # Linear regression
│   └── visualization/
│       └── loss_landscape.jl      # Loss landscape visualization
├── test/                          # Test suite
├── experiments/                   # Interactive demos and benchmarks
│   ├── paper/                     # Paper experiment infrastructure
│   └── *.ipynb                    # Jupyter notebooks
└── archive/                       # Legacy code
```

## Core Concepts

### Mathematical Foundation
- **p-adic fields**: Non-Archimedean valued fields (e.g., 2-adic, 3-adic numbers)
- **Polydiscs**: Multi-dimensional balls in p-adic space (product of discs)
- **Valuation**: Distance measurement using `v(x)` instead of `|x|`
- **Tree structure**: p-adic space has natural tree geometry

### Key Design Decisions

1. **Polydisc Implementation**: Uses `NTuple{N,S}` for performance (changed from `Vector{S}` in PR #27)
2. **Valuation vs Absolute Value**: Radius measured via valuation, not norm
3. **Compositional Functions**: Build complex functions from simple pieces (lazy evaluation)
4. **Mutable OptimSetup**: Allows in-place parameter updates
5. **Closure-based Loss**: Data captured in closures, not stored in structs
6. **DAG Structure**: Polydisc space forms a DAG; multiple paths to same state (addressed by DAG-MCTS)

---

## Part 1: Basic Structures (`src/basic/`)

### 1.1 Valuation (`valuation.jl`)

Generic implementations for p-adic valuation and absolute value.

**Functions:**
```julia
valuation(a::padic) -> Int
```
For `a = p^v · u` where `u` is a p-adic unit, returns `v`.

```julia
Base.abs(a::padic) -> Float64
```
Returns `p^(-v(a))` where `v(a)` is the valuation.

**Design**: Generic functions extensible to custom types via multiple dispatch.

### 1.2 Polydisc (`polydisc.jl`)

Core structure for p-adic balls.

**Types:**
```julia
ValuationPolydisc{S,T,N}
    center::NTuple{N,S}  # Center coordinates
    radius::NTuple{N,T}  # Radius (measured via valuation)

AbsPolydisc{S,T}         # Norm-based radius (rarely used)
```

**Construction:**
```julia
# Using tuples (preferred for performance)
p = ValuationPolydisc{padic,Int,2}((K(1), K(2)), (0, 0))

# Using vectors (convenience constructor converts to tuples)
p = ValuationPolydisc([K(1), K(2)], [0, 0])
```

**Key Functions:**
```julia
center(p)                    # Get center tuple
radius(p)                    # Get radius tuple
dim(p)                       # Dimension N
prime(p)                     # Prime p of the field

children(p, degree)          # Generate p^degree children
children_along_branch(p, i)  # Children along branch i

NAML.join(p1, p2)           # Intersection (not exported, conflicts with Base.join)
dist(p1, p2)                # Distance between polydiscs
concatenate(p1, p2)         # Combine into higher dimension

canonical_center(p)          # Canonical representative for hashing
```

**Equality:**
Implements Berkovich equality: `p1 == p2` if they have the same radius and centers differ by elements with valuation ≥ radius.

**Hashing:**
Supports `hash(p)` and can be used as Dict keys (uses canonical center).

**Tree Structure:**
Each polydisc has `p^degree` children. The `children` function generates child polydiscs by shrinking radii in different coordinate directions.

### 1.3 Tangent Vectors (`tangent_vector.jl`)

Represent directions in polydisc space.

**Type:**
```julia
ValuationTangent{S,T,N}
    point::ValuationPolydisc{S,T,N}  # Base point
    direction::NTuple{N,S}           # Direction vector
    magnitude::NTuple{N,T}           # Magnitude per coordinate
```

**Key Functions:**
```julia
NAML.zero(P::ValuationPolydisc, Q::ValuationPolydisc)  # Zero tangent at P toward Q
NAML.basis_vector(P, Q, i)                              # i-th basis vector
Base.:+(v1, v2)                                         # Vector addition
```

**Note**: `zero` and `basis_vector` not exported (conflict with Base).

### 1.4 Functions (`functions.jl`)

Rich compositional algebra for building functions on polydisc spaces.

**Abstract Base:**
```julia
PolydiscFunction{S}
```
All functions inherit from this. Supports:
- Evaluation on polydiscs
- Directional derivatives
- Automatic differentiation

**Concrete Types:**

**Polynomials:**
```julia
AbsolutePolynomialSum{S}
    polys::Vector{AbstractAlgebra.Generic.MPoly{S}}

LinearPolynomial{S}
    coefficients::Vector{S}   # [a₁, ..., aₙ]
    constant::S               # b
    # Represents: a₁T₁ + ... + aₙTₙ + b

LinearAbsolutePolynomialSum{S}
    polys::Vector{LinearPolynomial{S}}
```

**Rational Functions:**
```julia
LinearRationalFunction{S}
    num::LinearPolynomial{S}
    den::LinearPolynomial{S}

LinearRationalFunctionSum{S}
    rats::Vector{LinearRationalFunction{S}}
```

**Compositional Operators:**
```julia
Add{S}        # f + g
Sub{S}        # f - g
Mul{S}        # f * g
Div{S}        # f / g
SMul{S}       # scalar * f
Comp{S}       # compose with real function
Constant{S}   # constant function
Lambda{S}     # custom function wrapper
```

**Operator Overloading:**
```julia
f + g    # Add{S}(f, g)
f - g    # Sub{S}(f, g)
f * g    # Mul{S}(f, g)
f / g    # Div{S}(f, g)
a * f    # SMul{S}(a, f)  where a is Number
f^n      # Repeated multiplication
-f       # SMul{S}(-1, f)
```

**Core Operations:**

```julia
NAML.evaluate(f::PolydiscFunction, p::ValuationPolydisc)
```
Evaluate function at polydisc. Returns interval/polydisc representing range.
**Note**: Not exported (conflicts with Oscar).

```julia
evaluate_abs(f, p)
```
Evaluate absolute value of function at polydisc.

```julia
eval_abs(f, val::Vector)
```
Evaluate at a point (returns scalar).

```julia
batch_evaluate_init(f, data)
```
Create efficient batch evaluator. Returns closure for repeated evaluation on same dataset.

**Calculus:**

```julia
directional_derivative(f::PolydiscFunction, v::ValuationTangent)
```
Compute derivative in direction `v`. Supports all compositional types via chain rule.

```julia
directional_exponent(f, v)
```
Find exponent vectors where maximum is attained (used in derivative computation).

```julia
grad(f, v)
```
Compute full gradient vector.

**Example:**
```julia
# Define variables
R, (x, y) = polynomial_ring(K, ["x", "y"])

# Create polynomial
poly = AbsolutePolynomialSum([x^2 + y^2])

# Compose with operators
f = poly^2 + 3 * poly  # Creates: Add(Mul(poly, poly), SMul(3, poly))

# Evaluate at polydisc
p = ValuationPolydisc([K(0), K(0)], [0, 0])
result = NAML.evaluate(f, p)

# Compute derivative
v = NAML.zero(p, p)
df = directional_derivative(f, v)
```

---

## Part 2: Optimization (`src/optimization/`)

### 2.1 Core API (`optim_setup.jl`)

**Loss Structure:**
```julia
Loss
    eval::Function   # (params::Vector{ValuationPolydisc}) -> Vector{value}
    grad::Function   # (tangents::Vector{ValuationTangent}) -> Vector{value}
```

Encapsulates evaluation and gradient as closures that capture data.

**OptimSetup (mutable):**
```julia
OptimSetup{S,T,U,V}
    loss::Loss                      # Loss function
    param::ValuationPolydisc{S,T}   # Current parameters
    optimiser::Function             # Optimization algorithm
    state::U                        # Algorithm state
    context::V                      # Hyperparameters/settings
```

**API:**
```julia
eval_loss(optim)               # Evaluate current loss
step!(optim)                   # Take one optimization step
update_param!(optim, param)    # Update parameters
update_state!(optim, state)    # Update optimizer state
```

**Design**: Mutable struct allows efficient in-place updates during optimization.

### 2.2 Loss Functions (`loss.jl`)

```julia
MSE_loss_init(model::Model, data) -> Loss
```
Mean Squared Error: `(1/n) Σ (ŷᵢ - yᵢ)²`

```julia
MPE_loss_init(model::Model, data, p::Int) -> Loss
```
Mean p-Power Error: `(1/n) Σ |ŷᵢ - yᵢ|^p`

**Data Format:**
```julia
data::Vector{Tuple{Vector{S}, S}}  # [(x₁, y₁), ..., (xₙ, yₙ)]
```

### 2.3 Models (`model.jl`)

**AbstractModel:**
```julia
AbstractModel{S}
    fun::PolydiscFunction{S}     # The function (e.g., polynomial)
    param_info::Vector{Bool}     # true = data variable, false = parameter
```

Separates function structure from parameter values.

**Model (mutable):**
```julia
Model{S,T}
    fun::AbstractModel{S}
    param::ValuationPolydisc{S,T}
```

Concrete model with parameter values.

**Functions:**
```julia
var_indices(m)                           # Indices of data variables
param_indices(m)                         # Indices of parameters
set_abstract_model_variable(m, x, θ)    # Interleave data x and parameters θ
eval_abs(m, x)                          # Evaluate at input x
batch_evaluate_init(m, data)            # Batch evaluator
```

**Example:**
```julia
R, (x, a, b, c) = polynomial_ring(K, ["x", "a", "b", "c"])
f = AbsolutePolynomialSum([(x-a)*(x-b)*(x-c)])

# x is data (index 1), a,b,c are parameters (indices 2,3,4)
model = AbstractModel(f, [true, false, false, false])

# Add parameter values
param = ValuationPolydisc([K(1), K(2), K(3)], [0, 0, 0])
concrete_model = Model(model, param)
```

---

## Part 3: Optimizers

The library includes multiple optimization algorithms. See **[OPTIMIZERS.md](OPTIMIZERS.md)** for detailed documentation.

**Classical Optimizers:**
- `greedy_descent` - Tree-based greedy optimization
- `gradient_descent` - Gradient-based discrete descent

**Tree Search Optimizers:**
- `mcts` - Monte Carlo Tree Search
- `uct` - Upper Confidence Trees
- `dag_mcts` - MCTS with transposition tables for DAG structure (NEW)
- `hoo` - Hierarchical Optimistic Optimization
- `modified_uct` - UCT variant
- `flat_ucb` - Flat UCB variant

**Basic Usage Pattern:**
```julia
# Initialize optimizer
config = MCTSConfig(num_simulations=100, exploration_constant=1.41)
optim = mcts_descent_init(param, loss, 1, config)

# Or for greedy descent
optim = greedy_descent_init(param, loss, 1, (false, 1))

# Optimize
for i in 1:100
    step!(optim)
end
```

---

## Part 4: Statistics & Visualization

### 4.1 Statistics (`src/statistics/`)

**Frechet Mean (`frechet.jl`):**
```julia
frechet_mean(X::Vector{Vector{PadicFieldElem}})
```
Coordinate-wise median (ℓ¹-metric).

```julia
frechet_mean(X::Vector{ValuationPolydisc}, prec::Int)
```
Polydisc Frechet mean using greedy descent.

**Least Squares (`least_squares.jl`):**
```julia
make_ordinary_least_squares_loss(data) -> Loss
```
Create OLS loss for linear regression. Data format: `Vector{Tuple{Vector{S}, Vector{T}}}`.

Parameter ordering: `[A₁₁, A₁₂, ..., A₁ₙ, A₂₁, ..., Aₘₙ, b₁, ..., bₘ]`

```julia
solve_linear_system(A, b, y) -> Loss
```
Loss for solving `Ax + b = y`.

### 4.2 Visualization (`src/visualization/`)

**Loss Landscape (`loss_landscape.jl`):**

```julia
ConvexHullTree{S,T,N}
```
Tree structure representing convex hull of polydiscs with containment edges.

```julia
convex_hull(discs::Vector{ValuationPolydisc}) -> ConvexHullTree
```
Compute convex hull tree of polydisc set.

```julia
sample_loss_landscape(tree, loss_fn, num_samples) -> Dict
```
Sample loss values along geodesics between tree nodes.

```julia
plot_tree_with_loss(tree, landscape; colormap=:viridis)
plot_tree_simple(tree)
```
Visualize tree structure with/without loss coloring.

**Usage Example:**
```julia
using Plots

# Define polydiscs
d1 = ValuationPolydisc([K(0)], [5])
d2 = ValuationPolydisc([K(8)], [5])
d3 = ValuationPolydisc([K(16)], [5])

# Build and visualize
tree = convex_hull([d1, d2, d3])
landscape = sample_loss_landscape(tree, loss_fn, 20)
plot_tree_with_loss(tree, landscape)
```

---

## Part 5: Testing & Experiments

### 5.1 Tests (`test/`)

Run all tests:
```bash
julia --project=. test/runtests.jl
```

**Test Files:**
- `polydisc.jl` - Basic polydisc operations, equality, hashing
- `tangent_vector.jl` - Tangent vector operations
- `functions.jl` - Basic polynomial evaluation
- `test_functions.jl` - Expanded function algebra tests
- `polynomial_learning.jl` - Full optimization pipeline
- `gradient_descent.jl` - Gradient descent tests
- `test_all_optimizers.jl` - Comprehensive optimizer tests
- `frechet.jl` - Frechet mean tests
- `least_squares.jl` - Linear regression tests
- `dag_mcts.jl` - DAG-MCTS tests
- `loss_landscape.jl` - Visualization tests
- `geodesic.jl` - Geodesic computation tests
- `convex_hull.jl` - Convex hull tests

### 5.2 Experiments

**Interactive Demos (`experiments/`):**
- `cubic_learning_experiment.ipynb` - Learn cubic roots
- `polynomial_learning.ipynb` - Arbitrary degree polynomials
- `linear_learning_experiment.ipynb` - Linear function learning
- `tree_colouring_learning.ipynb` - Tree coloring task

**Visualization Examples:**
- `loss_landscape_plotting_example.jl`
- `tree_visualization_demo.jl`
- `clean_tree_example.jl`
- `cubic_roots_example.jl`
- `six_leaf_example.jl`

**Paper Experiments (`experiments/paper/`):**
Structured benchmarking infrastructure for research. See **[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md)** for details.

Quick overview:
- `run_benchmarks.jl` - Main benchmarking script
- `benchmark_config.jl` - Configuration
- `util.jl` - Utilities (random p-adic generation, cutoff functions, etc.)
- `test_util.jl` - Testing utilities

---

## Part 6: Dependencies & Environment

**Project.toml Dependencies:**
- `Oscar` - Comprehensive algebra/geometry library (includes Nemo for p-adics)
- `AbstractAlgebra` - Polynomial rings and operations
- `LinearAlgebra` - Vector operations
- `Nemo` - Number theory (p-adic implementation)
- `Printf` - Formatted output
- `Test` - Testing framework
- `BenchmarkTools` - Performance benchmarking

**Module Structure:**
```julia
module NAML
    using Oscar
    using LinearAlgebra
    using Printf

    # Includes and exports...
end
```

Not a registered package - use `include("src/NAML.jl"); using .NAML`.

---

## Part 7: Key Exported API

**Types:**
```julia
# Basic
ValuationPolydisc, AbsPolydisc
ValuationTangent
PolydiscFunction, AbsolutePolynomialSum, LinearPolynomial
LinearAbsolutePolynomialSum, LinearRationalFunction, LinearRationalFunctionSum

# Optimization
Loss, OptimSetup
AbstractModel, Model

# Tree Search
MCTSNode, MCTSConfig, MCTSState, SelectionMode, VisitCount, BestValue
UCTNode, UCTConfig, UCTState
DAGMCTSNode, DAGMCTSConfig, DAGMCTSState
HOONode, HOOConfig, HOOState
ModifiedUCTNode, ModifiedUCTConfig, ModifiedUCTState
FlatUCBNode, FlatUCBConfig, FlatUCBState

# Visualization
ConvexHullTree
```

**Functions:**
```julia
# Basic operations
valuation, center, radius, dim, prime
dist, children, children_along_branch, concatenate, canonical_center
directional_derivative, directional_exponent, grad, eval_abs

# Optimization
eval_loss, update_param!, step!
MSE_loss_init, MPE_loss_init
greedy_descent, greedy_descent_init
gradient_descent, gradient_descent_init
gradient_param
var_indices, param_indices, set_abstract_model_variable, batch_evaluate_init

# Tree search optimizers (see OPTIMIZERS.md)
mcts_descent, mcts_descent_init
uct_descent, uct_descent_init
dag_mcts_descent, dag_mcts_descent_init
hoo_descent, hoo_descent_init
modified_uct_descent, modified_uct_descent_init
flat_ucb_descent, flat_ucb_descent_init

# DAG-MCTS utilities
get_dag_stats, print_dag_stats, verify_transposition_table

# HOO utilities
get_tree_size, get_visited_nodes, get_leaf_nodes

# Statistics & Visualization
frechet_mean
make_ordinary_least_squares_loss, solve_linear_system
convex_hull, sample_loss_landscape
plot_tree_with_loss, plot_tree_simple
print_landscape_summary, plot_loss_landscape, export_landscape_csv
```

**Non-exported (use NAML.function_name):**
```julia
NAML.join          # Conflicts with Base.join
NAML.zero          # Conflicts with Base.zero
NAML.basis_vector  # Avoid Base conflict
NAML.evaluate      # Conflicts with Oscar
```

---

## Part 8: Common Patterns

### Creating Training Data
```julia
# Generate data
K = PadicField(2, 20)
data = [
    ([K(1)], K(2)),
    ([K(2)], K(4)),
    ([K(3)], K(6))
]  # [(x₁, y₁), ...]
```

### Defining Models
```julia
# 1. Create polynomial ring
R, (x, a, b) = polynomial_ring(K, ["x", "a", "b"])

# 2. Define function
f = AbsolutePolynomialSum([x^2 - a*x - b])

# 3. Create abstract model (x is data, a,b are params)
model = AbstractModel(f, [true, false, false])
```

### Running Optimization
```julia
# 1. Create loss
loss = MSE_loss_init(model, data)

# 2. Initialize parameters
param = ValuationPolydisc([K(0), K(0)], [5, 5])

# 3. Choose optimizer
config = DAGMCTSConfig(num_simulations=100)
optim = dag_mcts_descent_init(param, loss, 1, config)

# 4. Optimize
for i in 1:50
    step!(optim)
    if i % 10 == 0
        println("Step $i: Loss = ", eval_loss(optim))
    end
end

# 5. Get result
final_param = optim.param
println("Final center: ", center(final_param))
```

### Using DAG-MCTS Statistics
```julia
# After optimization, inspect the search graph
stats = get_dag_stats(optim.state)
println("Total nodes: $(stats[:total_nodes])")
println("Total edges: $(stats[:total_edges])")
println("Reuse count: $(stats[:reuse_count])")

# Verify transposition table consistency
verify_transposition_table(optim.state)
```

### Batch Evaluation
```julia
# For repeated evaluation on same data
batch_eval = batch_evaluate_init(model, data)

# Evaluate on multiple parameter sets efficiently
params = [param1, param2, param3]
losses = batch_eval(params)
```

---

## Part 9: Implementation Notes

### Polydisc Hashing
Polydiscs support hashing and can be used as Dict keys. The hash is computed using the canonical center (via `canonical_center(p)`), ensuring that Berkovich-equal polydiscs hash to the same value.

```julia
# Works correctly
table = Dict{ValuationPolydisc{padic,Int,2}, Int}()
table[p1] = 1
table[p2] = 2  # If p1 == p2 (Berkovich), this overwrites
```

### DAG Structure
The polydisc tree is actually a DAG: multiple sequences of coordinate refinements can lead to the same polydisc. DAG-MCTS exploits this via transposition tables, significantly reducing memory and computation.

### Performance Tips
1. Use `NTuple` directly when creating many polydiscs
2. Use `batch_evaluate_init` for repeated evaluations
3. Set `persist_table=true` in DAG-MCTS to reuse search across steps
4. Prefer `dag_mcts` over `mcts` for high-dimensional problems

### Precision Management
P-adic precision is fixed at field creation. Higher precision = more accuracy but slower computation. Typical range: 15-30.

```julia
K = PadicField(2, 20)  # 20 digits of 2-adic precision
```

---

## Additional Documentation

- **[OPTIMIZERS.md](OPTIMIZERS.md)** - Detailed optimizer documentation, algorithm descriptions, hyperparameter tuning
- **[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md)** - Paper experiment infrastructure and benchmarking utilities
