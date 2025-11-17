# NAML (Non-Archimedean Machine Learning) Repository - Comprehensive Documentation

## Overview

This is a specialized machine learning library implementing algorithms for non-Archimedean mathematics, specifically targeting p-adic numbers and polydiscs over non-Archimedean fields.

**Key Insight**: The repository implements machine learning optimization techniques that work over p-adic numbers (non-Archimedean fields) instead of real/complex numbers, with applications to understanding problem structure through Berkovich space geometry.

---

## 1. Repository Purpose and Scope

### What It Does
The repository provides infrastructure for:
- **Non-Archimedean Machine Learning**: Training parametrized models where inputs and parameters live in p-adic fields or polydiscs
- **Optimization Algorithms**: Implementing greedy descent and gradient descent in non-Archimedean spaces
- **Loss Functions**: Computing mean squared error (MSE) and mean p-power error (MPE) losses
- **Statistical Tools**: Computing Frechet means in polydisc space
- **Phylogenetic Applications**: Embedding trees into disc space for phylogenetic analysis

### Mathematical Foundation
- Works with p-adic fields (e.g., 2-adic numbers, 3-adic numbers)
- Operates on **polydiscs** (multi-dimensional balls in p-adic spaces) rather than Euclidean balls
- Uses **valuation** rather than absolute value for distance measurement
- Implements directional derivatives and gradients adapted for non-Archimedean geometry

### Core Applications
1. **Polynomial Root Learning**: Learn parameters of polynomials whose roots depend on parameters
2. **Tree Learning**: Learn embeddings where opposite elements are far apart
3. **Frechet Mean Computation**: Statistical center of point clouds in polydisc space
4. **Phylogenetic Tree Analysis**: Embed phylogenetic trees into disc spaces

---

## 2. Directory Structure and Organization

```
wordnet/
├── README.md                    # High-level project overview
├── .gitignore                   # Ignore .sage.py and .json files
├── src/                         # Main source code
│   ├── naml.jl                  # Main entry point - includes all modules
│   ├── basic/                   # Core mathematical structures
│   │   ├── polydisc.jl          # Polydisc definitions and operations
│   │   ├── tangent_vector.jl    # Tangent vector structures on polydiscs
│   │   └── functions.jl         # Polynomial functions and derivatives
│   ├── optim/                   # Optimization infrastructure
│   │   ├── basic.jl             # Loss struct and OptimSetup (core API)
│   │   ├── loss.jl              # MSE and MPE loss implementations
│   │   ├── greedy_descent.jl    # Greedy descent optimizer
│   │   ├── gradient_descent.jl  # Gradient descent optimizer
│   │   └── model.jl             # AbstractModel and Model structures
│   └── statistics/              # Statistical functions
│       └── frechet.jl           # Frechet mean computation
├── test/                        # Test files and demo experiments
│   ├── polydisc.jl              # Basic polydisc operations test
│   ├── tangent_vector.jl        # Tangent vector operations test
│   ├── functions.jl             # Polynomial evaluation test
│   ├── polynomial_learning.jl   # Polynomial root learning demo
│   ├── gradient_descent.jl      # Gradient descent demo
│   ├── frechet.jl               # Frechet mean test
│   ├── wordnet.jl               # WordNet embedding experiment (incomplete)
│   ├── polynomial_learning_visualization.jl  # Visualization utilities
│   ├── cubic_learning_experiment.ipynb       # Interactive demo: cubic root learning
│   ├── polynomial_learning.ipynb             # Interactive demo: arbitrary degree polynomial learning
│   └── tree_colouring_learning.ipynb         # Interactive demo: tree coloring task
├── phylogenetics/               # Phylogenetic applications
│   ├── util/
│   │   └── embeddings.jl        # Tree embedding into disc space
│   ├── test/
│   │   └── util.jl              # Tests for tree embedding
│   └── frechet_mean.ipynb       # Interactive demo: Frechet mean computation
└── archive/                     # Legacy code and historical implementations
    ├── berkovich.jl             # Earlier Berkovich point implementations
    ├── disc.jl                  # Disc space implementation
    ├── wordnet.jl               # WordNet experiment code
    ├── polydisk_optimisation.jl # Legacy optimization
    ├── valuation_disk.jl        # Valuation-based disc implementation
    ├── worddescent.ipynb        # Legacy notebook
    ├── paths_and_norms.sage     # SageMath implementation
    └── wordnet_data/            # WordNet dataset
        ├── mammals.jl           # Mammal hierarchy script
        └── wordnet_mammal_hypernyms.tsv  # WordNet mammal data
```

---

## 3. Key Files and Components Explained

### 3.1 Basic Mathematical Structures (`src/basic/`)

#### **polydisc.jl** - Core Polydisc Implementation
**Purpose**: Define polydiscs (p-adic balls) and basic operations

**Key Structures**:
- `ValuationPolydisc{S,T}`: A polydisc in a valued field
  - `center::Vector{S}`: Center coordinates in p-adic field
  - `radius::Vector{T}`: Radius measured via valuation (not absolute value)
- `AbsPolydisc{S,T}`: Polydisc with radius measured via norm (less commonly used)

**Key Functions**:
- `dim(p)`: Dimension of the polydisc
- `prime(p)`: Get the prime p of the p-adic field
- `center(p)`, `radius(p)`: Accessor functions
- `join(b1, b2)`: Compute intersection of two polydiscs
- `dist(b1, b2)`: Distance metric between polydiscs
- `children(p, degree)`: Generate child nodes in the tree (shrink radii in different directions)
- `children_along_branch(p, branch_index)`: Generate children along a single branch
- `concatenate(p, q)`: Combine two polydiscs into higher dimension

**Mathematical Context**:
- Polydiscs are the fundamental objects over p-adic numbers
- Unlike Euclidean geometry, they have a tree-like structure
- Each polydisc has `p^degree` children (branching factor equals the prime)

#### **tangent_vector.jl** - Tangent Spaces
**Purpose**: Represent directions at points in polydisc space

**Key Structures**:
- `ValuationTangent{S,T}`: A tangent vector at a polydisc
  - `point::ValuationPolydisc`: Base point
  - `direction::Vector{S}`: Direction of movement
  - `magnitude::Vector{T}`: Magnitude along each coordinate

**Key Functions**:
- `zero(P, Q)`: Zero tangent vector at P in direction of Q
- `basis_vector(P, Q, i)`: i-th standard basis vector
- `Base.:+(v1, v2)`: Add two tangent vectors

#### **functions.jl** - Polynomial Functions and Calculus
**Purpose**: Define polynomials on polydisc space and compute their derivatives

**Key Structures**:
- `PolydiscFunction{S}`: A function defined by a sum of absolute polynomials
  - `polys::Vector{...}`: Vector of multivariate polynomials

**Key Functions**:
- `evaluate_abs(f, p)`: Evaluate absolute value of polynomial at polydisc
- `directional_exponent(f, v)`: Find exponent vectors where maximum is attained
- `directional_derivative(f, v)`: Compute derivative in tangent direction (core for optimization)
- `grad(f, v)`: Compute gradient vector
- `eval_abs(fun, var)`: Evaluate polynomial sum at a point

---

### 3.2 Optimization Infrastructure (`src/optim/`)

#### **basic.jl** - Core Optimization API
**Purpose**: Provide the foundational structures for optimization

**Key Structures**:
- `Loss`: Wraps evaluation and gradient functions
  - `eval::Function`: Evaluate loss (param) → scalar
  - `grad::Function`: Evaluate gradient (tangent_vector) → scalar

- `OptimSetup{S,T,U,V}`: Main optimization container (mutable)
  - `loss::Loss`: The loss function
  - `param::ValuationPolydisc`: Current parameter values
  - `optimiser::Function`: The optimization algorithm
  - `state::U`: State tracking for stateful optimizers
  - `context::V`: Settings/hyperparameters

**Key Functions**:
- `eval_loss(optim)`: Evaluate current loss
- `update_param!(optim, param)`: Update parameters
- `step!(optim)`: Take one optimization step

**Design Philosophy**: OptimSetup is mutable to allow in-place updates, following functional+imperative hybrid style.

#### **loss.jl** - Standard Loss Functions
**Purpose**: Implement common loss functions for non-Archimedean setting

**Key Functions**:
- `MSE_loss_init(model, data)`: Mean Squared Error loss
  - Computes: `1/n * Σ(eval(model, xi, param) - yi)^2`
  - Returns closure capturing data and model

- `MPE_loss_init(model, data, p)`: Mean p-Power Error loss
  - Generalization of MSE using ℓp norm instead of ℓ2
  - Computes: `1/n * Σ|eval(model, xi, param) - yi|^p`

**Design Note**: Both return a `Loss` struct with eval and grad closures that capture data.

#### **greedy_descent.jl** - Greedy Descent Algorithm
**Purpose**: Implement tree-based greedy optimization

**How It Works**:
1. Generate all children of current polydisc
2. Evaluate loss at each child
3. Move to child with minimum loss
4. Repeat until convergence

**Key Functions**:
- `greedy_descent(loss, param, next_branch, settings)`: Single optimization step
  - `next_branch`: Which branch to descend (allows sequential descent)
  - `settings`: Tuple of (strict, degree)
    - `strict=true`: Descend along one branch at a time
    - `strict=false`: Consider all degree-dimensional branching directions
  - Returns: (next_param, updated_next_branch)

- `greedy_descent_init(param, loss, next_branch, settings)`: Initialize OptimSetup

**Advantages**: Simple, stable, follows tree structure naturally
**Limitations**: May get stuck in local minima; doesn't use gradient information

#### **gradient_descent.jl** - Gradient Descent Algorithm
**Purpose**: Implement gradient-based optimization (alternative to greedy)

**How It Works**:
1. Generate all children of current polydisc
2. Compute gradients at each child direction
3. Move toward child in direction of steepest descent
4. Repeat until convergence

**Key Functions**:
- `gradient_param(m, val, v)`: Compute gradient w.r.t. model parameters
  - Uses partial_gradient restricted to parameter indices
  - Handles mixed data/parameter variables

- `gradient_descent(loss, param, state, degree)`: Single optimization step
  - Returns: (best_child, unchanged_state)

- `gradient_descent_init(param, loss, state, degree)`: Initialize OptimSetup

**Advantages**: Uses directional information; more sophisticated
**Limitations**: Still limited to discrete children (no continuous descent)

#### **model.jl** - Model Structure
**Purpose**: Represent parametrized models

**Key Structures**:
- `AbstractModel{S}`: Model without parameter values
  - `fun::PolydiscFunction`: The function (e.g., polynomial sum)
  - `param_info`: Boolean vector: true = data variable, false = parameter

- `Model{S,T}`: Model with parameter values (mutable)
  - `fun::AbstractModel`: The abstract function
  - `param::ValuationPolydisc`: Current parameter values

**Key Functions**:
- `var_indices(m)`: Get indices of data variables
- `param_indices(m)`: Get indices of parameters
- `set_abstract_model_variable(m, val, param)`: Interleave data and param values
- `eval_abs(m, val)`: Evaluate model at input val

**Example**:
```julia
# Model: f(x, θ, y, φ) where param_info = [true, false, true, false]
# Data variables: x (index 1), y (index 3)
# Parameters: θ (index 2), φ (index 4)
```

---

### 3.3 Statistical Tools (`src/statistics/`)

#### **frechet.jl** - Frechet Mean
**Purpose**: Compute statistical center of point sets in polydisc space

**Key Functions**:
- `frechet_mean(X::Vector{Vector{PadicFieldElem}})`: Coordinate-wise median
  - Uses ℓ¹-metric
  - Returns point minimizing sum of distances

- `frechet_mean(X::Vector{ValuationPolydisc}, prec)`: Polydisc Frechet mean
  - Uses greedy descent internally
  - `prec`: Number of iterations
  - Returns: Polydisc minimizing sum of distances

---

### 3.4 Applications (`phylogenetics/`)

#### **util/embeddings.jl** - Tree Embedding
**Purpose**: Embed phylogenetic trees into disc space

**Key Structures**:
- `Node`: Tree node
  - `name`: Node label
  - `children::Vector{Node}`: Child nodes

**Key Functions**:
- `parse_newick(str::String)`: Parse Newick format strings
  - Converts `"(A,(B,C));"` format to `Node` tree structure
  - Note: Implementation has known bugs (commented in code)

- `disc_embedding(tree, basepoint)`: Recursively embed tree into disc space
  - Root maps to basepoint
  - Each child subtree maps to child disc
  - Returns embedded tree with discs at nodes

- `field_embedding(tree, basepoint)`: Field-based embedding (stub implementation)

---

## 4. Programming Languages and Technologies

### Primary Language
- **Julia** (v1.x): Main implementation language
  - Chosen for mathematical computing capabilities
  - Excellent for numerical/algebraic work
  - Good library support for p-adic numbers

### Dependencies
- **Oscar.jl**: Comprehensive algebra/geometry library
  - Provides `PadicField`, polynomial rings, etc.
  - Used for field construction and polynomial operations
  - Imported as: `using Oscar`

- **AbstractAlgebra.jl**: Abstract algebra structures
  - Multivariate polynomial rings
  - Polynomial operations
  - Module depends on Oscar, no separate import needed

- **LinearAlgebra.jl**: Standard Julia linear algebra
  - Used for vector operations, norms
  - Imported as: `using LinearAlgebra`

### Numerical Packages
- **Nemo.jl**: Number theory library (part of Oscar ecosystem)
  - p-adic number implementation
  - Used for `padic` type and operations

### Notebook Support
- **Jupyter Notebooks** (.ipynb files)
  - Used for interactive demonstrations
  - Run with Julia kernel
  - Useful for exploratory analysis and learning

### Legacy Technologies
- **SageMath** (.sage files in archive): Earlier mathematical work

---

## 5. Build and Run Process

### Setup
```bash
# Clone repository
git clone <repo-url>
cd wordnet

# Julia environment setup (if using Project.toml, uncomment if exists)
# julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

### Running Tests
```bash
# Run individual test files
julia test/polydisc.jl
julia test/functions.jl
julia test/polynomial_learning.jl
julia test/gradient_descent.jl

# Run notebook demos
# Open in Jupyter with Julia kernel:
jupyter notebook test/cubic_learning_experiment.ipynb
jupyter notebook test/polynomial_learning.ipynb
jupyter notebook phylogenetics/frechet_mean.ipynb
```

### Basic Usage Pattern
```julia
# 1. Include main module
include("src/naml.jl")

# 2. Set up field and polynomial ring
prec = 20  # p-adic precision
K = PadicField(2, prec)  # 2-adic field
R, (x, a, b, c) = polynomial_ring(K, ["x", "a", "b", "c"])

# 3. Define model
fun = PolydiscFunction([(x-a)*(x-b)*(x-c)])
model = AbstractModel(fun, [true, false, false, false])  # x is data, a,b,c are params

# 4. Create loss
loss = MSE_loss_init(model, data)

# 5. Initialize parameter point
param = ValuationPolydisc([K(1), K(2), K(3)], [0, 0, 0])

# 6. Set up optimization
optim = greedy_descent_init(param, loss, 1, (false, 1))

# 7. Optimize
for i in 1:100
    step!(optim)
end
```

### Performance Considerations
- Greedy descent generates `p^degree` children per step (exponential in degree)
- Gradient descent is more stable but still tree-based
- Precision parameter affects computation time/accuracy trade-off
- Code includes profiling/timing (see test files using `time()`)

---

## 6. Test Files and Test Structure

### Unit Test Files

| File | Purpose | Status |
|------|---------|--------|
| `test/polydisc.jl` | Basic polydisc operations (join, children, eachindex) | Working |
| `test/tangent_vector.jl` | Tangent vector addition and basis operations | Working |
| `test/functions.jl` | Polynomial evaluation and directional derivatives | Working |
| `test/frechet.jl` | Frechet mean computation | Working |
| `test/polynomial_learning.jl` | Full optimization pipeline for cubic learning | Working |
| `test/gradient_descent.jl` | Gradient descent optimization | Working (some commented API) |
| `phylogenetics/test/util.jl` | Tree parsing and embedding (mostly disabled @test) | Partial |
| `test/wordnet.jl` | WordNet embedding experiment | **Incomplete** (stub only) |

### Interactive Notebook Demos

| Notebook | Content | Audience |
|----------|---------|----------|
| `test/cubic_learning_experiment.ipynb` | Learn roots of parametrized cubic (2-adic) | Learning/Demonstration |
| `test/polynomial_learning.ipynb` | Learn roots of random arbitrary-degree polynomials | Learning/Demonstration |
| `test/tree_colouring_learning.ipynb` | Learn tree coloring with embedding | Application Demo |
| `phylogenetics/frechet_mean.ipynb` | Compute Frechet mean in polydisc space | Statistics Demo |

### Test Characteristics
- Tests are primarily **exploratory** (using `@show` macros)
- Not comprehensive unit test coverage
- Tests are **standalone scripts** that include source and run independently
- Notebooks useful for **understanding the mathematics** and experiments

---

## 7. Configuration Files and Their Purposes

### Version Control
- **`.git/`**: Git repository metadata
  - Currently on branch: `wordnet`
  - Main branch: `main`
  - Recent work on: batch processing, descent algorithms

### Build/Environment
- **`.gitignore`**: Ignores generated files
  - `*.sage.py`: SageMath compiled files
  - `*.json`: JSON output files (likely from Jupyter)

### Documentation
- **`README.md`**: Main project documentation
  - High-level overview of project goals
  - Pointers to demo notebooks
  - Content organization

### Project Structure (No package.toml found)
- **NOT a registered Julia package** (no Project.toml/Manifest.toml)
- Uses direct includes: `include("src/naml.jl")`
- Manual dependency management (requires Oscar.jl installed globally)

---

## 8. Scripts and Utilities

### Main Entry Point
- **`src/naml.jl`**: Master include file
  - Imports Oscar and LinearAlgebra
  - Sequentially includes all submodules
  - Establishes full namespace for library

### Demonstration Scripts
- **`test/polynomial_learning_visualization.jl`**: Visualization utilities
  - Creates plots for learning experiments
  - Dependencies: Uses polynomial learning results

### Data Files
- **`archive/wordnet_data/wordnet_mammal_hypernyms.tsv`**: WordNet mammal hierarchy
  - Tab-separated format
  - Used by legacy WordNet experiment
  - Format: hypernym/hyponym relationships

- **`archive/wordnet_data/mammals.jl`**: Script to process mammal data
  - Integrates WordNet data into experiments
  - Fixed in recent commit: "fix: mammals.jl script"

### Utility Scripts in phylogenetics
- **`phylogenetics/util/embeddings.jl`**: Reusable tree embedding code
- **`phylogenetics/test/util.jl`**: Tests for embeddings

---

## 9. Recent Development and Git History

### Recent Commits (Last 5)
1. **80f1546** (Oct 2024): "Merge pull request #4 - batch-rfc"
   - Major feature: batch processing for gradient/loss evaluation

2. **3264f8d**: "rfc: make gradient and loss evaluation work with batches"
   - Refactoring for batch operations

3. **5efbe4c**: "stashing few changes"
   - Temporary commits

4. **ae6f69b**: "fix: mammals.jl script"
   - Bug fix in WordNet processing

5. **f45a61e**: "Experiments and profiling with naive matrix method"
   - Performance optimization work

### Major Refactoring History
- **Descent RFC** (pull #2): Single-branch descent implementation
- **Optimization RFC** (pull #1): Refactored dependency on data in OptimSetup
- **Package management**: Moved to Julia package structure (`d48d1b1`)

### Current Status
- **Active development** on batch processing features
- **Some incomplete features** (WordNet embedding stub, tree parsing bugs)
- **Well-structured codebase** with clear separation of concerns

---

## 10. Known Issues and Limitations

### Code Quality
1. **Tree parsing** (`phylogenetics/embeddings.jl`): Known bugs, implementation incomplete
2. **WordNet experiment** (`test/wordnet.jl`): Only skeleton implemented
3. **Test coverage**: Exploratory rather than comprehensive

### Performance
1. **Exponential branching**: `p^degree` children per step explodes for large degrees
2. **No caching**: Distances/evaluations recomputed each iteration
3. **Memory**: Full child enumeration may be prohibitive in high dimensions

### Documentation
1. **Minimal inline documentation**: Code has sparse docstrings
2. **Mathematical background**: No detailed theory explanation
3. **API guide**: Limited usage examples in comments

---

## 11. Architecture Highlights

### Design Patterns
1. **Closure-based loss functions**: Data captured in closures, not stored in structs
2. **Mutable optimization state**: OptimSetup allows in-place parameter updates
3. **Modular component design**: Easy to swap optimizers or loss functions
4. **Tree-based algorithm**: Leverage p-adic geometry's natural tree structure

### Separation of Concerns
```
Basic Math       (polydisc, tangent_vector, functions)
    ↓
Models           (AbstractModel, Model)
    ↓
Loss Functions   (Loss struct, MSE/MPE implementations)
    ↓
Optimizers       (greedy_descent, gradient_descent)
    ↓
Applications     (frechet_mean, tree embedding, polynomial learning)
```

### Key Abstractions
1. **PolydiscFunction**: Any function on polydisc space (currently polynomials)
2. **Model/AbstractModel**: Separates function from parameter values
3. **Loss**: Encapsulates evaluation and gradient computation
4. **OptimSetup**: Unified optimization interface for all algorithms

---

## 12. Quick Reference: Key Types and Functions

### Data Types
```julia
ValuationPolydisc{S,T}         # p-adic ball (center + radius)
ValuationTangent{S,T}           # Direction at a point
PolydiscFunction{S}             # Sum of absolute value polynomials
AbstractModel{S}                # Function + parameter structure
Model{S,T}                      # Model + parameter values
Loss                            # (eval, grad) function pair
OptimSetup{S,T,U,V}            # Container for optimization
Node                            # Tree node for phylogenetics
```

### Core Functions
```julia
# Polydisc operations
children(p, degree)             # Generate child nodes
children_along_branch(p, i)     # Generate branch-i children
join(b1, b2)                    # Intersection of polydiscs
dist(b1, b2)                    # Polydisc distance

# Derivatives
directional_derivative(f, v)    # Compute ∂f along tangent
directional_exponent(f, v)      # Find maximal term direction

# Optimization
eval_loss(optim)                # Current loss value
step!(optim)                    # Single optimization step
greedy_descent_init(...)        # Initialize greedy optimizer
gradient_descent_init(...)      # Initialize gradient optimizer

# Utilities
frechet_mean(X, prec)           # Statistical center
parse_newick(str)               # Parse tree format
disc_embedding(tree, p)         # Embed tree into discs
```

---

## 13. Entry Points for New Development

### For Adding New Optimizers
1. Implement function: `my_optimizer(loss, param, state, context) → (new_param, new_state)`
2. Wrap in OptimSetup: `MyOptim(loss, param, (l,p,s,c) → my_optimizer(...), state, context)`
3. Call: `step!(optim)` as usual

### For New Loss Functions
1. Create: `MyLoss_loss_init(model, data, [params]) → Loss`
2. Return: `Loss(eval_closure, grad_closure)`
3. Use: `loss = MyLoss_loss_init(...); optim = MyOptim_init(..., loss)`

### For New Applications
1. Create submodule in `src/` or separate directory
2. Include through `src/naml.jl`
3. Reference existing models/optimizers

### For Performance Improvements
1. **Caching**: Cache distance computations (children are deterministic)
2. **Batch processing**: Recent work (branch `batch-rfc`) adds this
3. **Continuous descent**: Implement directional movement in Berkovich space

---

## 14. Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | 34 |
| **Julia Source Files** | 20 |
| **Jupyter Notebooks** | 5 |
| **Lines of Code (src+test)** | ~900 |
| **Core Modules** | 7 |
| **Test Files** | 8 |
| **Archive Files** | 8 |
| **Programming Language** | Julia |
| **Primary Dependencies** | Oscar.jl, AbstractAlgebra.jl, LinearAlgebra.jl |
| **Latest Commit** | Nov 2024 (batch processing) |
| **Git Branches** | main, wordnet |

---

## Conclusion

The NAML repository is a sophisticated implementation of machine learning over non-Archimedean fields, with particular focus on:
- **Theoretical rigor** in p-adic mathematics
- **Practical optimization** using tree-based algorithms
- **Clear architectural design** separating math, models, and optimization
- **Interactive demonstrations** through Jupyter notebooks

It's well-suited for researchers exploring non-Archimedean geometry, p-adic analysis, and alternative optimization landscapes for machine learning problems.
