# NAML (Non-Archimedean Machine Learning) Repository - Comprehensive Documentation

## Overview

This is a specialized machine learning library implementing algorithms for non-Archimedean mathematics, specifically targeting p-adic numbers and polydiscs over non-Archimedean fields.

**Key Insight**: The repository implements machine learning optimization techniques that work over p-adic numbers (non-Archimedean fields) instead of real/complex numbers.

---

## 1. Repository Purpose and Scope

### What It Does
The repository provides infrastructure for:
- **Non-Archimedean Machine Learning**: Training parametrized models where inputs and parameters live in p-adic fields or polydiscs
- **Tree Search Optimization Algorithms**: Implementing Monte Carlo Tree Search (MCTS), Upper Confidence Trees (UCT), Hierarchical Optimistic Optimization (HOO), and variants
- **Classical Optimizers**: Greedy descent and gradient descent in non-Archimedean spaces
- **Loss Functions**: Computing mean squared error (MSE) and mean p-power error (MPE) losses
- **Compositional Functions**: Rich function composition algebra with automatic differentiation
- **Statistical Tools**: Computing Frechet means in polydisc space
- **Phylogenetic Applications**: Embedding trees into disc space for phylogenetic analysis

### Mathematical Foundation
- Works with p-adic fields (e.g., 2-adic numbers, 3-adic numbers)
- Operates on **polydiscs** (multi-dimensional balls in p-adic spaces) rather than Euclidean balls
- Uses **valuation** rather than absolute value for distance measurement
- Implements directional derivatives and gradients adapted for non-Archimedean geometry
- Exploits tree structure of p-adic spaces for optimization

### Core Applications
1. **Polynomial Root Learning**: Learn parameters of polynomials whose roots depend on parameters
2. **Tree Learning**: Learn embeddings where opposite elements are far apart
3. **Frechet Mean Computation**: Statistical center of point clouds in polydisc space
4. **Phylogenetic Tree Analysis**: Embed phylogenetic trees into disc spaces
5. **Global Optimization**: Use tree search algorithms to find global minima in non-Archimedean spaces

---

## 2. Directory Structure and Organization

```
naml-experiments/
├── README.md                    # High-level project overview
├── .gitignore                   # Ignore .sage.py and .json files
├── src/                         # Main source code
│   ├── NAML.jl                  # Main module entry point with explicit exports
│   ├── basic/                   # Core mathematical structures
│   │   ├── valuation.jl         # Generic valuation and abs() for p-adic numbers
│   │   ├── polydisc.jl          # Polydisc definitions and operations
│   │   ├── tangent_vector.jl    # Tangent vector structures on polydiscs
│   │   └── functions.jl         # Function algebra: polynomials, rationals, composition (614 lines)
│   ├── optimization/            # Optimization infrastructure (renamed from optim/)
│   │   ├── optim_setup.jl       # Loss struct and OptimSetup (core API, renamed from basic.jl)
│   │   ├── loss.jl              # MSE and MPE loss implementations
│   │   ├── model.jl             # AbstractModel and Model structures
│   │   └── optimizers/          # All optimizer implementations
│   │       ├── greedy_descent.jl      # Greedy descent optimizer
│   │       ├── gradient_descent.jl    # Gradient descent optimizer
│   │       └── tree_search/           # Tree search algorithms (NEW)
│   │           ├── mcts.jl            # Monte Carlo Tree Search
│   │           ├── uct.jl             # Upper Confidence Trees
│   │           ├── modified_uct.jl    # Modified UCT variant
│   │           ├── flat_ucb.jl        # Flat UCB variant
│   │           └── hoo.jl             # Hierarchical Optimistic Optimization
│   └── statistics/              # Statistical functions
│       └── frechet.jl           # Frechet mean computation
├── test/                        # Formal test suite
│   ├── Project.toml             # Test dependencies
│   ├── runtests.jl              # Main test runner
│   ├── polydisc.jl              # Basic polydisc operations test
│   ├── tangent_vector.jl        # Tangent vector operations test
│   ├── functions.jl             # Polynomial evaluation test
│   ├── test_functions.jl        # Expanded function tests
│   ├── polynomial_learning.jl   # Polynomial root learning test
│   ├── gradient_descent.jl      # Gradient descent test
│   ├── frechet.jl               # Frechet mean test
│   ├── test_all_optimizers.jl   # Comprehensive optimizer tests
│   └── docker_test.jl           # Docker environment tests
├── experiments/                 # Interactive demonstrations (moved from test/)
│   ├── cubic_learning_experiment.ipynb       # Demo: cubic root learning
│   ├── linear_learning_experiment.ipynb      # Demo: linear function learning
│   ├── polynomial_learning.ipynb             # Demo: arbitrary degree polynomial learning
│   ├── tree_colouring_learning.ipynb         # Demo: tree coloring task
│   ├── testing_notebook.ipynb                # General testing playground
│   ├── wordnet.ipynb                         # WordNet embedding experiment
│   ├── wordnet.jl                            # WordNet utilities
│   └── polynomial_learning_visualization.jl  # Visualization utilities
├── outputs/                     # Output files from experiments
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

#### **valuation.jl** - Generic Valuation and Absolute Value (NEW)
**Purpose**: Provide generic implementations for p-adic valuation and absolute value

**Key Functions**:
- `valuation(a::padic)`: Compute p-adic valuation of a p-adic number
  - For `a = p^v · u` where `u` is a p-adic unit, returns the integer `v`
  - Wraps Nemo/Oscar implementation
  - Extensible: users can define `valuation(a::YourType)` for custom types

- `Base.abs(a::padic)`: Compute p-adic absolute value
  - For valuation `v(a)`, returns `p^(-v(a))` as Float64
  - Extends `Base.abs` to make codebase generic
  - Formula: `|a|_p = p^(-v(a))`

**Design Philosophy**: By defining these generic functions, the codebase can work with any type implementing valuation, not just Oscar's padic type.

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

#### **functions.jl** - Massively Expanded Function Algebra (NEW: 614 lines)
**Purpose**: Define a rich algebra of functions on polydisc space with automatic differentiation

**Abstract Base Type**:
- `PolydiscFunction{S}`: Base type for all functions on polydisc spaces
  - Supports evaluation on polydiscs
  - Supports directional derivative computation

**Concrete Function Types**:

1. **Polynomial Functions**:
   - `AbsolutePolynomialSum{S}`: Sum of multivariate polynomials
     - `polys::Vector{AbstractAlgebra.Generic.MPoly{S}}`
   - `LinearPolynomial{S}`: Linear polynomial `a₁T₁ + ... + aₙTₙ + b`
     - `coefficients::Vector{S}`: Coefficients for each variable
     - `constant::S`: Constant term
   - `LinearAbsolutePolynomialSum{S}`: Sum of linear polynomials
     - `polys::Vector{LinearPolynomial{S}}`

2. **Rational Functions**:
   - `LinearRationalFunction{S}`: Ratio of linear polynomials
     - `num::LinearPolynomial{S}`: Numerator
     - `den::LinearPolynomial{S}`: Denominator
   - `LinearRationalFunctionSum{S}`: Sum of rational functions
     - `rats::Vector{LinearRationalFunction{S}}`

3. **Compositional Types** (Operator Algebra):
   - `Add{S}`: Addition of two functions
     - `left::PolydiscFunction{S}`, `right::PolydiscFunction{S}`
   - `Sub{S}`: Subtraction of two functions
   - `Mul{S}`: Multiplication of two functions
   - `Div{S}`: Division of two functions
     - `top::PolydiscFunction{S}`, `bottom::PolydiscFunction{S}`
   - `SMul{S}`: Scalar multiplication
     - `left::Number`, `right::PolydiscFunction{S}`
   - `Comp{S}`: Composition with real function
     - `left::Function`, `right::PolydiscFunction{S}`
   - `Constant{S}`: Constant function
     - `value::Number`
   - `Lambda{S}`: Custom function wrapper
     - `func::Function`

**Operator Overloading**:
```julia
# Automatic algebraic operations
f + g    # Add{S}(f, g)
f - g    # Sub{S}(f, g)
f * g    # Mul{S}(f, g)
f / g    # Div{S}(f, g)
a * f    # SMul{S}(a, f)  where a is Number
f^n      # Repeated multiplication
-f       # SMul{S}(-1, f)
f + c    # Add{S}(f, Constant{S}(c))
```

**Core Evaluation Functions**:
- `evaluate(f::PolydiscFunction, p::ValuationPolydisc)`: Evaluate function at polydisc
  - Returns interval/polydisc representing function range
- `evaluate_abs(f, p)`: Evaluate absolute value of polynomial at polydisc
- `batch_evaluate_init(f, data)`: Create efficient batch evaluator
  - Returns closure for batch evaluation
  - Optimizes repeated evaluation on same data

**Calculus Operations**:
- `directional_exponent(f, v)`: Find exponent vectors where maximum is attained
  - Core for computing derivatives in non-Archimedean setting
- `directional_derivative(f, v)`: Compute derivative in tangent direction
  - Supports all compositional types via chain rule
  - Key for gradient-based optimization
- `grad(f, v)`: Compute gradient vector
  - Returns full gradient as tangent vector

**Key Design Features**:
- **Lazy evaluation**: Compositions are represented symbolically
- **Type-generic**: Works with any coefficient type `S`
- **Automatic differentiation**: Directional derivatives computed recursively
- **Batch optimization**: Specialized functions for efficient batch evaluation

---

### 3.2 Optimization Infrastructure (`src/optimization/`)

#### **optim_setup.jl** - Core Optimization API (renamed from basic.jl)
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
- `update_state!(optim, state)`: Update optimizer state
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
- `batch_evaluate_init(m, data)`: Create batch evaluator for model

**Example**:
```julia
# Model: f(x, θ, y, φ) where param_info = [true, false, true, false]
# Data variables: x (index 1), y (index 3)
# Parameters: θ (index 2), φ (index 4)
```

---

### 3.3 Optimizer Implementations (`src/optimization/optimizers/`)

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

---

### 3.4 Tree Search Optimizers (`src/optimization/optimizers/tree_search/`) - NEW

These are advanced bandit-based tree search algorithms adapted for non-Archimedean optimization.

#### **mcts.jl** - Monte Carlo Tree Search
**Purpose**: Use Monte Carlo simulations to explore the polydisc tree and find optimal regions

**Algorithm**: Classic MCTS with four phases per simulation:
1. **Selection**: Use UCB1 to traverse tree toward promising nodes
2. **Expansion**: Add new child nodes to the tree
3. **Simulation**: Random rollout from newly expanded node
4. **Backpropagation**: Update statistics along path to root

**Key Structures**:
- `MCTSNode{S,T}`: Tree node
  - `polydisc::ValuationPolydisc{S,T}`: Region at this node
  - `parent::Union{MCTSNode{S,T}, Nothing}`: Parent link
  - `children::Vector{MCTSNode{S,T}}`: Expanded children
  - `visits::Int`: Visit count
  - `total_value::Float64`: Cumulative value
  - `is_expanded::Bool`: Expansion status

- `MCTSConfig`: Configuration
  - `num_simulations::Int`: Simulations per step (default: 100)
  - `exploration_constant::Float64`: UCB1 constant (default: √2 ≈ 1.41)
  - `degree::Int`: Child generation degree
  - `max_children::Union{Int, Nothing}`: Limit children per node
  - `strict::Bool`: Single-branch mode
  - `value_transform::Function`: Loss → value (default: `loss -> 1/(loss + 1e-10)`)
  - `selection_mode::SelectionMode`: VisitCount or BestValue

- `MCTSState{S,T}`: Persistent state
  - `root::MCTSNode{S,T}`: Current search tree root
  - `next_branch::Int`: For strict mode
  - `step_count::Int`: Optimization steps taken

**Key Functions**:
- `ucb1_score(node, parent_visits, c)`: UCB1 formula for balancing exploration/exploitation
- `select_leaf(root, config)`: Selection phase
- `expand_node!(node, loss, config)`: Expansion phase
- `simulate(node, loss, config)`: Random rollout
- `backpropagate!(node, value)`: Update statistics
- `mcts_descent(loss, param, state, config)`: Single MCTS step
- `mcts_descent_init(param, loss, next_branch, config)`: Initialize optimizer

**When to Use**:
- When loss landscape is complex with many local minima
- When exploration is critical
- When you have computational budget for many simulations

#### **uct.jl** - Upper Confidence Trees
**Purpose**: Bandit-based tree search with depth limits and UCB scoring

**Algorithm**: Similar to MCTS but with:
- Depth-limited exploration (parameter `max_depth`)
- UCB scoring based on parent visits: `B = X̄ + c√(2ln(parent_visits)/visits)`
- Direct best-value selection (no rollout simulation)

**Key Structures**:
- `UCTNode{S,T}`: Tree node
  - Same as MCTSNode plus:
  - `depth::Int`: Node depth in tree (0 for root)

- `UCTConfig`: Configuration
  - `max_depth::Int`: Maximum tree depth D (default: 10)
  - `num_simulations::Int`: Simulations per step (default: 100)
  - `exploration_constant::Float64`: UCB constant c (default: √2)
  - `degree::Int`: Child generation degree
  - `strict::Bool`: Single-branch mode
  - `value_transform::Function`: Loss → value

- `UCTState{S,T}`: State
  - `root::UCTNode{S,T}`: Search tree root
  - `step_count::Int`: Steps taken

**Key Functions**:
- `ucb_score(node, parent_visits, c)`: Compute UCB score
  - Formula: `X̄ + c√(2ln(p)/n)` where p = parent visits, n = node visits
- `select_and_expand!(root, loss, config)`: Combined selection + expansion
- `backpropagate!(node, value)`: Update path statistics
- `uct_descent(loss, param, state, config)`: Single UCT step
- `uct_descent_init(param, loss, state, config)`: Initialize optimizer

**Advantages over MCTS**:
- Simpler (no random rollout)
- Depth control prevents infinite expansion
- Often faster convergence

**When to Use**:
- When you want simpler alternative to MCTS
- When depth control is important
- For smoother loss landscapes

#### **modified_uct.jl** - Modified UCT Variant
**Purpose**: UCT variant with modified exploration bonus

**Key Differences from Standard UCT**:
- Different UCB formula or exploration strategy
- May use different value aggregation
- Experimental variant for performance tuning

**Structures**: Similar to `uct.jl` with Modified prefix:
- `ModifiedUCTNode{S,T}`
- `ModifiedUCTConfig`
- `ModifiedUCTState{S,T}`

**Key Functions**:
- `modified_uct_descent(loss, param, state, config)`
- `modified_uct_descent_init(param, loss, state, config)`

#### **flat_ucb.jl** - Flat UCB Variant
**Purpose**: UCB-based search without tree structure (flat bandit)

**Algorithm**:
- Treats all candidate polydiscs as independent arms
- No tree hierarchy
- Pure multi-armed bandit approach with UCB scoring

**Structures**:
- `FlatUCBNode{S,T}`
- `FlatUCBConfig`
- `FlatUCBState{S,T}`

**Key Functions**:
- `flat_ucb_descent(loss, param, state, config)`
- `flat_ucb_descent_init(param, loss, state, config)`

**When to Use**:
- For comparison baseline
- When tree structure not helpful
- For very shallow optimization

#### **hoo.jl** - Hierarchical Optimistic Optimization
**Purpose**: Implement HOO algorithm from Bubeck et al. (2011) - theoretically principled global optimization

**Algorithm**: Maintains optimistic upper bounds B-values for each node:
1. Compute B-value: `B(h,i,n) = μ̂(h,i) + √(2ln(n)/N(h,i)) + ν₁ρʰ`
   - `μ̂(h,i)`: Empirical mean at node (h,i)
   - `n`: Total samples taken
   - `N(h,i)`: Visits to node (h,i)
   - `ρ`: Shrinkage rate (diameter shrinks as ρʰ)
   - `ν₁`: Smoothness parameter
   - `h`: Depth in tree
2. Select node with maximum B-value
3. Sample at that node
4. Update statistics
5. Repeat

**Key Structures**:
- `HOONode{S,T}`: Tree node
  - `polydisc::ValuationPolydisc{S,T}`: Region
  - `depth::Int`: Depth h (root = 0)
  - `position::Int`: Position i among siblings
  - `parent::Union{HOONode{S,T}, Nothing}`
  - `children::Vector{HOONode{S,T}}`
  - `visits::Int`: N(h,i)
  - `sum_values::Float64`: For computing μ̂
  - `is_expanded::Bool`

- `HOOConfig`: Configuration
  - `rho::Float64`: Shrinkage rate ρ ∈ (0,1) (default: 0.5)
  - `nu1::Float64`: Smoothness ν₁ > 0 (default: 0.1)
  - `max_depth::Int`: Maximum tree depth (default: 10)
  - `degree::Int`: Child generation degree
  - `strict::Bool`: Single-branch mode
  - `value_transform::Function`: Loss → reward

- `HOOState{S,T}`: State
  - `root::HOONode{S,T}`: Tree root
  - `total_samples::Int`: Total n for B-value formula
  - `next_branch::Int`: For strict mode
  - `step_count::Int`

**Key Functions**:
- `empirical_mean(node)`: Compute μ̂(h,i)
- `b_value(node, total_samples, config)`: Compute B-value (optimistic upper bound)
  - Formula: `μ̂ + √(2ln(n)/N) + ν₁ρʰ`
- `select_node(root, total_samples, config)`: Choose node with max B-value
- `expand_node!(node, loss, config)`: Generate and evaluate children
- `hoo_descent(loss, param, state, config)`: Single HOO step
- `hoo_descent_init(param, loss, next_branch, config)`: Initialize optimizer

**Theoretical Guarantees**:
- Provably converges to global optimum under smoothness assumptions
- Convergence rate depends on ρ, ν₁, and function smoothness

**When to Use**:
- When you want theoretical guarantees
- For global optimization problems
- When you can estimate smoothness (ν₁) and shrinkage (ρ)
- For benchmarking against theoretically sound baseline

**Tuning Advice**:
- `rho`: Set based on how fast regions shrink (0.5 means halving each level)
- `nu1`: Set based on how much function varies within regions (smaller = smoother)
- Start with defaults and adjust based on performance

---

### 3.5 Statistical Tools (`src/statistics/`)

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

### 3.6 Applications (`phylogenetics/`)

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
  - Module structure with explicit exports

### Dependencies
- **Oscar.jl**: Comprehensive algebra/geometry library
  - Provides `PadicField`, polynomial rings, etc.
  - Used for field construction and polynomial operations
  - Imported in module: `using Oscar`

- **AbstractAlgebra.jl**: Abstract algebra structures
  - Multivariate polynomial rings
  - Polynomial operations
  - Module depends on Oscar, no separate import needed

- **LinearAlgebra.jl**: Standard Julia linear algebra
  - Used for vector operations, norms
  - Imported in module: `using LinearAlgebra`

- **Printf.jl**: Formatted output
  - Used for logging and debugging

### Numerical Packages
- **Nemo.jl**: Number theory library (part of Oscar ecosystem)
  - p-adic number implementation
  - Used for `padic` type and operations

### Test Infrastructure
- **Test.jl**: Julia's standard testing framework
  - Used in `test/runtests.jl`
  - Provides `@testset` and `@test` macros

### Notebook Support
- **Jupyter Notebooks** (.ipynb files)
  - Used for interactive demonstrations in `experiments/`
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
cd naml-experiments

# Julia environment setup
# Note: Currently no Project.toml at root, but test/ has Project.toml
julia -e 'using Pkg; Pkg.add("Oscar"); Pkg.add("LinearAlgebra")'
```

### Using the Module
```julia
# Load the NAML module
include("src/NAML.jl")
using .NAML

# Or if installed as package:
using NAML

# Access exported functions directly
K = PadicField(2, 20)
p = ValuationPolydisc([K(1), K(2)], [0, 0])

# Use non-exported functions with NAML prefix
result = NAML.join(p1, p2)
```

### Running Tests
```bash
# Run formal test suite
cd test
julia --project runtests.jl

# Run individual test files
julia --project polydisc.jl
julia --project functions.jl
julia --project test_all_optimizers.jl

# Run notebook demos
cd ../experiments
jupyter notebook cubic_learning_experiment.ipynb
jupyter notebook polynomial_learning.ipynb
```

### Basic Usage Pattern
```julia
# 1. Include main module
include("src/NAML.jl")
using .NAML

# 2. Set up field and polynomial ring
prec = 20  # p-adic precision
K = PadicField(2, prec)  # 2-adic field
R, (x, a, b, c) = polynomial_ring(K, ["x", "a", "b", "c"])

# 3. Define model using compositional algebra
fun = AbsolutePolynomialSum([(x-a)*(x-b)*(x-c)])
model = AbstractModel(fun, [true, false, false, false])  # x is data, a,b,c are params

# 4. Create loss
loss = MSE_loss_init(model, data)

# 5. Initialize parameter point
param = ValuationPolydisc([K(1), K(2), K(3)], [0, 0, 0])

# 6. Choose optimizer - example with HOO
hoo_config = HOOConfig(rho=0.5, nu1=0.1, max_depth=15)
optim = hoo_descent_init(param, loss, 1, hoo_config)

# 7. Optimize
for i in 1:100
    step!(optim)
    if i % 10 == 0
        println("Step $i: Loss = ", eval_loss(optim))
    end
end

# Alternative: Use MCTS
mcts_config = MCTSConfig(num_simulations=200, exploration_constant=1.41)
optim_mcts = mcts_descent_init(param, loss, 1, mcts_config)
for i in 1:50
    step!(optim_mcts)
end
```

### Performance Considerations
- **Greedy descent**: Generates `p^degree` children per step (exponential in degree)
- **Tree search algorithms**: Computational cost controlled by `num_simulations` parameter
- **HOO**: Simpler per-step than MCTS (no rollout), but maintains larger tree
- **Batch evaluation**: Use `batch_evaluate_init` for repeated evaluations on same data
- **Precision parameter**: Affects computation time/accuracy trade-off
- Code includes profiling/timing in `logs_local/` experiments

---

## 6. Test Files and Experimental Structure

### Formal Test Suite (`test/`)

| File | Purpose | Status |
|------|---------|--------|
| `test/runtests.jl` | Main test runner with testsets | Working |
| `test/Project.toml` | Test dependencies (Test, Oscar, LinearAlgebra) | Working |
| `test/polydisc.jl` | Basic polydisc operations (join, children, eachindex) | Working |
| `test/tangent_vector.jl` | Tangent vector addition and basis operations | Working |
| `test/functions.jl` | Basic polynomial evaluation and derivatives | Working |
| `test/test_functions.jl` | Expanded function tests for new algebra | Working |
| `test/frechet.jl` | Frechet mean computation | Working |
| `test/polynomial_learning.jl` | Full optimization pipeline for cubic learning | Working |
| `test/gradient_descent.jl` | Gradient descent optimization | Working |
| `test/test_all_optimizers.jl` | Comprehensive tests for all optimizers | Working |
| `test/docker_test.jl` | Docker environment tests | Working |
| `phylogenetics/test/util.jl` | Tree parsing and embedding (mostly disabled @test) | Partial |

### Interactive Notebook Demos (`experiments/`)

| Notebook | Content | Audience |
|----------|---------|----------|
| `cubic_learning_experiment.ipynb` | Learn roots of parametrized cubic (2-adic) | Learning/Demonstration |
| `linear_learning_experiment.ipynb` | Learn linear function parameters | Learning/Demonstration |
| `polynomial_learning.ipynb` | Learn roots of random arbitrary-degree polynomials | Learning/Demonstration |
| `tree_colouring_learning.ipynb` | Learn tree coloring with embedding | Application Demo |
| `testing_notebook.ipynb` | General playground for testing features | Development |
| `wordnet.ipynb` | WordNet embedding experiment | Application (Experimental) |

**Visualization**:
- `polynomial_learning_visualization.jl`: Utilities for plotting results

### Experimental Scripts (`logs_local/`)

These are research-grade scripts for algorithm development and tuning:

| Script | Purpose |
|--------|---------|
| `experiment.jl` | General experimentation |
| `flat_ucb_comparison.jl` | Compare Flat UCB performance |
| `hoo_experiment.jl` | HOO algorithm experiments |
| `large_scale_experiment.jl` | Large-scale performance tests |
| `linear_combination_experiment.jl` | Test linear combinations |
| `mcts_tuning_experiment.jl` | Tune MCTS hyperparameters |
| `modified_uct_comparison.jl` | Compare Modified UCT variants |
| `polynomial_experiment.jl` | Polynomial learning experiments |
| `polynomial_learning_hoo.jl` | HOO for polynomial root finding |
| `root_finding_experiment.jl` | General root finding tests |
| `test_flat_ucb_basic.jl` | Basic Flat UCB functionality |
| `test_greedy_mcts.jl` | Greedy MCTS comparison |
| `test_modified_uct_basic.jl` | Basic Modified UCT tests |
| `test_uct_basic.jl` | Basic UCT functionality |
| `uct_vs_mcts_comparison.jl` | Direct UCT vs MCTS comparison |

### Test Characteristics
- **Formal tests**: Use `@test` and `@testset` macros
- **Comprehensive coverage**: All major optimizers tested
- **Exploratory demos**: Notebooks for understanding mathematics
- **Research scripts**: `logs_local/` for algorithm development
- **Standalone**: Tests include source and run independently

---

## 7. Configuration Files and Their Purposes

### Version Control
- **`.git/`**: Git repository metadata
  - Currently on branch: `update-CLAUDE.md`
  - Main branch: `main`
  - Recent work on: tree search algorithms, directory restructuring

### Build/Environment
- **`.gitignore`**: Ignores generated files
  - `*.sage.py`: SageMath compiled files
  - `*.json`: JSON output files (likely from Jupyter)

- **`test/Project.toml`**: Test environment dependencies
  - Test.jl: Testing framework
  - Oscar.jl: Algebra library
  - LinearAlgebra.jl: Linear algebra

### Documentation
- **`README.md`**: Main project documentation
  - High-level overview of project goals
  - Pointers to demo notebooks
  - Content organization (outdated - refers to old `src/optim` paths)

- **`CLAUDE.md`**: This comprehensive documentation file

### Module Structure
- **`src/NAML.jl`**: Module definition with explicit exports
  - Defines `module NAML`
  - Includes all submodules
  - Exports public API
  - NOT a registered Julia package (no Project.toml at root)

---

## 8. Scripts and Utilities

### Main Entry Point
- **`src/NAML.jl`**: Module definition
  - Imports Oscar, LinearAlgebra, Printf
  - Includes all submodules in correct dependency order
  - Exports public API (types and functions)
  - Provides namespace for library
  - Note: Some paths in includes are outdated (still reference `optim/` instead of `optimization/`)

### Demonstration Scripts
- **`experiments/polynomial_learning_visualization.jl`**: Visualization utilities
  - Creates plots for learning experiments
  - Dependencies: Uses polynomial learning results

- **`experiments/wordnet.jl`**: WordNet experiment utilities
  - Process WordNet data for embedding experiments

### Data Files
- **`archive/wordnet_data/wordnet_mammal_hypernyms.tsv`**: WordNet mammal hierarchy
  - Tab-separated format
  - Used by legacy WordNet experiment
  - Format: hypernym/hyponym relationships

- **`archive/wordnet_data/mammals.jl`**: Script to process mammal data
  - Integrates WordNet data into experiments

### Utility Scripts in phylogenetics
- **`phylogenetics/util/embeddings.jl`**: Reusable tree embedding code
- **`phylogenetics/test/util.jl`**: Tests for embeddings

---

## 9. Recent Development and Git History

### Major Recent Changes

1. **Directory Restructuring** (Current branch: update-CLAUDE.md):
   - Renamed `src/optim/` → `src/optimization/`
   - Renamed `src/optim/basic.jl` → `src/optimization/optim_setup.jl`
   - Created `src/optimization/optimizers/` subdirectory
   - Moved optimizers to subdirectory

2. **Tree Search Algorithms** (PR #22):
   - Implemented 5 new tree search optimizers:
     - MCTS (Monte Carlo Tree Search)
     - UCT (Upper Confidence Trees)
     - Modified UCT variant
     - Flat UCB variant
     - HOO (Hierarchical Optimistic Optimization)
   - Created `src/optimization/optimizers/tree_search/` directory
   - Added comprehensive experimental scripts in `logs_local/`

3. **Function Algebra Expansion**:
   - Massively expanded `src/basic/functions.jl` to 614 lines
   - Added compositional types: Add, Mul, Sub, Div, SMul, Comp, Constant, Lambda
   - Implemented operator overloading (+, -, *, /, ^)
   - Added batch evaluation infrastructure
   - Full calculus support for all types

4. **Generic Valuation Support**:
   - Added `src/basic/valuation.jl`
   - Generic `valuation()` and `abs()` for p-adic types
   - Extensible to custom implementations

5. **Experiments Directory** (PR #23):
   - Moved interactive notebooks from `test/` to `experiments/`
   - Created dedicated space for demonstrations
   - Separated formal tests from exploratory notebooks

6. **Test Infrastructure**:
   - Added `test/Project.toml` for dependency management
   - Created `test/runtests.jl` as main test runner
   - Added `test/test_all_optimizers.jl` for comprehensive optimizer testing
   - Expanded `test/test_functions.jl` for new function types

### Git History Highlights
Based on git status:
- Recent commits include:
  - "Move experiments to new directory" (#23)
  - "feat: implement various tree search algorithms" (#22)
  - "feat: add GEMINI.md"
  - Various test and proper-tests work (#20, #21)

### Current Development Status
- **Active development** on documentation (CLAUDE.md update)
- **Stable codebase** with comprehensive optimizer suite
- **Well-tested** optimizers with experimental validation
- **Some incomplete features** (WordNet embedding, tree parsing bugs)

---

## 10. Known Issues and Limitations

### Code Quality
1. **Tree parsing** (`phylogenetics/embeddings.jl`): Known bugs, implementation incomplete
2. **WordNet experiment**: Experimental/incomplete
3. **Path inconsistency**: `src/NAML.jl` still references old `optim/` paths instead of `optimization/`

### Performance
1. **Exponential branching**: `p^degree` children per step explodes for large degrees
2. **Tree search memory**: MCTS/UCT/HOO maintain growing trees (can be memory-intensive)
3. **No tree pruning**: Search trees not pruned between steps
4. **Batch optimization**: Available but not used by all components

### Documentation
1. **README.md outdated**: Still references old `src/optim` structure
2. **Inline documentation**: Sparse docstrings in some files
3. **Mathematical background**: Limited theory explanation
4. **Hyperparameter tuning**: Limited guidance for tree search parameters

### Algorithm Limitations
1. **Greedy descent**: Local minima traps
2. **Gradient descent**: Discrete steps only (no continuous descent)
3. **Tree search**: Require tuning of exploration constants
4. **HOO**: Requires knowledge of smoothness parameters (ρ, ν₁)

---

## 11. Architecture Highlights

### Design Patterns
1. **Closure-based loss functions**: Data captured in closures, not stored in structs
2. **Mutable optimization state**: OptimSetup allows in-place parameter updates
3. **Modular component design**: Easy to swap optimizers or loss functions
4. **Tree-based algorithms**: Leverage p-adic geometry's natural tree structure
5. **Compositional function algebra**: Build complex functions from simple pieces
6. **Lazy evaluation**: Function compositions represented symbolically

### Separation of Concerns
```
Basic Math           (valuation, polydisc, tangent_vector, functions)
    ↓
Models               (AbstractModel, Model)
    ↓
Loss Functions       (Loss struct, MSE/MPE implementations)
    ↓
Optimizers           (Classical: greedy, gradient)
    ↓                (Tree Search: MCTS, UCT, HOO, variants)
    ↓
Applications         (frechet_mean, tree embedding, polynomial learning)
```

### Key Abstractions
1. **PolydiscFunction**: Base type for all functions on polydisc space
   - Compositional: Add, Mul, Sub, Div, etc.
   - Concrete: AbsolutePolynomialSum, LinearRationalFunction, etc.
2. **Model/AbstractModel**: Separates function from parameter values
3. **Loss**: Encapsulates evaluation and gradient computation
4. **OptimSetup**: Unified optimization interface for all algorithms
5. **Tree Search Nodes**: Node{S,T} pattern for each algorithm (MCTSNode, UCTNode, HOONode)
6. **Config/State separation**: Configuration (immutable) vs State (mutable)

---

## 12. Quick Reference: Key Types and Functions

### Data Types

**Basic Structures**:
```julia
ValuationPolydisc{S,T}         # p-adic ball (center + radius)
ValuationTangent{S,T}           # Direction at a point
```

**Function Types**:
```julia
PolydiscFunction{S}             # Abstract base type
AbsolutePolynomialSum{S}        # Sum of polynomials
LinearPolynomial{S}             # Linear polynomial
LinearAbsolutePolynomialSum{S}  # Sum of linear polynomials
LinearRationalFunction{S}       # Rational function
LinearRationalFunctionSum{S}    # Sum of rationals
Add{S}, Sub{S}, Mul{S}, Div{S} # Compositional types
SMul{S}, Comp{S}                # Scalar mult, composition
Constant{S}, Lambda{S}          # Constants, custom functions
```

**Model Types**:
```julia
AbstractModel{S}                # Function + parameter structure
Model{S,T}                      # Model + parameter values
```

**Optimization Types**:
```julia
Loss                            # (eval, grad) function pair
OptimSetup{S,T,U,V}            # Complete optimization setup
```

**Tree Search Types**:
```julia
# MCTS
MCTSNode{S,T}, MCTSConfig, MCTSState{S,T}
SelectionMode                   # Enum: VisitCount, BestValue

# UCT
UCTNode{S,T}, UCTConfig, UCTState{S,T}

# Modified UCT
ModifiedUCTNode{S,T}, ModifiedUCTConfig, ModifiedUCTState{S,T}

# Flat UCB
FlatUCBNode{S,T}, FlatUCBConfig, FlatUCBState{S,T}

# HOO
HOONode{S,T}, HOOConfig, HOOState{S,T}
```

**Phylogenetics Types**:
```julia
Node                            # Tree node
```

### Core Functions

**Valuation** (NEW):
```julia
valuation(a::padic)             # p-adic valuation
Base.abs(a::padic)              # p-adic absolute value |a|_p
```

**Polydisc Operations**:
```julia
children(p, degree)             # Generate child nodes
children_along_branch(p, i)     # Generate branch-i children
NAML.join(b1, b2)               # Intersection (not exported)
dist(b1, b2)                    # Polydisc distance
concatenate(p, q)               # Combine polydiscs
dim(p), prime(p)                # Accessors
center(p), radius(p)            # Accessors
```

**Function Operations** (NEW/Expanded):
```julia
# Evaluation
evaluate(f, p)                  # Evaluate at polydisc
evaluate_abs(f, p)              # Absolute value evaluation
eval_abs(f, val)                # Point evaluation
batch_evaluate_init(f, data)    # Create batch evaluator

# Calculus
directional_derivative(f, v)    # ∂f along tangent
directional_exponent(f, v)      # Find maximal term
grad(f, v)                      # Gradient vector

# Composition (via operators)
f + g, f - g, f * g, f / g     # Arithmetic
a * f, f * a                    # Scalar multiplication
f^n                             # Exponentiation
comp(func, f)                   # Function composition
```

**Model Operations**:
```julia
var_indices(m)                  # Data variable indices
param_indices(m)                # Parameter indices
set_abstract_model_variable(m, val, param)  # Combine data + params
batch_evaluate_init(m, data)    # Batch evaluator
```

**Optimization Core**:
```julia
eval_loss(optim)                # Current loss value
step!(optim)                    # Single optimization step
update_param!(optim, param)     # Update parameters
update_state!(optim, state)     # Update state
```

**Loss Functions**:
```julia
MSE_loss_init(model, data)      # Mean squared error
MPE_loss_init(model, data, p)   # Mean p-power error
```

**Classical Optimizers**:
```julia
greedy_descent_init(param, loss, next_branch, settings)
gradient_descent_init(param, loss, state, degree)
```

**Tree Search Optimizers** (NEW):
```julia
# MCTS
mcts_descent_init(param, loss, next_branch, config)
MCTSConfig(; num_simulations=100, exploration_constant=1.41, ...)

# UCT
uct_descent_init(param, loss, state, config)
UCTConfig(; max_depth=10, num_simulations=100, ...)

# Modified UCT
modified_uct_descent_init(param, loss, state, config)

# Flat UCB
flat_ucb_descent_init(param, loss, state, config)

# HOO
hoo_descent_init(param, loss, next_branch, config)
HOOConfig(; rho=0.5, nu1=0.1, max_depth=10, ...)

# Utility functions
get_tree_size(state)            # HOO tree statistics
get_visited_nodes(state)
get_leaf_nodes(state)
```

**Statistical Functions**:
```julia
frechet_mean(X, prec)           # Polydisc Frechet mean
frechet_mean(X)                 # Coordinate-wise median
```

**Phylogenetics**:
```julia
parse_newick(str)               # Parse Newick tree format
disc_embedding(tree, p)         # Embed tree into discs
field_embedding(tree, p)        # Field-based embedding
```

---

## 13. Entry Points for New Development

### For Adding New Optimizers
1. Create file in `src/optimization/optimizers/` (or `tree_search/` subdirectory)
2. Define node structure (if tree-based): `MyOptimizerNode{S,T}`
3. Define config structure: `MyOptimizerConfig` with constructor
4. Define state structure: `MyOptimizerState{S,T}`
5. Implement: `my_optimizer_descent(loss, param, state, context) → (new_param, new_state)`
6. Implement: `my_optimizer_descent_init(param, loss, state, context) → OptimSetup`
7. Add exports to `src/NAML.jl`
8. Add tests to `test/test_all_optimizers.jl`
9. Add experiments to `logs_local/`

### For New Loss Functions
1. Add to `src/optimization/loss.jl`
2. Create: `MyLoss_loss_init(model, data, [params]) → Loss`
3. Return: `Loss(eval_closure, grad_closure)`
4. Add export to `src/NAML.jl`
5. Use: `loss = MyLoss_loss_init(...); optim = SomeOptimizer_init(..., loss)`

### For New Function Types
1. Add to `src/basic/functions.jl`
2. Define struct inheriting from `PolydiscFunction{S}`
3. Implement `evaluate(f::MyFunction, p::ValuationPolydisc)`
4. Implement `directional_derivative(f::MyFunction, v::ValuationTangent)`
5. Optionally implement `batch_evaluate_init`
6. Add export to `src/NAML.jl`

### For New Applications
1. Create submodule in `src/` or separate directory
2. Include through `src/NAML.jl`
3. Reference existing models/optimizers
4. Add demo notebook in `experiments/`

### For Performance Improvements
1. **Caching**: Cache distance computations (children are deterministic)
2. **Tree pruning**: Prune tree search nodes between steps
3. **Batch processing**: Extend batch evaluation to more components
4. **Continuous descent**: Implement directional movement in polydisc space
5. **Parallelization**: Parallelize child evaluation in tree search

---

## 14. Algorithm Selection Guide

### When to Use Each Optimizer

**Greedy Descent**:
- **Best for**: Simple problems, baseline comparisons
- **Pros**: Simple, fast per-step, no hyperparameters
- **Cons**: Gets stuck in local minima, no exploration
- **Use when**: Loss landscape is smooth, quick results needed

**Gradient Descent**:
- **Best for**: Smooth landscapes with useful gradients
- **Pros**: Uses directional information, more sophisticated than greedy
- **Cons**: Still discrete steps, can get stuck
- **Use when**: Gradients are informative, moderate complexity

**MCTS (Monte Carlo Tree Search)**:
- **Best for**: Complex landscapes, need exploration
- **Pros**: Balances exploration/exploitation, proven in game AI
- **Cons**: Computationally expensive, many hyperparameters
- **Hyperparameters**:
  - `num_simulations`: 100-1000 (more = better but slower)
  - `exploration_constant`: 1.41 (√2) standard, increase for more exploration
  - `selection_mode`: VisitCount (robust) or BestValue (greedy)
- **Use when**: Computational budget available, complex optimization

**UCT (Upper Confidence Trees)**:
- **Best for**: Simpler alternative to MCTS with depth control
- **Pros**: Simpler than MCTS (no rollout), depth-limited
- **Cons**: Less exploration than MCTS
- **Hyperparameters**:
  - `max_depth`: 10-20 (controls exploration depth)
  - `num_simulations`: 100-500
  - `exploration_constant`: √2 standard
- **Use when**: Want MCTS-like behavior with less complexity

**HOO (Hierarchical Optimistic Optimization)**:
- **Best for**: Global optimization with theoretical guarantees
- **Pros**: Provably converges, theoretically sound
- **Cons**: Requires smoothness knowledge, maintains large tree
- **Hyperparameters**:
  - `rho`: 0.5 typical (region shrinkage rate)
  - `nu1`: 0.1 typical (smoothness parameter)
  - `max_depth`: 10-15
- **Use when**: Need global optimum guarantees, can estimate smoothness

**Modified UCT / Flat UCB**:
- **Best for**: Experimental comparisons, specific use cases
- **Pros**: Variants for specialized scenarios
- **Cons**: Less tested than core algorithms
- **Use when**: Standard algorithms insufficient, research purposes

### Hyperparameter Tuning Advice

**General**:
- Start with default values
- Increase `num_simulations` if time permits
- Decrease `exploration_constant` if too random, increase if too greedy

**For Tree Search (MCTS/UCT)**:
- Monitor tree size with `get_tree_size(state)`
- If tree too large: decrease `max_depth` or `num_simulations`
- If not exploring enough: increase `exploration_constant`

**For HOO**:
- Set `rho` based on problem: 0.5 if regions halve each level
- Set `nu1` based on smoothness: smaller for smoother functions
- Increase `max_depth` for finer-grained search

---

## 15. Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Julia Files** | ~45 |
| **Core Source Files** | 16 |
| **Optimizer Implementations** | 7 (2 classical + 5 tree search) |
| **Jupyter Notebooks** | 7 |
| **Test Files** | 11 |
| **Experimental Scripts** | 14 (in logs_local/) |
| **Lines in functions.jl** | 614 |
| **Core Modules** | 8 (basic, optimization, optimizers, tree_search, statistics) |
| **Archive Files** | 8 |
| **Programming Language** | Julia 1.x |
| **Primary Dependencies** | Oscar.jl, AbstractAlgebra.jl, LinearAlgebra.jl, Printf.jl |
| **Git Branches** | main, update-CLAUDE.md |
| **Latest Major Feature** | Tree search algorithms (PR #22) |
| **Module Structure** | Explicit exports via NAML module |

---

## 16. Exported API Reference

The NAML module explicitly exports the following symbols (from `src/NAML.jl`):

### From `basic/valuation.jl`:
- `valuation`

### From `basic/polydisc.jl`:
- `ValuationPolydisc`, `AbsPolydisc`
- `center`, `radius`, `dim`, `prime`
- `dist`, `children`, `children_along_branch`, `concatenate`
- Note: `join` NOT exported (conflicts with Base.join) - use `NAML.join`

### From `basic/tangent_vector.jl`:
- `ValuationTangent`
- Note: `zero`, `basis_vector` NOT exported (conflict with Base) - use `NAML.zero`, `NAML.basis_vector`

### From `basic/functions.jl`:
- `PolydiscFunction`, `AbsolutePolynomialSum`
- `evaluate_abs`, `directional_exponent`, `directional_derivative`, `grad`, `eval_abs`
- Note: `evaluate` NOT exported (conflicts with Oscar) - use `NAML.evaluate`

### From `optimization/model.jl`:
- `AbstractModel`, `Model`
- `var_indices`, `param_indices`, `set_abstract_model_variable`, `batch_evaluate_init`

### From `optimization/optim_setup.jl`:
- `Loss`, `OptimSetup`
- `eval_loss`, `update_param!`, `step!`

### From `optimization/loss.jl`:
- `MSE_loss_init`, `MPE_loss_init`

### From `optimization/optimizers/greedy_descent.jl`:
- `greedy_descent`, `greedy_descent_init`

### From `optimization/optimizers/gradient_descent.jl`:
- `gradient_param`, `gradient_descent`, `gradient_descent_init`

### From `optimization/optimizers/tree_search/hoo.jl`:
- `HOONode`, `HOOConfig`, `HOOState`
- `hoo_descent`, `hoo_descent_init`
- `get_tree_size`, `get_visited_nodes`, `get_leaf_nodes`

### From `optimization/optimizers/tree_search/mcts.jl`:
- `MCTSNode`, `MCTSConfig`, `MCTSState`
- `SelectionMode`, `VisitCount`, `BestValue`
- `mcts_descent`, `mcts_descent_init`

### From `optimization/optimizers/tree_search/uct.jl`:
- `UCTNode`, `UCTConfig`, `UCTState`
- `uct_descent`, `uct_descent_init`

### From `optimization/optimizers/tree_search/modified_uct.jl`:
- `ModifiedUCTNode`, `ModifiedUCTConfig`, `ModifiedUCTState`
- `modified_uct_descent`, `modified_uct_descent_init`

### From `optimization/optimizers/tree_search/flat_ucb.jl`:
- `FlatUCBNode`, `FlatUCBConfig`, `FlatUCBState`
- `flat_ucb_descent`, `flat_ucb_descent_init`

### From `statistics/frechet.jl`:
- `frechet_mean`

---

## Conclusion

The NAML repository is a sophisticated and actively developed implementation of machine learning over non-Archimedean fields, with particular focus on:

- **Advanced optimization** using state-of-the-art tree search algorithms (MCTS, UCT, HOO)
- **Flexible function algebra** with compositional types and automatic differentiation
- **Clear architectural design** separating math, models, and optimization
- **Comprehensive optimizer suite** from simple greedy descent to theoretically-grounded HOO
- **Interactive demonstrations** through Jupyter notebooks

**Recent Major Advances**:
1. Tree search algorithm suite (5 new optimizers)
2. Massively expanded function algebra (614 lines)
3. Generic valuation support
4. Restructured directory organization
5. Formal test infrastructure

The repository is well-suited for:
- **Researchers** exploring non-Archimedean geometry and p-adic analysis
- **ML practitioners** interested in alternative optimization landscapes
- **Algorithm developers** working on tree search methods
- **Applied mathematicians** studying phylogenetics and tree embeddings

**Current State**: Production-ready core with active research extensions. The codebase is mature enough for serious use while remaining open to experimental features.
