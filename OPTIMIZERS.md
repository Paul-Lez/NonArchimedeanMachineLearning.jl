# NAML Optimizers - Detailed Documentation

This document provides detailed information about all optimization algorithms available in NAML.

## Table of Contents
1. [Classical Optimizers](#classical-optimizers)
2. [Tree Search Optimizers](#tree-search-optimizers)
3. [Algorithm Comparison](#algorithm-comparison)
4. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Classical Optimizers

### Greedy Descent

**File:** `src/optimization/optimizers/greedy_descent.jl`

**Algorithm:**
1. Generate all children of current polydisc
2. Evaluate loss at each child
3. Move to child with minimum loss
4. Repeat until convergence

**Initialization:**
```julia
greedy_descent_init(param, loss, next_branch, settings)
```

**Parameters:**
- `param::ValuationPolydisc` - Initial parameter polydisc
- `loss::Loss` - Loss function
- `next_branch::Int` - Which branch to descend (for strict mode)
- `settings::Tuple{Bool, Int}` - `(strict, degree)`
  - `strict=true`: Descend along one branch at a time (sequential coordinates)
  - `strict=false`: Consider all degree-dimensional branching directions
  - `degree`: Child generation degree

**Pros:**
- Simple and stable
- No hyperparameters to tune
- Fast per-step
- Follows tree structure naturally

**Cons:**
- Gets stuck in local minima
- No exploration
- No gradient information used

**Use when:**
- Quick baseline needed
- Loss landscape is smooth
- Simple problems

**Example:**
```julia
# Non-strict mode: explore all directions
optim = greedy_descent_init(param, loss, 1, (false, 1))

# Strict mode: descend one coordinate at a time
optim = greedy_descent_init(param, loss, 1, (true, 1))

for i in 1:100
    step!(optim)
end
```

---

### Gradient Descent

**File:** `src/optimization/optimizers/gradient_descent.jl`

**Algorithm:**
1. Generate all children of current polydisc
2. Compute gradients at each child direction
3. Move toward child in direction of steepest descent
4. Repeat until convergence

**Initialization:**
```julia
gradient_descent_init(param, loss, state, degree)
```

**Parameters:**
- `param::ValuationPolydisc` - Initial parameters
- `loss::Loss` - Loss function
- `state` - Optimizer state (typically `nothing`)
- `degree::Int` - Child generation degree

**Functions:**
```julia
gradient_param(model, val, tangent)
```
Compute gradient w.r.t. model parameters.

**Pros:**
- Uses directional information
- More sophisticated than greedy
- Works well for smooth landscapes

**Cons:**
- Still limited to discrete children
- Can get stuck in local minima
- No continuous descent

**Use when:**
- Gradients are informative
- Moderate problem complexity
- Smooth loss landscape

**Example:**
```julia
optim = gradient_descent_init(param, loss, nothing, 1)

for i in 1:100
    step!(optim)
end
```

---

## Tree Search Optimizers

All tree search algorithms balance **exploration** (trying new regions) and **exploitation** (refining promising regions).

### MCTS (Monte Carlo Tree Search)

**File:** `src/optimization/optimizers/tree_search/mcts.jl`

**Algorithm:** Classic MCTS with four phases:
1. **Selection**: Use UCB1 to traverse tree toward promising nodes
2. **Expansion**: Add new child nodes to tree
3. **Simulation**: Random rollout from expanded node
4. **Backpropagation**: Update statistics along path to root

**Configuration:**
```julia
MCTSConfig(;
    num_simulations=100,
    exploration_constant=1.41,
    degree=1,
    max_children=nothing,
    strict=false,
    value_transform=default_value_transform,
    selection_mode=VisitCount
)
```

**Parameters:**
- `num_simulations::Int` - Simulations per optimization step (default: 100)
- `exploration_constant::Float64` - UCB1 constant `c`, typically √2 ≈ 1.41
- `degree::Int` - Child generation degree
- `max_children::Union{Int,Nothing}` - Limit children per node
- `strict::Bool` - Single-branch mode
- `value_transform::Function` - Loss → value mapping (default: `loss -> 1/(loss + 1e-10)`)
- `selection_mode::SelectionMode` - `VisitCount` or `BestValue`

**UCB1 Formula:**
```
UCB1(node) = X̄(node) + c * √(2 * ln(parent_visits) / visits(node))
```
where `X̄` is average value, `c` is exploration constant.

**Selection Modes:**
- `VisitCount`: Select node with most visits (robust)
- `BestValue`: Select node with best average value (greedy)

**Pros:**
- Proven in game AI (AlphaGo, etc.)
- Good exploration/exploitation balance
- Robust to noise

**Cons:**
- Computationally expensive
- Many hyperparameters
- Memory grows with tree

**Use when:**
- Computational budget available
- Complex optimization landscape
- Need strong exploration

**Example:**
```julia
config = MCTSConfig(
    num_simulations=200,
    exploration_constant=1.41,
    degree=1,
    selection_mode=VisitCount
)
optim = mcts_descent_init(param, loss, 1, config)

for i in 1:50
    step!(optim)
end
```

---

### DAG-MCTS (MCTS with Transposition Tables)

**File:** `src/optimization/optimizers/tree_search/dag_mcts.jl`

**Key Difference:** Recognizes that polydisc space is a DAG, not a tree. Multiple action sequences can reach the same polydisc state.

**Algorithm:** MCTS adapted for DAGs:
1. **Global transposition table**: Maps polydisc → node instance
2. **No single parent**: Nodes track all parents (or none for pure path-based approach)
3. **Path-based backpropagation**: Uses explicit path stack instead of parent pointers
4. **Shared statistics**: All paths to same node share visit/value counts

**Configuration:**
```julia
DAGMCTSConfig(;
    num_simulations=100,
    exploration_constant=1.41,
    degree=1,
    value_transform=default_value_transform,
    persist_table=false,
    selection_mode=VisitCount
)
```

**Parameters:**
- Similar to MCTS, plus:
- `persist_table::Bool` - Persist transposition table across steps
  - `true`: Reuse learned information (may grow large)
  - `false`: Fresh search each step

**Data Structure:**
```julia
DAGMCTSNode{S,T,N}
    polydisc::ValuationPolydisc{S,T,N}
    parents::Vector{DAGMCTSNode{S,T,N}}
    children::Dict{Int, DAGMCTSNode{S,T,N}}
    visits::Int
    total_value::Float64
    is_expanded::Bool
```

**Utilities:**
```julia
get_dag_stats(state) -> Dict
```
Returns:
- `:total_nodes` - Number of unique polydiscs explored
- `:total_edges` - Number of parent-child relationships
- `:reuse_count` - How many times nodes were reused
- `:max_parents` - Maximum parents any node has

```julia
print_dag_stats(state)
```
Pretty-print DAG statistics.

```julia
verify_transposition_table(state)
```
Check transposition table consistency (debugging).

**Advantages over standard MCTS:**
- **Massive memory savings**: Single node for polydiscs reached via multiple paths
- **Better exploration**: Aggregated statistics from all paths
- **Scales to higher dimensions**: Crucial for problems where many paths converge

**Use when:**
- High-dimensional problems (≥4D)
- Memory is a concern
- Many action sequences lead to same state

**Example:**
```julia
config = DAGMCTSConfig(
    num_simulations=100,
    exploration_constant=1.41,
    degree=1,
    persist_table=true
)
optim = dag_mcts_descent_init(param, loss, 1, config)

for i in 1:50
    step!(optim)
    if i % 10 == 0
        stats = get_dag_stats(optim.state)
        println("Nodes: $(stats[:total_nodes]), Reuse: $(stats[:reuse_count])")
    end
end
```

---

### UCT (Upper Confidence Trees)

**File:** `src/optimization/optimizers/tree_search/uct.jl`

**Algorithm:** Simpler than MCTS - no random rollout phase.
1. **Selection + Expansion**: Traverse tree using UCB, expand at leaf
2. **Direct Evaluation**: Evaluate loss at expanded node
3. **Backpropagation**: Update path statistics

**Configuration:**
```julia
UCTConfig(;
    max_depth=10,
    num_simulations=100,
    exploration_constant=1.41,
    degree=1,
    strict=false,
    value_transform=default_value_transform
)
```

**Parameters:**
- `max_depth::Int` - Maximum tree depth (default: 10)
- Other parameters similar to MCTS

**Data Structure:**
```julia
UCTNode{S,T}
    polydisc::ValuationPolydisc{S,T}
    parent::Union{UCTNode{S,T}, Nothing}
    children::Vector{UCTNode{S,T}}
    depth::Int
    visits::Int
    total_value::Float64
    is_expanded::Bool
```

**Pros:**
- Simpler than MCTS (no rollout)
- Depth control prevents unbounded expansion
- Often faster convergence for smooth landscapes

**Cons:**
- Less exploration than MCTS
- Depth limit can be restrictive

**Use when:**
- Want MCTS-like behavior with less complexity
- Depth control is important
- Smooth loss landscapes

**Example:**
```julia
config = UCTConfig(
    max_depth=15,
    num_simulations=100,
    exploration_constant=1.41
)
optim = uct_descent_init(param, loss, nothing, config)

for i in 1:50
    step!(optim)
end
```

---

### HOO (Hierarchical Optimistic Optimization)

**File:** `src/optimization/optimizers/tree_search/hoo.jl`

**Algorithm:** Theoretically-grounded global optimization from Bubeck et al. (2011).

Maintains optimistic upper bounds (B-values) for each node:
```
B(h,i,n) = μ̂(h,i) + √(2ln(n)/N(h,i)) + ν₁ρʰ
```
where:
- `μ̂(h,i)`: Empirical mean at node (h,i)
- `n`: Total samples taken
- `N(h,i)`: Visits to node (h,i)
- `ρ`: Shrinkage rate (0 < ρ < 1)
- `ν₁`: Smoothness parameter (ν₁ > 0)
- `h`: Depth in tree

**Algorithm Steps:**
1. Compute B-value for all nodes
2. Select node with maximum B-value
3. Evaluate loss at that node
4. Update statistics
5. Repeat

**Configuration:**
```julia
HOOConfig(;
    rho=0.5,
    nu1=0.1,
    max_depth=10,
    degree=1,
    strict=false,
    value_transform=default_value_transform
)
```

**Parameters:**
- `rho::Float64` - Shrinkage rate ρ ∈ (0,1) (default: 0.5)
  - Rate at which polydisc diameter shrinks: `diameter(depth h) ~ ρʰ`
  - Smaller ρ = faster shrinkage
- `nu1::Float64` - Smoothness parameter ν₁ > 0 (default: 0.1)
  - Function variation within polydiscs: `|f(x) - f(y)| ≤ ν₁ρʰ`
  - Smaller ν₁ = smoother function
- `max_depth::Int` - Maximum tree depth (default: 10)

**Data Structure:**
```julia
HOONode{S,T}
    polydisc::ValuationPolydisc{S,T}
    depth::Int
    position::Int
    parent::Union{HOONode{S,T}, Nothing}
    children::Vector{HOONode{S,T}}
    visits::Int
    sum_values::Float64
    is_expanded::Bool
```

**Utilities:**
```julia
get_tree_size(state) -> Int
get_visited_nodes(state) -> Int
get_leaf_nodes(state) -> Int
```

**Theoretical Guarantees:**
Provably converges to global optimum under smoothness assumptions. Convergence rate depends on ρ, ν₁, and function smoothness.

**Pros:**
- Theoretically sound
- Guaranteed convergence to global optimum
- No random rollout

**Cons:**
- Requires knowledge of smoothness (ν₁) and shrinkage (ρ)
- Less empirically tuned than MCTS
- Maintains large tree

**Use when:**
- Want theoretical guarantees
- Can estimate smoothness parameters
- Global optimization required
- Benchmarking against principled baseline

**Tuning Advice:**
- `rho=0.5`: Diameter halves each level (typical for p-adic branching)
- `nu1`: Estimate from loss function variation
  - Start with 0.1
  - Decrease if function is smoother
  - Increase if function varies rapidly
- `max_depth`: Set based on desired precision

**Example:**
```julia
config = HOOConfig(
    rho=0.5,
    nu1=0.1,
    max_depth=15
)
optim = hoo_descent_init(param, loss, 1, config)

for i in 1:100
    step!(optim)
    if i % 20 == 0
        tree_size = get_tree_size(optim.state)
        println("Step $i: Loss = $(eval_loss(optim)), Tree size = $tree_size")
    end
end
```

---

### Modified UCT

**File:** `src/optimization/optimizers/tree_search/modified_uct.jl`

**Description:** Experimental variant of UCT with modified exploration strategy.

**Configuration:**
```julia
ModifiedUCTConfig(;
    max_depth=10,
    num_simulations=100,
    exploration_constant=1.41,
    degree=1,
    strict=false,
    value_transform=default_value_transform
)
```

**Use when:** Experimental comparisons, research purposes.

---

### Flat UCB

**File:** `src/optimization/optimizers/tree_search/flat_ucb.jl`

**Algorithm:** UCB-based search without tree structure. Treats all candidate polydiscs as independent arms (multi-armed bandit).

**Configuration:**
```julia
FlatUCBConfig(;
    num_simulations=100,
    exploration_constant=1.41,
    degree=1,
    strict=false,
    value_transform=default_value_transform
)
```

**Pros:**
- Simple
- No tree overhead

**Cons:**
- No hierarchical structure
- Less efficient than tree methods

**Use when:**
- Comparison baseline
- Very shallow optimization
- Tree structure not helpful

---

## Algorithm Comparison

### Performance Characteristics

| Algorithm | Complexity/Step | Memory | Exploration | Theoretical Guarantees |
|-----------|----------------|--------|-------------|----------------------|
| Greedy Descent | O(p^degree) | O(1) | None | None |
| Gradient Descent | O(p^degree) | O(1) | Gradient-based | None |
| MCTS | O(num_sims × depth) | O(tree_size) | Strong | Empirical |
| DAG-MCTS | O(num_sims × depth) | O(unique_states) | Strong | Empirical |
| UCT | O(num_sims × depth) | O(tree_size) | Moderate | None |
| HOO | O(num_sims) | O(tree_size) | Balanced | Yes (under assumptions) |
| Flat UCB | O(num_sims) | O(num_arms) | Uniform | Multi-armed bandit |

### When to Use Each

**Greedy Descent:**
- ✓ Quick baseline
- ✓ Smooth landscapes
- ✗ Local minima problems

**Gradient Descent:**
- ✓ Informative gradients
- ✓ Moderate complexity
- ✗ Discrete step limitations

**MCTS:**
- ✓ Complex landscapes
- ✓ Computational budget available
- ✗ Memory intensive

**DAG-MCTS:**
- ✓ High-dimensional problems (≥4D)
- ✓ Memory concerns
- ✓ Many paths to same state
- ✗ Overhead for low dimensions

**UCT:**
- ✓ Simpler than MCTS
- ✓ Depth control needed
- ✗ Less exploration

**HOO:**
- ✓ Theoretical guarantees needed
- ✓ Can estimate smoothness
- ✗ Parameter tuning required

---

## Hyperparameter Tuning

### General Guidelines

**num_simulations** (MCTS, UCT, DAG-MCTS):
- Start: 100
- Range: 50-1000
- Higher = better but slower
- Diminishing returns after ~500

**exploration_constant** (c):
- Default: √2 ≈ 1.41
- Range: 0.5-3.0
- Higher = more exploration
- Lower = more exploitation
- Tune based on convergence behavior:
  - Too random → decrease
  - Stuck in local minimum → increase

**degree**:
- Default: 1
- Higher = more children per node
- Exponential growth: p^degree children
- Usually keep at 1

**max_depth** (UCT, HOO):
- Default: 10-15
- Higher = finer-grained search
- Memory grows with depth
- Set based on problem precision needs

**rho** (HOO):
- Default: 0.5
- Should match polydisc diameter shrinkage
- For standard p-adic children: 0.5 is correct

**nu1** (HOO):
- Default: 0.1
- Estimate from function variation
- Start with 0.1, adjust if:
  - Convergence too slow → decrease
  - Too greedy → increase

### Tuning Workflow

1. **Start with defaults:**
   ```julia
   config = DAGMCTSConfig()  # All defaults
   ```

2. **Run short test:**
   ```julia
   optim = dag_mcts_descent_init(param, loss, 1, config)
   for i in 1:20
       step!(optim)
   end
   ```

3. **Check behavior:**
   - Converging too slowly → increase num_simulations
   - Random/unstable → decrease exploration_constant
   - Memory issues → decrease max_depth or use DAG-MCTS

4. **Iterate:**
   - Adjust one parameter at a time
   - Compare performance on held-out validation

### Value Transform

All tree search algorithms support custom value transforms:
```julia
value_transform::Function  # loss::Float64 -> value::Float64
```

**Default:** `loss -> 1 / (loss + 1e-10)`

**Custom examples:**
```julia
# Negative loss (for minimization with max selection)
value_transform = loss -> -loss

# Exponential weighting
value_transform = loss -> exp(-loss)

# Clipped
value_transform = loss -> max(0.0, 1.0 / (loss + 1e-10))
```

**Design consideration:** Higher values should be better. Transform loss (minimize) to value (maximize).

---

## Debugging Tips

### MCTS/UCT Stuck?
- Check `exploration_constant` - may be too low
- Increase `num_simulations`
- Verify loss function is evaluating correctly

### DAG-MCTS Using Too Much Memory?
- Set `persist_table=false`
- Reduce `num_simulations`
- Check for polydisc equality issues (verify hashing)

### HOO Not Converging?
- Adjust `nu1` - may be too large
- Increase `max_depth`
- Verify `rho` matches problem geometry

### General Debugging:
```julia
# Print loss every step
for i in 1:100
    step!(optim)
    println("Step $i: Loss = $(eval_loss(optim))")
end

# Inspect state
println("Current param: ", center(optim.param))
println("Current radius: ", radius(optim.param))

# DAG-MCTS specific
print_dag_stats(optim.state)
verify_transposition_table(optim.state)

# HOO specific
println("Tree size: ", get_tree_size(optim.state))
println("Leaf nodes: ", get_leaf_nodes(optim.state))
```

---

## References

- **MCTS**: Browne et al. (2012) "A Survey of Monte Carlo Tree Search Methods"
- **UCT**: Kocsis & Szepesvári (2006) "Bandit based Monte-Carlo Planning"
- **HOO**: Bubeck et al. (2011) "X-Armed Bandits" / "Optimistic Optimization of Deterministic Functions"
- **DAG-MCTS**: Extension for polydisc optimization (this library)

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Main library documentation
- [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) - Benchmarking infrastructure
- `experiments/paper/benchmark_config.jl` - Example configurations
