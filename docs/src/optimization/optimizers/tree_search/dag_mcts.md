# DAG-MCTS

`DAGMCTSConfig` supports several internal variants:

- `PathStatsDAGMCTS` (default): visit
  counts and values are tracked for root-to-node paths, selection uses DUCB, and
  a transposition cache shares objective evaluations by polydisc.
- `NodeStatsDAGMCTS`: visit counts and
  values are stored directly on shared DAG nodes.
- `UCT1DAGMCTS`: the UCT1 transposition rule from Childs, Brodeur, and Kocsis.
  Selection uses local move statistics `Q(s,a)` and `N(s,a)` stored on each
  shared node.
- `UCT2DAGMCTS`: the UCT2 transposition rule from the same paper. Selection
  uses the shared child-node value `Q(g(s,a))` for exploitation and the local
  move count `N(s,a)` for exploration.
- `UCTMaxDAGMCTS`: a max-value UCT variant. Selection uses the maximum value
  observed for each shared child node in place of its average value.

Select the variant when constructing the optimizer:

```julia
path_stats_config = DAGMCTSConfig(variant = PathStatsDAGMCTS)
node_stats_config = DAGMCTSConfig(variant = NodeStatsDAGMCTS)
uct1_config = DAGMCTSConfig(variant = UCT1DAGMCTS)
uct2_config = DAGMCTSConfig(variant = UCT2DAGMCTS)
uctmax_config = DAGMCTSConfig(variant = UCTMaxDAGMCTS)
```

```@autodocs
Modules = [NonArchimedeanMachineLearning]
Pages   = ["optimization/optimizers/tree_search/dag_mcts.jl"]
```
