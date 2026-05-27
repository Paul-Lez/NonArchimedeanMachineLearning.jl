# DAG-MCTS

`DAGMCTSConfig` supports two internal variants:

- `PathStatsDAGMCTS` (default): visit
  counts and values are tracked for root-to-node paths, selection uses DUCB, and
  a transposition cache shares objective evaluations by polydisc.
- `NodeStatsDAGMCTS`: visit counts and
  values are stored directly on shared DAG nodes.

Select the variant when constructing the optimizer:

```julia
path_stats_config = DAGMCTSConfig(variant = PathStatsDAGMCTS)
node_stats_config = DAGMCTSConfig(variant = NodeStatsDAGMCTS)
```

```@autodocs
Modules = [NonArchimedeanMachineLearning]
Pages   = ["optimization/optimizers/tree_search/dag_mcts.jl"]
```
