# Loss Landscape Visualization Module

This module provides tools for visualizing loss landscapes on polydisc parameter spaces in non-Archimedean machine learning.

For now this just works for dimension 1.

## Overview

The loss landscape visualization helps understand optimization in p-adic parameter spaces by:

1. **Building a tree structure** (convex hull) from polydiscs representing parameter regions
2. **Sampling loss values** along geodesics between connected nodes
3. **Visualizing the tree** with edges colored by loss values

## Key Components

### Data Structures

- **`ConvexHullTree{S,T}`**: Represents the convex hull of polydiscs as a tree
  - `nodes`: All polydiscs (input + joins)
  - `children`, `parents`: Tree edges based on containment
  - `leaf_indices`: Original input polydiscs

### Main Functions

#### Convex Hull Computation

- **`convex_hull(discs)`**: Compute the convex hull tree of polydiscs
  - Computes all pairwise joins iteratively
  - Removes duplicates using Berkovich equality
  - Builds parent-child relationships based on containment

#### Loss Sampling

- **`sample_loss_landscape(tree, loss_fn, num_samples)`**: Sample loss along tree edges
  - Interpolates along geodesics between nodes
  - Returns Dict mapping (parent, child) → [(x, loss), ...]

#### Visualization

- **`plot_tree_with_loss(tree, landscape)`**: Main visualization function
  - Nodes positioned by radius level (y-axis)
  - Edges colored by loss values
  - Equal aspect ratio for correct visual levels

- **`plot_tree_simple(tree)`**: Tree structure without loss coloring

## File Organization

The `loss_landscape.jl` file is organized into 5 sections:

1. **Data Structures**: Core types (ConvexHullTree)
2. **Convex Hull Computation**: Building the tree structure
3. **Geodesic Interpolation and Loss Sampling**: Computing loss along paths
4. **Output and Reporting**: Text summaries and CSV export
5. **Tree Visualization**: Plotting functions
