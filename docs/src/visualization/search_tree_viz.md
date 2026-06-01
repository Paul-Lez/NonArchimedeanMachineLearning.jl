# Search Tree Viz

The package provides two complementary tree-search visualizations:

- `visualize_search_tree` builds the existing interactive D3 tree view.
- `plot_search_tree_cone` draws a 3D cone-style view of a fixed or live search
  tree. This first-pass layout is useful for 2D 2-adic examples, where each
  search depth is placed on a lower, wider circle.

```julia
using Oscar
using Plots
using NonArchimedeanMachineLearning

K = PadicField(2, 20)
root_disc = ValuationPolydisc([K(0), K(0)], [0, 0])
root = build_fixed_search_dag(root_disc; max_depth=4, degree=1)

plot_search_tree_cone(root; max_depth=4)
```

By default, cone plots colour nodes by visit count using a rainbow scheme:
violet/blue for low-visit nodes, then green/yellow/orange, up to red for the
most visited nodes. Edges are drawn as segmented gradients from the parent node
colour to the child node colour. The default visit colour scale is logarithmic,
which gives non-root nodes more visible range when the root has many more visits
than the rest of the tree. Use `visit_color_max` to keep the same colour scale
across multiple optimizer snapshots. Pass `visit_color_scheme=:gradient` to
recover the earlier two-colour gradient. Pass `visited_only=true` to hide
unvisited nodes and their incident edges. The cone layout uses a small
parent/child barycenter ordering heuristic so DAG nodes with shared children are
usually drawn close together. The cone surface is hidden by default; pass
`show_surface=true` to draw it as a faint guide. Axis labels, ticks, grid, and
frame are hidden by default; pass `show_axes=true` to show them.

For a rotatable browser/notebook view, load WGLMakie and use the interactive
variant:

```julia
using WGLMakie

fig = plot_search_tree_cone_interactive(root; max_depth=4, visited_only=true)
served = serve_search_tree_cone_interactive(fig)
println(served.url)
wait(served.server)
```

This live Bonito server is the recommended local workflow for interaction. The
Julia process must keep running while the browser tab is open.

You can also export a standalone HTML snapshot:

```julia
using WGLMakie

fig = plot_search_tree_cone_interactive(root; max_depth=4, visited_only=true,
                                        offline=true)
save_search_tree_cone_interactive_html("search_tree.html", fig)
```

If your browser refuses to run local HTML scripts from a double-clicked file, serve
the directory and open the page through localhost:

```bash
python3 -m http.server 8765
```

Then visit `http://localhost:8765/search_tree.html`.

```@autodocs
Modules = [NonArchimedeanMachineLearning]
Pages   = ["visualization/search_tree_viz.jl"]
```
