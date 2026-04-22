# Release cleanup report

Generated: 2026-04-22

Scope: `src/`, the `src/visualization/README.md` file, and the Documenter entrypoints that expose source docstrings. This began as a review report; the status notes below track follow-up cleanup work.

## Resolved

- Point 1 in the suggested order is resolved: `eval_abs` was unexported, optimizer return docs were updated to include `converged::Bool`, `gradient_data` was removed, and the `OptimSetup` optimiser contract docs now describe `(new_param, new_state, converged)`.
- Point 2 in the suggested order is resolved: the UCT, modified UCT, and flat UCB source-level test helpers were removed from `src/` and converted into testsets in `test/tree_search_algorithms.jl`.

## Release blockers

- `src/NAML.jl:1` has no module docstring. Since this is the only Julia module in the package, add a package-level docstring before `module NAML`.
- The docs build completed but emitted a Documenter warning for 82 docstrings not included in the manual. Most are from exported tree-search APIs in `hoo.jl`, `uct.jl`, `modified_uct.jl`, and `flat_ucb.jl`, which have no docs pages in `docs/make.jl`.

## Formatting and documentation conventions

- Add a `.JuliaFormatter.toml` or document the formatting style. Current source has many lines over 100 characters and section-banner styles vary by file.
- Remove trailing whitespace in:
  - `src/basic/functions.jl:834`
  - `src/optimization/optim_setup.jl:28`
  - `src/optimization/optim_setup.jl:30`
- Decide whether public docs should use ASCII-only identifiers in examples. Several files use Unicode math symbols and identifiers (`θ`, `φ`, `c₀`, `aᵢ`, arrows, etc.). This is readable, but it should be an intentional style.
- Replace in-progress prose in public docstrings and comments: `TODO`, `For now`, `NEW INTERFACE`, `DEPRECATED`, `dummy`, `placeholder`, `doesn't make sense`, `less cursed`, and similar phrases should become precise limitations or linked issues.
- Prefer `@doc raw"""..."""` consistently for docstrings that contain LaTeX. `doo.jl` and `loss_landscape.jl` use bare triple-quoted strings while most of the package uses `@doc raw`.

## `src/NAML.jl`

- Add a module docstring that explains the package, the p-adic polydisc convention, and the main API groups.
- Re-check exports against implemented and documented public APIs. Several exported tree-search symbols are not present in the manual.
- The export list mixes stable user APIs with low-level evaluator structs. Decide which evaluator types are truly public, then document or unexport the rest.

## `src/basic/`

### `valuation.jl`

- Convert the file header comments into a docstring or a short internal comment that points to the module-level docs.
- `unit(a::padic)` is documented as generic, but the implementation returns `a.u` (`src/basic/valuation.jl:77`). Clarify that it is Oscar/Nemo-specific or add a truly generic extension point.
- The `Base.abs(a::padic)` docs describe custom type support, but only `padic` is implemented here. Make the extension guidance explicit.

### `valued_point.jl`

- Address or move the release-facing TODO at `src/basic/valued_point.jl:22` about making `ValuedFieldPoint` a proper `AbstractAlgebra.jl` ring.
- Add docstrings or internal-only guidance for the Base arithmetic, comparison, `hash`, `show`, conversion, and promotion methods. These are user-visible once `ValuedFieldPoint` is public.
- Consider grouping the Oscar interop methods (`Oscar.zero`, `Oscar.one`, `lift`) under one documented compatibility section.

### `polydisc.jl`

- Add docs for the generic vector constructor at `src/basic/polydisc.jl:35`.
- Fix return-type docs for `center` and `radius`: both return `NTuple`s, but the docstrings say `Vector` (`src/basic/polydisc.jl:156` and `src/basic/polydisc.jl:171`).
- `canonical_center` says centers differing by valuation `> radius` canonicalize together (`src/basic/polydisc.jl:265`), while equality uses `>= radius` (`src/basic/polydisc.jl:228`). Align the mathematical statement.
- Document or mark internal: `base_field`, `subdisc`, `components`, `Base.:<=`, and the `HashedPolydisc` forwarding methods.
- Remove stale commented-out `residue_size` and `aggregate` blocks unless they are deliberately kept as design notes.
- Replace informal comments: `check correctness (max vs min)` (`src/basic/polydisc.jl:431`) and `less cursed implementation` (`src/basic/polydisc.jl:700`).
- Resolve or file issues for TODOs about negative valuations and the possible abstract polydisc hierarchy.
- Constructors mention 0-dimensional polydiscs, but accessors such as `prime` and `base_field` index `center[1]`. Document the limitation or harden the implementation.

### `tangent_vector.jl`

- Add a real file/module docstring; the current two-line header is only a comment.
- Fix docstring field/signature mismatches:
  - `direction` is documented as `Vector{S}` but stored as `ValuationPolydisc{S,T,N}`.
  - `zero(P, Q)` and `basis_vector(P, Q, i)` document `Q::Vector{S}`, but implementations require `Q::ValuationPolydisc`.
- Clarify whether `zero` and `basis_vector` are package-private replacements or public `NAML.zero`/`NAML.basis_vector` helpers, since they are intentionally not exported.

### `functions.jl`

- Fix the file header typo: "contains the basic of functions".
- Add docstrings for the function-composition types at `src/basic/functions.jl:63-114`: `LinearRationalFunction`, `Add`, `Mul`, `Sub`, `Div`, `SMul`, `DifferentiableFunction`, `Comp`, `Constant`, and `Lambda`.
- Review recursive fallback methods:
  - `parent(F::PolydiscFunction)` returns `parent(F)`.
  - `evaluate(f::PolydiscFunction, p::ValuationPolydisc)` returns `evaluate(f, p)`.
  - `batch_evaluate_init(f::PolydiscFunction)` returns `batch_evaluate_init(f)`.
  These should either be removed or changed to clear `MethodError`/`error` fallbacks.
- Remove the public-doc artifact at `src/basic/functions.jl:234`: "Removed duplicate stub definition".
- Replace TODO/in-progress comments around exponentiation, parent semantics, typed evaluators, and directional derivatives with precise issues or release-ready limitations.
- The legacy `batch_evaluate_init` docs say `DEPRECATED`, but there is no deprecation warning. Either add a deprecation path or reword as "legacy compatibility".
- `directional_derivative(fun::PolydiscFunction)` documents all polydisc functions, but the implementation assumes `fun.polys`. Narrow the method/docs or add a proper abstract fallback.
- Add docs for exported evaluator structs and their directional-derivative methods, or make them internal.

## `src/optimization/`

### `optim_setup.jl`

- Replace the all-caps banner at `src/optimization/optim_setup.jl:1` with a concise file docstring; it also has the typo "THIS SECTIONS".
- `OptimSetup` docs omit type parameters `L` and `O`.
- Document `Base.:+(::Loss, ::Loss)` or keep it private.
- `eval_loss` documents a scalar loss, but the implementation assumes `loss.eval` is a batch function returning an indexable collection. Make this convention explicit.

### `loss.jl`

- The header says `Loss` is defined in `basic.jl`, but it is defined in `optimization/optim_setup.jl`.
- Decide whether `MSE_loss_init_new` is release API. Its docstring says "Experimental" and the body has a profiling TODO.
- Remove TODO text from public argument docs for finite `p`/sup loss, or add a clear issue-backed limitation.
- Add docstrings for the `ValuedFieldPoint` lifting overloads at `src/optimization/loss.jl:267` and `src/optimization/loss.jl:282`.
- Several loss definitions use very long nested comprehensions. Reformat for readability before release.

### `model.jl`

- Add a file-level docstring.
- `AbstractModel` is a concrete struct. Either rename it or explicitly explain that "abstract" is mathematical terminology, not a Julia `abstract type`.
- Type and validate `param_info` as a boolean vector; the docs imply `Vector{Bool}` but the field is untyped.
- The `Model` docstring advertises `Model{S,T,N}`, while the actual type is `Model{FS,PS,T,N}`.
- Add docs or internal markers for the `specialise` overloads on `Add`, `Sub`, `Mul`, `Div`, `SMul`, `Comp`, and `Constant`.
- Replace TODOs at `src/optimization/model.jl:377` and `src/optimization/model.jl:483` with release-ready notes or tracked issues.
- `ModelEvaluator` docs describe data and parameter dimensions, but the constructors instantiate `N1` and `N2` as `0,0`. Clarify whether these parameters are placeholders or remove them.
- Replace "NEW TYPED INTERFACE" language in public docs with stable API wording.

## `src/optimization/optimizers/`

### `gradient_descent.jl`

- Document the `ValuedFieldPoint` overload of `gradient_param`.

### `greedy_descent.jl`

- Update the return docs to include `converged::Bool`.
- Fix typo `amond` at `src/optimization/optimizers/greedy_descent.jl:40`.
- Resolve or move the TODO at `src/optimization/optimizers/greedy_descent.jl:75`.

### `random_descent.jl`

- Update return docs to include `converged::Bool`.
- Tone down public wording such as "Expected to perform poorly" and "Should be much better", or move it to examples/benchmarks.
- Confirm whether this baseline optimizer should remain exported for the release.

## `src/optimization/optimizers/tree_search/`

### `doo.jl`

- Convert bare triple-quoted docstrings to the package's `@doc raw"""..."""` style.
- Type signatures in docs often use `DOONode{S,T}`/`DOOState{S,T}` while the code uses `{S,T,N}`.
- Update `doo_descent` return docs to include `converged::Bool`.
- TODOs about `PriorityQueue` appear in both comments and public docs. Keep the limitation but move implementation tasks to an issue.

### `hoo.jl`

- Add a Documenter page and include it in `docs/make.jl`; exported HOO docstrings currently trigger Documenter warnings.
- `HOOConfig` field docs say the value transform defaults to identity, but the constructor defaults to `loss -> 1.0 / (loss + 1e-10)`.
- Update `hoo_descent` return docs to include `converged::Bool`.
- Decide whether debugging helpers such as `print_tree_stats` are public API, then document or hide them consistently.

### `mcts.jl`

- `MCTSConfig` docs say `persist_tree::Bool=false`, but the constructor default is `persist_tree=true`.
- Selection-mode docs mention only `VisitCount` and `BestValue` in some places, but `BestLoss` is implemented and exported.
- Resolve or reword the convergence TODO at `src/optimization/optimizers/tree_search/mcts.jl:687`.

### `uct.jl`

- Add a Documenter page and include it in `docs/make.jl`.
- Replace "shouldn't happen" comments with explicit fallback rationale.

### `modified_uct.jl`

- Add a Documenter page and include it in `docs/make.jl`.
- Document that `total_nodes` assumes a binary tree approximation, even though p-adic branching may differ.
- Replace "shouldn't happen" comments with explicit fallback rationale.

### `flat_ucb.jl`

- Add a Documenter page and include it in `docs/make.jl`.
- Document the approximation behind `total_nodes = 2^(max_depth + 1)`.
- Replace "should not happen in practice" comments with precise invariants or error handling.

### `value_transforms.jl`

- Add a docstring for `DEFAULT_VALUE_TRANSFORM`.
- Document transform domains and numerical edge cases. For example, `inverse_transform` can still misbehave if `loss + epsilon <= 0`.
- State consistently that transforms map loss to a value/reward where larger is better.

### `dag_mcts.jl`

- `DAGMCTSConfig` field docs omit `track_parents`, although the struct and constructor include it.
- Selection-mode docs mention only `VisitCount` and `BestValue` in some places, but `BestLoss` is implemented and exported.
- Remove TODO text from the public config docstring about strict mode and `max_children`; reword as current limitations.
- Update `dag_mcts_descent` return docs to include `converged::Bool`.
- Reword the TODO at `src/optimization/optimizers/tree_search/dag_mcts.jl:834`.

## `src/statistics/`

### `frechet.jl`

- Add a file-level docstring and use the accented spelling consistently: `Fréchet`.
- Remove informal prose at `src/statistics/frechet.jl:27`: "surely there's a cleaner way of doing this."
- The second method's docs mention a "workaround" and "dummy model". Reword as an implementation note or refactor before release.
- Remove unused locals (`mean_point`, `mean_radius`, and the polynomial ring binding) unless they are placeholders for intended work.
- Add argument validation for empty input and inconsistent dimensions.

### `least_squares.jl`

- Remove the leading blank line and add a file-level docstring.
- Clarify that "least squares" here is implemented through the package's absolute-polynomial/loss machinery, not ordinary Euclidean floating-point least squares.
- Add validation for empty data and mismatched input/output dimensions.
- Consider typing `loss_terms` and `residual_polys` to avoid `Vector{Any}` in release code.

## `src/visualization/`

### `README.md`

- Replace "For now this just works for dimension 1: in higher dimensions the pictures looks weird" with a precise limitation.
- Update `ConvexHullTree{S,T}` to `ConvexHullTree{S,T,N}`.
- Keep the file-organization section in sync with `loss_landscape.jl`, or remove it to avoid drift.

### `loss_landscape.jl`

- The top docstring is useful, but because there is no visualization submodule it is effectively a file docstring. Decide whether to create a submodule or keep this as internal orientation.
- Replace the placeholder example `loss_fn(disc) = ... # your loss function` with a runnable example.
- `geodesic_interpolation` returns a `ValuationPolydisc` with `Float64` radii representing actual radii, while the rest of the package convention says radii are valuation radii. Clarify the convention or rename/adjust the function.
- `extract_spanning_tree` docs mention a virtual root, but the implementation picks a canonical root and attaches other roots to it.
- `plot_tree_with_loss` accepts `leaf_labels`, but the docstring signature and keyword list omit it.
- Remove unused variables such as `mean_loss` in `plot_loss_landscape` and `z_range` in `plot_tree_with_loss`.
- Replace the "dummy heatmap" colorbar comment with a precise plotting workaround note.
- The functions require `Plots` to be loaded in `Main`; keep this documented, and consider an optional dependency/extension approach for release.

### `search_tree_viz.jl`

- Add docstrings or internal comments for the normalization helpers if they remain visible in generated docs.
- Fix the public example: it calls `mcts_descent_init(param, loss, 1, config)`, but the current signature is `mcts_descent_init(param, loss, config)`.
- Clarify that `D3Trees` is a hard dependency of loading `NAML`, not just an optional visualization helper.

## Documentation build notes

- Command run: `julia --project=docs docs/make.jl`.
- Result: build completed.
- Important warning: 82 docstrings are not included in the manual. This is mainly because `docs/make.jl` lacks pages for `hoo.jl`, `uct.jl`, `modified_uct.jl`, and `flat_ucb.jl`.
- The docs pages are currently almost all bare `@autodocs` stubs. Before full release, add short narrative pages for the main user workflows, especially polydiscs, loss functions, models, and optimizer selection.

## Suggested order of work

1. [x] Fix the hard API/doc mismatches: missing `eval_abs`, optimizer return docs, `gradient_data`, and `OptimSetup` contract docs.
2. [x] Move in-source tests from tree-search files into `test/`.
3. [ ] Add missing tree-search docs pages and rerun Documenter until the "docstrings not included" warning is understood or gone.
4. [ ] Clean TODO/slop language and stale comments in `basic/functions.jl`, `polydisc.jl`, `loss.jl`, `model.jl`, and visualization.
5. [ ] Add a formatting config and run JuliaFormatter once the style is agreed.
