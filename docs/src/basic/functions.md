# Functions

## Batch evaluation

Typed evaluators created with `batch_evaluate_init` accept either a single
polydisc or a vector of polydiscs:

```julia
evaluator = batch_evaluate_init(f, ValuationPolydisc{S, T, N})

scalar_value = evaluator(point)
batch_values = evaluator([point1, point2])
```

The batch form returns one value for each input polydisc, in the same order.
Typed evaluators also support batched directional derivatives:

```julia
derivatives = directional_derivative(evaluator, tangent_vectors)
```

This is equivalent to applying the scalar derivative method to each tangent
vector individually.

```@autodocs
Modules = [NonArchimedeanMachineLearning]
Pages   = ["basic/functions.jl"]
```
