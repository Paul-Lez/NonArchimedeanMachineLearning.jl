"""
Utility functions for absolute sum minimization experiments.

Provides:
1. Random polynomial generation (both AbsolutePolynomialSum and LinearAbsolutePolynomialSum)
2. Loss function creation for minimizing sums of absolute polynomials
"""

using Oscar

"""
    create_absolute_sum_loss(f::NAML.PolydiscFunction{S}, ::Type{VP}) where {S, VP<:NAML.ValuationPolydisc}
    -> NAML.Loss

Create a loss function from a PolydiscFunction using typed evaluators (zero dynamic dispatch).

# Arguments
- `f::NAML.PolydiscFunction{S}`: The function to minimize (e.g., LinearAbsolutePolynomialSum)
- `::Type{VP}`: The polydisc type for typed evaluator construction

# Returns
- A `NAML.Loss` struct that evaluates using typed evaluators
"""
function create_absolute_sum_loss(f::NAML.PolydiscFunction{S}, ::Type{VP}) where {S, VP<:NAML.ValuationPolydisc}
    batch_eval = NAML.batch_evaluate_init(f, VP)
    batch_fn = (params) -> map(batch_eval, params)
    grad_fn = (vs) -> [NAML.directional_derivative(batch_eval, v) for v in vs]
    return NAML.Loss(batch_fn, grad_fn)
end

# Legacy fallback for backward compatibility
function create_absolute_sum_loss(polys::Vector{<:NAML.PolydiscFunction{S}}) where S
    total_sum = sum(polys)
    batch_eval = NAML.batch_evaluate_init(total_sum)
    batch_fn = (params) -> map(batch_eval, params)
    grad_fn = (vs) -> [NAML.directional_derivative(batch_eval, v) for v in vs]
    return NAML.Loss(batch_fn, grad_fn)
end


"""
    generate_random_absolute_sum_problem(p::Int, prec::Int, num_polys::Int,
                                        num_vars::Int, degree::Int)
    -> NAML.Loss

Generate a random absolute sum minimization problem.

Creates random polynomials f₁, ..., fₙ and returns a loss function L(x) = |f₁(x)| + ... + |fₙ(x)|

For linear polynomials (degree=1), uses LinearAbsolutePolynomialSum for efficiency.
For higher degrees, uses AbsolutePolynomialSum.

# Arguments
- `p::Int`: Prime for p-adic field
- `prec::Int`: p-adic precision
- `num_polys::Int`: Number of polynomials in the sum
- `num_vars::Int`: Number of variables (dimension)
- `degree::Int`: Degree of each polynomial (1=linear, 2=quadratic, etc.)

# Returns
- A `NAML.Loss` function that can be minimized
"""
function generate_random_absolute_sum_problem(p::Int, prec::Int, num_polys::Int,
                                             num_vars::Int, degree::Int)
    K = PadicField(p, prec)

    # Construct polydisc type for typed evaluators
    VP = NAML.ValuationPolydisc{PadicFieldElem, Int, num_vars}

    if degree == 1
        # Flatten all linear polynomials into a single LinearAbsolutePolynomialSum
        linear_polys = [generate_random_linear_polynomial(K, num_vars) for _ in 1:num_polys]
        total_func = NAML.LinearAbsolutePolynomialSum(linear_polys)
    else
        # Flatten all polynomials into a single AbsolutePolynomialSum
        var_names = ["x$i" for i in 1:num_vars]
        R, _ = polynomial_ring(K, var_names)

        polys = [generate_random_polynomial(K, num_vars, degree, var_names) for _ in 1:num_polys]
        total_func = NAML.AbsolutePolynomialSum(polys)
    end

    return create_absolute_sum_loss(total_func, VP)
end


"""
    generate_initial_point(num_vars::Int, K::PadicField) -> NAML.ValuationPolydisc

Generate an initial point for optimization (Gauss point).

# Arguments
- `num_vars::Int`: Dimension of the space
- `K::PadicField`: The p-adic field

# Returns
- A `ValuationPolydisc` centered at (1, 1, ..., 1) with radius 0
"""
function generate_initial_point(num_vars::Int, K::PadicField)
    center = ntuple(i -> K(1), num_vars)
    radius = ntuple(i -> 0, num_vars)
    return NAML.ValuationPolydisc(center, radius)
end
