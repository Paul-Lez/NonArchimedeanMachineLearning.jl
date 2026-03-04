"""
Utility functions for general experiments with the NAML library.

This module provides:
1. Random p-adic number generation with configurable exponent ranges
2. General loss function creation with cutoff functions
3. Polynomial-to-linear problem transformation
4. Cross-entropy loss functions for smooth optimization
5. MSE loss functions
6. Gauss point generation
"""

using Oscar

"""
    generate_random_padic(p::Int, prec::Int, min_exp::Int, num_terms::Int) -> PadicFieldElem

Generate a random p-adic number with specified precision and exponent range.

# Arguments
- `p::Int`: The prime for the p-adic field
- `prec::Int`: The p-adic precision
- `min_exp::Int`: The minimal exponent of p (can be negative, e.g., -2)
- `num_terms::Int`: The number of terms to include in the expansion

# Returns
- A p-adic number of the form Σ aᵢ * p^(min_exp + i) where aᵢ ∈ {0, ..., p-1}

# Example
```julia
# Generate a 2-adic number with precision 20, starting from p^(-2), with 5 terms
x = generate_random_padic(2, 20, -2, 5)
# This gives a number like a₀ * 2^(-2) + a₁ * 2^(-1) + a₂ * 2^0 + a₃ * 2^1 + a₄ * 2^2
```
"""
function generate_random_padic(p::Int, prec::Int, min_exp::Int, num_terms::Int)
    K = PadicField(p, prec)
    p_adic = K(p)  # Cast p to p-adic field for proper inversion when min_exp < 0
    coeffs = rand(0:(p-1), num_terms)
    return sum([K(coeffs[i]) * p_adic^(min_exp + i - 1) for i in 1:num_terms])
end


"""
    mk_cutoff(cutoff_val::Float64) -> Function

Create a step function that returns 0 below cutoff_val and 1 above.

# Arguments
- `cutoff_val::Float64`: The threshold value

# Returns
- A function f: ℝ → {0, 1} where f(x) = 0 if x < cutoff_val, else 1
"""
function mk_cutoff(cutoff_val::Float64)::NAML.DifferentiableFunction
    return NAML.DifferentiableFunction(
        x -> x < cutoff_val ? 0.0 : 1.0,
        x -> 0.0  # Step function has zero derivative
    )
end


"""
    sigmoid(x::Float64, threshold::Float64, scale::Float64) -> Float64

Smooth sigmoid function centered at threshold with given scale.

Maps R → (0, 1) with:
- sigmoid(threshold, threshold, scale) = 0.5
- sigmoid → 0 as x → -∞
- sigmoid → 1 as x → +∞

The scale parameter controls steepness (smaller = steeper).
"""
function sigmoid(x::Float64, threshold::Float64, scale::Float64)::Float64
    return 1.0 / (1.0 + exp(-(x - threshold) / scale))
end


"""
    binary_cross_entropy(p::Float64, y::Float64) -> Float64

Binary cross-entropy loss: -[y * log(p) + (1-y) * log(1-p)]

Arguments:
- p: Predicted probability in (0, 1)
- y: True label in {0, 1}

Returns:
- Cross-entropy loss value (non-negative)

Note: Uses log with epsilon clipping to avoid log(0).
"""
function binary_cross_entropy(p::Float64, y::Float64)::Float64
    # Clip p to avoid log(0)
    eps = 1e-10
    p_clipped = clamp(p, eps, 1.0 - eps)

    if y == 1.0
        return -log(p_clipped)
    elseif y == 0.0
        return -log(1.0 - p_clipped)
    else
        # General case (though y should be 0 or 1)
        return -(y * log(p_clipped) + (1.0 - y) * log(1.0 - p_clipped))
    end
end


"""
    make_cutoff_loss(f::NAML.PolydiscFunction{S},
                     data::Vector{Tuple{T, U}},
                     cutoff_val::Float64) where {S, T, U<:Real} -> NAML.Loss

Create a loss function of the form: L(θ) = Σᵢ |cutoff(|f(xᵢ, θ)|) - yᵢ|²

This is the general loss function for function learning where:
- f is a PolydiscFunction that depends on data variables and parameters
- data is a vector of (input, output) pairs where output is a real number
- cutoff is a step function that thresholds the output

# Arguments
- `f::NAML.PolydiscFunction{S}`: The function to optimize
- `data::Vector{Tuple{T, U}}`: Training data as (input, output) pairs
- `cutoff_val::Float64`: The cutoff threshold value

# Returns
- A `NAML.Loss` struct with evaluation and gradient functions

# Example
```julia
K = PadicField(2, 20)
R, (x, a) = polynomial_ring(K, ["x", "a"])
f = NAML.AbsolutePolynomialSum([x - a])  # Learn parameter a
data = [(K(1), 0.5), (K(2), 1.0)]
loss = make_cutoff_loss(f, data, 0.25)
```
"""
function make_cutoff_loss(f::NAML.PolydiscFunction{S},
                          data::Vector{Tuple{T, U}},
                          cutoff_val::Float64) where {S, T, U<:Real}
    @assert length(data) > 0 "Empty training data"

    cutoff_fun = mk_cutoff(cutoff_val)

    # Create sum of squared errors with cutoff
    loss_terms = Vector{NAML.PolydiscFunction{S}}()
    for (x_val, y_val) in data
        # For each data point, create: (cutoff(|f(x)|) - y)²
        # We need to compose: cutoff with f evaluated at x_val
        # This assumes f can be partially evaluated at x_val
        term = (NAML.Comp(cutoff_fun, f) - NAML.Constant{S}(y_val))^2
        push!(loss_terms, term)
    end

    # Sum all loss terms
    total_loss = sum(loss_terms)

    # Create batch evaluator
    batch_eval = NAML.batch_evaluate_init(total_loss)
    batch_fn = function(params::Vector{NAML.ValuationPolydisc{S, Int64, M}}) where M 
        return map(batch_eval, params)
    end

    return NAML.Loss(batch_fn, x -> 0)
end


"""
    polynomial_to_linear_loss(data::Vector{Tuple{S, T}},
                              degree::Int,
                              cutoff_val::Union{Float64, Nothing} = nothing) where {S<:PadicFieldElem, T}
                              -> NAML.Loss

Transform a polynomial learning problem into a linear learning problem.

Given a polynomial f = a₀ + a₁x + ... + aₙx^n and data points (x, y), this creates
an optimization problem that is linear in the coefficients (a₀, ..., aₙ).

Two cases:
1. **y is p-adic**: Use LinearAbsolutePolynomialSum directly
   Loss term: |a₀ + a₁x + ... + aₙx^n - y|²

2. **y is real**: Use cutoff composition
   Loss term: |cutoff(|a₀ + a₁x + ... + aₙx^n|) - y|²

# Arguments
- `data::Vector{Tuple{S, T}}`: Training data as (x, y) pairs
- `degree::Int`: Degree of the polynomial to fit
- `cutoff_val::Union{Float64, Nothing}`: Cutoff value (required if y is real, ignored if y is p-adic)

# Returns
- A `NAML.Loss` struct suitable for optimization

# Example
```julia
# Case 1: p-adic output
K = PadicField(2, 20)
data_padic = [(K(1), K(2)), (K(2), K(5))]
loss1 = polynomial_to_linear_loss(data_padic, 3, nothing)

# Case 2: real output (requires cutoff)
data_real = [(K(1), 0.0), (K(2), 1.0)]
loss2 = polynomial_to_linear_loss(data_real, 3, 0.25)
```
"""
function polynomial_to_linear_loss(data::Vector{Tuple{S, T}},
                                   degree::Int,
                                   cutoff_val::Union{Float64, Nothing} = nothing) where {S<:PadicFieldElem, T}
    @assert length(data) > 0 "Empty training data"

    # Determine if outputs are p-adic or real
    first_output = data[1][2]
    is_padic_output = first_output isa PadicFieldElem

    if !is_padic_output && isnothing(cutoff_val)
        error("cutoff_val must be provided when y is a real number")
    end

    # Transform data: (x, y) -> ([1, x, x², ..., x^degree], y)
    # The linear function maps (a₀, ..., aₙ) -> a₀·1 + a₁·x + ... + aₙ·x^n
    transformed_data = Vector{Tuple{Vector{S}, T}}()
    for (x, y) in data
        x_powers = [x^i for i in 0:degree]
        push!(transformed_data, (x_powers, y))
    end

    if is_padic_output
        # Case 1: p-adic output - use LinearAbsolutePolynomialSum
        # Create loss: Σᵢ |a₀ + a₁xᵢ + ... + aₙxᵢⁿ - yᵢ|²
        K = parent(first_output)
        loss_terms = Vector{NAML.PolydiscFunction{S}}()

        for (x_powers, y) in transformed_data
            # LinearPolynomial: coefficients for (a₀, ..., aₙ), constant is -y
            linear_poly = NAML.LinearPolynomial(x_powers, -y)
            linear_sum = NAML.LinearAbsolutePolynomialSum([linear_poly])
            # Square the result
            term = linear_sum^2
            push!(loss_terms, term)
        end

        total_loss = sum(loss_terms)
        batch_eval = NAML.batch_evaluate_init(total_loss)
        batch_fn = function(params::Vector{NAML.ValuationPolydisc{S, Int64, M}}) where M
            return map(batch_eval, params)
        end

        return NAML.Loss(batch_fn, x -> 0)
    else
        # Case 2: real output - use cutoff composition
        # Create loss: Σᵢ |cutoff(|a₀ + a₁xᵢ + ... + aₙxᵢⁿ|) - yᵢ|²
        K = parent(data[1][1])
        cutoff_fun = mk_cutoff(cutoff_val)
        loss_terms = Vector{NAML.PolydiscFunction{S}}()

        for (x_powers, y) in transformed_data
            # LinearPolynomial: a₀ + a₁xᵢ + ... + aₙxᵢⁿ
            linear_poly = NAML.LinearPolynomial(x_powers, K(0))
            linear_sum = NAML.LinearAbsolutePolynomialSum([linear_poly])
            # Apply cutoff and subtract y
            term = (NAML.Comp(cutoff_fun, linear_sum) - NAML.Constant{S}(y))^2
            push!(loss_terms, term)
        end

        total_loss = sum(loss_terms)
        batch_eval = NAML.batch_evaluate_init(total_loss)
        batch_fn = function(params::Vector{NAML.ValuationPolydisc{S, Int64, M}}) where M
            map(batch_eval, params)
        end

        return NAML.Loss(batch_fn, x -> 0)
    end
end


"""
    polynomial_to_crossentropy_loss(data::Vector{Tuple{S, T}},
                                     degree::Int,
                                     threshold::Float64,
                                     scale::Float64 = 0.01) where {S<:PadicFieldElem, T<:Real}
                                     -> NAML.Loss

Create a cross-entropy loss function for polynomial learning.

Instead of using a binary cutoff, this uses:
1. Map |f(x)| to probability: p = sigmoid(|f(x)|, threshold, scale)
2. Compute cross-entropy: L = -[y * log(p) + (1-y) * log(1-p)]

This provides a SMOOTH loss landscape compared to the binary cutoff.

# Arguments
- `data::Vector{Tuple{S, T}}`: Training data as (x, y) pairs where y ∈ {0, 1}
- `degree::Int`: Degree of the polynomial to fit
- `threshold::Float64`: Center point of sigmoid (similar to cutoff_val)
- `scale::Float64`: Steepness of sigmoid (default: 0.01, smaller = steeper)

# Returns
- A `NAML.Loss` struct suitable for optimization

# Example
```julia
K = PadicField(2, 20)
data = [(K(1), 0.0), (K(2), 1.0)]
loss = polynomial_to_crossentropy_loss(data, 3, 0.25, 0.01)
```
"""
function polynomial_to_crossentropy_loss(
    data::Vector{Tuple{S, T}},
    degree::Int,
    threshold::Float64,
    scale::Float64 = 0.01
) where {S<:PadicFieldElem, T<:Real}

    @assert length(data) > 0 "Empty training data"
    @assert all(y -> y == 0.0 || y == 1.0, [y for (x, y) in data]) "y values must be 0.0 or 1.0"

    # Transform data: (x, y) -> ([1, x, x², ..., x^degree], y)
    transformed_data = Vector{Tuple{Vector{S}, T}}()
    for (x, y) in data
        x_powers = [x^i for i in 0:degree]
        push!(transformed_data, (x_powers, y))
    end

    K = parent(data[1][1])

    # Create LinearAbsolutePolynomialSum for each data point
    # and compose with sigmoid and cross-entropy

    # For cross-entropy loss, we need to:
    # 1. Evaluate |a₀ + a₁x + ... + aₙx^n|
    # 2. Apply sigmoid to get probability
    # 3. Compute cross-entropy with target y

    # Since we need to compose non-p-adic functions (sigmoid, log),
    # we'll use NAML.Comp and create a custom composition

    loss_terms = Vector{NAML.PolydiscFunction{S}}()

    for (x_powers, y) in transformed_data
        # Create linear polynomial: a₀·1 + a₁·x + ... + aₙ·x^n
        linear_poly = NAML.LinearPolynomial(x_powers, K(0))
        linear_sum = NAML.LinearAbsolutePolynomialSum([linear_poly])

        # Create composition: cross_entropy(sigmoid(|poly|))
        sigmoid_crossent = NAML.DifferentiableFunction(
            abs_val -> binary_cross_entropy(sigmoid(abs_val, threshold, scale), y),
            abs_val -> (sigmoid(abs_val, threshold, scale) - y) / scale
        )

        # Compose with the linear sum
        term = NAML.Comp(sigmoid_crossent, linear_sum)
        push!(loss_terms, term)
    end

    # Sum all loss terms
    total_loss = sum(loss_terms)

    # Create batch evaluator
    batch_eval = NAML.batch_evaluate_init(total_loss)
    batch_fn = function(params::Vector{NAML.ValuationPolydisc{S, Int64, M}}) where M
        return map(batch_eval, params)
    end

    return NAML.Loss(batch_fn, x -> 0)
end

function polynomial_to_valuation_crossentropy_loss(
    data::Vector{Tuple{S, T}},
    degree::Int,
    prime::Int,
    threshold::Float64,
    scale::Float64 = 0.01
) where {S<:PadicFieldElem, T<:Real}

    @assert length(data) > 0 "Empty training data"
    @assert all(y -> y == 0.0 || y == 1.0, [y for (x, y) in data]) "y values must be 0.0 or 1.0"

    # Transform data: (x, y) -> ([1, x, x², ..., x^degree], y)
    transformed_data = Vector{Tuple{Vector{S}, T}}()
    for (x, y) in data
        x_powers = [x^i for i in 0:degree]
        push!(transformed_data, (x_powers, y))
    end

    K = parent(data[1][1])

    # Create LinearAbsolutePolynomialSum for each data point
    # and compose with sigmoid and cross-entropy

    # For cross-entropy loss, we need to:
    # 1. Evaluate |a₀ + a₁x + ... + aₙx^n|
    # 2. Apply sigmoid to get probability
    # 3. Compute cross-entropy with target y

    # Since we need to compose non-p-adic functions (sigmoid, log),
    # we'll use NAML.Comp and create a custom composition

    loss_terms = Vector{NAML.PolydiscFunction{S}}()

    for (x_powers, y) in transformed_data
        # Create linear polynomial: a₀·1 + a₁·x + ... + aₙ·x^n
        linear_poly = NAML.LinearPolynomial(x_powers, K(0))
        linear_sum = NAML.LinearAbsolutePolynomialSum([linear_poly])
        log_fn = NAML.DifferentiableFunction(
            x -> log(1/prime, x),
            x -> -1.0 / (x * log(prime))
        )
        valuative_linear_sum = NAML.Comp(log_fn, linear_sum)

        # Create composition: cross_entropy(sigmoid(|poly|))
        sigmoid_crossent = NAML.DifferentiableFunction(
            abs_val -> binary_cross_entropy(sigmoid(abs_val, threshold, scale), y),
            abs_val -> (sigmoid(abs_val, threshold, scale) - y) / scale
        )

        # Compose with the linear sum
        term = NAML.Comp(sigmoid_crossent, valuative_linear_sum)
        push!(loss_terms, term)
    end

    # Sum all loss terms
    total_loss = sum(loss_terms)

    # Create batch evaluator
    batch_eval = NAML.batch_evaluate_init(total_loss)
    batch_fn = function(params::Vector{NAML.ValuationPolydisc{S, Int64, M}}) where M
        return map(batch_eval, params)
    end

    return NAML.Loss(batch_fn, x -> 0)
end


"""
    polynomial_to_mse_loss(data::Vector{Tuple{S, T}},
                          degree::Int) where {S<:PadicFieldElem, T<:Real}
                          -> NAML.Loss

Create a simple MSE loss function for polynomial learning.

This is a smooth alternative that directly minimizes:
L = Σᵢ (|a₀ + a₁xᵢ + ... + aₙxᵢⁿ| - yᵢ)²

# Arguments
- `data::Vector{Tuple{S, T}}`: Training data as (x, y) pairs
- `degree::Int`: Degree of the polynomial to fit

# Returns
- A `NAML.Loss` struct suitable for optimization

# Example
```julia
K = PadicField(2, 20)
data = [(K(1), 0.5), (K(2), 1.0)]
loss = polynomial_to_mse_loss(data, 3)
```
"""
function polynomial_to_mse_loss(
    data::Vector{Tuple{S, T}},
    degree::Int
) where {S<:PadicFieldElem, T<:Real}

    @assert length(data) > 0 "Empty training data"

    # Transform data
    transformed_data = Vector{Tuple{Vector{S}, T}}()
    for (x, y) in data
        x_powers = [x^i for i in 0:degree]
        push!(transformed_data, (x_powers, y))
    end

    K = parent(data[1][1])
    loss_terms = Vector{NAML.PolydiscFunction{S}}()

    for (x_powers, y) in transformed_data
        # Create linear polynomial
        linear_poly = NAML.LinearPolynomial(x_powers, K(0))
        linear_sum = NAML.LinearAbsolutePolynomialSum([linear_poly])

        # Create MSE term: (|poly| - y)²
        mse_func = NAML.DifferentiableFunction(
            abs_val -> (abs_val - y)^2,
            abs_val -> 2.0 * (abs_val - y)
        )

        term = NAML.Comp(mse_func, linear_sum)
        push!(loss_terms, term)
    end

    # Sum all loss terms
    total_loss = sum(loss_terms)

    # Create batch evaluator
    batch_eval = NAML.batch_evaluate_init(total_loss)
    batch_fn = function(params::Vector{NAML.ValuationPolydisc{S, Int64, M}}) where M
        return map(batch_eval, params)
    end 
    return NAML.Loss(batch_fn, x -> 0)
end


"""
    generate_gauss_point(n::Int, K::PadicField) -> NAML.ValuationPolydisc

Generate the Gauss point in n-dimensional space over field K.

The Gauss point is the polydisc centered at (1, 1, ..., 1) with all radii equal to 0.
This is a standard starting point for optimization in p-adic spaces.

# Arguments
- `n::Int`: Number of dimensions
- `K::PadicField`: The p-adic field

# Returns
- A `ValuationPolydisc` representing the Gauss point

# Example
```julia
K = PadicField(2, 20)
gauss = generate_gauss_point(5, K)
# Returns: Polydisc with center [1, 1, 1, 1, 1] and radius [0, 0, 0, 0, 0]
```
"""
function generate_gauss_point(n::Int, K::PadicField)
    return NAML.ValuationPolydisc(ntuple(i -> K(0), n), ntuple(i -> 0, n))
end


"""
    generate_gauss_point(n::Int, K::PadicField, T::Type) -> NAML.ValuationPolydisc

Generate the Gauss point with custom radius type.

# Arguments
- `n::Int`: Number of dimensions
- `K::PadicField`: The p-adic field
- `T::Type`: Type for the radius (e.g., Int64, Float64)

# Returns
- A `ValuationPolydisc{PadicFieldElem, T}`

# Example
```julia
K = PadicField(2, 20)
gauss = generate_gauss_point(5, K, Float64)
```
"""
function generate_gauss_point(n::Int, K::PadicField, T::Type)
    return NAML.ValuationPolydisc(ntuple(i -> K(0), n), ntuple(i -> zero(0), n))
end


"""
    generate_gauss_point(n_dims::Int, K::PadicField, radius_val::Int) -> NAML.ValuationPolydisc

Generate a Gauss point with custom radius value.

The Gauss point is centered at (1, 1, ..., 1) but with all radii equal to the specified value.
This is useful for starting optimization at different scales.

# Arguments
- `n_dims::Int`: Number of dimensions
- `K::PadicField`: The p-adic field
- `radius_val::Int`: The radius value for all coordinates

# Returns
- A `ValuationPolydisc` with center [1, 1, ..., 1] and radius [radius_val, radius_val, ..., radius_val]

# Example
```julia
K = PadicField(2, 20)
gauss = generate_gauss_point(5, K, 3)
# Returns: Polydisc with center [1, 1, 1, 1, 1] and radius [3, 3, 3, 3, 3]
```
"""
function generate_gauss_point(n::Int, K::PadicField, radius_val::Int)
    return NAML.ValuationPolydisc(ntuple(i -> K(0), n), ntuple(i -> radius_val, n))
end


"""
    polynomial_fitting_data(x::S, y::T, degree::Int) where {S<:PadicFieldElem, T}
                           -> Tuple{Vector{S}, T}

Transform a single data point (x, y) for polynomial fitting.

Converts (x, y) into ([1, x, x², ..., x^degree], y) for use in linear regression.

# Arguments
- `x::S`: Input p-adic value
- `y::T`: Output value (p-adic or real)
- `degree::Int`: Maximum degree of polynomial

# Returns
- A tuple ([1, x, x², ..., x^degree], y)

# Example
```julia
K = PadicField(2, 20)
x = K(3)
y = K(7)
powers, output = polynomial_fitting_data(x, y, 3)
# powers = [1, 3, 9, 27] (in 2-adics)
```
"""
function polynomial_fitting_data(x::S, y::T, degree::Int) where {S<:PadicFieldElem, T}
    x_powers = [x^i for i in 0:degree]
    return (x_powers, y)
end


"""
    generate_random_binary_function(n::Int) -> Vector{Float64}

Generate a random binary-valued function (outputs in {0, 1}).

# Arguments
- `n::Int`: Number of output values to generate

# Returns
- A vector of Float64 values, each either 0.0 or 1.0

# Example
```julia
outputs = generate_random_binary_function(10)
# Might return [0.0, 1.0, 1.0, 0.0, ...]
```
"""
function generate_random_binary_function(n::Int)::Vector{Float64}
    return Float64.(rand(0:1, n))
end


"""
    generate_polynomial_learning_data(p::Int, prec::Int, n_points::Int,
                                     min_exp::Int = 0, num_terms::Int = 10)
                                     -> Vector{Tuple{PadicFieldElem, Float64}}

Generate random data for polynomial learning experiments with distinct x values.

Creates n_points data points where:
- x values are distinct random p-adic numbers
- y values are random binary outputs {0, 1}

# Arguments
- `p::Int`: Prime for p-adic field
- `prec::Int`: p-adic precision
- `n_points::Int`: Number of data points to generate
- `min_exp::Int`: Minimal exponent for p-adic numbers (default: 0)
- `num_terms::Int`: Number of terms in p-adic expansion (default: 10)

# Returns
- Vector of (x, y) tuples with distinct x values

# Example
```julia
data = generate_polynomial_learning_data(2, 20, 5)
# Generates 5 random points with distinct x values
```

# Note
To get more distinct values, increase `num_terms`. The number of possible distinct
values is approximately p^num_terms, so for p=2 and num_terms=10, you can generate
up to ~1000 distinct values.
"""
function generate_polynomial_learning_data(p::Int, prec::Int, n_points::Int,
                                          min_exp::Int = 0, num_terms::Int = 10)
    K = PadicField(p, prec)

    # Generate distinct x values
    x_values = Vector{PadicFieldElem}()
    max_attempts = n_points * 100  # Prevent infinite loop
    attempts = 0

    while length(x_values) < n_points && attempts < max_attempts
        x = generate_random_padic(p, prec, min_exp, num_terms)

        # Check if this x is already in our list
        if !any(existing_x -> existing_x == x, x_values)
            push!(x_values, x)
        end

        attempts += 1
    end

    if length(x_values) < n_points
        error("Could not generate $n_points distinct p-adic numbers after $max_attempts attempts. " *
              "Try increasing num_terms or adjusting the exponent range.")
    end

    # Generate random binary outputs
    y_values = generate_random_binary_function(n_points)

    return [(x, y) for (x, y) in zip(x_values, y_values)]
end


"""
    compute_classification_accuracy(model::NAML.AbstractModel,
                                     data::Vector{Tuple{S, Float64}},
                                     param::NAML.ValuationPolydisc,
                                     threshold::Float64,
                                     scale::Float64) where {S}
                                     -> Float64

Compute classification accuracy for binary classification with sigmoid threshold.

Evaluates the model at each data point with given parameters, applies sigmoid
transformation, and classifies based on 0.5 probability threshold.

# Arguments
- `model::NAML.AbstractModel`: The model to evaluate
- `data::Vector{Tuple{S, Float64}}`: Training data as (input, label) pairs where label ∈ {0, 1}
- `param::NAML.ValuationPolydisc`: Current parameter values
- `threshold::Float64`: Threshold for sigmoid transformation
- `scale::Float64`: Scale for sigmoid transformation

# Returns
- Accuracy as percentage (0.0 to 100.0)

# Example
```julia
K = PadicField(3, 20)
R, (x, a) = polynomial_ring(K, ["x", "a"])
fun = NAML.AbsolutePolynomialSum([x - a])
model = NAML.AbstractModel(fun, [true, false])
data = [(K(1), 0.0), (K(2), 1.0)]
param = NAML.ValuationPolydisc([K(1)], [0])
acc = compute_classification_accuracy(model, data, param, 0.7, 2.0)
```
"""
function compute_classification_accuracy(
    model::NAML.AbstractModel,
    data::Vector{Tuple{S, Float64}},
    param::NAML.ValuationPolydisc,
    threshold::Float64,
    scale::Float64
) where {S}
    # Specialize the model at each data point
    specialized_models = [NAML.specialise(model, [val]) for (val, _) in data]

    # Initialize batch evaluation for each specialized model
    batch_evals = [NAML.batch_evaluate_init(specialized_models[i]) for i in eachindex(specialized_models)]

    # Count correct predictions
    correct = 0
    for (i, (_, y)) in enumerate(data)
        # Evaluate model at current parameters
        pred_float = Float64(batch_evals[i](param))

        # Apply sigmoid transformation (sigmoid already handles the z transformation)
        prob = sigmoid(pred_float, threshold, scale)

        # Classify based on 0.5 threshold
        pred_label = prob >= 0.5 ? 1.0 : 0.0

        # Check if correct
        if pred_label == y
            correct += 1
        end
    end

    # Return accuracy as percentage
    return 100.0 * correct / length(data)
end
