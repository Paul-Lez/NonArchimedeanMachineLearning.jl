
@doc raw"""
    make_ordinary_least_squares_loss(data::Vector{Tuple{Vector{S}, Vector{T}}})::Loss where {S, T}

Create an ordinary least squares loss function for linear regression.

Given training data `{(x₁, y₁), ..., (xₙ, yₙ)}`, constructs a loss function
`L(A, b) = Σᵢ ||Axᵢ + b - yᵢ||²` where the parameters are matrix A and vector b.

# Arguments
- `data::Vector{Tuple{Vector{S}, Vector{T}}}`: Training data as `(input_vector, output_vector)` pairs

# Returns
`Loss`: Loss structure with closures for evaluation and gradient computation

# Parameter Ordering
For m-dimensional outputs and n-dimensional inputs, the parameters are ordered as:
- Indices 1 to m*n: Matrix A entries in row-major order [A₁₁, A₁₂, ..., A₁ₙ, A₂₁, ..., Aₘₙ]
- Indices m*n+1 to m*n+m: Vector b entries [b₁, ..., bₘ]

# Notes
The loss function is constructed symbolically using LinearPolynomial and PolydiscFunction operations.
"""
function make_ordinary_least_squares_loss(data::Vector{Tuple{
        Vector{S}, Vector{T}}})::Loss where {S, T}
    # Get dimensions from first data point
    n = length(data[1][1])  # input dimension
    m = length(data[1][2])  # output dimension
    num_params = m * (n + 1)  # m*n for A, m for b

    # Get field elements for constructing zero and one
    zero_elem = data[1][1][1] - data[1][1][1]  # 0 in the field
    one_elem = data[1][1][1] / data[1][1][1]   # 1 in the field

    # Build loss as sum of squared residuals over all data points and output dimensions
    loss_terms = []

    for (x, y) in data
        for i in 1:m
            # Build coefficients for LinearPolynomial representing (Ax + b)ᵢ - yᵢ
            # This is: A_{i,1}*x₁ + ... + A_{i,n}*xₙ + bᵢ - yᵢ
            all_coeffs = []

            for param_idx in 1:num_params
                if param_idx <= m*n
                    # This parameter is A_{row,col}
                    row = div(param_idx - 1, n) + 1
                    col = mod(param_idx - 1, n) + 1
                    if row == i
                        # Coefficient of A_{i,col} is x[col]
                        push!(all_coeffs, x[col])
                    else
                        push!(all_coeffs, zero_elem)
                    end
                elseif param_idx == m*n + i
                    # This parameter is b_i, coefficient is 1
                    push!(all_coeffs, one_elem)
                else
                    # This parameter doesn't appear in this equation
                    push!(all_coeffs, zero_elem)
                end
            end

            # Create polynomial and add its square to loss terms
            poly = LinearPolynomial{S}(all_coeffs, -y[i])
            poly_sum = LinearAbsolutePolynomialSum{S}([poly])
            push!(loss_terms, poly_sum^2)
        end
    end

    # Total loss function: sum of all squared residuals
    loss_function = sum(loss_terms)
    batch_eval = batch_evaluate_init(loss_function)

    function loss_eval(params::Vector{ValuationPolydisc{S, U, N}}) where {U, N}
        return map(batch_eval, params)
    end

    function loss_grad(vs::Vector{ValuationTangent{S, U, N}}) where {U, N}
        return [directional_derivative(loss_function, v) for v in vs]
    end

    return Loss(loss_eval, loss_grad)
end

@doc raw"""
    solve_linear_system(A::Matrix{S}, b::Vector{S}, y::Vector{S})::Loss where S

Create a least squares loss function for solving a linear system.

Given matrix A, vectors b and y, constructs a loss function `L(x) = ||Ax + b - y||²`
where x is the parameter to optimize.

# Arguments
- `A::Matrix{S}`: Coefficient matrix (m × n)
- `b::Vector{S}`: Offset vector (m-dimensional)
- `y::Vector{S}`: Target vector (m-dimensional)

# Returns
`Loss`: Loss structure with closures for evaluation and gradient computation

# Notes
The parameter x is n-dimensional, where n is the number of columns of A.
The loss measures the squared Euclidean norm of the residual (Ax + b - y).
"""
function solve_linear_system(A::Matrix{S}, b::Vector{S}, y::Vector{S})::Loss where {S}
    m, n = size(A)  # m equations, n unknowns

    # Build residual polynomials: (Ax + b - y)ᵢ for each equation i
    residual_polys = []

    for i in 1:m
        # For equation i: A[i,1]*x₁ + ... + A[i,n]*xₙ + (b[i] - y[i])
        coeffs = [A[i, j] for j in 1:n]
        constant = b[i] - y[i]
        poly = LinearPolynomial{S}(coeffs, constant)
        push!(residual_polys, LinearAbsolutePolynomialSum{S}([poly]))
    end

    # Total loss: sum of squared residuals
    loss_function = sum([r^2 for r in residual_polys])
    batch_eval = batch_evaluate_init(loss_function)

    function loss_eval(params::Vector{ValuationPolydisc{S, T, N}}) where {T, N}
        return map(batch_eval, params)
    end

    function loss_grad(vs::Vector{ValuationTangent{S, T, N}}) where {T, N}
        return [directional_derivative(loss_function, v) for v in vs]
    end

    return Loss(loss_eval, loss_grad)
end
