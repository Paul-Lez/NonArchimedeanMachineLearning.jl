module NAML

using Oscar
using LinearAlgebra
using AbstractAlgebra
using Nemo

# Include all source files
include("basic/polydisc.jl")
include("basic/tangent_vector.jl")
include("basic/functions.jl")
include("optim/model.jl")
include("optim/basic.jl")
include("optim/gradient_descent.jl")
include("optim/greedy_descent.jl")
include("optim/loss.jl")
include("statistics/frechet.jl")

# Export types
export ValuationPolydisc, AbsPolydisc
export ValuationTangent
export PolydiscFunction, AbsolutePolynomialSum, LinearPolynomial, LinearAbsolutePolynomialSum
export LinearRationalFunction, LinearRationalFunctionSum
export Add, Mul, Sub, Div, SMul, Comp, Constant
export AbstractModel, Model
export Loss, OptimSetup

# Export polydisc functions
export dim, prime, center, radius, base_ring
export join, dist, children, children_along_branch, concatenate, subdisc, components
export padic_abs

# Export tangent vector functions
export zero, basis_vector

# Export function evaluation
export evaluate, evaluate_abs, directional_derivative, directional_exponent
export grad, partial_gradient, batch_evaluate_init

# Export model functions
export var_indices, param_indices, getkeys
export set_abstract_model_variable, set_model_variable
export specialise, update_weights!

# Export optimization functions
export eval_loss, update_param!, step!
export greedy_descent, greedy_descent_init
export gradient_descent, gradient_descent_init
export gradient_param

# Export loss functions
export MSE_loss_init, MPE_loss_init

# Export statistics
export frechet_mean

end # module NAML
