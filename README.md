# Non-Archimedean Machine Learning 

This repository implements some algorithms for machine learning with inputs 
and parameters in some non-Archimedean field (or more generally in some 
polydisc space over a non-Archimedean field). 

## Content

- Basic structures (polydisc, tangent vectors, functions) and API are implemented in the files in folder `src/basic`:
    - `valuation.jl` provides generic valuation and absolute value implementations
    - `polydisc.jl` defines polydiscs and their operations
    - `tangent_vector.jl` implements tangent vectors on polydisc spaces
    - `functions.jl` provides a compositional function algebra for building complex functions
- The folder `src/optimization` contains the infrastructure for training non-Archimedean models:
    - `optim_setup.jl` develops core objects and API for setting up and training models
    - `loss.jl` implements several "standard" loss functions (MSE, MPE)
    - `model.jl` defines AbstractModel and Model structures
    - `optimizers/` subdirectory contains optimization algorithms:
        - `greedy_descent.jl` implements a "greedy" descent algorithm
        - `gradient_descent.jl` implements a version of gradient descent
        - `tree_search/` contains tree search algorithms (MCTS, UCT, HOO, etc.)
- The folder `src/statistics` contains statistical tools like Frechet mean computation

## See how this works

To see this in practice, run the demos in the `experiments/` directory:
- `cubic_learning_experiment.ipynb` - learn the roots (which depend on some parameter `a`) of a cubic
- `polynomial_learning.ipynb` - learn the roots of "random" polynomials of arbitrary degree
- `linear_learning_experiment.ipynb` - linear learning experiments
- `tree_colouring_learning.ipynb` - tree coloring task demonstration
