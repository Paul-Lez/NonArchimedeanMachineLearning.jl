# Non-Archimedean Machine Learning 

This repository implements some algorithms for machine learning with inputs 
and parameters in some non-Archimedean field (or more generally in some 
polydisc space over a non-Archimedean field). 

## Content

- Basic structures (polydisc, tangent vectors, absolute polynomials) and API are implemented in the files in folder `src/basic`
- The folder `src/optim` contains the infrastructure for training non-Archimedean models. 
    - `basic.jl` develops some objects and API for setting up and training models,
    - `loss.jl` implements several "standard" loss functions,
    - `greedy_descent.jl` implements a "greedy" descent algorithm,
    - `gradient_descent.jl` implements a version of gradient descent.

## See how this works

To see this in practice, run the demo in `test/cubic_learning_experiment.ipynb`. This notebook contains code for learning the roots (which depend on some parameter `a`) of a cubic.
Another demo experiment can be found in the file `test/polynomial_learning.ipynb`. This second experiment contains code for learning the roots of "random" polynomials of arbitrary degree.
