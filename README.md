# Orthojax - Orthogonal bases in Jax

This is a package for orthogonal bases in Jax. Currently, it supports:

- Orthogonal polynomials (polynomial chaos) with respect to an arbitrary measure. This is achieved by reimplementing in Python some of the functions in the `orthpol`` package by Walter Gautschi. The original Fortran code can be found [here](http://www.cs.purdue.edu/archives/2002/wxg/codes/orthpol.f).
- Tensor products of orthogonal bases (product basis).


## Examples

- You can learn more about how the package works [here](https://predictivesciencelab.github.io/advanced-scientific-machine-learning/polynomial_chaos/04_orthpol_demo.html).
- You can find an example to uncertainty propagation [here](https://predictivesciencelab.github.io/advanced-scientific-machine-learning/polynomial_chaos/05_pc_ode_1d.html).
- An example on how to use a tensor product basis can be found in `examples/tensor_product.py`.

## Todo

- Add Fourier basis.
- Add documentation of API.