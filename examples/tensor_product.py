"""Tests the tensor product of two functions."""
import numpy as np

import jax.numpy as jnp

import orthojax as ojax


bases = (
    ojax.make_legendre_polynomial(3),
    ojax.make_legendre_polynomial(3),
    ojax.make_legendre_polynomial(3),
)

total_degree = 3
tp = ojax.TensorProduct(total_degree, bases)

print("Total degree: ", tp.total_degree)
print("Number of basis functions: ", tp.num_basis)
print("Terms: ", tp.terms)
print("Number of terms: ", tp.num_terms)

x = np.random.randn(100, 3)
all_phis = tp(x)
print(all_phis.shape)
