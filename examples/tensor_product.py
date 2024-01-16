"""Tests the tensor product of two functions."""
import numpy as np

import jax
import jax.numpy as jnp

import orthojax as ojax


bases = (
    ojax.make_legendre_polynomial(3),
    ojax.make_legendre_polynomial(3),
    ojax.make_legendre_polynomial(3),
)

tp = ojax.TensorProduct(bases)
x = np.random.randn(100, 3)
all_phis = tp(x)
print(all_phis.shape)
