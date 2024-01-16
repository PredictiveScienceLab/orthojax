"""Tensor product of orthonormal basis."""

__all__ = ["TensorProduct"]


import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
import equinox as eqx


class TensorProduct(eqx.Module):
    """Tensor product of orthonormal basis.
    
    Args:
        bases: list of bases
    """
    
    bases: list
    num_dim: int

    def __init__(self, bases):
        self.bases = bases
        self.num_dim = len(bases)

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0))
    def __call__(self, x):
        return self.eval(x)
    
    @eqx.filter_jit
    def eval(self, x):
        n = len(self.bases)
        phis = [b.eval(xi) for b, xi in zip(self.bases, x)]
        tmp = jnp.kron(phis[0], phis[1])
        for i in range(2, n):
            tmp = jnp.kron(tmp, phis[i])
        return tmp