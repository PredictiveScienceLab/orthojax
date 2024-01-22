"""Tensor product of orthonormal basis."""

__all__ = ["TensorProduct", "_compute_basis_terms"]


import numpy as np
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
import equinox as eqx


def _compute_basis_terms(num_dim, total_degree, degrees):
    """Compute the basis terms.

    The following is taken from Stokhos.

    The approach here for ordering the terms is inductive on the total
    order p.  We get the terms of total order p from the terms of total
    order p-1 by incrementing the orders of the first dimension by 1.
    We then increment the orders of the second dimension by 1 for all of the
    terms whose first dimension order is 0.  We then repeat for the third
    dimension whose first and second dimension orders are 0, and so on.
    How this is done is most easily illustrated by an example of dimension 3:

    Order  terms   cnt  Order  terms   cnt
    0    0 0 0          4    4 0 0  15 5 1
                            3 1 0
    1    1 0 0  3 2 1        3 0 1
        0 1 0               2 2 0
        0 0 1               2 1 1
                            2 0 2
    2    2 0 0  6 3 1        1 3 0
        1 1 0               1 2 1
        1 0 1               1 1 2
        0 2 0               1 0 3
        0 1 1               0 4 0
        0 0 2               0 3 1
                            0 2 2
    3    3 0 0  10 4 1       0 1 3
        2 1 0               0 0 4
        2 0 1
        1 2 0
        1 1 1
        1 0 2
        0 3 0
        0 2 1
        0 1 2
        0 0 3
    """
    # Temporary array of terms grouped in terms of same order
    terms_order = [[] for i in range(total_degree + 1)]

    # Store number of terms up to each order
    num_terms = np.zeros(total_degree + 2, dtype='i')

    # Set order zero
    terms_order[0] = ([np.zeros(num_dim, dtype='i')])
    num_terms[0] = 1

    # The array cnt stores the number of terms we need to
    # increment for each dimension.
    cnt = np.zeros(num_dim, dtype='i')
    for j, degree in zip(range(num_dim), degrees):
        if degree >= 1:
            cnt[j] = 1

    cnt_next = np.zeros(num_dim, dtype='i')
    term = np.zeros(num_dim, dtype='i')

    # Number of basis functions
    num_basis = 1

    # Loop over orders
    for k in range(1, total_degree + 1):
        num_terms[k] = num_terms[k - 1]
        # Stores the inde of the term we are copying
        prev = 0
        # Loop over dimensions
        for j, degree in zip(range(num_dim), degrees):
            # Increment orders of cnt[j] terms for dimension j
            for i in range(cnt[j]):
                if terms_order[k - 1][prev + i][j] < degree:
                    term = terms_order[k - 1][prev + i].copy()
                    term[j] += 1
                    terms_order[k].append(term)
                    num_basis += 1
                    num_terms[k] += 1
                    for l in range(j + 1):
                        cnt_next[l] += 1
            if j < num_dim - 1:
                prev += cnt[j] - cnt[j + 1]
        cnt[:] = cnt_next
        cnt_next[:] = 0
    num_terms[total_degree + 1] = num_basis
    # Copy into final terms array
    terms = []
    for k in range(total_degree + 1):
        num_k = len(terms_order[k])
        for j in range(num_k):
            terms.append(terms_order[k][j])
    terms = np.array(terms, dtype='i')
    return num_basis, terms, num_terms


class TensorProduct(eqx.Module):
    """Tensor product of orthonormal basis.
    
    Args:
        bases: list of bases
    """
    
    bases: list
    num_basis: int
    num_dim: int
    total_degree: int
    terms: list
    num_terms: list

    def __init__(self, total_degree, bases):
        self.total_degree = total_degree
        self.bases = bases
        self.num_dim = len(bases)
        num_basis, terms, num_terms = _compute_basis_terms(
            self.num_dim, total_degree, [b.degree for b in self.bases]
        )
        self.num_basis = num_basis
        self.terms = terms
        self.num_terms = num_terms

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0))
    def __call__(self, x):
        return self.eval(x)
    
    @eqx.filter_jit
    def eval(self, x):
        n = len(self.bases)
        phis = [b.eval(xi) for b, xi in zip(self.bases, x)]
        tmp = jnp.ones(self.num_basis)
        for i in range(self.num_dim):
            tmp = tmp * phis[i][self.terms[:, i]]
        return tmp
    