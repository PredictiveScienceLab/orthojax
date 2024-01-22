"""This is where we reimplement orthpol.
Note that for th ebasic algorithms (fejer, lancz), I asked 
ChatGPT to translate the Fortran code to Python.
"""

__all__ = [
    "make_quadrature", "make_orthogonal_polynomial", "make_legendre_polynomial", 
    "make_hermite_polynomial",
    "QuadratureRule",
    "OrthogonalPolynomial", "lancz", "fejer", "symtr", "tr"
]


from collections import namedtuple

import numpy as np
import math

import jax.numpy as jnp
import jax
from jax import value_and_grad, vmap, jit
from jax import lax

from functools import partial

import equinox as eqx



def symtr(t):
    """Implements a tranformation of [-1, 1] to [-Infinity, Infinity]."""
    return t / (1. - t * t)


def tr(t):
    """Implements a transformation of [-1, 1] to [0, Infinity]."""
    return (1. + t) / (1. - t)


QuadratureRule = namedtuple('QuadratureRule', "x w")


def fejer(n):
    """Implements the Fejer algorithm for computing the nodes and weights
    of the n-th degree Fejer quadrature formula.

    Args:
        n (int): The degree of the Fejer quadrature formula.

    Returns:
        Tuple: A tuple containing the nodes and weights of the Fejer quadrature formula.

    Caution: This was translated from Fortran by ChatGPT.
    """
    dpi = 4.0 * np.arctan(1.0)
    nh = n // 2
    np1h = (n + 1) // 2
    dn = float(n)
    
    # Calculate x using cosine function and symmetry properties
    k = np.arange(1, nh + 1)
    x_nh = np.cos(0.5 * (2 * k - 1) * dpi / dn)
    x = np.concatenate((-x_nh[::-1], [0] * (n % 2), x_nh))
    
    # Initialize weights
    w = np.zeros(n)
    
    # Calculate weights using recurrence relations and symmetry
    for k in range(1, np1h + 1):
        dc1 = 1.0
        dc0 = 2.0 * x[k - 1]**2 - 1.0
        dt = 2.0 * dc0
        dsum = dc0 / 3.0
        
        m = np.arange(2, nh + 1)
        dc2 = dc1
        dc1 = dc0
        dc0 = dt * dc1 - dc2
        dsum += np.sum(dc0 / (4 * m**2 - 1))
        
        w[k - 1] = 2.0 * (1.0 - 2.0 * dsum) / dn
        if k != np1h or n % 2 == 0:  # avoid double counting the center for odd n
            w[n - k] = w[k - 1]
    
    return QuadratureRule(x, w)


def make_quadrature(n, left=0, right=1, wf = lambda t: 1.0):
    """Make a quadrature rule for the interval [left, right] with n nodes."""
    x, w = fejer(n)
    if math.isinf(left) and math.isinf(right):
        transformation = symtr
    elif math.isinf(right):
        transformation = lambda t: left + tr(t)
    elif math.isinf(left):
        transformation = lambda t: right - tr(-t)
    else:
        transformation = lambda t: 0.5 * ((right - left) * t + right + left)
    transformation = jit(vmap(value_and_grad(transformation)))
    x_prime, dphi = transformation(x)
    w_prime = w * wf(x_prime) * dphi

    return QuadratureRule(x_prime, w_prime)


class OrthogonalPolynomial(eqx.Module):

    alpha: jax.Array
    beta: jax.Array
    gamma: jax.Array
    quad: QuadratureRule

    @property
    def num_basis(self):
        return self.alpha.shape[0]

    @property
    def degree(self):
        return self.alpha.shape[0] - 1
    
    @property
    def num_terms(self):
        return self.alpha.shape[0]

    def __init__(self, alpha, beta, gamma, quad):
        self.alpha = jnp.array(alpha)
        self.beta = jnp.array(beta)
        self.gamma = jnp.array(gamma)
        self.quad = quad

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0))
    def __call__(self, xi):
        return self.eval(xi)
    
    @eqx.filter_jit
    def eval(self, xi):
        init = (xi, 1.0 / self.gamma[0], (xi - self.alpha[0]) / self.gamma[1])
        xs = jnp.hstack(
            [self.alpha[1:-1, None], self.beta[1:-1, None], self.gamma[2:, None]]
        )
        def f(carry, x):
            xi, phi_prev, phi = carry
            alpha, beta, gamma = x
            phi_next = ((xi - alpha) * phi - beta * phi_prev) / gamma
            return (xi, phi, phi_next), phi_next
        _, phi = lax.scan(f, init, xs)
        return jnp.hstack([[init[1], init[2]], phi])
    
    def normalize(self):
        beta = np.copy(self.beta)
        p = beta.shape[0]
        gamma = np.copy(self.gamma)
        beta[0] = np.sqrt(beta[0])
        gamma[0] = beta[0]
        for i in range(p):
            beta[i] = np.sqrt(beta[i] * gamma[i])
            gamma[i] = beta[i]

        return OrthogonalPolynomial(self.alpha, beta, gamma, self.quad)


def lancz(n, quadrature_rule):
    """Implements the Lanczos algorithm for computing the coefficients of the
    recurrent relations of the orthogonal polynomials.
    
    This was generated by CoPilot from the Fortran code of Gautschi.
    """
    x = quadrature_rule.x
    w = quadrature_rule.w
    alpha = np.zeros(n)
    beta = np.zeros(n)
    dp0 = np.zeros(len(x))
    dp1 = np.zeros(len(x))
    if n <= 0 or n > len(x):
        raise ValueError('n must be between 1 and len(x).')
    else:
        for i in range(len(x)):
            dp0[i] = x[i]
            dp1[i] = 0.0
        dp1[0] = w[0]
        for i in range(len(x) - 1):
            dpi = w[i + 1]
            dgam = 1.0
            dsig = 0.0
            dt = 0.0
            xlam = x[i + 1]
            for k in range(i + 1):
                drho = dp1[k] + dpi
                dtmp = dgam * drho
                dtsig = dsig
                if drho <= 0.0:
                    dgam = 1.0
                    dsig = 0.0
                else:
                    dgam = dp1[k] / drho
                    dsig = dpi / drho
                dtk = dsig * (dp0[k] - xlam) - dgam * dt
                dp0[k] = dp0[k] - (dtk - dt)
                dt = dtk
                if dsig <= 0.0:
                    dpi = dtsig * dp1[k]
                else:
                    dpi = (dt ** 2) / dsig
                dtsig = dsig
                dp1[k] = dtmp
        for k in range(n):
            alpha[k] = dp0[k]
            beta[k] = dp1[k]

        gamma = np.ones(n)

        return OrthogonalPolynomial(alpha, beta, gamma, quadrature_rule)
    

def make_orthogonal_polynomial(degree, left=0.0, right=1.0, wf=lambda xi: 1.0, ncap=100):
    """Make an orthogonal polynomial of degree degree."""
    quadrature_rule = make_quadrature(ncap, left, right, wf)
    return lancz(degree + 1, quadrature_rule).normalize()


def make_legendre_polynomial(degree, ncap=100):
    """Make a Legendre polynomial of degree degree."""
    return make_orthogonal_polynomial(degree, left=-1.0, right=1.0, wf=lambda t: 0.5, ncap=ncap)


def make_hermite_polynomial(degree, ncap=100):
    """Make a Hermite polynomial of degree degree."""
    return make_orthogonal_polynomial(degree, left=-math.inf, right=math.inf, 
                                      wf=lambda t: np.exp(-t * t / 2.0) / np.sqrt(2.0 * np.pi), ncap=ncap)
