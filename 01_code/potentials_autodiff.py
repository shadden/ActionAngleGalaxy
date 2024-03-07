import jax.numpy as jnp
from math import sqrt
import jax
from celmech.poisson_series import PSTerm, PoissonSeries

def create_tuple(n, m):
    """
    Create a tuple with n zeros followed by m ones.

    Parameters:
    n (int): The number of zeros.
    m (int): The number of ones.

    Returns:
    tuple: A tuple containing n zeros followed by m ones.
    """
    # Create the tuple using tuple concatenation
    return (0,) * n + (1,) * m


def get_derivs(f,x0,dmax):
    """
    Compute the partial derivatives of the function, `f`, of a two-dimensional
    input evaluated at the point `x0` up to a maximum order, `dmax`.

    Parameters
    ----------
    f : function
        function to take derivatives of
    x0 : jnp.array
        point at whcih to evaluate derivatives
    dmax : int
        maximum order of derivatives

    Returns
    -------
    dict
        Dictionary storing partial derivative values.
    """
    derivs = {(0,0):f(x0)}
    Dn_f = jax.jacrev(f)
    Dn_f_val = Dn_f(x0)
    derivs[(1,0)] = Dn_f_val[create_tuple(1,0)]
    derivs[(0,1)] = Dn_f_val[create_tuple(0,1)]
    for d in range(2,dmax+1):
        Dn_f = jax.jacfwd(Dn_f)
        Dn_f_val = Dn_f(x0)
        for j in range(d+1):
            i = d-j
            derivs[(i,j)] = Dn_f_val[create_tuple(i,j)]
    return derivs

def get_derivs_func_dict(f,dmax):
    """
    Compute the partial derivatives of the function, `f`, of a two-dimensional
    input evaluated at the point `x0` up to a maximum order, `dmax`.

    Parameters
    ----------
    f : function
        function to take derivatives of
    x0 : jnp.array
        point at whcih to evaluate derivatives
    dmax : int
        maximum order of derivatives

    Returns
    -------
    dict
        Dictionary storing partial derivative values.
    """
    deriv_fns = {(0,0):f}
    for d in range(1,dmax+1):
        for n1 in range(d+1):
            if d-n1 >= 1:
                deriv_fns[(d-n1,n1)] = jax.jacfwd(deriv_fns[(d-n1-1,n1)],argnums=0)
            else:
                deriv_fns[(d-n1,n1)] = jax.jacfwd(deriv_fns[(d-n1,n1-1)],argnums=1)
    return deriv_fns

from scipy.special import binom, gamma
factorial = lambda n: gamma(n+1)
from math import sqrt
def PotentialDerivativesToPoissonSeries(derivs_dict,nmax):
    r"""
    Convert a dictionary containing mixed partial derivatives up to order `nmax`
    of a 2D effective potential expressed in polar coordinates into a collection
    of Poisson series in complex canonical variables with terms grouped by
    order. 

    Parameters
    ----------
    derivs_dict : dict
        Dictionary of partial derivatives with the key (l,m) denoting .. math::
            \frac{\partial^{l+m}} {\partial R^{l}\partial z^m}\Phi_\mathrm{eff}

    nmax : int
        maximum order of derivatives appearing in `derivs_dict`

    Returns
    -------
    dict
        A dictionary with integer keys and PoissonSeries values containing terms
        of a given order, specified by the key.
    """
    kappa_sq = derivs_dict[(2,0)]
    nu_sq = derivs_dict[(0,2)]
    kappa = sqrt(kappa_sq)
    nu = sqrt(nu_sq)
    dR_factor = sqrt(0.5 / kappa)
    dz_factor = sqrt(0.5 / nu)
    Hdict = dict()
    Hdict[2] = PoissonSeries.from_PSTerms(
        [
            PSTerm(kappa,[1,0],[1,0],[],[]),
            PSTerm(nu,[0,1],[0,1],[],[])
        ]
    )
    for n in range(3,nmax+1):
        term_list = []
        for m in range(0,n+1):
            if (n-m)%2:
                continue
            prefactor_nm = 1/factorial(m)/factorial(n-m)
            prefactor_nm *= derivs_dict[(m,n-m)]
            prefactor_nm *= dR_factor**m
            prefactor_nm *= dz_factor**(n-m)
            for p in range(m+1):
                binom_p = binom(m,p)
                for q in range(0,n-m+1):
                    binom_q = binom(n-m,q)
                    C = float(binom_p*binom_q*prefactor_nm)
                    term = PSTerm(C,[p,q],[m-p,n-m-q],[],[])
                    term_list.append(term)
        Hdict[n] = PoissonSeries.from_PSTerms(term_list)
    return Hdict
#
from collections import defaultdict
from celmech.poisson_series import expLinv
def soln(ics,kappa,nu,Hav,chi):
    xR_series,xz_series = [ defaultdict(lambda: PoissonSeries(2,0)) for _ in range(2)]
    xR_series[1]=PoissonSeries.from_PSTerms([PSTerm(1,[1,0],[0,0],[],[])])
    xz_series[1]=PoissonSeries.from_PSTerms([PSTerm(1,[0,1],[0,0],[],[])])

    xR_transformed = PoissonSeries(2,0)
    for order,val in expLinv(xR_series,chi):
        xR_transformed+=val
    xz_transformed = PoissonSeries(2,0)
    for order,val in expLinv(xz_series,chi):
        xz_transformed+=val
