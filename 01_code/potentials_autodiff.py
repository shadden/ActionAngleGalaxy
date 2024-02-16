import jax.numpy as jnp
import jax
from celmech.poisson_series_manipulate import PSTerm, PoissonSeries

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
    x0 : jjnp.array
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

from scipy.special import binom, gamma
factorial = lambda n: gamma(n+1)

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
    kappa = jnp.sqrt(kappa_sq)
    nu = jnp.sqrt(nu_sq)
    dR_factor = jnp.sqrt(0.5 / kappa)
    dz_factor = jnp.sqrt(0.5 / nu)
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
def lm_lims_to_ELsqI3(l1,l2,mstar,e,F,Fargs=()):
    esq = e*e
    F1 = F(l1,*Fargs)
    F2 = F(l2,*Fargs)
    Fmu = F(mstar,*Fargs)
    A = jnp.array([
        [l1, -0.5 * esq /(l1 - 1), -1],
        [l2, -0.5 * esq /(l2 - 1), -1],
        [mstar, 0.5 * esq / (1-mstar), -1]
    ])
    y = jnp.array([-F1,-F2,Fmu])
    soln = jnp.linalg.solve(A,y)
    return soln

def ELsqI3_to_p0s(E,Lsq,I3,mu0,lmbda0,e,F,Fargs=()):
    numerator_lmbda = lmbda0 * E  - I3  - 0.5 * e*e*Lsq / (lmbda0-1)  + F(lmbda0,*Fargs)
    denom_lmbda = 2 * (lmbda0-1) * (lmbda0-1 + e*e)
    p_lmbda_sq = numerator_lmbda / denom_lmbda
    numerator_mu = I3 - mu0 * E - 0.5 *e*e*Lsq/(1-mu0) + F(mu0,*Fargs)
    denom_mu = 2 * (1-mu0) * (mu0 - 1 + e*e)
    p_mu_sq = numerator_mu / denom_mu
    return p_lmbda_sq, p_mu_sq



from scipy.integrate import quad
def uv_lims_to_actions_freqs(u1,u2,nu,e,U,V):
    E,L,I3 = uv_lims_to_ELI3(u1,u2,nu,e,U,V)
    G = lambda x: Gfn(x,e)
    Jr_ig = lambda u: np.sqrt(E*np.sinh(u)**2 - I3 - 0.5 * L*L / np.sinh(u)**2 / e**2 - U(u,e))
    Jz_ig = lambda v: np.sqrt(E*np.sin(v)**2  + I3 - 0.5 * L *L / np.sin(v)**2 / e**2 + V(v,e))
    
    dJr_dE_ig = lambda u: 0.5 * np.sinh(u)**2 / np.sqrt(E*np.sinh(u)**2 - I3 - 0.5 * L*L / np.sinh(u)**2 / e**2 - U(u,e))
    dJz_dE_ig = lambda v: 0.5 * np.sin(v)**2  / np.sqrt(E*np.sin(v)**2  + I3 - 0.5 * L *L / np.sin(v)**2 / e**2 + V(v,e))

    dJr_dI3_ig = lambda u: -0.5 / np.sqrt(E*np.sinh(u)**2 - I3 - 0.5 * L*L / np.sinh(u)**2 / e**2 - U(u,e))
    dJz_dI3_ig = lambda v:  0.5 / np.sqrt(E*np.sin(v)**2  + I3 - 0.5 * L *L / np.sin(v)**2 / e**2 + V(v,e))

    dJr_dL_ig = lambda u: (- 0.5 * L / np.sinh(u)**2 / e**2) / np.sqrt(E*np.sinh(u)**2 - I3 - 0.5 * L*L / np.sinh(u)**2 / e**2 - U(u,e))
    dJz_dL_ig = lambda v: (- 0.5 * L / np.sin(v)**2 / e**2)  / np.sqrt(E*np.sin(v)**2  + I3 - 0.5 * L*L / np.sin(v)**2 / e**2 + V(v,e))

    Jr,dJr_dE,dJr_dI3,dJr_dL = (np.sqrt(2)*e/np.pi) * np.array([quad(f,u1,u2)[0] for f in [Jr_ig,dJr_dE_ig,dJr_dI3_ig,dJr_dL_ig]])
    Jz,dJz_dE,dJz_dI3,dJz_dL = (2*np.sqrt(2)*e/np.pi) * np.array([quad(f,0.5*np.pi,nu)[0] for f in [Jz_ig,dJz_dE_ig,dJz_dI3_ig,dJz_dL_ig]])

    M = np.array([
        [dJr_dE,dJr_dI3,dJr_dL],
        [dJz_dE,dJz_dI3,dJz_dL],
        [0,0,1]    
    ])
    return np.array([Jr,Jz]), np.linalg.inv(M)