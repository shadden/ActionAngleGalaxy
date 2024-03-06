import sympy as sp
from sympy import sqrt
import numpy as np
import celmech as cm
from collections import defaultdict
from PoissonSeriesPerturbationTheory import PoissonSeries, PSTerm
from scipy.special import binom
import jax
from jax import config
config.update("jax_enable_x64",True)
import jax.numpy as jnp
from scipy.optimize import root_scalar


def MiyamotoNagai_get_hamiltonian(ics,NL,Na,Nb):
    r"""
    Get Hamiltonian for orbits in a Miyamato-Nagai potential,

    .. math::
        \Phi(R,z) =  -\frac{1}{\sqrt{R^2} + (a + \sqrt{z^2+ b^2})^2}

    Parameters
    ----------
    ics : ndarray
        Array of initial conditions for (R,z,p_R,p_z)
    NL : float
        Value of the angular momentum
    Na : float
        value of the parameter a
    Nb : float
        value of the parameter b

    Returns
    -------
    celmech.hamiltonian.Hamiltonian
        Hamiltonian object
    """
    R,z,pR,pz = sp.symbols("R,z,p_R,p_z",real=True)
    a,b,L = sp.symbols("a,b,L",real=True)
    KE = (pR * pR + pz * pz + (L/R) * (L/R))/2
    z_term = a + sp.sqrt(z*z + b*b)
    PE = -1/sp.sqrt(R * R + z_term * z_term)
    H = KE + PE
    pars = {a:Na,b:Nb,L:NL}
    state = cm.PhaseSpaceState((R,z,pR,pz),ics)
    ham = cm.Hamiltonian(H,pars,state)
    return ham
    

def MiyamotoNagai_L_to_Rc(L,a,b):
    """
    Given a value of angular momentum and potential parameters a and b for a
    Miyamoto-Nagai potential, compute the radius of a circular orbit.

    Parameters
    ----------
    L : float
        angular momentum
    a : float
        potential parameter
    b : float
        potential parameter

    Returns
    -------
    _type_
        _description_
    """
    c = a + b
    PhiEff = lambda R: 0.5 * L*L/R/R - 1/jnp.sqrt(R*R + c*c)
    dPhiEff_dR = jax.grad(PhiEff)
    d2PhiEff_d2R = jax.grad(dPhiEff_dR)
    root_result = root_scalar(dPhiEff_dR,x0 = L*L,fprime=d2PhiEff_d2R)
    assert root_result.converged, "Root finding failed!"
    return root_result.root

from scipy.integrate import quad
def MiyamotoNagai_planar_orbit_action_and_freq(E,L,a,b):
    """
    Compute the action and frequency values for a planar orbit in the
    Miyamoto-Nagai potential

    Parameters
    ----------
    E : float
        energy of the orbit
    L : float
        angular momentum of the orbit
    a : float
        potential parameter
    b : float
        potential parameter

    Returns
    -------
    tuple
        pericenter, apocenter, radial action, and radial frequency of orbit.
    """
    c = a + b
    csq = c * c
    Lsq= L * L
    Esq = E * E
    L4 = Lsq * Lsq
    action_ig = lambda R: jnp.sqrt(2 * (E - 0.5*L*L/R/R + 1/jnp.sqrt(R*R+c*c)))
    freq_ig = lambda R: 1/action_ig(R)
    poly = np.array([1,csq - Lsq/E-1/Esq,0.25*L4/Esq - csq * Lsq / E,0.25*L4*csq/Esq])
    Rp,Rm = np.sqrt(np.roots(poly)[:2])
    Jr = 2 * quad(action_ig,Rm,Rp)[0]/(2*np.pi)
    kappa_inv = 2 * quad(freq_ig,Rm,Rp)[0]/(2*np.pi)
    return Rm,Rp,Jr,1/kappa_inv
from scipy.linalg import solve as lin_solve
from warnings import warn
def MiyamotoNagai_find_vertical_periodic_orbit(pz0,ham,**kwargs):
    """
    Find the initial conditions and frequency of the vertical periodic orbit
    with a given initial vertical momentum, pz0, when z=0.

    Parameters
    ----------
    pz0 : float
        initial vertical momentum
    ham : celmech.hamiltonian.Hamiltonian
        Hamiltonian describing MN potential orbits

    Optional keyword arguments
    --------------------------
    atol : float
        Absolute tolerance of Newton method for finding initial conditions
    rtol : float
        Relative tolerance for Newton's method. Default is sqrt(EPS) where EPS
        is machine preceision.
    max_iter : int
        maximum number of Newton iterations to try. Default 20.

    Returns 
    ------- ndarray
        initial phase space coordinate for vertical periodic orbit
    float
        frequency of orbit
    """
    f = lambda t,y: ham.flow_func(*y).reshape(-1)
    Df = lambda t,y: ham.jacobian_func(*y)
    a,b,L = [ham.H_params[s] for s in sp.symbols("a,b,L",real=True)]
    Rc = MiyamotoNagai_L_to_Rc(L,a,b)
    X = np.array((Rc,0))
    max_iter=kwargs.get('max_iter',20)
    EPS = np.finfo(float).eps
    atol = kwargs.get('atol',1e-10 * Rc)
    rtol = kwargs.get('rtol',np.sqrt(EPS))
    proj_R = np.array([[1,0,0,0],[0,0,1,0]])
    eye2 = np.eye(2)
    for i in range(max_iter):
        y,M,T = integrate_tangent_eq(X,pz0,f,Df)
        g = X - proj_R@y
        Dg = eye2 - proj_R@M@proj_R.T
        dX = lin_solve(Dg,-1 * g)
        if np.all(np.abs(dX) < rtol * np.abs(X) + atol):
            break
        X+=dX
    else:
        warn("Newton method failed to converge in {} iterations".format(max_iter))
    return y,2*np.pi/T        

def F_from_fDf(t,Y,N,f,Df):
    y = Y[:N]
    M = Y[N:].reshape((N,N))
    return np.concatenate((f(t,y),(Df(t,y)@M).reshape(-1)))

from scipy.integrate import solve_ivp

def integrate_tangent_eq(X,pz0,f,Df,return_soln = False):
    y0 = (X[0],0,X[1],pz0)
    N = len(y0)
    M0 = np.eye(N)
    Y0 = np.concatenate((y0,M0.reshape(-1)))
    F = lambda t,Y: F_from_fDf(t,Y,N,f,Df)
    event = lambda t,Y: Y[1] if t>0 else F(0,Y0)[1]
    event.direction = F(0,Y0)[1]
    event.terminal = True
    soln = solve_ivp(F,(0,np.infty),Y0,events=event,dense_output=return_soln,method='DOP853')
    Yf = soln.y_events[0][0]
    Tf = soln.t_events[0][0]
    yf,Mf = Yf[:N],Yf[N:].reshape((N,N))
    if return_soln:
        return (yf,Mf,Tf),soln
    return yf,Mf,Tf

def record_z0_section_points(ic,ham,Tfin,**kwargs):
    f = lambda t,y: ham.flow_func(*y).reshape(-1)
    Df = lambda t,y: ham.jacobian_func(*y)
    N = ham.N_dim
    event = lambda t,Y: Y[1] if t>0 else f(0,Y)[1]
    event.direction = f(0,ic)[1]
    event.terminal = False
    soln = solve_ivp(f,(0,Tfin),ic,events=event,dense_output=True,jac= Df,**kwargs)
    t=soln.t_events[0]
    return soln.sol(t)