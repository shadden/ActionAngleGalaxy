from math import sqrt
import jax.numpy as jnp
import jax

def R_z_to_lambda_mu(R,z,e):
    """
    Convert polar coordinates to elliptic coordiantes.
    """
    esq = e*e
    D = esq*esq + 2 * esq * (R*R - z*z) + (R*R+z*z)**2
    lmbda = (2 - esq + R*R + z*z + sqrt(D))/2
    mu = (2 - esq + R*R + z*z - sqrt(D))/2
    return lmbda,mu

def lambda_mu_to_R_z(lmbda,mu,e):
    bsq = 1-e*e
    R = sqrt((lmbda-1)*(1-mu))/e
    z = sqrt((lmbda-bsq)*(mu-bsq))/e
    return R,z

def R_z_pR_pz_to_lambda_mu_plambda_pmu(R,z,pR,pz,e):
    esq = e*e
    D = esq*esq + 2 * esq * (R*R - z*z) + (R*R+z*z)**2
    lmbda = (2 - esq + R*R + z*z + sqrt(D))/2
    mu = (2 - esq + R*R + z*z - sqrt(D))/2
    bsq = 1-esq
    T = jnp.array([
        [sqrt((1-mu)/(lmbda-1)),sqrt((mu - bsq)/(lmbda-bsq))],
        [-1*sqrt((lmbda-1)/(1-mu)),sqrt((lmbda - bsq)/(mu-bsq))]
    ])
    plambda,pmu = T@jnp.array((pR,pz))
    return lmbda,mu,plambda,pmu

def lambda_mu_plambda_pmu_to_R_z_pR_pz(lmbda,mu,plmbda,pmu,e):
    bsq = 1-e*e
    R = sqrt((lmbda-1)*(1-mu))/e
    z = sqrt((lmbda-bsq)*(mu-bsq))/e
    T = jnp.array([
        [sqrt((1-mu)/(lmbda-1)),sqrt((mu - bsq)/(lmbda-bsq))],
        [-1*sqrt((lmbda-1)/(1-mu)),sqrt((lmbda - bsq)/(mu-bsq))]
    ])
    pR,pz = jnp.linalg.inv(T) @ jnp.array((plmbda,pmu))
    return R,z,pR,pz


def ELsqI3_to_p0s(E,Lsq,I3,lmbda0,mu0,e,F,Fargs=()):
    pl_fn = lambda x: jnp.sqrt((x * E - I3 - 0.5*e*e*Lsq/(x-1) - F(x,*Fargs))/(2*(x-1)*(x-1+e*e)))
    pmu_fn = lambda x: jnp.sqrt((I3 - x * E - 0.5*e*e*Lsq/(1-x) + F(x,*Fargs))/(2*(1-x)*(x-1+e*e)))
    return pl_fn(lmbda0), pmu_fn(mu0)


def Rz_lims_to_ics(Rmin,Rmax,zmax,e,F,Fargs=(),**kwargs):
    lmin,_ = R_z_to_lambda_mu(Rmin,0,e)
    lmax,_ = R_z_to_lambda_mu(Rmax,0,e)
    _,mu_star = R_z_to_lambda_mu(Rmin,zmax,e)
    E,Lsq,I3 = lm_lims_to_ELsqI3(lmin,lmax,mu_star,e,F,Fargs)
    R0 = kwargs.get("R0",Rmin)
    z0 = kwargs.get("z0",0)    
    l0,mu0 = R_z_to_lambda_mu(R0,z0,e)
    pl,pmu = ELsqI3_to_p0s(E,Lsq,I3,l0,mu0,e,F,Fargs)
    R0,z0,pR,pz = lambda_mu_plambda_pmu_to_R_z_pR_pz(l0,mu0,pl,pmu,e)
    return jnp.array((R0,z0,pR,pz)),sqrt(Lsq)

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
    y = jnp.array([F1,F2,Fmu])
    soln = jnp.linalg.solve(A,y)
    return soln

def Rz_lims_to_ELsqI3(Rmin,Rmax,zmax,e,F,Fargs=()):
    lmin,_ = R_z_to_lambda_mu(Rmin,0,e)
    lmax,_ = R_z_to_lambda_mu(Rmax,0,e)
    _,mu_star = R_z_to_lambda_mu(Rmin,zmax,e)
    return lm_lims_to_ELsqI3(lmin,lmax,mu_star,e,F,Fargs)

from scipy.integrate import quad
def lm_lims_to_actions_and_frequencies(l1,l2,mstar,e,F,Fargs=()):
    E,Lsq,I3 = lm_lims_to_ELsqI3(l1,l2,mstar,e,F,Fargs)
    Jr_ig = lambda x: jnp.sqrt((x * E - I3 - 0.5*e*e*Lsq/(x-1) - F(x,*Fargs))/(2*(x-1)*(x-1+e*e)))
    Jz_ig = lambda x: jnp.sqrt((I3 - x * E - 0.5*e*e*Lsq/(1-x) + F(x,*Fargs))/(2*(1-x)*(x-1+e*e)))
    Jr = quad(Jr_ig,l1,l2)[0]
    Jr /= jnp.pi
    Jz = quad(Jz_ig,1-e*e,mstar)[0]
    Jz *= 2/jnp.pi
    L = jnp.sqrt(Lsq)
    
    dJr_dE_ig = lambda x:  0.5 * x / jnp.sqrt((x * E - I3 - 0.5*e*e*Lsq/(x-1) - F(x,*Fargs))) / jnp.sqrt((2*(x-1)*(x-1+e*e)))
    dJr_dI3_ig = lambda x: -0.5    / jnp.sqrt((x * E - I3 - 0.5*e*e*Lsq/(x-1) - F(x,*Fargs))) / jnp.sqrt((2*(x-1)*(x-1+e*e)))
    dJr_dL_ig = lambda x: -0.5 * (e*e*L/(x-1))  / jnp.sqrt((x * E - I3 - 0.5*e*e*Lsq/(x-1) - F(x,*Fargs))) / jnp.sqrt((2*(x-1)*(x-1+e*e)))

    dJz_dE_ig =  lambda x: -0.5 * x  / jnp.sqrt((I3 - x * E - 0.5*e*e*Lsq/(1-x) + F(x,*Fargs))) / jnp.sqrt((2*(1-x)*(x-1+e*e)))
    dJz_dI3_ig = lambda x:  0.5       / jnp.sqrt((I3 - x * E - 0.5*e*e*Lsq/(1-x) + F(x,*Fargs))) / jnp.sqrt((2*(1-x)*(x-1+e*e)))
    dJz_dL_ig =  lambda x: -0.5 * (e*e*L/(1-x)) /   jnp.sqrt((I3 - x * E - 0.5*e*e*Lsq/(1-x) + F(x,*Fargs))) /  jnp.sqrt((2*(1-x)*(x-1+e*e))) 
    
    Jr,dJr_dE,dJr_dI3,dJr_dL = [(1/jnp.pi)*quad(f,l1,l2)[0] for f in (Jr_ig,dJr_dE_ig,dJr_dI3_ig,dJr_dL_ig)]
    Jz,dJz_dE,dJz_dI3,dJz_dL = [(2/jnp.pi)*quad(f,1-e*e,mstar)[0] for f in (Jz_ig,dJz_dE_ig,dJz_dI3_ig,dJz_dL_ig)]
    
    
    mtrx = jnp.array([
        [dJr_dE,dJr_dI3,dJr_dL],
        [dJz_dE,dJz_dI3,dJz_dL],
        [0,0,1]
    ])
    
    return jnp.array((Jr,Jz,L)),jnp.linalg.inv(mtrx)[0]

def Rz_lims_to_actions_and_frequencies(Rmin,Rmax,zmax,e,F,Fargs=()):
    lmin,_ = R_z_to_lambda_mu(Rmin,0,e)
    lmax,_ = R_z_to_lambda_mu(Rmax,0,e)
    _,mstar = R_z_to_lambda_mu(Rmin,zmax,e)
    actions,freqs = lm_lims_to_actions_and_frequencies(lmin,lmax,mstar,e,F,Fargs)
    return actions,freqs
from scipy.optimize import root_scalar

def L_to_Rc(R_guess,L,e,F,Fargs=()):
    Fprime = jax.grad(F)
    fn = lambda x: (x - 1 + e*e) * Fprime(x,*Fargs) + F(1-e*e,*Fargs) - 0.5 * ((x - 1 + e*e) / (x-1))**2 * L * L - F(x,*Fargs)
    lambda_guess = 1 + R_guess**2
    root_soln = root_scalar(fn,x0 = lambda_guess,fprime=jax.grad(fn))
    assert root_soln.converged, "Failed to converge!"
    lambda_root = root_soln.root
    R_root = jnp.sqrt(lambda_root - 1)
    return R_root