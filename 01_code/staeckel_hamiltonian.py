import celmech as cm
import numpy as np
import sympy as sp

def Gfn(x,e):
    return -1 * (x/e) * np.arctan(e * x / np.sqrt(1-e*e))

def F(tau,e):
    """
    The functions appearing in the oblate perfect spheroid potential, which is
    given by 

    .. math::
        \phi(\lambda,\mu) = -(F(\lambda) - F(\mu))/(\lambda - \mu)

    Parameters
    ---------- 
    tau : float
        coordinate symbol
    e : float
        eccentricity of oblate spheroids
    """
    bsq = 1-e*e
    return sp.sqrt(tau-bsq) * sp.acos(sp.sqrt(bsq/tau))

def get_POS_potential_alt(e,u,v,eps):
    pot_lm = lambda l,m : -1 * (F(l+eps,e) - F(m+eps,e))/(l-m)
    l = 1 + e**2 * sp.sinh(u)**2
    m = 1 - e**2 * sp.sin(v)**2
    return pot_lm(l,m)

def get_POS_potential(e,u,v):
    coshu = sp.cosh(u)
    cosv = sp.cos(v)
    sinhu = sp.sinh(u)
    sinv = sp.sin(v)
    U = -coshu * sp.atan(e*coshu/sp.sqrt(1-e*e))/e
    V = -cosv * sp.atan(e*cosv/sp.sqrt(1-e*e))/e
    Phi = (U-V)/(sinv*sinv + sinhu*sinhu)
    return Phi

def get_Staeckel_hamiltonian(ecc,u1,u2,v_star,U,V):
    
    # Set up functional form of Hamiltonian
    u,v = sp.symbols("u,v",real=True)
    pu,pv = sp.symbols("p_u,p_v",real=True)
    e,L = sp.symbols("e,L",positive=True)
    eps = sp.symbols("epsilon",positive = True)
    sinhu = sp.sinh(u)
    sinv = sp.sin(v)
    Phi = (U(u,e) - V(v,e))/(sinv*sinv + sinhu*sinhu)
    T = (pu*pu + pv*pv)/2/e/e/(sinv*sinv + sinhu*sinhu) + L*L/2/e/e/sinhu**2/sinv**2
    H = T + Phi
    # Set initial conditions
    u0=u1
    v0 = np.pi/2
    NU = sp.lambdify((u,e),U(u,e))
    NV = sp.lambdify((v,e),V(v,e))
    NE,NL,NI3 = uv_lims_to_ELI3(u1,u2,v_star,ecc,NU,NV)
    pu_sq,pv_sq = ELI3_to_pu_sq_pv_sq(NE,NL,NI3,u0,v0,ecc,NU,NV)
    inits = np.array([u0,v0,np.sqrt(pu_sq),np.sqrt(pv_sq)])
    state = cm.hamiltonian.PhaseSpaceState([u,v,pu,pv],inits)
    
    # Set values of parameters
    params = {L:NL,e:ecc}

    # Get Hamiltonian object
    ham = cm.Hamiltonian(H,params,state)
    return ham


def uv_lims_to_ELI3(u1,u2,nu,e,U,V):
    sinh1_sq = np.sinh(u1)**2
    sinh2_sq = np.sinh(u2)**2
    sinv_sq = np.sin(nu)**2
    e_sq =e*e
    A=np.array([
        [sinh1_sq, -0.5/sinh1_sq/e_sq,  -1],
        [sinh2_sq, -0.5/sinh2_sq/e_sq,  -1],
        [sinv_sq , -0.5/sinv_sq/e_sq,   +1]
    ])
    U1 = U(u1,e)
    U2 = U(u2,e)
    Vstar = V(nu,e)
    b = np.array([U1,U2,-Vstar])
    E,Lsq,I3 = np.linalg.solve(A,b)
    L =np.sqrt(Lsq)
    return E, L, I3

def ELI3_to_pu_sq_pv_sq(E,L,I3,u0,v0,e,U,V):
    sh_sq = np.sinh(u0)**2
    s_sq = np.sin(v0)**2
    Lsq = L*L
    esq= e*e
    pu_sq = 2 * esq * (E * sh_sq - I3 - 0.5* Lsq / esq / sh_sq - U(u0,e))
    pv_sq = 2 * esq * (E *  s_sq + I3 - 0.5* Lsq / esq / s_sq  + V(v0,e))
    return pu_sq, pv_sq

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


# def get_POSP_hamiltonian_full(ecc,u1,u2,v_star):
    
#     # Set up functional form of Hamiltonian
#     u,v,phi = sp.symbols("u,v,phi",real=True)
#     pu,pv,L = sp.symbols("p_u,p_v,L",real=True)
#     e = sp.symbols("e",positive=True)
#     coshu = sp.cosh(u)
#     cosv = sp.cos(v)
#     sinhu = sp.sinh(u,e)
#     sinv = sp.sin(v,e)
#     U = -1 * coshu * sp.atan(e*coshu/sp.sqrt(1-e*e))/e
#     V = -1 * cosv * sp.atan(e*cosv/sp.sqrt(1-e*e))/,ee
#     Phi = (U-V)/(sinv*sinv + sinhu*sinhu,e)
#     T = (pu*pu + pv*pv)/2/e/e/(sinv*sinv + sinhu*sinhu) + L*L/2/e/e/sinhu**2/sinv**2
#     H = T + Ph,ei
#    ,e 
#     # Set initial conditions
#     u0=u,e1
#     v0 =,e np.pi/2
#     NE,NL,NI3 = uv_lims_to_ELI3(u1,u2,v_star,ecc)
#     pu_sq,pv_sq = ELI3_to_pu_sq_pv_sq(NE,NL,NI3,u0,v0,ecc)
#     inits = np.array([u0,v0,0.,np.sqrt(pu_sq),np.sqrt(pv_sq),NL])
#     state = cm.hamiltonian.PhaseSpaceState([u,v,phi,pu,pv,L],inits)
    
#     # Set values of parameters
#     params = {e:ecc}

#     # Get Hamiltonian object
#     ham = cm.Hamiltonian(H,params,state)
#     return ham
