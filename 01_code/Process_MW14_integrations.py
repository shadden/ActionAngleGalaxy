from SeriesUtils import *
from MWPotential2014 import MWPotential2014_get_hamiltonian
from celmech.poisson_series import PoissonSeries,PSTerm
from celmech.poisson_series import expL,expLinv, bracket
from celmech.poisson_series import birkhoff_normalize

srcdir = "/fs/lustre/cita/hadden/05_action_angle_galaxy/03_results/" # directory of integration results
finame_template = "mw14_integations_{}.npz"
finame3d_template = "mw14_3d_integrations_{}.npz"
staeck_actions_finame = "mw14_staeckel_actions_{}.npz"
lmax = 10 # maximum expansion order

# Set up Hamiltonian
mw2014ham = MWPotential2014_get_hamiltonian([1.,0,0,0],1.)
nu0 = np.sqrt(-1 * mw2014ham.calculate_jacobian()[3,1])
kappa0 = np.sqrt(-1 * mw2014ham.calculate_jacobian()[2,0])
R,z,pR,pz = mw2014ham.qp_vars
rule = {R:1.,z:0.,pR:0.,pz:0.}

# read in derivatives of the potential
import pickle
derivs_file = srcdir+"mw14_derivs.pkl"
try:
    with open(derivs_file,'rb') as fi:
        derivs = pickle.load(fi)
except:
    derivs = dict()
    for n in range(2,lmax+1):
        for l in range(n+1):
            nR = n-l
            nz = l
            derivs[(nR,nz)]=float(sp.diff(mw2014ham.N_H,R,nR,z,nz).xreplace(rule))
    with open(derivs_file,"wb") as fi:
        pickle.dump(derivs,fi)

# get Hamiltonian as Poisson series
ham_series = PotentialDerivativesToPoissonSeries(derivs,lmax)

# run Birkoff normalization 
omega = np.real((ham_series[2][(1,0,1,0)],ham_series[2][(0,1,0,1)]))
chi,hav = birkhoff_normalize(omega,ham_series,lmax)

# Series for Omega_i,x_i, x'_i
Omega_R,Omega_z = h_series_to_omega_series(pert_series_to_PoissonSeries(hav))
xR_series,xz_series = get_xseries(2)
xz_osc_to_mean = pert_series_to_PoissonSeries(expLinv(xz_series,chi))
xR_osc_to_mean = pert_series_to_PoissonSeries(expLinv(xR_series,chi))

# Series arranged by powers sqrt(Jz)
Omega_z_by_rtJz = collect_by_rt_action_power(Omega_z,1)
Omega_R_by_rtJz = collect_by_rt_action_power(Omega_R,1)
xR_osc_to_mean_by_rtJz = collect_by_rt_action_power(xR_osc_to_mean,1)
xz_osc_to_mean_by_rtJz = collect_by_rt_action_power(xz_osc_to_mean,1)

# create osculating-to-mean functions that take array input
def poisson_series_to_vectorized_func(pseries):
    f1 = sp.lambdify(pseries.cvar_symbols,pseries.as_expression())
    f2 = lambda xR,xz: f1(xR,xz,np.conjugate(xR),np.conjugate(xz))
    return f2
xR_osc_to_mean_f,xz_osc_to_mean_f = [poisson_series_to_vectorized_func(p) for p in (xR_osc_to_mean,xz_osc_to_mean)]

# create osculating-to-mean pade approximations
def poisson_series_to_pade_vectorized_func(pseries,padeify_func):
    f1 = padeify_func(pseries)
    f2 = lambda xR,xz: f1(xR,np.conjugate(xR),np.abs(xz)**2,xz/np.abs(xz))
    return f2

# generate pade approximants to an input series
pade_fs = (to_pade1_approx_function,to_pade2_approx_function)
get_pade_1_and_2 = lambda series: [poisson_series_to_pade_vectorized_func(series,f) for f in pade_fs]

# xR pade approximation
xR_osc_to_mean_pade1, xR_osc_to_mean_pade2 = get_pade_1_and_2(xR_osc_to_mean)

# xz pade approximation
newterms = [PSTerm(term.C,term.k-np.array((0,1)),term.kbar,term.p,term.q) for term in xz_osc_to_mean.terms]
xz_osc_to_mean_by_xz = PoissonSeries.from_PSTerms(newterms)
xz_osc_to_mean_by_xz_pade1,xz_osc_to_mean_by_xz_pade2 = get_pade_1_and_2(xz_osc_to_mean_by_xz)
xz_osc_to_mean_pade1 = lambda xR,xz: xz * xz_osc_to_mean_by_xz_pade1(xR,xz)
xz_osc_to_mean_pade2 = lambda xR,xz: xz * xz_osc_to_mean_by_xz_pade2(xR,xz)


# Omega_R pade approximation
Omega_R_pade1_f,Omega_R_pade2_f = get_pade_1_and_2(Omega_R)

# Omega_z pade approximation
Omega_z_pade1_f,Omega_z_pade2_f = get_pade_1_and_2(Omega_z)

# Omega_phi
Rc = 1.
Omega_phi_0 = 1.0
phi_dot = pert_series_to_PoissonSeries(expL(generate_phidot_series(Rc,kappa0,lmax),chi))
is_secular = lambda term: np.allclose(term.k,term.kbar)
Omega_phi=Omega_phi_0 * PoissonSeries.from_PSTerms([term for term in phi_dot.terms if is_secular(term)])
Omega_phi_pade1_f, Omega_phi_pade2_f = get_pade_1_and_2(Omega_phi)


# file processing
import sys
i = int(sys.argv[1])
finame = srcdir+finame_template.format(i)
data = np.load(finame)
staeck_data = np.load(srcdir + staeck_actions_finame.format(i))
data3d = srcdir+finame3d_template.format(i)

#   birkhoff normalization quantities
xRs = data['xR'][1:]
xzs = data['xz'][1:]
expIphi = data['orbit'][1:,0,:]
time = data['times']
xRs_mean = xR_osc_to_mean_f(xRs,xzs)
xzs_mean = xz_osc_to_mean_f(xRs,xzs)
JRs_staeck = staeck_data['Jr']
Jzs_staeck = staeck_data['J']
xRs_pade1 = xR_osc_to_mean_pade1(xRs,xzs)
xRs_pade2 = xR_osc_to_mean_pade2(xRs,xzs)
xzs_pade1 = xz_osc_to_mean_pade1(xRs,xzs)
xzs_pade2 = xz_osc_to_mean_pade2(xRs,xzs)
Omega_R_pade1 = np.real(Omega_R_pade1_f(xRs.T[0],xzs.T[0]))
Omega_R_pade2 = np.real(Omega_R_pade2_f(xRs.T[0],xzs.T[0]))
Omega_z_pade1 = np.real(Omega_z_pade1_f(xRs.T[0],xzs.T[0]))
Omega_z_pade2 = np.real(Omega_z_pade2_f(xRs.T[0],xzs.T[0]))
Omega_phi_pade1 = np.real(Omega_phi_pade1_f(xRs.T[0],xzs.T[0]))
Omega_phi_pade2 = np.real(Omega_phi_pade2_f(xRs.T[0],xzs.T[0]))

#  celmech fmft frequencies
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
Omega_R_N,Omega_z_N,Omega_phi_N = np.zeros((3,49))
for j,xR,xz,eIphi in zip(range(49),xRs,xzs,expIphi):
    Omega_R_N[j] = -1*list(fmft(time,xR,5))[0]
    Omega_z_N[j] = -1*list(fmft(time,xz,5))[0]
    Omega_phi_N[j] = list(fmft(time,eIphi,5))[0]

# galpy staeckel approx frequencies
from galpy.potential import MWPotential2014 as mwp
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
Delta = estimateDeltaStaeckel(mwp,1.,0.)
aAS = actionAngleStaeckel(pot=mwp,delta = Delta, c=True)
R,z,vR,vz = data['orbit'][1:,:,0].T
_,_,_,Omega_R_staeck, Omega_z_staeck, Omega_phi_staeck = aAS.actionsFreqs(R,vR,1/R,z,vz)

save_name = srcdir+"mw14_integations_and_actions_{}.npz".format(i)
np.savez_compressed(
    save_name,
    times=data['times'],
    orbit=data['orbit'],
    xR_epi = xRs,
    xz_epi = xzs,
    xR_mean = xRs_mean,
    xz_mean = xzs_mean,
    xR_pade1 = xRs_pade1,
    xz_pade1 = xzs_pade1,
    xR_pade2 = xRs_pade2,
    xz_pade2 = xzs_pade2,
    JR_staeckel = JRs_staeck,
    Jz_staeckel = Jzs_staeck
)
