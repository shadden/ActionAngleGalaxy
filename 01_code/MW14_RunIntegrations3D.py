import numpy as np
from MWPotential2014 import *
import sys
savedir = "/fs/lustre/cita/hadden/05_action_angle_galaxy/03_results/"
I = int(sys.argv[1])
v_factor = np.linspace(0,0.25)[I]
N_angles = 50

Rc = 1.
NL = 1.
ham = MWPotential2014_get_hamiltonian_full([0.,Rc,0.,NL,0.,0.])
kappa0 = np.sqrt(-1 * ham.calculate_jacobian()[4,1])
nu0 = np.sqrt(-1 * ham.calculate_jacobian()[5,2])

Ntimes = 512
T = 2 * np.pi / kappa0
times = np.linspace(0,10,Ntimes) * T

vc = NL/Rc
v_init =  v_factor * vc

all_xR,all_xz = np.zeros((2,N_angles,Ntimes),dtype = np.complex128)
all_orbits = np.zeros((N_angles,ham.N_dim,Ntimes))
angles = np.linspace(0,0.5 * np.pi,N_angles)
for J,theta in enumerate(angles):
    print(J)
    ics = np.array([0.,Rc,0.,NL,v_init * np.cos(theta), v_init * np.sin(theta)])
    ham.state.values = ics
    ham.state.t = 0
    for i,t in enumerate(times):
        ham.integrate(t)
        all_orbits[J,:,i] = ham.state.values
    orbit = all_orbits[J]
    all_xR[J] = np.sqrt(0.5*kappa0) * ((orbit[0] - Rc) + 1j * orbit[2]/kappa0)
    all_xz[J] = np.sqrt(0.5*nu0) * (orbit[1] + 1j * orbit[3]/nu0)
np.savez_compressed(savedir + "mw14_3d_integations_{}".format(I),orbit=all_orbits,times=times,xR=all_xR,xz=all_xz)
