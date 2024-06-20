from galpy.potential import MWPotential2014
import numpy as np
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
import sys
I = int(sys.argv[1])
mwp = MWPotential2014
Rc = 1.
vc = 1.
L = Rc * vc
Delta = estimateDeltaStaeckel(mwp,Rc,0.)
aAS = actionAngleStaeckel(pot=mwp,delta = Delta, c=True)
srcdir = "/fs/lustre/cita/hadden/05_action_angle_galaxy/03_results/"
finame = srcdir+"mw14_integations_{}.npz".format(I)
data = np.load(finame)
times= data['times']
R,z,vR,vz = np.transpose(data['orbit'],(1,0,2))
J1,J2,J3 = np.zeros((3,50,512))
for j in range(50):
    J1[j],J2[j],J3[j] = aAS(R[j],vR[j],L/R[j],z[j],vz[j])
np.savez_compressed(srcdir + "mw14_staeckel_actions_{}.npz".format(I),Jr = J1, L = J2, J = J3)
