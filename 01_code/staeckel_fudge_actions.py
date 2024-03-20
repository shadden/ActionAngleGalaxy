from galpy.potential import MiyamotoNagaiPotential
import numpy as np
from MiyamotoNagaiPotential import MiyamotoNagai_L_to_Rc
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
import sys


I = int(sys.argv[1])

L = 3.
mp = MiyamotoNagaiPotential(a = 3,b=0.3)
Rc = MiyamotoNagai_L_to_Rc(L,3,0.3)
vc = mp.vcirc(Rc)
Delta = estimateDeltaStaeckel(mp,Rc,0.)
aAS = actionAngleStaeckel(pot=mp,delta = Delta, c=False)
srcdir = "/fs/lustre/cita/hadden/05_action_angle_galaxy/01_code/"

finame = srcdir+"integations_{}.npz".format(I)
data = np.load(finame)
times= data['times']
R,z,vR,vz = np.transpose(data['orbit'],(1,0,2))
J1,J2,J3 = np.zeros((3,50,512))

for j in range(50):
    J1[j],J2[j],J3[j] = aAS(R[j],vR[j],L/R[j],z[j],vz[j])

np.savez_compressed(srcdir + "staeckel_actions_{}.npz".format(I),Jr = J1, L = J2, J = J3) 