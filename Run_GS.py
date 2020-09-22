from quspin.operators import hamiltonian, exp_op, quantum_operator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
from numpy.random import ranf, seed, uniform, choice # pseudo random numbers
import scipy
import scipy.linalg as _la
from scipy.sparse import linalg
import math
import sys,os
from time import time
import h_functions as hf
import params

LOCAL = os.path.abspath('.')
PATH_now = LOCAL

ti=time()

L=params.L
w=params.D

### INITIAL STATE ###
in_flag  = params.In_flag

### SYSTEM SIZE ###
N = L//2

Jxy = 1.0 # hopping
Jz = 1.0 # zz interaction

### RUN ###
ti = time() # start timing function for each realization

sp_basis = spin_basis_1d(L,L//2)
int_list, hop_list = hf.coupling_list(L, Jxy, Jz)
oplist_static = [["+-",hop_list],["-+",hop_list],["zz",int_list]]
Dim = sp_basis.Ns
t_i, t_f, t_steps = 0.0, 100.0, 200
t_tab = np.linspace(t_i,t_f,num=t_steps,endpoint=True)

operator_dict = dict(H0 = oplist_static)
for i in range(L):
    operator_dict["z"+str(i)] = [["z", [[1.0, i]]]]

no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
H_dict = quantum_operator(operator_dict, basis = sp_basis, **no_checks)
params_dict = dict(H0=1.0)

seed = np.random.randint(0, 100000)
np.random.seed(seed)
dis_table_1=np.random.rand(L)*2-1.0

for j in range(L):
    params_dict["z"+str(j)] = w*dis_table_1[j] # create random fields list

HAM_1 = H_dict.tohamiltonian(params_dict) # build initial Hamiltonian through H_dict
HAM_1 = np.real(HAM_1.tocsc())
psi_1 = linalg.eigsh(HAM_1, k=1)[1].flatten() # initialize system in GS of pre-quench Hamiltonian

epsilon=params.Epsilon

dis_table_2=np.random.rand(L)

for j in range(L):
    params_dict["z"+str(j)] = w*dis_table_1[j]+w*epsilon*dis_table_2[j] # create random fields list

HAM_2 = H_dict.tohamiltonian(params_dict) # build initial Hamiltonian through H_dict
HAM_2 = np.real(HAM_2.tocsc())
psi_2 = linalg.eigsh(HAM_2, k=1)[1].flatten() # initialize system in GS of pre-quench Hamiltonian

Overlap = (abs(np.vdot(psi_1.ravel(), psi_2.ravel()))**2)

directory = '../DATA/Overlap_GS/L'+str(L)+'/D'+str(w)+'/'
PATH_now = LOCAL+os.sep+directory+os.sep
if not os.path.exists(PATH_now):
    os.makedirs(PATH_now)

nomefile = str(PATH_now+'L_'+str(L)+'D_'+str(w)+'seed'+str(seed)+'.dat')
with open(nomefile, 'w') as f:
  f.write('%.9f' % Overlap)
print("Realization completed in {:2f} s".format(time()-ti))
