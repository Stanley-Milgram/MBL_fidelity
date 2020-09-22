from quspin.operators import hamiltonian, quantum_operator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
from numpy.random import ranf, seed, uniform# pseudo random numbers
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

# LL = #1
L=params.L
# DD = #2
w=params.D

in_flag = params.In_flag

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
t_i, t_f, t_steps = 0.0, 20.0, 400
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

epsilon=params.Epsilon
params_dict = dict(H0=1.0)

dis_table_2=np.random.rand(L)*2.0-1.0

for j in range(L):
    params_dict["z"+str(j)] = w*dis_table_1[j]+w*epsilon*dis_table_2[j] # create random fields list

HAM_2 = H_dict.tohamiltonian(params_dict) # build initial Hamiltonian through H_dict

ind = hf.Psi_0(Dim, L, in_flag)
i_0 = sp_basis.index(ind) # find index of product state
psi_0 = np.zeros(sp_basis.Ns, dtype = np.complex64) # create empty state vector
psi_0[i_0] = 1.0# set MB state to be |10101010>
psi_0 = psi_0.flatten()

psi_t_1=HAM_1.evolve(psi_0, 0.0, t_tab)
psi_t_2=HAM_2.evolve(psi_0, 0.0, t_tab)

Losch=[]
for i in range(psi_t_1.shape[1]):
    Losch.append(abs(np.vdot(psi_t_1[:,i].ravel(), psi_t_2[:,i].ravel()))**2)

data = np.c_[t_tab, Losch]

directory = '../DATA/FidRandom/epsilon'+str(epsilon)+'/L'+str(L)+'/D'+str(w)+'/'
PATH_now = LOCAL+os.sep+directory+os.sep
if not os.path.exists(PATH_now):
    os.makedirs(PATH_now)

nomefile = str(PATH_now+'FidL_'+str(L)+'D_'+str(w)+'seed'+str(seed)+'.dat')
np.savetxt(nomefile, data, fmt = '%.9f')

print("Realization completed in {:2f} s".format(time()-ti))
