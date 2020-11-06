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

ti=time()

# LL = #1
L=params.L
# DD = #2
w=params.D

in_flag = 1

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

t_tab=np.linspace(0.0, 20.0, 50, endpoint=True)
psi_t_1=HAM_1.evolve(psi_0, 0.0, t_tab)
psi_t_2=HAM_2.evolve(psi_0, 0.0, t_tab)

def overlap(a, b):
    OVRLAP=[]
    for i in range(a.shape[1]):
        OVRLAP.append(abs(np.vdot(a[:,i].ravel(), b[:,i].ravel()))**2)
    return(OVRLAP)

Losch=overlap(psi_t_1, psi_t_2)

E_1, V_1=HAM_1.eigh()
E_2, V_2=HAM_2.eigh()
dE=E_1-E_2

params_dict = dict(H0=0.0)
for j in range(L):
    params_dict["z"+str(j)] = w*dis_table_1[j]+w*epsilon*dis_table_2[j] # create random fields list

Sz_i=H_dict.tohamiltonian(params_dict) # build initial Hamiltonian through H_dict

UGA=[]
OVRLAP_1=np.abs(np.array([np.vdot(psi_0, V_1[:,i]) for i in range(Dim)]))
OVRLAP_2=np.abs(np.array([np.vdot(psi_0, V_2[:,i]) for i in range(Dim)]))

for t in t_tab:
    uga=[(OVRLAP_1[n]**2)*(np.exp(-1j*(Sz_i.expt_value(V_1[:,n])+Sz_i.expt_value(V_2[:,n]))*epsilon*(t/2)))
         for n in range(sp_basis.Ns)]
    UGA.append(np.abs(np.sum(uga)**2))

data1=np.c_[t_tab, Losch, UGA]
data2=np.c_[E_1, E_2, OVRLAP_1, OVRLAP_2]

directory = '../DATA/OVRLP/epsilon'+str(epsilon)+'/L'+str(L)+'/D'+str(w)+'/'
PATH_now = LOCAL+os.sep+directory+os.sep
if not os.path.exists(PATH_now):
    os.makedirs(PATH_now)

nomefile1 = str(PATH_now+'FidL_'+str(L)+'D_'+str(w)+'seed'+str(seed)+'.dat')
nomefile2 = str(PATH_now+'SpectL_'+str(L)+'D_'+str(w)+'seed'+str(seed)+'.dat')

np.savetxt(nomefile1, data1, fmt = '%.9f')
np.savetxt(nomefile2, data2, fmt = '%.9f')

print("Realization completed in {:2f} s".format(time()-ti))
