from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy, diag_ensemble # entropies
import numpy as np # generic math functions
from numpy.random import ranf,seed # pseudo random numbers
import scipy.sparse as _sp
import scipy.linalg as _la
import itertools
from itertools import combinations_with_replacement, combinations
from math import factorial

def mat_format(A):
    """Find the type of a matrix(dense or sparse)
    Args:
        A(2d array of floats)           = Dense or sparse Matrix
    Returns:
        mat_type(string)                = Type of A, "Sparse" or "Dense"
    """
    form = str(type(A))
    spar_str = "<class 's"
    if form.startswith(spar_str):
        mat_type = "Sparse"
    else:
        mat_type = "Dense"
    return mat_type

### FROM CONFIGURATION TO BIN NUMBER ###
def TO_bin(xx):
    return int(xx,2)

### FROM BIN NUMBER TO CONFIGURATION ###
def TO_con(x,L):
    x1=int(x)
    L1=int(L)
    return np.binary_repr(x1, width=L1)

### BINOMIAL ###
def comb(n, k):
	kk = factorial(n) / factorial(k) / factorial(n - k)
	uga= int(kk)
	return uga

### BASIS CREATION ###
def basis_prep(L):
    basis = spin_basis_1d(L, Nup = L//2, pauli=False)
    return basis

def Base_states(n,k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        result.append(''.join(s))
    return result

### COUPLING (HOPPING AND INTERACTION) LISTS ###
def coupling_list(L, Jxy, Jzz):
    J_zz = [[Jzz,i,i+1] for i in range(L-1)] # OBC
    J_xy = [[Jxy/2,i,i+1] for i in range(L-1)] # OBC
    return J_zz, J_xy

### RANDOM FIELD LIST ###
def field_list(L, W, Dis_type, seed = None):
    if seed is None:
        seed = np.random.randint(1,10000)
    if seed is not None:
        np.random.seed(seed)
    if Dis_type == 0:
        dis = 2*np.random.rand(L)-1.0
        mag_field = [[dis[i], i] for i in range(L)]
        dis_hz = [["z", mag_field]]
    else:
        for i in range(L):
            mag_field = [np.cos(2*math.pi*0.721*i/L), i]
        dis_hz = [["z", mag_field]]

    return mag_field, dis_hz

### OPERATORS ASSOCIATED TO COUPLINGS ###
def op_list(Jxy, Jzz):
    oplist_static = [["+-",Jxy],["-+",Jxy],["zz",Jzz]]
    oplist_dynamic = [] # empty if Hamiltonian is time-independent
    return oplist_static, oplist_dynamic

### RANDOM HEISENBERG HAMILTONIAN ###
def ham(L, basis, hop, int, W, Dis_type, seed = None):
    op_static, op_dynamic = op_list(hop, int)
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
    clean_ham = hamiltonian(op_static, op_dynamic, basis=basis, dtype=np.float64,**no_checks) #clean XXZ Hamiltonian with no disorder
    h_z, dis_static = field_list(L, Dis_type, seed)
    dis_ham = hamiltonian(dis_static, op_dynamic, basis=basis,dtype=np.float64,**no_checks)
    tot_ham = clean_ham + W*dis_ham #disordered XXZ Hamiltonian
    return tot_ham

def eigval(A, targ = None):
    """Diagonalize the Hamiltonian
    Args:
        A(2d array of floats)           = Matrix to be diagonalized (Hamiltonian)
    Returns:
        E(1d array of floats)           = Eigenvalues
        V(2d array of floats)           = Eigenvectors (in columns!)
    """
    type = mat_format(A)
    if type == 'Sparse':
        if targ is not None:
            E,V = _sp.linalg.eigsh(A, k = 1, sigma = targ, which='SA', return_eigenvectors=True)
        else:
            E,V = _sp.linalg.eigsh(A, k = 1, which='SA', return_eigenvectors=True)
    else:
        E, V = _la.eigh(A)
    return E, V

### LEVEL STATISTICS (HUSE OBSERVABLE) ###
def levstat(E):
    gap = E[1:]-E[:-1]
    r = list(map(lambda x,y:min(x,y)*1./max(x,y), gap[1:], gap[0:-1]))
    return np.mean(r)

def Psi_0(Dim, L, in_flag):
    """Index of the initial state for time evolution
    Args:
        Dim(int)                        = Dimension of Hilbert space
        L(int)                          = Size fo the chain
        Base_num(1d array of str)       = Basis states in binary representation
        in_flag(int)                    = Flag for initial state (0 ---> random,
                                                                  1 ---> 1010101010,
                                                                  2 ---> 1111100000)
    Returns:
        n(int)                          = Index of the chosen initial state in Base_num
    """
    if in_flag == 0:
        n = np.random.randint(0,Dim-1)
    elif in_flag == 1:
        n = TO_con(sum([2**i for i in range(1, L, 2)]), L)
    else:
        n = TO_con(sum([2**i for i in range(0,L//2, 1)]),L)[::-1]
    return n

def overlap(psi_1, psi_2):
    return np.abs(np.dot(psi_1,psi_2))**2

### DATA DIRECTORIES AND DATA FILES CREATION ###

def generate_directory(basename, para, namesub):
    #namesub is a string example namesub = 'L'
    path = os.getcwd()
    path_now = os.path.join(basename,namesub+str(para))
    if not os.path.exists(path_now):
        os.makedirs(path_now)
    return path_now

def generate_filename(basename):

    unix_timestamp = int(time.time())
    local_time = str(int(round(time.time() * 1000)))
    xx = basename + local_time + ".dat"
    if os.path.isfile(xx):
        time.sleep(1)
        return generate_filename(basename)
    return xx

def creation_all_subdirectory(L, W):
    PATH = os.getcwd()
    PATH_L=generate_directory(PATH, L, 'L_')
    PATH_W= generate_directory(PATH_alpha, W, 'W_')
    return 0
