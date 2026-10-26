"""Simulation of the quantum wastooth map"""

# %% 
"""Import""" 

import numpy as np
import h5py
from scipy.fft import fft, fftshift, ifft, ifftshift


# %% 


"Check the normalization of the system's wavefunction"
def norm_check(psi): 

    err = np.abs(np.vdot(psi, psi) - 1.)
    
    if err > 1e-12: 
        
        raise ValueError(rf'Normalization Problems: err = {err}')

"Create the kick operator"
def U_k(theta_list, k_val): 

    return np.exp(1.j * 0.5 * k_val * (theta_list - np.pi)**2 )

"Create the free evolution operator"
def U_T(n_list, T_val): 

    return np.exp(-1.j * 0.5 * T_val * (n_list)**2 )

"""Using fftshift and iffshift ensures the operator acts correctly on the wavefunction"""

"Change the wavefunction basis from angles to impulses"
def rappr_ang_to_n(psi_ang): 

    return fftshift(fft(psi_ang, norm='ortho'))

"Change the wavefunction basis from impulses to angles"
def rappr_n_to_ang(psi_n):

    return ifft(ifftshift(psi_n), norm='ortho')

"Aplly the correct sequence for one iteration of the QSM"
def qswm_pass(psi, Opp_Uk, Opp_UT):

    psi = Opp_Uk * psi

    psi = rappr_ang_to_n(psi)

    psi = Opp_UT * psi

    psi =  rappr_n_to_ang(psi)

    return psi


#%% 


"Number of qubits for the simulation. For nq >= 16 the output data file can be large"
nq = 10 

"Hilbert Space of for the system. For nq = 10 dim_H = 1024 elements"
dim_H = 2 ** nq 

"Number of the iteration of the QSM. For longer iteration the output can be large"
n_iterations = 1000

"""The K parameter is the chaos parameter for the QSM, and controls the evolution of the system. 
For -4 < K < 0, the evolution of the map is stable, instead for K < -4 or K > 0 the evolution is chaotic"""
K_val = np.sqrt(2)

"Period parameter"
LL = 67
T_val = 2 * np.pi * LL / dim_H

"Kick parameter"
k_val = K_val / T_val


"Theoric Localization length. Dynamical localization can be observed only in the chaotic regime"
loc_len = np.square(k_val * np.pi ) / 3

"The eta parameter expresses the condition for the dynamical localization of the wavefunction. For eta << 1, the dynamical localization is possible; no chaotic behavior"
eta = loc_len / dim_H


print(rf'nq:{nq:n}, dim_H:{dim_H:n}, k:{k_val:.2e}, T:{T_val:.2e}, eta:{eta:.2e}')

#%%

"""Creation arrays for the eigenvalues of the system"""

"Creation of the impulses eigenvalues array. The impulse n is -N/2 <= n <= N/2 - 1, with N  = dim_H"
n_eigvl = np.arange(dim_H) - dim_H // 2

"Creation of the angles eigenvalues array. The angle theta is in 0<= theta <=  2 \pi * (N-1)/N, with N = dim_H"
ang_eigvl =  2 * np.pi * np.arange(dim_H) / dim_H

# print(r'n:', n_eigvl)
# print(r'theta:', ang_eigvl)

"Creation of the kick and free evolution operator given the correct order of the eigenvalues"
Opp_Uk = U_k(ang_eigvl, k_val)
Opp_UT = U_T(n_eigvl, T_val)


#%% 

"""Initialization of the wavefunction in the impulse basis"""

"Selection of the initial impulse. A common choice is n = 0, that is in the dim_H // 2 position of the n_eigvl array"

"With this choice the initial impulse is peaked in n0 = 0"
n0_list = [dim_H //2]

psi_n0 = np.zeros(shape=dim_H, dtype=np.complex128)
psi_n0[n0_list] = 1.
psi_n0 = psi_n0 / np.linalg.norm(psi_n0)

# print(psi_n0)

#%%

"""Creation of the Dataset File"""

"The dataset file will be created in the Data repository.The file contains the wavefunction in the impulse basis at each iteration of the map. The name of the file is given by the nq, k_val and T_val"

"Creation of the file in the Data repository"
with h5py.File(f'Data/qsm_nq{nq:n}_K{K_val:.2f}_LL{LL:n}_iter{n_iterations:n}.h5', 'w') as f:

        
    grp = f.create_dataset('psi', shape = (1, dim_H), maxshape = (None, dim_H), 
        dtype = 'complex128', chunks = (1, dim_H), compression='lzf')

    "Save the important parameters"
    grp.attrs['nq'] = nq
    grp.attrs['k'] = k_val
    grp.attrs['T'] = T_val

    "Save the wavefunction at t = 0"
    grp[0] = psi_n0

    psi_now = rappr_n_to_ang(psi_n0)

    "Evolutions of the wavefunction for n_iterations"
    for t in range(n_iterations):

        psi_next = qswm_pass(psi_now, Opp_Uk, Opp_UT)

        "Updating the old wavefunction for the next iteration"
        psi_now = psi_next

        psi_next = rappr_ang_to_n(psi_next) 

        "Updating the dataset size"
        grp.resize(grp.shape[0] + 1, axis = 0)

        "Save the the wavefunction after the map iteration"    
        grp[-1] = psi_next

        "Numerical errors can accumulate, causing the wavefunction to lose normalization."
        if (t % 3000 == 0):
            psi_now /= np.linalg.norm(psi_now)
