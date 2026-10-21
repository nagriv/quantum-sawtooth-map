"""Simulation of the quantum wastooth map"""

# %% Import 

import numpy as np
import h5py
from scipy.fft import fft, fftshift, ifft, ifftshift


# %% Definition for some usefull function

def norm_check(psi): # Check the normalization of the system's wavefunction

    err = np.abs(np.vdot(psi, psi) - 1.)
    
    if err > 1e-12: 
        
        raise ValueError(rf'Normalization Problems: err = {err}')

# Create the kick operator
def U_k(theta_list, k_val): 

    return np.exp(1.j * 0.5 * k_val * (theta_list - np.pi)**2 )

# Create the free evolution operator
def U_T(n_list, T_val): 

    return np.exp(-1.j * 0.5 * T_val * (n_list)**2 )

## Using fftshift and iffshift ensures the operator acts correctly on the wavefunction

# Change the wavefunction basis from angles to impulses
def rappr_ang_to_n(psi_ang): 

    return fftshift(fft(psi_ang, norm='ortho'))

# Change the wavefunction basis from impulses to angles
def rappr_n_to_ang(psi_n):

    return ifft(ifftshift(psi_n), norm='ortho')

# Aplly the correct sequence for one iteration of the QSM
def qswm_pass(psi, Opp_Uk, Opp_UT):

    psi = Opp_Uk * psi

    psi = rappr_ang_to_n(psi)

    psi = Opp_UT * psi

    psi =  rappr_n_to_ang(psi)

    return psi


#%% Setting of the parameters for the simulation

ifsave = True # change to false for not create the dataset file

nq = 10 # number of qubits for the simulation. For nq >= 16 the output data file can be large.

dim_H = 2 ** nq # Hilbert Space of for the system. For nq = 10 dim_H = 1024 elements

n_iterations = 5000 # number of the iteration of the QSM


k_val = np.sqrt(3) # kick parameter

T_val = np.sqrt(2/3) # period parameter 

# The K parameter is the chaos parameter for the QSM, and controls the evolution of the system. 
# For -4 < K < 0, the evolution of the map is stable, instead for K < -4 or K > 0 the evolution is chaotic.
# Choose k_val and T_val correctly for the right evolution

K_val = k_val * T_val

# theoric localization length 
loc_len = np.square(k_val * np.pi ) / 3

# The eta parameter expresses the condition for the dynamical localization of the wavefunction. For eta << 1, the dynamical localization is possible; no chaotic behavior.

eta = loc_len/dim_H


print(rf'nq:{nq:n}, dim_H:{dim_H:n}, k:{k_val:.2e}, T:{T_val:.2e}, eta:{eta:.2e}')

#%%

# Creation of the impulses eigenvalues array. 
# The impulse n is -N/2 <= n <= N/2 - 1, with N  = dim_H
n_eigvl = np.arange(dim_H) - dim_H // 2

# Creation of the angles eigenvalues array. 
# The angle theta is in 0<= theta <=  2 \pi * (N-1)/N, with N = dim_H
ang_eigvl =  2 * np.pi * np.arange(dim_H) / dim_H

# print(r'n:', n_eigvl)
# print(r'theta:', ang_eigvl)

#%% 
# Selection of the initial impulse.
#  A common choice is n = 0, that is in the dim_H // 2  position of the n_eigvl array.  

n0_list = [dim_H //2] # n0 is pinned in the n = 0 eigenvalue

# n0_list = np.arange(dim_H) # n0 are all possible eigenvalues, the initial state will be spread. 

psi_n0 = np.zeros(shape=dim_H, dtype=np.complex128)
psi_n0[n0_list] = 1.
psi_n0 = psi_n0 / np.linalg.norm(psi_n0)

Opp_Uk = U_k(ang_eigvl, k_val)
Opp_UT = U_T(n_eigvl, T_val)

# print(psi_n0)


