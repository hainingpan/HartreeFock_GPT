from HF import *

import numpy as np
from typing import Any
from scipy.linalg import block_diag

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict[str, Any]): Dictionary of model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int = 10, parameters: dict[str, Any] = {'gamma_0': 1.0, 'gamma_1': 1.0, 'gamma_2': 1.0, 'gamma_3': 1.0, 'gamma_N': 1.0, 'U_H':1.0, 'U_X':1.0}, filling_factor: float = 0.5):
    self.lattice = 'triangular'
    self.D = (6,)
    self.basis_order = {
        '0': 'orbital_spin',
        '1': 'orbital_spin',
        '2': 'orbital_spin',
        '3': 'orbital_spin',
        '4': 'orbital_spin',
        '5': 'orbital_spin'}
    # Order for each flavor:
    # 0: (0, spin_up)
    # 1: (0, spin_down)
    # 2: (1, spin_up)
    # 3: (1, spin_down)
    # 4: (2, spin_up)
    # 5: (2, spin_down)

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Default temperature
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.a =  2.46 # Lattice constant in Angstroms
    self.A = self.a**2 * np.sqrt(3) / 2.0 # Area of the unit-cell in Angstroms^2.

    # Model parameters
    self.gamma_0 = parameters.get('gamma_0', 1.0)  # Default value for gamma_0
    self.gamma_1 = parameters.get('gamma_1', 1.0)  # Default value for gamma_1
    self.gamma_2 = parameters.get('gamma_2', 1.0)  # Default value for gamma_2
    self.gamma_3 = parameters.get('gamma_3', 1.0)  # Default value for gamma_3
    self.gamma_N = parameters.get('gamma_N', 1.0)  # Default value for gamma_N
    self.U_H = parameters.get('U_H', 1.0) # Default value for U_H
    self.U_X = parameters.get('U_X', 1.0) # Default value for U_X

    return

  def f_k(self, k):
    """Calculates the f(k) function."""
    kx, ky = k[:, 0], k[:, 1]
    return np.exp(1j * ky * self.a / np.sqrt(3)) * (1 + 2 * np.exp(-1j * 3 * ky * self.a / (2 * np.sqrt(3))) * np.cos(kx * self.a / 2))

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((6, 6, N_k), dtype=np.complex128)
    f = self.f_k(self.k_space)

    H_nonint[0, 1, :] = -self.gamma_0 * f       # -gamma_0 f         
    H_nonint[1, 0, :] = -self.gamma_0 * np.conj(f)   # gamma_0 f^*
    H_nonint[1, 2, :] = -self.gamma_1             # gamma_1     
    H_nonint[2, 1, :] = -self.gamma_1             # gamma_1
    H_nonint[0, 5, :] = -self.gamma_2             # gamma_2    
    H_nonint[5, 0, :] = -self.gamma_2             # gamma_2
    H_nonint[0, 3, :] = -self.gamma_3 * np.conj(f)  # gamma_3 f^*
    H_nonint[3, 0, :] = -self.gamma_3 * f        # gamma_3 f  
    H_nonint[2, 3, :] = -self.gamma_0 * f       # gamma_0 f     
    H_nonint[3, 2, :] = -self.gamma_0 * np.conj(f)   # gamma_0 f^*     
    H_nonint[3, 4, :] = -self.gamma_1             # gamma_1    
    H_nonint[4, 3, :] = -self.gamma_1             # gamma_1
    H_nonint[2, 5, :] = -self.gamma_3 * np.conj(f)  # gamma_3 f^*   
    H_nonint[5, 2, :] = -self.gamma_3 * f        # gamma_3 f      
    H_nonint[4, 5, :] = -self.gamma_0 * f       # gamma_0 f   
    H_nonint[5, 4, :] = -self.gamma_0 * np.conj(f)   # gamma_0 f^*  
    H_nonint[0, 0, :] = -self.gamma_N             # gamma_N
    H_nonint[3, 3, :] = -np.conj(self.gamma_N) # gamma_N^*

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros((6, 6, N_k), dtype=np.complex128)

    for l in range(6):
        for lp in range(6):
            H_int[l, l, :] += self.U_H * np.mean(exp_val[lp, lp, :]) #Hartree Term
            for kp in range(N_k): # kp = k'
                 H_int[lp, l, :] -= self.U_X * exp_val[l, lp, kp] # Exchange Term. U_X should be q = k' - k dependent

    return H_int


  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool = True) -> np.ndarray:
    """Generates the total Hartree-Fock Hamiltonian."""
    N_k = exp_val.shape[-1]
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
      return self.flatten(H_total)
    else:
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D), np.prod(self.D), self.k_space.shape[0]))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0], self.D[0], self.k_space.shape[0]))


