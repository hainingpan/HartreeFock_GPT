from HF import *

import numpy as np
from typing import Any
from scipy.linalg import block_diag

# Assuming generate_k_space is defined elsewhere and returns a (N_k, 2) array

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict[str, Any]={'gamma_0': 1.0, 'gamma_1': 1.0, 'gamma_2': 1.0, 'gamma_3': 1.0, 'gamma_N': 1.0, 'U_H':1.0, 'U_X':1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2, 3) # Spin, sublattice
    self.basis_order = {'0': 'spin', '1': 'sublattice'}
    # 0: spin. Order: up, down
    # 1: sublattice. Order: A, B, C


    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Temperature set to 0
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]
    self.a = 2.46  # Lattice constant in Angstroms

    # Model parameters with default values
    self.gamma_0 = parameters.get('gamma_0', 1.0)
    self.gamma_1 = parameters.get('gamma_1', 1.0)
    self.gamma_2 = parameters.get('gamma_2', 1.0)
    self.gamma_3 = parameters.get('gamma_3', 1.0)
    self.gamma_N = parameters.get('gamma_N', 1.0)  
    self.U_H = parameters.get('U_H', 1.0)
    self.U_X = parameters.get('U_X', 1.0)

    return


  def f_k(self, k):
    """Calculates the f(k) function."""
    kx, ky = k
    return np.exp(1j * ky * self.a / np.sqrt(3)) * (1 + 2 * np.exp(-1j * 3 * ky * self.a / (2 * np.sqrt(3))) * np.cos(kx * self.a / 2))


  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""

    Nk = self.k_space.shape[0]
    H_nonint = np.zeros(((2, 3, 2, 3, Nk)), dtype=np.complex128)


    f = self.f_k(self.k_space.T)  # Calculate f(k) for all k-points

    # Spin up block
    H_nonint[0, 0, 0, 1, :] = self.gamma_0 * f       # gamma_0 f
    H_nonint[0, 0, 0, 3, :] = self.gamma_3 * np.conj(f) + self.gamma_N  # gamma_3 f* + gamma_N
    H_nonint[0, 0, 0, 4, :] = self.gamma_2             # gamma_2

    H_nonint[0, 1, 0, 0, :] = self.gamma_0 * np.conj(f)  # gamma_0 f*
    H_nonint[0, 1, 0, 2, :] = self.gamma_1              # gamma_1

    H_nonint[0, 2, 0, 1, :] = self.gamma_1              # gamma_1
    H_nonint[0, 2, 0, 3, :] = self.gamma_0 * f       # gamma_0 f


    H_nonint[0, 3, 0, 0, :] = self.gamma_3 * f + np.conj(self.gamma_N)  # gamma_3 f + gamma_N*
    H_nonint[0, 3, 0, 2, :] = self.gamma_0 * np.conj(f)  # gamma_0 f*
    H_nonint[0, 3, 0, 4, :] = self.gamma_1    # gamma_1

    H_nonint[0, 4, 0, 0, :] = self.gamma_2 #gamma_2 term between A up and E up
    H_nonint[0, 4, 0, 3, :] = self.gamma_1             # gamma_1  between D up and E up
    H_nonint[0, 4, 0, 5, :] = self.gamma_0 * f # gamma0 f between E up and F up

    H_nonint[0, 5, 0, 2, :] = self.gamma_3 * f       # gamma_3 f between F up and C up
    H_nonint[0, 5, 0, 4, :] = self.gamma_0 * np.conj(f)       # gamma_0 f* between F up and E up


    # Spin down block (identical structure to spin up)
    H_nonint[1, :, 1, :, :] = H_nonint[0, :, 0, :, :]

    return H_nonint


  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val) #exp_val.reshape(self.D + self.D + (self.Nk,))
    Nk = self.k_space.shape[0]
    H_int = np.zeros(self.D + self.D + (Nk,), dtype=np.complex128)

    # Hartree term
    for l in range(self.D[1]):  # Iterate over sublattices
       for s in range(self.D[0]):  # Iterate over spins
            for lp in range(self.D[1]): # Iterate over sublattices
                n_lp_s = np.mean(exp_val[s, lp, s, lp,:]) # Mean for same spin, other sublattice
                H_int[s, l, s, l, :] += self.U_H * n_lp_s  # Hartree term


    # Exchange term
    for l in range(3):
      for s in range(2):
        for lp in range(3):
          for kp in range(Nk):
            H_int[s, l, s, lp,:] -= self.U_X * exp_val[s, lp, s, l, kp] # Assuming U_X is k-independent for now.
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
    """Generates the total Hamiltonian."""

    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int

    if flatten:
        return self.flatten(H_total)
    else:
        return H_total

  def flatten(self, ham):
        return ham.reshape((np.prod(self.D), np.prod(self.D), self.Nk))

  def expand(self, exp_val):
        return exp_val.reshape((self.D + self.D + (self.Nk,)))

