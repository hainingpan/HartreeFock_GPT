from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_kx (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 'gamma_0', 'gamma_1', 'gamma_3', 'gamma_4', 'a', and interaction potential 'V'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'gamma_0': 1.0, 'gamma_1': 1.0, 'gamma_3': 1.0, 'gamma_4': 1.0, 'a': 1.0, 'V': lambda k: 1.0}, filling_factor: float=0.5): # Assuming V(k) is a function
    self.lattice = 'triangular'
    self.D = (4,)  # Number of orbitals
    self.basis_order = {'0': 'orbital'}
    # Order for each flavor:
    # 0: orbital 0, 1, 2, 3

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0 # Assuming zero temperature as a default
    self.k_space = generate_k_space(self.lattice, N_shell)

    # Model parameters
    self.gamma_0 = parameters['gamma_0']
    self.gamma_1 = parameters['gamma_1']
    self.gamma_3 = parameters['gamma_3']
    self.gamma_4 = parameters['gamma_4']
    self.a = parameters['a']  # Lattice constant
    self.V = parameters['V'] # Interaction potential as a function of k
    self.aM = self.a # lattice spacing parameter used for Area computation. Assumed to be the same as self.a, but can be different.

    return


  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((self.D[0], self.D[0], N_k), dtype=np.complex128) # Use complex dtype for f(k)
    kx = self.k_space[:, 0]
    ky = self.k_space[:, 1]

    f = np.exp(1j * ky * self.a / np.sqrt(3)) * (1 + 2 * np.exp(-1j * 3 * ky * self.a / (2 * np.sqrt(3))) * np.cos(kx * self.a / 2))

    H_nonint[0, 1, :] = self.gamma_0 * f      # Orbital 1 to 0
    H_nonint[1, 0, :] = self.gamma_0 * np.conj(f) # Orbital 0 to 1
    H_nonint[0, 2, :] = self.gamma_4 * f      # Orbital 2 to 0
    H_nonint[2, 0, :] = self.gamma_4 * np.conj(f) # Orbital 0 to 2
    H_nonint[0, 3, :] = self.gamma_3 * np.conj(f) # Orbital 3 to 0
    H_nonint[3, 0, :] = self.gamma_3 * f       # Orbital 0 to 3
    H_nonint[1, 2, :] = self.gamma_1          # Orbital 2 to 1
    H_nonint[2, 1, :] = self.gamma_1          # Orbital 1 to 2
    H_nonint[1, 3, :] = self.gamma_4 * f      # Orbital 3 to 1
    H_nonint[3, 1, :] = self.gamma_4 * np.conj(f) # Orbital 1 to 3
    H_nonint[2, 3, :] = self.gamma_0 * f      # Orbital 3 to 2
    H_nonint[3, 2, :] = self.gamma_0 * np.conj(f) # Orbital 2 to 3


    return H_nonint


  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros((self.D[0], self.D[0], N_k), dtype=np.complex128) # Using complex128 to be safe. V(k) might introduce imaginary parts.

    A = self.get_area()

    for l1 in range(self.D[0]): # 0, 1, 2, 3
      for l2 in range(self.D[0]): # 0, 1, 2, 3
        for k2 in range(N_k):
          # Hartree term
          hartree_sum = 0
          for k1 in range(N_k):
            for l in range(self.D[0]):
               hartree_sum += exp_val[l, l, k1] * self.V(0) # All expectation values with matching creation and annihilation operators are on the diagonal (H[l, l, k1])
          H_int[l2, l2, k2] += (1/A) * hartree_sum

          # Fock term
          fock_sum = 0
          for k1 in range(N_k): # Momentum k1 is summed over
            fock_sum += exp_val[l1, l2, k1] * self.V(np.array(self.k_space[k1])-np.array(self.k_space[k2]))
          H_int[l1, l2, k2] -= (1/A) * fock_sum # Minus sign for the Fock term

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
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.k_space.shape[0]))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0],self.D[0], self.k_space.shape[0]))

  def get_area(self):
    return (np.sqrt(3)/2)*self.aM**2 # Returns the area of the triangular unit cell.

