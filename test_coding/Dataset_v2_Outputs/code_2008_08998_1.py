from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters 't' and 'U'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int = 10, parameters: dict = {'t': 1.0, 'U': 1.0}, filling_factor: float = 0.5):
    self.lattice = 'triangular'
    self.D = (2,)
    self.basis_order = {'0': 'spin'}
    # Order for each flavor:
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Temperature
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]
    #N_k = 2*N_shell*(N_shell+1) +1 for a triangular lattice

    # Model parameters
    self.t = parameters.get('t', 1.0)  # Hopping parameter
    self.U = parameters.get('U', 1.0)  # Interaction strength
    self.aM = 1 # Setting the lattice constant to 1

    return

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""
    H_nonint = np.zeros((self.D[0], self.D[0], self.Nk), dtype=np.float32)

    # Kinetic energy term for both spins
    for s in range(self.D[0]):
        for k in range(self.Nk): # Iterating over all k-points.
            H_nonint[s, s, k] = -2 * self.t * (np.cos(self.k_space[k, 0]) + np.cos(self.k_space[k, 1]) + np.cos(self.k_space[k,0] - self.k_space[k, 1])) # Assuming nearest neighbor hopping.

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    H_int = np.zeros((self.D[0], self.D[0], self.Nk), dtype=np.float32)

    # Hartree and Fock terms
    # Simplified interaction: only considering on-site interaction for now.
    for p in range(self.Nk):
        for s in range(self.D[0]):
            for sp in range(self.D[0]):
                # Hartree Term
                H_int[sp, sp, p] += self.U * exp_val[s, s, p] / self.Nk

                # Fock Term
                H_int[sp, s, p] -= self.U * exp_val[s, sp, p] / self.Nk

    return H_int


  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool = True) -> np.ndarray:
    """Generates the total Hartree-Fock Hamiltonian."""
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
    return exp_val.reshape((self.D[0], self.D[0], self.Nk))


