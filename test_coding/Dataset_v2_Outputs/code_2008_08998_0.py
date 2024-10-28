from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters 't', 'U', and 'n'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'t': 1.0, 'U': 1.0, 'n': [1.0]}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2,)  # spin up, spin down
    self.basis_order = {'0': 'spin'}

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Assuming T=0
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]

    # Model parameters
    self.t = parameters.get('t', 1.0)  # Hopping parameter. Default to 1.0
    self.U_n = parameters.get('n', np.array([1.0])) # Interaction strengths in real space. Default to [1.0].
    self.N = len(self.U_n) # Maximum extent of interaction. Default to extent of U_n = 1.
    self.aM = 1 # Assuming the lattice constant is 1.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""

    H_nonint = np.zeros((self.D[0], self.D[0], self.Nk), dtype=np.float32)

    # Kinetic energy terms
    for s in range(self.D[0]):
        H_nonint[s, s, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]) + np.cos(self.k_space[:,0] - self.k_space[:,1]))

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""

    exp_val = self.expand(exp_val)
    H_int = np.zeros((self.D[0], self.D[0], self.Nk), dtype=np.float32)

    # Compute U(k)
    U_k = np.fft.fft(self.U_n, n=self.Nk, axis=0)

    for p in range(self.Nk):
      for s in range(self.D[0]):
        for sp in range(self.D[0]):
          # Hartree term
          H_int[sp, sp, p] += (1/self.Nk)*np.sum(U_k*exp_val[s, s, :])
          # Fock term
          H_int[sp, s, p] -= (1/self.Nk)*U_k[p]*exp_val[s, sp, p]

    return H_int


  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
    """Generates the total Hartree-Fock Hamiltonian."""
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
        return self.flatten(H_total)
    else:
        return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0],self.D[0], self.Nk))



