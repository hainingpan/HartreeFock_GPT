from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'t':1.0, 'U':1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2,) # spin
    self.basis_order = {'0': 'spin'}
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]
    self.N = self.Nk # Assuming N = Nk for simplicity here. Clarify if different.

    # Model parameters
    self.t = parameters.get('t', 1.0) # Hopping parameter.
    self.U = parameters.get('U', 1.0) # Interaction strength.
    # Assuming U(k) = U for simplicity here.  Needs modification for k-dependent interaction
    self.a = 1.0 # Lattice constant, defaulting to 1.0.  Adjust as needed.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((self.D[0], self.D[0], self.Nk), dtype=np.float32)

    for s in range(self.D[0]):  # Iterate over spin flavors
      for k_idx in range(self.Nk):
        k = self.k_space[k_idx]
        # Define nearest neighbor vectors for the triangular lattice
        n_vectors = np.array([[1, 0], [-1/2, np.sqrt(3)/2], [-1/2, -np.sqrt(3)/2]])
        E_s_k = 0
        for n in n_vectors:
            E_s_k += self.t * np.exp(-1j * np.dot(k, n))
        H_nonint[s, s, k_idx] = -E_s_k.real # Taking the real part, assuming Hermitian Hamiltonian
    return H_nonint


  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D, D, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    #exp_val = self.expand(exp_val) # Expand exp_val to (s, s', k)
    N_k = exp_val.shape[-1]
    H_int = np.zeros((self.D[0], self.D[0], N_k), dtype=np.float32)

    for s in range(self.D[0]):
      for sp in range(self.D[0]):
        for k3_idx in range(self.Nk):
          # Hartree Term
          n_s = np.mean(exp_val[s, s, :])  # <c_{k1,s}^\dagger c_{k1,s}>
          H_int[sp, sp, k3_idx] += (self.U / self.N) * n_s


        for k2_idx in range(self.Nk):
          # Fock Term
          n_s_sp = np.mean(exp_val[s, sp, :]) # <c_{k1,s}^\dagger c_{k1,sp}>
          H_int[sp, s, k2_idx] -= (self.U/self.N) * n_s_sp # Assuming U(k) is constant. Modify for k dependence.

    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D, D, N_k).

    Returns:
      np.ndarray: The total Hamiltonian with shape (D, D, N_k) or flattened if flatten=True.
    """
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
      return exp_val.reshape(self.D + self.D + (self.Nk,))


