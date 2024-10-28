from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters 't_s', 'U', and 'n'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'t_s':1.0, 'U':1.0, 'n': [1.0]}, filling_factor: float=0.5):
    self.lattice = 'triangular'  # Lattice symmetry ('square' or 'triangular').
    self.D = (1,)  # Number of flavors (levels).
    self.basis_order = {'0': 'level'}
    # Order for each flavor:
    # 0: level s

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0 # Assumed to be zero temperature.
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]


    # Model parameters
    self.t_s = parameters.get('t_s', 1.0)  # Hopping parameters, default to 1.0 if not provided.
    self.U_n = parameters.get('U', np.ones(self.k_space.shape[0])) # Interaction strengths. Assumed constant if scalar.
    self.n_vec = parameters.get('n', np.array([0.0, 0.0])) # Vectors n.


    self.aM = 1.0  # Lattice constant (Used for area). Default to 1.0

    # Calculate U(k) based on U(n)
    self.U_k = np.zeros(self.Nk, dtype=complex)
    for k_idx in range(self.Nk):
        k = self.k_space[k_idx]
        for n_idx, n in enumerate(self.n_vec):
           self.U_k[k_idx] += self.U_n[n_idx] * np.exp(-1j * np.dot(k, n))
    self.U_0 = np.sum(self.U_n) # U(k=0)


    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.Nk
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)

    # Kinetic energy term
    for k_idx in range(N_k):
        k = self.k_space[k_idx]
        E_s = 0
        for n_idx, n in enumerate(self.n_vec):
            E_s += self.t_s[n_idx] * np.exp(-1j * np.dot(k, n))
        H_nonint[0, 0, k_idx] = E_s # Assuming s is the only level
    return H_nonint


  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D, D, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # D, D, N_k
    N_k = self.Nk
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)

    for s in range(self.D[0]): # Loop over all levels
        for k2_idx in range(N_k):
            # Hartree Term
            hartree_term = 0.0
            for s_prime in range(self.D[0]):
                for k1_idx in range(N_k):
                   hartree_term += (1/N_k) * self.U_0 * exp_val[s_prime, s_prime, k1_idx] # <c_s'(k1)^\dagger c_s'(k1)>
            H_int[s, s, k2_idx] += hartree_term

            # Fock Term
            fock_term = 0.0
            for s_prime in range(self.D[0]):
                for k1_idx in range(N_k):
                    fock_term += -(1/N_k) * self.U_k[(k1_idx-k2_idx)%N_k] * exp_val[s_prime, s, k1_idx] # <c_s'(k1)^\dagger c_s(k1)>
            H_int[s, s, k2_idx] += fock_term
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The total Hamiltonian with shape (D, D, N_k) or flattened if `flatten=True`.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
      return self.flatten(H_total)
    else:
      return H_total #l1, s1, q1, ....k

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D + self.D + (self.Nk,)))

