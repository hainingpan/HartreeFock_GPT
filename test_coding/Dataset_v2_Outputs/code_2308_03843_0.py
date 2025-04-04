from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells. Determines the k-point grid size.
    parameters (dict): Dictionary containing model parameters. Requires 'V', 'mu', and hopping 'h'.
                       'V' should be a function that takes a momentum difference as argument and returns the interaction potential.
                       'h' should be a function of k that returns a matrix of shape (Nb, Nb).
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict[str, Any]={'V': lambda x: 1.0, 'mu': 0.0, 'h': lambda k: np.eye(2)}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.Nb = 2 # Number of orbitals (assuming 2 for this example, update if needed)
    self.D = (self.Nb, 2)  # (orbital, spin)
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # Order for each flavor:
    # 0: orbital 0, orbital 1, ...
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Temperature is assumed to be zero
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]

    # Model parameters
    self.V = parameters['V']  # Interaction potential function
    self.mu = parameters['mu'] # Chemical potential
    self.h = parameters['h'] # Hopping function.


    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (Nb, Nb, spin, Nk).
    """
    H_nonint = np.zeros((self.Nb, self.Nb, 2, self.Nk), dtype=np.complex128)
    for spin in range(2):  # spin: 0 (up), 1 (down)
        for k_idx in range(self.Nk):
            k = self.k_space[k_idx]
            H_nonint[:, :, spin, k_idx] = self.h(k) - self.mu * np.eye(self.Nb)

    return H_nonint


  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
        exp_val (np.ndarray): Expectation value array with shape (Nb, Nb, spin, Nk).

    Returns:
        np.ndarray: The interacting Hamiltonian with shape (Nb, Nb, spin, Nk).
    """

    N_k = self.k_space.shape[0] # exp_val.shape[-1] is not reliable as exp_val is flattened
    H_int = np.zeros((self.Nb, self.Nb, 2, N_k), dtype=np.complex128)
    # exp_val has shape (Nb, Nb, 2, Nk)
   
    for spin in range(2): # 0: spin up, 1: spin down
      for alpha in range(self.Nb):
        for beta in range(self.Nb):
          for k_idx in range(self.Nk):
              k = self.k_space[k_idx]

              # Hartree Term
              hartree_sum = 0.0
              for gamma in range(self.Nb):
                  for k_prime_idx in range(N_k):
                      for spin_prime in range(2):
                        hartree_sum += self.V(0) * exp_val[gamma, gamma, spin_prime, k_prime_idx]
              H_int[alpha, beta, spin, k_idx] += hartree_sum * (alpha == beta)

              # Fock Term
              fock_sum = 0.0
              for k_prime_idx in range(N_k):
                  k_prime = self.k_space[k_prime_idx]
                  fock_sum += self.V(k - k_prime) * exp_val[alpha, beta, spin, k_prime_idx]
              H_int[alpha, beta, spin, k_idx] -= fock_sum # Minus sign for the fock term.

    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
    """Generates the total Hartree-Fock Hamiltonian.

    Args:
        exp_val (np.ndarray): Expectation value array with shape (Nb, Nb, spin, Nk) or flattened version.
        flatten (bool, optional): Whether to flatten the output. Defaults to True.

    Returns:
        np.ndarray: The total Hamiltonian, flattened or unflattened.
    """
    exp_val = exp_val.reshape((self.Nb, self.Nb, 2, self.Nk))
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
    return exp_val.reshape((self.Nb,self.Nb, 2, self.Nk))



