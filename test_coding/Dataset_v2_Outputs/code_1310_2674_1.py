from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a multi-orbital, multi-spin system.

  Args:
    N_shell (int): Number of k-point shells to include.
    parameters (dict[str, Any]): Dictionary of model parameters.
        Must include 'epsilon', 't', and 'U'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int = 10, parameters: dict[str, Any] = {'epsilon': [0.0, 0.0], 't': [[1.0, 0.5], [0.5, 1.0]], 'U': [[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.25], [0.25, 0.5]]], [[[0.5, 0.25], [0.25, 0.5]], [[1.0, 0.5], [0.5, 1.0]]]]}, filling_factor: float = 0.5):
    self.lattice = 'cubic'  # Lattice type
    self.D = (2, 2)  # (orbital, spin)
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # 0: orbital. Order: orbital_0, orbital_1
    # 1: spin. Order: spin_up, spin_down


    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Temperature
    self.k_space = generate_k_space(self.lattice, N_shell)

    # Model parameters
    self.epsilon = np.array(parameters['epsilon'])  # Orbital energies
    self.t = np.array(parameters['t'])  # Hopping parameters
    self.U = np.array(parameters['U'])  # Interaction strengths


  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""

    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)  # Use self.D

    for alpha in range(self.D[0]):  # Orbital index
        for sigma in range(self.D[1]):  # Spin index
            H_nonint[alpha, sigma, alpha, sigma, :] = self.epsilon[alpha]  # On-site energy

            for beta in range(self.D[0]):  # Orbital index
                # Assuming t is k-independent for this example.  Modify as needed.
                H_nonint[alpha, sigma, beta, sigma, :] -= self.t[alpha, beta]

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)  # Use self.D

    for alpha in range(self.D[0]):
        for alpha_prime in range(self.D[0]):
            for beta in range(self.D[0]):
                for beta_prime in range(self.D[0]):
                    for sigma in range(self.D[1]):
                        for sigma_prime in range(self.D[1]):
                            # Hartree term
                            H_int[alpha_prime, sigma_prime, beta_prime, sigma_prime, :] += self.U[sigma, sigma_prime, alpha, alpha_prime, beta, beta_prime] * exp_val[alpha, sigma, beta, sigma, :]
                            # Fock term
                            H_int[alpha_prime, sigma_prime, beta, sigma, :] -= self.U[sigma, sigma_prime, alpha, alpha_prime, beta, beta_prime] * exp_val[alpha, sigma, beta_prime, sigma_prime, :]


    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
    """Generates the total Hamiltonian."""
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
    return exp_val.reshape(self.D + self.D + (self.k_space.shape[0],))



