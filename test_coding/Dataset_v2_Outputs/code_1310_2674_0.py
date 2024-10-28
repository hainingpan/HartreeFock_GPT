from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a two-orbital model on a 3D cubic lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters 'epsilon', 't', and 'U'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'epsilon': [0.0, 0.0], 't': [[1.0, 0.5], [0.5, 1.0]], 'U': [[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 1.0], [1.0, 0.5]]], [[[0.5, 1.0], [1.0, 0.5]], [[1.0, 0.5], [0.5, 1.0]]]]}, filling_factor: float=0.5): # LM Task: Modified init
    self.lattice = 'cubic'  # Lattice symmetry ('cubic').
    self.D = (2, 2)          # Number of orbitals and spins.
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # Order for each flavor:
    # 0: orbital 0, orbital 1
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0             # Temperature.
    self.k_space = generate_k_space(self.lattice, N_shell) # LM Task: Call generate_k_space

    # Model parameters
    self.epsilon = np.array(parameters['epsilon'])  # On-site energies for each orbital.
    self.t = np.array(parameters['t'])          # Hopping parameters between orbitals.
    self.U = np.array(parameters['U'])          # Interaction strengths.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)  # Initialize with correct dimensions

    for alpha in range(self.D[0]):
      for sigma in range(self.D[1]):
        # On-site energy term
        H_nonint[alpha, sigma, alpha, sigma, :] = self.epsilon[alpha]

        for beta in range(self.D[0]):
          # Hopping term
          # Assuming t^{\alpha\beta}_{ij} are provided and Fourier transformed beforehand
          t_k = t_alpha_beta # LM TASK: implement Fourier transform here and replace this variable.
          H_nonint[alpha, sigma, beta, sigma, :] -= t_k # Minus sign from Hamiltonian

    return H_nonint


  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D, D, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    for alpha in range(self.D[0]):
      for alpha_prime in range(self.D[0]):
        for beta in range(self.D[0]):
          for beta_prime in range(self.D[0]):
            for sigma in range(self.D[1]):
              for sigma_prime in range(self.D[1]):
                # Hartree term
                expectation_val_Hartree = exp_val[alpha, sigma, beta, sigma, :]
                H_int[alpha_prime, sigma_prime, beta_prime, sigma_prime, :] += self.U[sigma, sigma_prime, alpha, alpha_prime, beta, beta_prime] * expectation_val_Hartree

                # Fock term
                expectation_val_Fock = exp_val[alpha, sigma, beta_prime, sigma_prime, :]
                H_int[alpha_prime, sigma_prime, beta, sigma, :] -= self.U[sigma, sigma_prime, alpha, alpha_prime, beta, beta_prime] * expectation_val_Fock

    return H_int


  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray: # LM Task: Modified generate_Htotal
    """
      Generates the total Hartree-Fock Hamiltonian.

      Args:
        exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

      Returns:
        np.ndarray: The total Hamiltonian with shape (D, D, N_k) or flattened.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
      return self.flatten(H_total)
    else:
      return H_total

  def flatten(self, ham): # LM Task: Modified flatten function.
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.k_space.shape[0])) # LM Task: Modified flatten function.

  def expand(self, exp_val): # LM Task: Modified expand function.
    return exp_val.reshape(self.D + self.D + (self.k_space.shape[0],)) # LM Task: Modified expand function.

