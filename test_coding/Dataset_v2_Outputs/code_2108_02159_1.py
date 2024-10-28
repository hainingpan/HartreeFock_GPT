from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular moiré lattice.

  Args:
    N_shell (int): Number of shells for generating k-points.
    parameters (dict): Dictionary of model parameters.
      Should include 'hbar', 'm_star', 'V_M', 'phi', and interaction parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'hbar':1.0, 'm_star':1.0, 'V_M':1.0, 'phi':0.0, 'V_int':1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2, 6) # (spin, reciprocal lattice vectors)
    self.basis_order = {'0': 'spin', '1': 'reciprocal_lattice_vector'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: b0, b1, b2, b3, b4, b5

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0
    self.k_space = generate_k_space(self.lattice, N_shell)
    #Any other problem specific parameters.
    self.aM = 1.0 # Moiré lattice constant. LM Task: Define this value.

    # Model parameters
    self.hbar = parameters['hbar'] # Reduced Planck constant
    self.m_star = parameters['m_star'] # Effective mass
    self.V_M = parameters['V_M'] # Moiré modulation strength
    self.phi = parameters['phi'] # Moiré modulation shape parameter
    self.V_int = parameters['V_int']  # Interaction strength


  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    for s in range(self.D[0]): # Spin
        for b in range(self.D[1]): # Reciprocal lattice vector
            for b_prime in range(self.D[1]): # Reciprocal lattice vector
                for k in range(N_k):
                    # Kinetic energy term
                    H_nonint[s, b, s, b_prime, k] += - (self.hbar**2/(2*self.m_star)) * np.dot(self.k_space[k,:] + reciprocal_lattice_vectors[b,:], self.k_space[k,:] + reciprocal_lattice_vectors[b,:]) * (b == b_prime)

                    # Moiré potential term
                    V_j = self.V_M * np.exp(1j * (-1)**(np.arange(6)) * self.phi) # Array of V_j values.

                    for j in range(6):
                        H_nonint[s, b, s, b_prime, k] += V_j[j] * (reciprocal_lattice_vectors[j,:] == reciprocal_lattice_vectors[b,:] - reciprocal_lattice_vectors[b_prime,:])

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)


    for alpha in range(self.D[0]):
        for beta in range(self.D[0]):
            for b in range(self.D[1]):
                for b_prime in range(self.D[1]):
                    for k in range(N_k):
                      # Hartree term
                      for alpha_prime in range(self.D[0]):
                        for b_double_prime in range(self.D[1]):
                          for k_prime in range(N_k):
                            H_int[alpha, b, beta, b_prime, k] += (alpha==beta)/self.aM * self.V_int * np.sum(exp_val[alpha_prime, b + b_double_prime, alpha_prime, b_prime + b_double_prime, k_prime])

                      # Fock term
                      for b_double_prime in range(self.D[1]):
                          for k_prime in range(N_k):
                            H_int[alpha, b, beta, b_prime, k] -= 1/self.aM * self.V_int * exp_val[alpha, b + b_double_prime, beta, b_prime + b_double_prime, k_prime]
    return H_int




  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
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
      return exp_val.reshape(self.D + self.D + (self.k_space.shape[0],))


