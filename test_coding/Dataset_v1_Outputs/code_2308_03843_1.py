from HF import *
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters:
        'mu': Chemical Potential
        'V': Interaction matrix
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'mu': 0.0, 'V':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.norb = 3 # Number of orbitals
    self.D = (2, self.norb) # Number of flavors identified: (spin, orbital)
    self.basis_order = {'0': 'spin', '1':'orbital'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: 0, 1, ... norb

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.mu = parameters['mu'] # Chemical potential
    self.V = parameters['V'] # Interaction strength

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    for s in range(self.D[0]): #spin
      for alpha in range(self.D[1]):
        for beta in range(self.D[1]):
          H_nonint[s, alpha, beta, :] =  self.h(self.k_space) # This needs to be defined based on the specific problem.
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # 2, norb, norb, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    for s in range(self.D[0]): # spin
      for alpha in range(self.D[1]): # orbital 1
        for beta in range(self.D[1]): # orbital 2
          for k in range(N_k):
            # Hartree Terms
            for gamma in range(self.D[1]): # orbital 3
              H_int[s, alpha, beta, k] += self.V * exp_val[s, gamma, gamma, k] # This expression for V may need to be modified depending on the specifics of the problem
            # Fock Terms
            for kp in range(N_k):
              H_int[s, alpha, beta, k] -= self.V * exp_val[s, alpha, beta, kp] # This expression for V may need to be modified depending on the specifics of the problem

    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) ->np.ndarray:
    """
      Generates the total Hartree-Fock Hamiltonian.

      Args:
          exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

      Returns:
          np.ndarray: The total Hamiltonian with shape (D, D, N_k).
    """
    N_k = exp_val.shape[-1]
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
    return exp_val.reshape((self.D + self.D + (self.Nk,)))
