import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'V':1.0, 'mu': 0.0}, filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2, 2) # Number of orbitals and spins.
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # Order for each flavor:
    # 0: orbital 0, orbital 1, ...
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0 # Assume T = 0
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.V = parameters['V'] # Interaction strength
    self.mu = parameters['mu'] # Chemical potential

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    # Kinetic energy
    # They are identical in this case, but we keep them separate for clarity
    # Since there is no k dependence in the non-interacting Hamiltonian, we will tile it later
    H_nonint[0, 0, 0, 0, 0, 0, 0] = - self.mu  
    H_nonint[1, 1, 0, 0, 1, 1, 0] = - self.mu
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # 2, 2, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Calculate the mean densities for spin up and spin down
    for alpha in range(self.D[0]):
      for beta in range(self.D[0]):
        for k in range(N_k):
          for spin in range(self.D[1]):
            for gamma in range(self.D[0]):
              for k_prime in range(N_k):
                for spin_prime in range(self.D[1]):
                  # Hartree Terms
                  H_int[alpha, beta, 0, spin, alpha, beta, 0, spin] += self.V * exp_val[gamma, gamma, k_prime, spin_prime] # Interaction of spin up with average spin down density
                  # Fock Terms
                  H_int[alpha, beta, k, spin, alpha, beta, k, spin] += -self.V*exp_val[alpha, beta, k_prime, spin]
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
    # Since there is no k dependence in the non-interacting Hamiltonian, we tile it here
    H_nonint = np.tile(H_nonint, (1, 1, 1, 1, 1, 1, N_k))
    H_total = H_nonint + H_int
    if flatten:
      return self.flatten(H_total)
    else:
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0], self.D[0], self.D[1], self.D[1], self.Nk))
