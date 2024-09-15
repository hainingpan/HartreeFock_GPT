import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 't' and 'U'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'t':1.0, 'U':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2,) # Number of flavors identified.
    self.basis_order = {'0': 'spin'}
    # Order for each flavor:
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell=N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.t = parameters['t'] # Hopping parameter
    self.U = parameters['U'] # Interaction strength
    # Assuming U(n) = U*delta(n,0)
    self.aM = 1.0 # Lattice constant.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((self.D[0], self.D[0], N_k), dtype=np.float32)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 0, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    H_nonint[1, 1, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
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
    # Hartree Terms
    for s in range(2):
      for sp in range(2):
        H_int[s, sp, :] += self.U/N_k * np.sum(np.diag(exp_val[:, :, :]), axis=0)
    # Fock Terms
    for k in range(N_k):
      for q in range(N_k):
        for s in range(2):
          for sp in range(2):
            H_int[s, sp, q] -= self.U/N_k*exp_val[s,sp,k]

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
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.k_space.shape[0]))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0],self.D[0], self.k_space.shape[0]))
