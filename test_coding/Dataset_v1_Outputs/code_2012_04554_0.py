from HF import *
import numpy as np
from typing import Any

def generate_k_space(symmetry: str='square', N_shell: int=1):
  """Generates a k-space grid for a given lattice symmetry.
  
  Args:
    symmetry (str, optional): The symmetry of the lattice.
    Can be 'square' or 'triangular'. Defaults to 'square'.
    N_shell (int, optional): Number of shells in the Brillouin zone. Defaults to 1.

  Returns:
    np.ndarray: An array of k-points with shape (N_k, 2).
  """
  if symmetry == 'square':
    N_kx = 2*(N_shell+1)
    k_space = np.zeros((N_kx, N_kx, 2))
    for i in range(N_kx):
      for j in range(N_kx):
        kx = (2 * np.pi * i / N_kx) - np.pi
        ky = (2 * np.pi * j / N_kx) - np.pi
        k_space[i, j, :] = [kx, ky]
    return k_space.reshape((N_kx * N_kx, 2))
  elif symmetry == 'triangular':
    raise NotImplementedError
  else:
    raise ValueError("Invalid lattice symmetry provided. Must be 'square' or 'triangular'.")

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'t_nn':1.0, 'U_0':1.0}, filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2,) # LM Task: Define the tuple that contains the dimensions of the flavors.
    self.basis_order = {'0': 'spin'}
    # Order for each flavor:
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0 # LM Task: Assume T = 0
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell=N_shell)
    self.N_sites = 1.0 # LM Task: Define the number of sites, used to normalize sums over k.
    self.Nk = self.k_space.shape[0]
    # N_kx = 2*(N_shell+1) for a square lattice

    # LM Task: Define all the parameters in the Hamiltonian that do not depend on exp_val here:
    # Model parameters
    self.t_nn = parameters['t_nn'] # Nearest neighbor hopping
    self.U_0 = parameters['U_0'] # On site interaction strength

    # LM Task: Define interaction parameters, such as U(k).
    self.U_k = np.zeros((self.Nk,))
    for i, k in enumerate(self.k_space):
      self.U_k[i] = self.U_0 # Assuming U(k) = U_0 = constant for now

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
    H_nonint[0, 0, :] = -2 * self.t_nn * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    H_nonint[1, 1, :] = -2 * self.t_nn * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    # LM Task: Update the interacting part of the Hamiltonian.
    exp_val = self.expand(exp_val) # 2, 2, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros((self.D[0], self.D[0], N_k), dtype=np.float32)
    for s in range(self.D[0]):
      for sp in range(self.D[0]):
        for k in range(N_k):
          # Hartree Terms
          H_int[sp, sp, k] += self.U_0 * np.mean(exp_val[s, s, :]) / self.N_sites # k indep
          # Fock Terms
          for k1 in range(N_k):
            H_int[s, sp, k] -= self.U_k[k - k1] * exp_val[s, sp, k1] / self.N_sites
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
      return H_total #l1, s1, q1, ....k

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0],self.D[0], self.Nk))
