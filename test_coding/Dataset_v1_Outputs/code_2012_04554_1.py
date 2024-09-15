from HF import *
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_kx (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 't_s(n)' and 'U(n)'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
    n (str | None, optional): Parameter used in chemical potential calculation. Defaults to None.
  """
  def __init__(self, N_kx: int=10, parameters: dict={'t_s(n)': {(0, 0): -1.0}, 'U(n)': {(0, 0): 1.0}}, filling_factor: float=0.5): # LM Task: Modify parameter_kwargs.
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2,) # LM Task: Define this tuple. Number of flavors identified.
    self.basis_order = {'0': 'spin'}
    # Order for each flavor:
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.n = n # Number of particles in the system.
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice
    self.Nk = self.k_space.shape[0]
    # Model parameters
    self.t_s = parameters['t_s(n)'] # Hopping parameter
    self.U = parameters['U(n)'] # Interaction strength
    self.a = 1 # LM Task: Define the lattice constant, used for the area.

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
    for s in range(self.D[0]):
        H_nonint[s, s, :] = -np.sum([self.t_s[n]*np.exp(-1j*np.dot(self.k_space, np.array(n))) for n in self.t_s.keys()], axis=0)  
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
    for s in range(self.D[0]):
        for sp in range(self.D[0]):
            H_int[s, sp, :] += 1/self.Nk*self.U[(0, 0)]*np.mean(exp_val[s, s, :])

    # Fock Terms
    for s in range(self.D[0]):
        for sp in range(self.D[0]):
            for k in range(self.Nk):
                H_int[s, sp, k] += -1/self.Nk*np.sum([self.U[tuple(np.array(k_1)-np.array(k))] * exp_val[s, sp, k_1] for k_1 in self.k_space])
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
      return flatten(H_total)
    else:
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0],self.D[0], self.Nk))
