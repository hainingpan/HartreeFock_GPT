from HF import *
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
  def __init__(self, N_shell: int=1, parameters: dict={'t':1.0, 'U':1.0, 'U_k': 1.0}, filling_factor: float=0.5): #LM Task: Modify parameter_kwargs
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2, N_shell) #LM Task: Define the tuple of flavors.
    self.basis_order = {'0': 'spin', '1': 'momentum'} #LM Task: Define the basis order.
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: k_0, k_1, ..., k_{N-1}

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell) #LM Task: Assuming generate_k_space is defined elsewhere
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.t = parameters['t'] # Hopping parameter
    self.U = parameters['U'] # Interaction strength
    self.U_k = parameters['U_k'] # Interaction strength in k space #LM Task: Add any other parameters needed.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32) #LM Task: Modify the size of the Hamiltonian.
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 0, :, 0, 0, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1])) # sum_{k} t_s c_{ks}^\dagger c_{ks}
    H_nonint[1, 1, :, 1, 1, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  # sum_{k} t_s c_{ks}^\dagger c_{ks}
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # 2, N_shell, 2, N_shell, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32) #LM Task: Modify the size of the Hamiltonian.

    # Calculate the mean densities for spin up and spin down
    # Hartree Terms
    H_int[0, :, 0, :, :] +=  self.U * np.mean(exp_val[0, :, 0, :, :], axis = 1) # sum_{p} U <c_{p,up}^\dagger c_{p,up}> c_{q,up}^\dagger c_{q,up}
    H_int[1, :, 1, :, :] += self.U * np.mean(exp_val[1, :, 1, :, :], axis = 1) # sum_{p} U <c_{p,down}^\dagger c_{p,down}> c_{q,down}^\dagger c_{q,down}
    H_int[0, :, 0, :, :] +=  self.U * np.mean(exp_val[1, :, 1, :, :], axis = 1) # sum_{p} U <c_{p,down}^\dagger c_{p,down}> c_{q,up}^\dagger c_{q,up}
    H_int[1, :, 1, :, :] +=  self.U * np.mean(exp_val[0, :, 0, :, :], axis = 1) # sum_{p} U <c_{p,up}^\dagger c_{p,up}> c_{q,down}^\dagger c_{q,down}

    #Fock Terms
    H_int[0, :, 1, :, :] += self.U_k * exp_val[0, :, 1, :, :] # sum_{p} U <c_{p,up}^\dagger c_{p,down}> c_{q,down}^\dagger c_{q,up}
    H_int[1, :, 0, :, :] += self.U_k * exp_val[1, :, 0, :, :] # sum_{p} U <c_{p,down}^\dagger c_{p,up}> c_{q,up}^\dagger c_{q,down}
    return H_int #LM Task: Add all terms in the interacting Hamiltonian

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
      return H_total #l1, s1, q1, ....k

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D,self.D, self.Nk))
