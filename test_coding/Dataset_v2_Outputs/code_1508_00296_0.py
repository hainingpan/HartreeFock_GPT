from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hamiltonian for a two-layer (N and B) system with spin.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'t_N': 1.0, 't_B': 1.0, 't_BN': 1.0, 'Delta': 0.0, 'U_N': 1.0, 'U_B': 1.0, 'V_B': 1.0, 'V_BN': 1.0}, filling_factor: float=0.5):
    self.lattice = 'square'
    self.D = (2, 2)  # layer, spin
    self.basis_order = {'0': 'layer', '1': 'spin'}
    # Order for each flavor:
    # 0: layer N, layer B
    # 1: spin up, spin down


    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Default temperature
    self.k_space = generate_k_space(self.lattice, N_shell)

    # Model parameters with default values
    self.t_N = parameters.get('t_N', 1.0)  # Hopping in layer N
    self.t_B = parameters.get('t_B', 1.0)  # Hopping in layer B
    self.t_BN = parameters.get('t_BN', 1.0)  # Hopping between layers N and B
    self.Delta = parameters.get('Delta', 0.0) # On-site energy difference
    self.U_N = parameters.get('U_N', 1.0)  # Interaction strength in layer N
    self.U_B = parameters.get('U_B', 1.0)  # Interaction strength in layer B
    self.V_B = parameters.get('V_B', 1.0)  # Intra-layer interaction in layer B
    self.V_BN = parameters.get('V_BN', 1.0)  # Inter-layer interaction between B and N

    self.aM = 1.0 # Assuming lattice constant to be 1

    return

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + (N_k,), dtype=np.float32)
    # Kinetic energy terms
    H_nonint[0, 0, :] = self.t_N * np.sum(np.exp(-1j*np.dot(self.k_space, n)) for n in self.NN_sites) + self.Delta # N layer, spin up and down
    H_nonint[0, 1, :] = self.t_BN * np.sum(np.exp(-1j*np.dot(self.k_space, n')) for n' in self.BN_sites) # N to B, spin up and down
    H_nonint[1, 0, :] = self.t_BN * np.sum(np.exp(1j*np.dot(self.k_space, n')) for n' in self.BN_sites) # B to N, spin up and down
    H_nonint[1, 1, :] = self.t_B * np.sum(np.exp(-1j*np.dot(self.k_space, n)) for n in self.NN_sites) # B layer, spin up and down

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + (N_k,), dtype=np.float32)

    # Calculate the mean densities
    n_B_up = np.mean(exp_val[1, 0, :])
    n_B_down = np.mean(exp_val[1, 1, :])
    n_N_up = np.mean(exp_val[0, 0, :])
    n_N_down = np.mean(exp_val[0, 1, :])


    # Interaction terms
    H_int[0, 0, :] = self.U_N / self.N_sites * n_N_down + self.V_BN / self.N_sites * (n_B_up + n_B_down) # N, up
    H_int[0, 1, :] = self.U_N / self.N_sites * n_N_up + self.V_BN / self.N_sites * (n_B_up + n_B_down) # N, down
    H_int[1, 0, :] = self.U_B / self.N_sites * n_B_down + 2*self.V_B / self.N_sites * n_B_down + self.V_BN / self.N_sites * (n_N_up+n_N_down)  # B, up
    H_int[1, 1, :] = self.U_B / self.N_sites * n_B_up + 2*self.V_B / self.N_sites * n_B_up + self.V_BN / self.N_sites * (n_N_up + n_N_down) # B, down

    return H_int

  # ... (rest of the class as before including generate_Htotal, flatten, and expand)

