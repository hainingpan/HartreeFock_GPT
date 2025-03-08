import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a square-centered lattice with N atoms at vertices 
  and B atoms at centers.
  
  Args:
    N_shell (int): Number of shells in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float): Filling factor of the system. Default is 0.5.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
    self.lattice = 'square'  # Square-centered lattice
    self.D = (2, 2)  # (atom_type, spin)
    self.basis_order = {'0': 'atom_type', '1': 'spin'}
    # atom_type: 0=N (vertices), 1=B (centers)
    # spin: 0=up, 1=down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Hopping parameters
    self.t_N = parameters.get('t_N', 1.0)  # Hopping parameter for N atoms
    self.t_B = parameters.get('t_B', 1.0)  # Hopping parameter for B atoms
    self.t_BN = parameters.get('t_BN', 0.5)  # Hopping parameter between B and N atoms
    self.Delta = parameters.get('Delta', 0.0)  # On-site energy for N atoms
    
    # Interaction parameters
    self.U_N = parameters.get('U_N', 3.0)  # On-site interaction for N atoms
    self.U_B = parameters.get('U_B', 3.0)  # On-site interaction for B atoms
    self.V_B = parameters.get('V_B', 0.65)  # Interaction between B atoms
    self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between B and N atoms

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Hopping term for N atoms at vertices (a operators)
    for spin in range(2):
      H_nonint[0, spin, 0, spin, :] = self.t_N * (2 * np.cos(self.k_space[:, 0]) + 2 * np.cos(self.k_space[:, 1])) + self.Delta
    
    # Hopping term for B atoms at centers (b operators)
    for spin in range(2):
      H_nonint[1, spin, 1, spin, :] = self.t_B * (2 * np.cos(self.k_space[:, 0]) + 2 * np.cos(self.k_space[:, 1]))
    
    # Hopping between B and N atoms
    for spin in range(2):
      phase = np.exp(-1j * (self.k_space[:, 0] + self.k_space[:, 1]) / 2)
      H_nonint[0, spin, 1, spin, :] = self.t_BN * phase
      H_nonint[1, spin, 0, spin, :] = self.t_BN * np.conj(phase)
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Calculate mean densities
    n_N_up = np.mean(exp_val[0, 0, 0, 0, :])  # <a†_up a_up>
    n_N_down = np.mean(exp_val[0, 1, 0, 1, :])  # <a†_down a_down>
    n_B_up = np.mean(exp_val[1, 0, 1, 0, :])  # <b†_up b_up>
    n_B_down = np.mean(exp_val[1, 1, 1, 1, :])  # <b†_down b_down>
    
    # Interaction between different spins on B sites (U_B term)
    H_int[1, 0, 1, 0, :] += self.U_B * n_B_down  # For spin up
    H_int[1, 1, 1, 1, :] += self.U_B * n_B_up  # For spin down
    
    # Interaction between different spins on N sites (U_N term)
    H_int[0, 0, 0, 0, :] += self.U_N * n_N_down  # For spin up
    H_int[0, 1, 0, 1, :] += self.U_N * n_N_up  # For spin down
    
    # Interaction between all spins on B sites (2V_B term)
    H_int[1, 0, 1, 0, :] += 2 * self.V_B * (n_B_up + n_B_down)  # For spin up
    H_int[1, 1, 1, 1, :] += 2 * self.V_B * (n_B_up + n_B_down)  # For spin down
    
    # Interaction between B and N sites (V_BN term)
    H_int[0, 0, 0, 0, :] += self.V_BN * (n_B_up + n_B_down)  # N up affected by B
    H_int[0, 1, 0, 1, :] += self.V_BN * (n_B_up + n_B_down)  # N down affected by B
    H_int[1, 0, 1, 0, :] += self.V_BN * (n_N_up + n_N_down)  # B up affected by N
    H_int[1, 1, 1, 1, :] += self.V_BN * (n_N_up + n_N_down)  # B down affected by N
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
      return_flat (bool): Whether to return the Hamiltonian in flattened form.

    Returns:
      np.ndarray: The total Hamiltonian.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
