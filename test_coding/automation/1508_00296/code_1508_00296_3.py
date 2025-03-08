import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """Hartree-Fock Hamiltonian for a square-centered lattice with N atoms at vertices and B atoms at centers.
  
  Args:
    N_shell (int): Number shell in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float): Filling factor. Default is 0.5.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
    self.lattice = 'square'   # Using square lattice for k-space generation
    self.D = (2, 2)  # (site/orbital, spin)
    self.basis_order = {'0': 'site', '1': 'spin'}
    # Order for each flavor:
    # site: a (N atom), b (B atom)
    # spin: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Hopping parameters
    self.t_N = parameters.get('t_N', 1.0)  # Hopping parameter for N atoms
    self.t_B = parameters.get('t_B', 1.0)  # Hopping parameter for B atoms
    self.t_BN = parameters.get('t_BN', 0.5)  # Hopping parameter between N and B atoms
    self.Delta = parameters.get('Delta', 0.0)  # On-site energy for N atoms
    
    # Interaction parameters
    self.U_N = parameters.get('U_N', 3.0)  # Interaction between different spins on N site
    self.U_B = parameters.get('U_B', 3.0)  # Interaction between different spins on B site
    self.V_B = parameters.get('V_B', 0.65)  # Self-interaction on B site
    self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between N and B sites

  def compute_form_factor_nn(self):
    """Compute the form factor for nearest neighbor hopping on the square lattice."""
    return 2 * (np.cos(self.k_space[:, 0] * self.a) + np.cos(self.k_space[:, 1] * self.a))

  def compute_form_factor_nb(self):
    """Compute the form factor for hopping between N and B atoms."""
    return 2 * (np.cos((self.k_space[:, 0] + self.k_space[:, 1]) * self.a/2) + 
                np.cos((self.k_space[:, 0] - self.k_space[:, 1]) * self.a/2))

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian.

    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Compute form factors
    ff_nn = self.compute_form_factor_nn()  # For N-N and B-B hopping
    ff_nb = self.compute_form_factor_nb()  # For N-B hopping
    
    # N atom hopping terms (t_N) for both spins
    H_nonint[0, 0, 0, 0, :] = self.t_N * ff_nn  # N atom, spin up
    H_nonint[0, 1, 0, 1, :] = self.t_N * ff_nn  # N atom, spin down
    
    # B atom hopping terms (t_B) for both spins
    H_nonint[1, 0, 1, 0, :] = self.t_B * ff_nn  # B atom, spin up
    H_nonint[1, 1, 1, 1, :] = self.t_B * ff_nn  # B atom, spin down
    
    # N-B hopping terms (t_BN) for both spins
    H_nonint[0, 0, 1, 0, :] = self.t_BN * ff_nb  # N to B, spin up
    H_nonint[0, 1, 1, 1, :] = self.t_BN * ff_nb  # N to B, spin down
    H_nonint[1, 0, 0, 0, :] = self.t_BN * ff_nb  # B to N, spin up
    H_nonint[1, 1, 0, 1, :] = self.t_BN * ff_nb  # B to N, spin down
    
    # On-site energy (Delta) for N atoms
    H_nonint[0, 0, 0, 0, :] += self.Delta  # N atom, spin up
    H_nonint[0, 1, 0, 1, :] += self.Delta  # N atom, spin down
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)

    # Calculate the mean densities for each site and spin
    n_a_up = np.mean(exp_val[0, 0, 0, 0, :])   # <a†_{k,up} a_{k,up}>
    n_a_down = np.mean(exp_val[0, 1, 0, 1, :]) # <a†_{k,down} a_{k,down}>
    n_b_up = np.mean(exp_val[1, 0, 1, 0, :])   # <b†_{k,up} b_{k,up}>
    n_b_down = np.mean(exp_val[1, 1, 1, 1, :]) # <b†_{k,down} b_{k,down}>

    # U_B term: Interaction between different spins on B site
    H_int[1, 0, 1, 0, :] += self.U_B / self.N_k * n_b_down  # B site, spin up affected by spin down
    H_int[1, 1, 1, 1, :] += self.U_B / self.N_k * n_b_up    # B site, spin down affected by spin up
    
    # U_N term: Interaction between different spins on N site
    H_int[0, 0, 0, 0, :] += self.U_N / self.N_k * n_a_down  # N site, spin up affected by spin down
    H_int[0, 1, 0, 1, :] += self.U_N / self.N_k * n_a_up    # N site, spin down affected by spin up
    
    # 2V_B term: Self-interaction on B site (for all spin pairs)
    H_int[1, 0, 1, 0, :] += 2 * self.V_B / self.N_k * (n_b_up + n_b_down)  # B site, spin up
    H_int[1, 1, 1, 1, :] += 2 * self.V_B / self.N_k * (n_b_up + n_b_down)  # B site, spin down
    
    # V_BN term: Interaction between N and B sites
    # B occupation affecting N site
    H_int[0, 0, 0, 0, :] += self.V_BN / self.N_k * (n_b_up + n_b_down)  # N site, spin up
    H_int[0, 1, 0, 1, :] += self.V_BN / self.N_k * (n_b_up + n_b_down)  # N site, spin down
    
    # N occupation affecting B site
    H_int[1, 0, 1, 0, :] += self.V_BN / self.N_k * (n_a_up + n_a_down)  # B site, spin up
    H_int[1, 1, 1, 1, :] += self.V_BN / self.N_k * (n_a_up + n_a_down)  # B site, spin down
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """Generates the total Hartree-Fock Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
      return_flat (bool): Whether to return the flattened version of the Hamiltonian.

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
